#include "fugu/Simulation.hpp"
#include "fugu/MpiComm.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace fugu {

bool Simulation::init(MPI_Comm comm, const std::string& config_file) {
    // 1. Chargement de la configuration JSON
    if (!config.load_from_json(config_file)) {
        return false;
    }

    int my_rank;
    MPI_Comm_rank(comm, &my_rank);
    if (my_rank == 0) {
        config.print();
    }

    // 2. Initialisation du maillage partitionné
    bool periodic_x = (config.bc.left.type == BCType::PERIODIC || config.bc.right.type == BCType::PERIODIC);
    bool periodic_y = (config.bc.bottom.type == BCType::PERIODIC || config.bc.top.type == BCType::PERIODIC);
    mesh.init(comm, config.nx, config.ny, config.lx, config.ly, periodic_x, periodic_y);

    // 3. Allocation des champs décalés MAC avec halos
    fields.allocate(mesh);

    // 4. Initialisation du module d'interface et des conditions initiales
    tracker.init(config);
    tracker.set_initial_condition(mesh, config.init_cond, fields);

    // Initialisation des conditions aux limites initiales
    MpiComm::apply_boundary_conditions(mesh, config.bc, fields, false);

    // Initialisation du canal amont pour profil POISEUILLE_STEP afin de garantir la continuité
    if (config.bc.left.type == BCType::DIRICHLET && config.bc.left.profile == "POISEUILLE_STEP") {
        double y_step = config.bc.left.y_step;
        double h = mesh.ly - y_step;
        double u_mean = config.bc.left.u_mean;
        for (int j = 1; j <= mesh.ny_local; ++j) {
            double y = mesh.yc(j);
            double u_val = 0.0;
            if (y >= y_step) {
                u_val = 6.0 * u_mean * (y - y_step) * (mesh.ly - y) / (h * h);
            }
            int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
            for (int i = 1; i <= u_nx; ++i) {
                double x = mesh.xu(i);
                if (x <= 4.0 + 1e-9) {
                    fields.u[mesh.u_idx(i, j)] = u_val;
                }
            }
        }
        MpiComm::sync_u(mesh, fields.u);
    }

    // Initialisation du champ thermique si activé
    if (config.enable_energy) {
        for (int j = 1; j <= mesh.ny_local; ++j) {
            for (int i = 1; i <= mesh.nx_local; ++i) {
                double frac = mesh.xc(i) / mesh.lx;
                fields.T[mesh.cell_idx(i, j)] = config.bc.left.T_val + frac * (config.bc.right.T_val - config.bc.left.T_val);
            }
        }
        MpiComm::sync_cell_scalar(mesh, fields.T);
        MpiComm::apply_thermal_boundary_conditions(mesh, config.bc, fields);
    }

    // 5. Initialisation du solveur Navier-Stokes et des préconditionneurs
    ns_solver.init(mesh, config);
    ns_solver.assemble_preconditioner_blocks(fields, config.dt);

    precond = PreconditionerFactory::create(config.solver.precond_type);
    precond->init(&ns_solver);

    linear_solver = SolverFactory::create(config.solver.type);
    linear_solver->init(config.solver);

    // 6. Initialisation de l'exportateur VTK et diagnostics CSV
    vtk_writer.init(mesh, config.output.output_dir, config.output.case_name);

    return true;
}

double Simulation::compute_time_step() const {
    if (!config.adaptive_dt) {
        return config.dt;
    }

    double local_max_speed = 1.0e-6;
    for (int j = 1; j <= mesh.ny_local; ++j) {
        for (int i = 1; i <= mesh.nx_local; ++i) {
            double u_mag = std::abs(fields.u[mesh.u_idx(i, j)]);
            double v_mag = std::abs(fields.v[mesh.v_idx(i, j)]);
            double speed = std::sqrt(u_mag * u_mag + v_mag * v_mag);
            if (speed > local_max_speed) local_max_speed = speed;
        }
    }
    double global_max_speed = MpiComm::global_max(mesh, local_max_speed);

    // 1. Critère CFL de convection
    double dt_cfl = config.cfl_max * std::min(mesh.dx, mesh.dy) / global_max_speed;

    // 2. Critère capillaire (onde de surface) si tension de surface activée
    double dt_sigma = 1.0e10;
    if (config.enable_surface_tension && config.sigma > 0.0) {
        double rho_sum = config.fluid1.rho + config.fluid2.rho;
        double h_min = std::min(mesh.dx, mesh.dy);
        dt_sigma = std::sqrt((rho_sum * h_min * h_min * h_min) / (4.0 * M_PI * config.sigma));
    }

    double dt_target = std::min(dt_cfl, dt_sigma);
    return std::min(config.dt, dt_target);
}

void Simulation::run() {
    if (config.mode == "LAPLACIAN") {
        run_laplacian();
        return;
    }
    if (config.mode == "VORTEX_ADVECTION") {
        run_vortex_advection();
        return;
    }

    double t = 0.0;
    int step = 0;

    // Sauvegarde initiale à t = 0.0
    vtk_writer.write_step(mesh, fields, step, t);
    vtk_writer.write_diagnostics(mesh, fields, step, t, config.dt);

    if (mesh.rank == 0) {
        std::cout << "\n>>> DEMARRAGE DU CALCUL FUGU 2D PARALLELE <<<\n"
                  << std::left << std::setw(8)  << "Pas"
                  << std::setw(14) << "Temps [s]"
                  << std::setw(12) << "dt [s]"
                  << std::setw(12) << "Iter Krylov"
                  << std::setw(16) << "Vitesse Max"
                  << std::setw(16) << "Div Max";
        if (config.enable_energy) {
            std::cout << std::setw(14) << "Nu Moyen";
        }
        std::cout << "\n" << std::string(85, '-') << "\n";
    }

    std::vector<double> state_vector(ns_solver.n_owned_total, 0.0);
    std::vector<double> rhs(ns_solver.n_owned_total, 0.0);
    std::vector<double> u_prev = fields.u;

    while (t < config.t_final - 1.0e-12 * (std::abs(config.t_final) + 1.0)) {
        step++;
        u_prev = fields.u;
        double dt = compute_time_step();
        if (t + dt > config.t_final) {
            dt = config.t_final - t;
        }
        if (dt < 1.0e-12) {
            break;
        }

        // 1. Transport de l'interface si écoulement diphasique
        if (config.enable_two_phase) {
            tracker.advect(mesh, fields, dt);
        }

        // 2. Assemblage des blocs du préconditionneur
        ns_solver.assemble_preconditioner_blocks(fields, dt);

        int total_iters = 0;
        int uzawa_max = ns_solver.is_augmented_lagrangian() ? ns_solver.uzawa_iter : 1;

        for (int m = 0; m < uzawa_max; ++m) {
            // 3. Construction du second membre rhs
            ns_solver.assemble_rhs(fields, dt, rhs);

            // 4. Initialisation du vecteur d'état à partir des champs courants
            ns_solver.pack_vector(fields, state_vector);

            // 5. Résolution du système linéaire par le solveur sélectionné
            int iters = linear_solver->solve(ns_solver, *precond, fields, dt, rhs, state_vector);
            total_iters += iters;

            // 6. Déballage de la vitesse résolue
            ns_solver.unpack_vector(state_vector, fields);

            // 7. Mise à jour de la pression par Lagrangien augmenté si applicable
            if (ns_solver.is_augmented_lagrangian()) {
                ns_solver.update_pressure_al(fields);
            }
        }

        // 7.5 Résolution de l'énergie thermique si activée
        if (config.enable_energy) {
            ns_solver.solve_energy(fields, dt);
        }

        t += dt;

        // Calcul des métriques de contrôle
        double local_max_u = 0.0;
        double local_max_div = 0.0;
        double local_du_max = 0.0;
        for (int j = 1; j <= mesh.ny_local; ++j) {
            for (int i = 1; i <= mesh.nx_local; ++i) {
                double speed = std::sqrt(fields.u[mesh.u_idx(i, j)] * fields.u[mesh.u_idx(i, j)] +
                                         fields.v[mesh.v_idx(i, j)] * fields.v[mesh.v_idx(i, j)]);
                if (speed > local_max_u) local_max_u = speed;

                double du = std::abs(fields.u[mesh.u_idx(i, j)] - u_prev[mesh.u_idx(i, j)]);
                if (du > local_du_max) local_du_max = du;

                double div = std::abs((fields.u[mesh.u_idx(i + 1, j)] - fields.u[mesh.u_idx(i, j)]) / mesh.dx +
                                      (fields.v[mesh.v_idx(i, j + 1)] - fields.v[mesh.v_idx(i, j)]) / mesh.dy);
                if (div > local_max_div) local_max_div = div;
            }
        }
        double global_max_u = MpiComm::global_max(mesh, local_max_u);
        double global_max_div = MpiComm::global_max(mesh, local_max_div);
        double global_du_max = MpiComm::global_max(mesh, local_du_max);

        double nu_mean = 0.0;
        if (config.enable_energy) {
            nu_mean = ns_solver.compute_mean_nusselt_hot_wall(fields);
        }

        // Détection de convergence stationnaire
        if (config.steady_state_tol > 0.0 && step > 10 && global_du_max < config.steady_state_tol) {
            if (mesh.rank == 0) {
                std::cout << "\n>>> CONVERGENCE STATIONNAIRE ATTEINTE : du_max = "
                          << global_du_max << " < " << config.steady_state_tol << " (t = " << t << " s) <<<\n";
                if (config.enable_energy) {
                    std::cout << ">>> NOMBRE DE NUSSELT FINAL : Nu = " << std::fixed << std::setprecision(5) << nu_mean << " <<<\n";
                }
            }
            vtk_writer.write_step(mesh, fields, step, t);
            break;
        }

        // Affichage console
        if (mesh.rank == 0 && (step % config.output.log_interval == 0)) {
            std::cout << std::left << std::setw(8)  << step
                      << std::setw(14) << std::fixed << std::setprecision(5) << t
                      << std::setw(12) << std::scientific << std::setprecision(3) << dt
                      << std::setw(12) << total_iters
                      << std::setw(16) << std::scientific << std::setprecision(4) << global_max_u
                      << std::setw(16) << std::scientific << std::setprecision(4) << global_max_div;
            if (config.enable_energy) {
                std::cout << std::setw(14) << std::fixed << std::setprecision(4) << nu_mean;
            }
            std::cout << "\n";
        }

        // Écriture des diagnostics temporels
        if (config.output.write_csv_history) {
            vtk_writer.write_diagnostics(mesh, fields, step, t, dt);
        }

        // Export des fichiers VTK
        if (step % config.output.vtk_interval == 0 || t >= config.t_final) {
            vtk_writer.write_step(mesh, fields, step, t);
        }
    }

    vtk_writer.finalize();

    if (mesh.rank == 0) {
        std::cout << std::string(78, '-') << "\n"
                  << ">>> SIMULATION FUGU 2D TERMINEE AVEC SUCCES A t = " << t << " s <<<\n"
                  << "Fichiers de visualisation VTK et CSV disponibles dans : " << config.output.output_dir << "\n";
    }
}

void Simulation::run_laplacian() {
    double a = config.lx;
    double b = config.ly;
    int nx = config.nx;
    int ny = config.ny;
    double dx = a / nx;
    double dy = b / ny;

    if (mesh.rank == 0) {
        std::cout << "\n========================================================\n"
                  << "  VERIFICATION DU CAS 3.1 : LAPLACIEN DELTA(PSI) = 1     \n"
                  << "  Domaine [" << a << " x " << b << "], Grille " << nx << " x " << ny << "\n"
                  << "========================================================\n";
    }

    // Inconnues nodales intérieures : 1 à nx-1, 1 à ny-1
    int n_int_x = nx - 1;
    int n_int_y = ny - 1;
    int n_dofs = n_int_x * n_int_y;

    CSRMatrix A;
    A.nrows = n_dofs;
    A.ncols = n_dofs;
    A.row_ptr.resize(n_dofs + 1, 0);

    auto dof_idx = [&](int i, int j) { return j * n_int_x + i; };

    // 1. Comptage des éléments non nuls
    for (int j = 0; j < n_int_y; ++j) {
        for (int i = 0; i < n_int_x; ++i) {
            int row = dof_idx(i, j);
            int count = 1;
            if (i > 0) count++;
            if (i < n_int_x - 1) count++;
            if (j > 0) count++;
            if (j < n_int_y - 1) count++;
            A.row_ptr[row + 1] = A.row_ptr[row] + count;
        }
    }

    A.col_idx.resize(A.row_ptr[n_dofs]);
    A.values.resize(A.row_ptr[n_dofs]);

    std::vector<double> rhs(n_dofs, -1.0); // -Delta Psi = -1
    std::vector<double> x(n_dofs, 0.0);

    int cur = 0;
    for (int j = 0; j < n_int_y; ++j) {
        for (int i = 0; i < n_int_x; ++i) {
            int row = dof_idx(i, j);
            if (j > 0) {
                A.col_idx[cur] = dof_idx(i, j - 1);
                A.values[cur] = -1.0 / (dy * dy);
                cur++;
            }
            if (i > 0) {
                A.col_idx[cur] = dof_idx(i - 1, j);
                A.values[cur] = -1.0 / (dx * dx);
                cur++;
            }
            A.col_idx[cur] = row;
            A.values[cur] = 2.0 / (dx * dx) + 2.0 / (dy * dy);
            cur++;
            if (i < n_int_x - 1) {
                A.col_idx[cur] = dof_idx(i + 1, j);
                A.values[cur] = -1.0 / (dx * dx);
                cur++;
            }
            if (j < n_int_y - 1) {
                A.col_idx[cur] = dof_idx(i, j + 1);
                A.values[cur] = -1.0 / (dy * dy);
                cur++;
            }
        }
    }

    // Solveur Gradient Conjugué préconditionné par Jacobi
    std::vector<double> r = rhs;
    std::vector<double> diag(n_dofs, 1.0);
    for (int i = 0; i < n_dofs; ++i) {
        for (int p_idx = A.row_ptr[i]; p_idx < A.row_ptr[i+1]; ++p_idx) {
            if (A.col_idx[p_idx] == i) { diag[i] = A.values[p_idx]; break; }
        }
    }

    std::vector<double> p_cg(n_dofs, 0.0), z(n_dofs, 0.0), Ap(n_dofs, 0.0);
    for (int i = 0; i < n_dofs; ++i) z[i] = r[i] / diag[i];
    p_cg = z;
    double rz_old = 0.0;
    for (int i = 0; i < n_dofs; ++i) rz_old += r[i] * z[i];

    int max_it = 20000;
    double tol = config.solver.tol > 0 ? config.solver.tol : 1e-12;
    int iters = 0;

    for (iters = 0; iters < max_it; ++iters) {
        A.matvec(p_cg, Ap);
        double pAp = 0.0;
        for (int i = 0; i < n_dofs; ++i) pAp += p_cg[i] * Ap[i];
        if (std::abs(pAp) < 1e-30) break;
        double alpha = rz_old / pAp;
        double r_norm = 0.0;
        for (int i = 0; i < n_dofs; ++i) {
            x[i] += alpha * p_cg[i];
            r[i] -= alpha * Ap[i];
            r_norm += r[i] * r[i];
        }
        r_norm = std::sqrt(r_norm);
        if (r_norm < tol) break;

        for (int i = 0; i < n_dofs; ++i) z[i] = r[i] / diag[i];
        double rz_new = 0.0;
        for (int i = 0; i < n_dofs; ++i) rz_new += r[i] * z[i];
        double beta = rz_new / rz_old;
        rz_old = rz_new;
        for (int i = 0; i < n_dofs; ++i) p_cg[i] = z[i] + beta * p_cg[i];
    }

    // Évaluation de la solution analytique exacte (formule de Thétis)
    double err_l1 = 0.0, err_l2 = 0.0, err_linf = 0.0;
    for (int j = 0; j < n_int_y; ++j) {
        for (int i = 0; i < n_int_x; ++i) {
            double xp = (i + 1) * dx;
            double yp = (j + 1) * dy;
            double psi_ex = 0.0;
            for (int k2 = 0; k2 < 60; ++k2) {
                for (int k1 = 0; k1 < 60; ++k1) {
                    double n1 = 2 * k1 + 1;
                    double n2 = 2 * k2 + 1;
                    double num = 16.0 * std::sin(n1 * M_PI * xp / a) * std::sin(n2 * M_PI * yp / b);
                    double den = (std::pow(M_PI, 4.0)) * n1 * n2 * (std::pow(n1 / a, 2.0) + std::pow(n2 / b, 2.0));
                    psi_ex -= num / den;
                }
            }
            double diff = std::abs(x[dof_idx(i, j)] - psi_ex);
            err_l1 += diff;
            err_l2 += diff * diff;
            if (diff > err_linf) err_linf = diff;
        }
    }
    err_l1 /= n_dofs;
    err_l2 = std::sqrt(err_l2 / n_dofs);

    if (mesh.rank == 0) {
        std::cout << ">>> RESULTAT RESOLUTION LAPLACIEN FUGU 2D <<<\n"
                  << "  Iterations : " << iters << "\n"
                  << "  Erreur L1   : " << std::scientific << std::setprecision(5) << err_l1 << "\n"
                  << "  Erreur L2   : " << std::scientific << std::setprecision(5) << err_l2 << "\n"
                  << "  Erreur Linf : " << std::scientific << std::setprecision(5) << err_linf << "\n"
                  << "========================================================\n";

        std::string err_file = config.output.output_dir + "/laplacian_error.csv";
        std::ofstream ofs(err_file);
        if (ofs.is_open()) {
            ofs << "nx,ny,dx,dy,iters,err_l1,err_l2,err_linf\n"
                << nx << "," << ny << "," << dx << "," << dy << ","
                << iters << "," << err_l1 << "," << err_l2 << "," << err_linf << "\n";
            ofs.close();
        }
    }
}

void Simulation::run_vortex_advection() {
    int nx = config.nx;
    int ny = config.ny;
    double T_period = (config.t_final > 0.0) ? config.t_final : 6.0;
    double dt = config.dt;

    if (mesh.rank == 0) {
        std::cout << "\n========================================================\n"
                  << "  VERIFICATION DU CAS 3.19 : ADVECTION VORTEX CISAILLEMENT\n"
                  << "  Grille : " << nx << " x " << ny << " | Periode T = " << T_period << " s | dt = " << dt << " s\n"
                  << "========================================================\n";
    }

    // Sauvegarder la condition initiale C_exact
    std::vector<double> C_initial = fields.C;

    // Export initial
    vtk_writer.write_step(mesh, fields, 0, 0.0);

    double t = 0.0;
    int step = 0;

    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);

    while (t < T_period - 1.0e-12) {
        step++;
        if (t + dt > T_period) dt = T_period - t;

        double cos_factor = std::cos(M_PI * t / T_period);

        // Imposer le champ de vitesse analytique sur les faces décalées
        for (int j = 1; j <= mesh.ny_local; ++j) {
            double y_c = mesh.yc(j);
            for (int i = 1; i <= u_nx; ++i) {
                double x_u = mesh.xu(i);
                double sx = std::sin(M_PI * x_u);
                double sy = std::sin(2.0 * M_PI * y_c);
                fields.u[mesh.u_idx(i, j)] = - (sx * sx) * sy * cos_factor;
            }
        }

        for (int j = 1; j <= v_ny; ++j) {
            double y_v = mesh.yv(j);
            for (int i = 1; i <= mesh.nx_local; ++i) {
                double x_c = mesh.xc(i);
                double sx = std::sin(2.0 * M_PI * x_c);
                double sy = std::sin(M_PI * y_v);
                fields.v[mesh.v_idx(i, j)] = sx * (sy * sy) * cos_factor;
            }
        }

        // Halos et CL
        MpiComm::sync_u(mesh, fields.u);
        MpiComm::sync_v(mesh, fields.v);

        // Advection TVD de C
        tracker.advect(mesh, fields, dt);

        t += dt;

        if (step % config.output.vtk_interval == 0 || t >= T_period || std::abs(t - 0.5 * T_period) < 0.5 * dt) {
            vtk_writer.write_step(mesh, fields, step, t);
        }

        if (mesh.rank == 0 && (step % config.output.log_interval == 0 || t >= T_period)) {
            std::cout << "  [Vortex] Pas " << step << " | t = " << std::fixed << std::setprecision(4) << t
                      << " s / " << T_period << " s\n";
        }
    }

    // Calcul de l'erreur L1 entre C_initial et C(T)
    double local_err_l1 = 0.0;
    double local_cells = mesh.nx_local * mesh.ny_local;
    for (int j = 1; j <= mesh.ny_local; ++j) {
        for (int i = 1; i <= mesh.nx_local; ++i) {
            local_err_l1 += std::abs(fields.C[mesh.cell_idx(i, j)] - C_initial[mesh.cell_idx(i, j)]);
        }
    }
    double global_err_l1 = MpiComm::global_sum(mesh, local_err_l1);
    double global_cells = MpiComm::global_sum(mesh, local_cells);
    double err_l1 = global_err_l1 / global_cells;

    if (mesh.rank == 0) {
        std::cout << "\n>>> RESULTAT DU CAS 3.19 : ADVECTION VORTEX FUGU 2D <<<\n"
                  << "  Maillage : " << nx << " x " << ny << "\n"
                  << "  Erreur L1 : " << std::scientific << std::setprecision(5) << err_l1 << "\n"
                  << "========================================================\n";

        std::string err_file = config.output.output_dir + "/vortex_error.csv";
        std::ofstream ofs(err_file);
        if (ofs.is_open()) {
            ofs << "nx,ny,dt,T,err_l1\n"
                << nx << "," << ny << "," << dt << "," << T_period << "," << err_l1 << "\n";
            ofs.close();
        }
    }

    vtk_writer.finalize();
}

} // namespace fugu

