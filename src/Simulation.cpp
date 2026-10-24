#include "covof/Simulation.hpp"
#include "covof/MpiComm.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

// Déclaration de variable typée
bool Simulation::init(MPI_Comm comm, const std::string& config_file) {
    // 1. Chargement de la configuration JSON
    if (!config.load_from_json(config_file)) {
        // Retour de fonction
        return false;
    }

    // Déclaration de variable typée
    int my_rank;
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Comm_rank(comm, &my_rank);
    // Évaluation de la condition de contrôle
    if (my_rank == 0) {
        config.print();
    }

    // 2. Initialisation du maillage partitionné
    bool periodic_x = (config.bc.left.type == BCType::PERIODIC || config.bc.right.type == BCType::PERIODIC);
    // Déclaration de variable typée
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
        // Déclaration de variable typée
        double y_step = config.bc.left.y_step;
        // Déclaration de variable typée
        double h = mesh.ly - y_step;
        // Déclaration de variable typée
        double u_mean = config.bc.left.u_mean;
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Déclaration de variable typée
            double y = mesh.yc(j);
            // Déclaration de variable typée
            double u_val = 0.0;
            // Évaluation de la condition de contrôle
            if (y >= y_step) {
                // Calcul et assignation numérique
                u_val = 6.0 * u_mean * (y - y_step) * (mesh.ly - y) / (h * h);
            }
            // Déclaration de variable typée
            int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= u_nx; ++i) {
                // Déclaration de variable typée
                double x = mesh.xu(i);
                // Évaluation de la condition de contrôle
                if (x <= 4.0 + 1e-9) {
                    // Calcul de la résolution géométrique de cellule
                    fields.u[mesh.u_idx(i, j)] = u_val;
                }
            }
        }
        MpiComm::sync_u(mesh, fields.u);
    }

    // Initialisation du champ de vitesse solénoïdal pour LIQUID_SHEET (Équations 17-18 Fluids 2021)
    if (config.init_cond.type == InitialConditionType::LIQUID_SHEET) {
        // Déclaration des variables physiques et géométriques
        double y_c = (config.init_cond.center_y > 0.0) ? config.init_cond.center_y : 0.5 * mesh.ly;
        // Déclaration de variable typée pour l'épaisseur de la nappe liquide
        double delta = (config.init_cond.thickness > 0.0) ? config.init_cond.thickness : 0.0003;
        // Déclaration de variable typée pour la longueur du domaine
        double L = mesh.lx;
        // Déclaration de variable typée pour le nombre d'onde
        double k_pert = 2.0 * M_PI / L;
        // Déclaration de variable typée pour l'amplitude de perturbation
        double amp = 0.04;
        // Facteur pour u assurant une divergence rigoureusement nulle (Eq. 17)
        double factor_u = amp * (L / (2.0 * M_PI)) * (2.0 / delta);

        // Composante horizontale u sur les faces décalées MAC en x
        int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Déclaration de variable typée
            double y = mesh.yc(j);
            // Déclaration de variable typée pour l'écart au centre de la nappe
            double dy_c = y - y_c;
            // Signe pour la divergence solénoïdale
            double sgn_y = (dy_c > 1.0e-12) ? 1.0 : ((dy_c < -1.0e-12) ? -1.0 : 0.0);
            // Décroissance exponentielle symétrique par rapport au plan médian
            double exp_decay = std::exp(-2.0 * std::abs(dy_c) / delta);
            // Boucle spatiale sur les faces u selon l'axe horizontal X
            for (int i = 1; i <= u_nx; ++i) {
                // Déclaration de variable typée
                double x = mesh.xu(i);
                // Vitesse moyenne convective A : 30 m/s dans le gaz, 2 m/s dans le liquide
                double c_u = 0.5 * (fields.C[mesh.cell_idx(i - 1, j)] + fields.C[mesh.cell_idx(i, j)]);
                // Interpolation de la vitesse moyenne de base A selon la présence de gaz/liquide
                double A_val = c_u * 30.0 + (1.0 - c_u) * 2.0;
                // Calcul et assignation numérique de u selon l'Équation 17
                fields.u[mesh.u_idx(i, j)] = A_val - sgn_y * factor_u * std::cos(k_pert * x) * exp_decay;
            }
        }

        // Composante verticale v sur les faces décalées MAC en y
        int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
        // Boucle spatiale sur les faces v selon l'axe vertical Y
        for (int j = 1; j <= v_ny; ++j) {
            // Déclaration de variable typée
            double y = mesh.yv(j);
            // Déclaration de variable typée pour l'écart au centre de la nappe
            double dy_c = y - y_c;
            // Décroissance exponentielle symétrique par rapport au plan médian
            double exp_decay = std::exp(-2.0 * std::abs(dy_c) / delta);
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                double x = mesh.xc(i);
                // Calcul et assignation numérique de v selon l'Équation 17
                fields.v[mesh.v_idx(i, j)] = amp * std::sin(k_pert * x) * exp_decay;
            }
        }
        MpiComm::sync_u(mesh, fields.u);
        MpiComm::sync_v(mesh, fields.v);
    }

    // Initialisation du champ thermique si activé
    if (config.enable_energy) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                double frac = mesh.xc(i) / mesh.lx;
                // Calcul de la résolution géométrique de cellule
                fields.T[mesh.cell_idx(i, j)] = config.bc.left.T_val + frac * (config.bc.right.T_val - config.bc.left.T_val);
            }
        }
        MpiComm::sync_cell_scalar(mesh, fields.T);
        MpiComm::apply_thermal_boundary_conditions(mesh, config.bc, fields);
    }

    // 5. Initialisation du solveur Navier-Stokes et des préconditionneurs
    ns_solver.init(mesh, config);
    ns_solver.assemble_preconditioner_blocks(fields, config.dt);

    // Calcul et assignation numérique
    precond = PreconditionerFactory::create(config.solver.precond_type);
    precond->init(&ns_solver);

    // Calcul et assignation numérique
    linear_solver = SolverFactory::create(config.solver.type);
    linear_solver->init(config.solver);

    // 6. Initialisation de l'exportateur VTK et diagnostics CSV
    vtk_writer.init(mesh, config.output.output_dir, config.output.case_name);

    // Retour de fonction
    return true;
}

// Déclaration de variable typée
double Simulation::compute_time_step() const {
    // Évaluation de la condition de contrôle
    if (!config.adaptive_dt) {
        // Retour de fonction
        return config.dt;
    }

    // Déclaration de variable typée
    double local_max_speed = 1.0e-6;
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= mesh.ny_local; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= mesh.nx_local; ++i) {
            // Déclaration de variable typée
            double u_mag = std::abs(fields.u[mesh.u_idx(i, j)]);
            // Déclaration de variable typée
            double v_mag = std::abs(fields.v[mesh.v_idx(i, j)]);
            // Déclaration de variable typée
            double speed = std::sqrt(u_mag * u_mag + v_mag * v_mag);
            // Évaluation de la condition de contrôle
            if (speed > local_max_speed) local_max_speed = speed;
        }
    }
    // Déclaration de variable typée
    double global_max_speed = MpiComm::global_max(mesh, local_max_speed);

    // 1. Critère CFL de convection
    double dt_cfl = config.cfl_max * std::min(mesh.dx, mesh.dy) / global_max_speed;

    // 2. Critère capillaire (onde de surface) si tension de surface activée
    double dt_sigma = 1.0e10;
    // Évaluation de la condition de contrôle
    if (config.enable_surface_tension && config.sigma > 0.0) {
        // Déclaration de variable typée
        double rho_sum = config.fluid1.rho + config.fluid2.rho;
        // Déclaration de variable typée
        double h_min = std::min(mesh.dx, mesh.dy);
        // Calcul et assignation numérique
        dt_sigma = std::sqrt((rho_sum * h_min * h_min * h_min) / (4.0 * M_PI * config.sigma));
    }

    // Déclaration de variable typée
    double dt_target = std::min(dt_cfl, dt_sigma);
    // Retour de fonction
    return std::min(config.dt, dt_target);
}

void Simulation::run() {
    // Évaluation de la condition de contrôle
    if (config.mode == "LAPLACIAN") {
        run_laplacian();
        // Retour de fonction
        return;
    }
    // Évaluation de la condition de contrôle
    if (config.mode == "VORTEX_ADVECTION") {
        run_vortex_advection();
        // Retour de fonction
        return;
    }

    // Déclaration de variable typée
    double t = 0.0;
    // Déclaration de variable typée
    int step = 0;

    // Sauvegarde initiale à t = 0.0
    vtk_writer.write_step(mesh, fields, step, t);
    vtk_writer.write_diagnostics(mesh, fields, step, t, config.dt);

    // Évaluation de la condition de contrôle
    if (mesh.rank == 0) {
        std::cout << "\n>>> DEMARRAGE DU CALCUL COVOF 2D PARALLELE <<<\n"
                  << std::left << std::setw(8)  << "Pas"
                  << std::setw(14) << "Temps [s]"
                  << std::setw(12) << "dt [s]"
                  << std::setw(12) << "Iter Krylov"
                  << std::setw(16) << "Vitesse Max"
                  << std::setw(16) << "Div Max";
        // Évaluation de la condition de contrôle
        if (config.enable_energy) {
            std::cout << std::setw(14) << "Nu Moyen";
        }
        std::cout << "\n" << std::string(85, '-') << "\n";
    }

    // Déclaration de variable typée
    std::vector<double> state_vector(ns_solver.n_owned_total, 0.0);
    // Déclaration de variable typée
    std::vector<double> rhs(ns_solver.n_owned_total, 0.0);
    // Déclaration de variable typée
    std::vector<double> u_prev = fields.u;

    // Boucle itérative conditionnelle
    while (t < config.t_final - 1.0e-12 * (std::abs(config.t_final) + 1.0)) {
        step++;
        // Calcul et assignation numérique
        u_prev = fields.u;
        // Déclaration de variable typée
        double dt = compute_time_step();
        // Évaluation de la condition de contrôle
        if (t + dt > config.t_final) {
            // Calcul et assignation numérique
            dt = config.t_final - t;
        }
        // Évaluation de la condition de contrôle
        if (dt < 1.0e-12) {
            break;
        }

        // 1. Transport de l'interface si écoulement diphasique
        if (config.enable_two_phase) {
            tracker.advect(mesh, fields, dt);
        }

        // 2. Assemblage des blocs du préconditionneur
        ns_solver.assemble_preconditioner_blocks(fields, dt);

        // Déclaration de variable typée
        int total_iters = 0;
        // Déclaration de variable typée
        int uzawa_max = ns_solver.is_augmented_lagrangian() ? ns_solver.uzawa_iter : 1;

        // Boucle d'itération algorithmique
        for (int m = 0; m < uzawa_max; ++m) {
            // 3. Construction du second membre rhs
            ns_solver.assemble_rhs(fields, dt, rhs);

            // 4. Initialisation du vecteur d'état à partir des champs courants
            ns_solver.pack_vector(fields, state_vector);

            // 5. Résolution du système linéaire par le solveur sélectionné
            int iters = linear_solver->solve(ns_solver, *precond, fields, dt, rhs, state_vector);
            // Calcul et assignation numérique
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

        // Calcul et assignation numérique
        t += dt;

        // Calcul des métriques de contrôle
        double local_max_u = 0.0;
        // Déclaration de variable typée
        double local_max_div = 0.0;
        // Déclaration de variable typée
        double local_du_max = 0.0;
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                double speed = std::sqrt(fields.u[mesh.u_idx(i, j)] * fields.u[mesh.u_idx(i, j)] +
                                         fields.v[mesh.v_idx(i, j)] * fields.v[mesh.v_idx(i, j)]);
                // Évaluation de la condition de contrôle
                if (speed > local_max_u) local_max_u = speed;

                // Déclaration de variable typée
                double du = std::abs(fields.u[mesh.u_idx(i, j)] - u_prev[mesh.u_idx(i, j)]);
                // Évaluation de la condition de contrôle
                if (du > local_du_max) local_du_max = du;

                // Déclaration de variable typée
                double div = std::abs((fields.u[mesh.u_idx(i + 1, j)] - fields.u[mesh.u_idx(i, j)]) / mesh.dx +
                                      (fields.v[mesh.v_idx(i, j + 1)] - fields.v[mesh.v_idx(i, j)]) / mesh.dy);
                // Évaluation de la condition de contrôle
                if (div > local_max_div) local_max_div = div;
            }
        }
        // Déclaration de variable typée
        double global_max_u = MpiComm::global_max(mesh, local_max_u);
        // Déclaration de variable typée
        double global_max_div = MpiComm::global_max(mesh, local_max_div);
        // Déclaration de variable typée
        double global_du_max = MpiComm::global_max(mesh, local_du_max);

        // Déclaration de variable typée
        double nu_mean = 0.0;
        // Évaluation de la condition de contrôle
        if (config.enable_energy) {
            // Calcul et assignation numérique
            nu_mean = ns_solver.compute_mean_nusselt_hot_wall(fields);
        }

        // Détection de convergence stationnaire
        if (config.steady_state_tol > 0.0 && step > 10 && global_du_max < config.steady_state_tol) {
            // Évaluation de la condition de contrôle
            if (mesh.rank == 0) {
                // Calcul et assignation numérique
                std::cout << "\n>>> CONVERGENCE STATIONNAIRE ATTEINTE : du_max = "
                          // Calcul de la résolution géométrique de cellule
                          << global_du_max << " < " << config.steady_state_tol << " (t = " << t << " s) <<<\n";
                // Évaluation de la condition de contrôle
                if (config.enable_energy) {
                    // Calcul et assignation numérique
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
            // Évaluation de la condition de contrôle
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

    // Évaluation de la condition de contrôle
    if (mesh.rank == 0) {
        std::cout << std::string(78, '-') << "\n"
                  // Calcul et assignation numérique
                  << ">>> SIMULATION COVOF 2D TERMINEE AVEC SUCCES A t = " << t << " s <<<\n"
                  << "Fichiers de visualisation VTK et CSV disponibles dans : " << config.output.output_dir << "\n";
    }
}

void Simulation::run_laplacian() {
    // Déclaration de variable typée
    double a = config.lx;
    // Déclaration de variable typée
    double b = config.ly;
    // Déclaration de variable typée
    int nx = config.nx;
    // Déclaration de variable typée
    int ny = config.ny;
    // Déclaration de variable typée
    double dx = a / nx;
    // Déclaration de variable typée
    double dy = b / ny;

    // Évaluation de la condition de contrôle
    if (mesh.rank == 0) {
        // Calcul et assignation numérique
        std::cout << "\n========================================================\n"
                  // Calcul et assignation numérique
                  << "  VERIFICATION DU CAS 3.1 : LAPLACIEN DELTA(PSI) = 1     \n"
                  << "  Domaine [" << a << " x " << b << "], Grille " << nx << " x " << ny << "\n"
                  // Calcul et assignation numérique
                  << "========================================================\n";
    }

    // Inconnues nodales intérieures : 1 à nx-1, 1 à ny-1
    int n_int_x = nx - 1;
    // Déclaration de variable typée
    int n_int_y = ny - 1;
    // Déclaration de variable typée
    int n_dofs = n_int_x * n_int_y;

    CSRMatrix A;
    // Calcul et assignation numérique
    A.nrows = n_dofs;
    // Calcul et assignation numérique
    A.ncols = n_dofs;
    A.row_ptr.resize(n_dofs + 1, 0);

    // Calcul de la résolution géométrique de cellule
    auto dof_idx = [&](int i, int j) { return j * n_int_x + i; };

    // 1. Comptage des éléments non nuls
    for (int j = 0; j < n_int_y; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n_int_x; ++i) {
            // Déclaration de variable typée
            int row = dof_idx(i, j);
            // Déclaration de variable typée
            int count = 1;
            // Évaluation de la condition de contrôle
            if (i > 0) count++;
            // Évaluation de la condition de contrôle
            if (i < n_int_x - 1) count++;
            // Évaluation de la condition de contrôle
            if (j > 0) count++;
            // Évaluation de la condition de contrôle
            if (j < n_int_y - 1) count++;
            // Calcul et assignation numérique
            A.row_ptr[row + 1] = A.row_ptr[row] + count;
        }
    }

    A.col_idx.resize(A.row_ptr[n_dofs]);
    A.values.resize(A.row_ptr[n_dofs]);

    // Déclaration de variable typée
    std::vector<double> rhs(n_dofs, -1.0); // -Delta Psi = -1
    // Déclaration de variable typée
    std::vector<double> x(n_dofs, 0.0);

    // Déclaration de variable typée
    int cur = 0;
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 0; j < n_int_y; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n_int_x; ++i) {
            // Déclaration de variable typée
            int row = dof_idx(i, j);
            // Évaluation de la condition de contrôle
            if (j > 0) {
                // Calcul de la résolution géométrique de cellule
                A.col_idx[cur] = dof_idx(i, j - 1);
                // Calcul et assignation numérique
                A.values[cur] = -1.0 / (dy * dy);
                cur++;
            }
            // Évaluation de la condition de contrôle
            if (i > 0) {
                // Calcul de la résolution géométrique de cellule
                A.col_idx[cur] = dof_idx(i - 1, j);
                // Calcul et assignation numérique
                A.values[cur] = -1.0 / (dx * dx);
                cur++;
            }
            // Calcul de la résolution géométrique de cellule
            A.col_idx[cur] = row;
            // Calcul et assignation numérique
            A.values[cur] = 2.0 / (dx * dx) + 2.0 / (dy * dy);
            cur++;
            // Évaluation de la condition de contrôle
            if (i < n_int_x - 1) {
                // Calcul de la résolution géométrique de cellule
                A.col_idx[cur] = dof_idx(i + 1, j);
                // Calcul et assignation numérique
                A.values[cur] = -1.0 / (dx * dx);
                cur++;
            }
            // Évaluation de la condition de contrôle
            if (j < n_int_y - 1) {
                // Calcul de la résolution géométrique de cellule
                A.col_idx[cur] = dof_idx(i, j + 1);
                // Calcul et assignation numérique
                A.values[cur] = -1.0 / (dy * dy);
                cur++;
            }
        }
    }

    // Solveur Gradient Conjugué préconditionné par Jacobi
    std::vector<double> r = rhs;
    // Déclaration de variable typée
    std::vector<double> diag(n_dofs, 1.0);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n_dofs; ++i) {
        // Boucle d'itération algorithmique
        for (int p_idx = A.row_ptr[i]; p_idx < A.row_ptr[i+1]; ++p_idx) {
            // Évaluation de la condition de contrôle
            if (A.col_idx[p_idx] == i) { diag[i] = A.values[p_idx]; break; }
        }
    }

    // Déclaration de variable typée
    std::vector<double> p_cg(n_dofs, 0.0), z(n_dofs, 0.0), Ap(n_dofs, 0.0);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n_dofs; ++i) z[i] = r[i] / diag[i];
    // Calcul et assignation numérique
    p_cg = z;
    // Déclaration de variable typée
    double rz_old = 0.0;
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n_dofs; ++i) rz_old += r[i] * z[i];

    // Déclaration de variable typée
    int max_it = 20000;
    // Déclaration de variable typée
    double tol = config.solver.tol > 0 ? config.solver.tol : 1e-12;
    // Déclaration de variable typée
    int iters = 0;

    // Boucle d'itération algorithmique
    for (iters = 0; iters < max_it; ++iters) {
        A.matvec(p_cg, Ap);
        // Déclaration de variable typée
        double pAp = 0.0;
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n_dofs; ++i) pAp += p_cg[i] * Ap[i];
        // Évaluation de la condition de contrôle
        if (std::abs(pAp) < 1e-30) break;
        // Déclaration de variable typée
        double alpha = rz_old / pAp;
        // Déclaration de variable typée
        double r_norm = 0.0;
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n_dofs; ++i) {
            // Calcul et assignation numérique
            x[i] += alpha * p_cg[i];
            // Calcul et assignation numérique
            r[i] -= alpha * Ap[i];
            // Calcul de norme euclidienne ou résidu de convergence
            r_norm += r[i] * r[i];
        }
        // Calcul de norme euclidienne ou résidu de convergence
        r_norm = std::sqrt(r_norm);
        // Évaluation de la condition de contrôle
        if (r_norm < tol) break;

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n_dofs; ++i) z[i] = r[i] / diag[i];
        // Déclaration de variable typée
        double rz_new = 0.0;
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n_dofs; ++i) rz_new += r[i] * z[i];
        // Déclaration de variable typée
        double beta = rz_new / rz_old;
        // Calcul et assignation numérique
        rz_old = rz_new;
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n_dofs; ++i) p_cg[i] = z[i] + beta * p_cg[i];
    }

    // Évaluation de la solution analytique exacte (formule analytique de référence)
    double err_l1 = 0.0, err_l2 = 0.0, err_linf = 0.0;
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 0; j < n_int_y; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n_int_x; ++i) {
            // Déclaration de variable typée
            double xp = (i + 1) * dx;
            // Déclaration de variable typée
            double yp = (j + 1) * dy;
            // Déclaration de variable typée
            double psi_ex = 0.0;
            // Boucle d'itération algorithmique
            for (int k2 = 0; k2 < 60; ++k2) {
                // Boucle d'itération algorithmique
                for (int k1 = 0; k1 < 60; ++k1) {
                    // Déclaration de variable typée
                    double n1 = 2 * k1 + 1;
                    // Déclaration de variable typée
                    double n2 = 2 * k2 + 1;
                    // Déclaration de variable typée
                    double num = 16.0 * std::sin(n1 * M_PI * xp / a) * std::sin(n2 * M_PI * yp / b);
                    // Déclaration de variable typée
                    double den = (std::pow(M_PI, 4.0)) * n1 * n2 * (std::pow(n1 / a, 2.0) + std::pow(n2 / b, 2.0));
                    // Calcul et assignation numérique
                    psi_ex -= num / den;
                }
            }
            // Déclaration de variable typée
            double diff = std::abs(x[dof_idx(i, j)] - psi_ex);
            // Calcul et assignation numérique
            err_l1 += diff;
            // Calcul et assignation numérique
            err_l2 += diff * diff;
            // Évaluation de la condition de contrôle
            if (diff > err_linf) err_linf = diff;
        }
    }
    // Calcul et assignation numérique
    err_l1 /= n_dofs;
    // Calcul et assignation numérique
    err_l2 = std::sqrt(err_l2 / n_dofs);

    // Évaluation de la condition de contrôle
    if (mesh.rank == 0) {
        std::cout << ">>> RESULTAT RESOLUTION LAPLACIEN COVOF 2D <<<\n"
                  << "  Iterations : " << iters << "\n"
                  << "  Erreur L1   : " << std::scientific << std::setprecision(5) << err_l1 << "\n"
                  << "  Erreur L2   : " << std::scientific << std::setprecision(5) << err_l2 << "\n"
                  << "  Erreur Linf : " << std::scientific << std::setprecision(5) << err_linf << "\n"
                  // Calcul et assignation numérique
                  << "========================================================\n";

        // Déclaration de variable typée
        std::string err_file = config.output.output_dir + "/laplacian_error.csv";
        std::ofstream ofs(err_file);
        // Évaluation de la condition de contrôle
        if (ofs.is_open()) {
            ofs << "nx,ny,dx,dy,iters,err_l1,err_l2,err_linf\n"
                << nx << "," << ny << "," << dx << "," << dy << ","
                << iters << "," << err_l1 << "," << err_l2 << "," << err_linf << "\n";
            ofs.close();
        }
    }
}

void Simulation::run_vortex_advection() {
    // Déclaration de variable typée
    int nx = config.nx;
    // Déclaration de variable typée
    int ny = config.ny;
    // Déclaration de variable typée
    double T_period = (config.t_final > 0.0) ? config.t_final : 6.0;
    // Déclaration de variable typée
    double dt = config.dt;

    // Évaluation de la condition de contrôle
    if (mesh.rank == 0) {
        // Calcul et assignation numérique
        std::cout << "\n========================================================\n"
                  << "  VERIFICATION DU CAS 3.19 : ADVECTION VORTEX CISAILLEMENT\n"
                  // Calcul et assignation numérique
                  << "  Grille : " << nx << " x " << ny << " | Periode T = " << T_period << " s | dt = " << dt << " s\n"
                  // Calcul et assignation numérique
                  << "========================================================\n";
    }

    // Sauvegarder la condition initiale C_exact
    std::vector<double> C_initial = fields.C;

    // Export initial
    vtk_writer.write_step(mesh, fields, 0, 0.0);

    // Déclaration de variable typée
    double t = 0.0;
    // Déclaration de variable typée
    int step = 0;

    // Déclaration de variable typée
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    // Déclaration de variable typée
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);

    // Boucle itérative conditionnelle
    while (t < T_period - 1.0e-12) {
        step++;
        // Évaluation de la condition de contrôle
        if (t + dt > T_period) dt = T_period - t;

        // Déclaration de variable typée
        double cos_factor = std::cos(M_PI * t / T_period);

        // Imposer le champ de vitesse analytique sur les faces décalées
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Déclaration de variable typée
            double y_c = mesh.yc(j);
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= u_nx; ++i) {
                // Déclaration de variable typée
                double x_u = mesh.xu(i);
                // Déclaration de variable typée
                double sx = std::sin(M_PI * x_u);
                // Déclaration de variable typée
                double sy = std::sin(2.0 * M_PI * y_c);
                // Calcul de la résolution géométrique de cellule
                fields.u[mesh.u_idx(i, j)] = - (sx * sx) * sy * cos_factor;
            }
        }

        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= v_ny; ++j) {
            // Déclaration de variable typée
            double y_v = mesh.yv(j);
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                double x_c = mesh.xc(i);
                // Déclaration de variable typée
                double sx = std::sin(2.0 * M_PI * x_c);
                // Déclaration de variable typée
                double sy = std::sin(M_PI * y_v);
                // Calcul de la résolution géométrique de cellule
                fields.v[mesh.v_idx(i, j)] = sx * (sy * sy) * cos_factor;
            }
        }

        // Halos et CL
        MpiComm::sync_u(mesh, fields.u);
        MpiComm::sync_v(mesh, fields.v);

        // Advection TVD de C
        tracker.advect(mesh, fields, dt);

        // Calcul et assignation numérique
        t += dt;

        // Évaluation de la condition de contrôle
        if (step % config.output.vtk_interval == 0 || t >= T_period || std::abs(t - 0.5 * T_period) < 0.5 * dt) {
            vtk_writer.write_step(mesh, fields, step, t);
        }

        // Évaluation de la condition de contrôle
        if (mesh.rank == 0 && (step % config.output.log_interval == 0 || t >= T_period)) {
            // Calcul et assignation numérique
            std::cout << "  [Vortex] Pas " << step << " | t = " << std::fixed << std::setprecision(4) << t
                      << " s / " << T_period << " s\n";
        }
    }

    // Calcul des erreurs L1, L2, Linf et du bilan de masse entre C_initial et C(T)
    double local_err_l1 = 0.0;
    // Déclaration de variable typée
    double local_err_l2 = 0.0;
    // Déclaration de variable typée
    double local_err_linf = 0.0;
    // Déclaration de variable typée
    double local_mass_init = 0.0;
    // Déclaration de variable typée
    double local_mass_final = 0.0;
    // Déclaration de variable typée
    double local_cells = mesh.nx_local * mesh.ny_local;
    // Déclaration de variable typée
    double cell_vol = mesh.dx * mesh.dy;

    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= mesh.ny_local; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= mesh.nx_local; ++i) {
            // Déclaration de variable typée
            int idx = mesh.cell_idx(i, j);
            // Déclaration de variable typée
            double diff = std::abs(fields.C[idx] - C_initial[idx]);
            // Calcul et assignation numérique
            local_err_l1 += diff;
            // Calcul et assignation numérique
            local_err_l2 += diff * diff;
            // Évaluation de la condition de contrôle
            if (diff > local_err_linf) local_err_linf = diff;
            // Calcul et assignation numérique
            local_mass_init += C_initial[idx] * cell_vol;
            // Calcul et assignation numérique
            local_mass_final += fields.C[idx] * cell_vol;
        }
    }

    // Réduction parallèle MPI globale de la somme de l'erreur L1
    double global_err_l1 = MpiComm::global_sum(mesh, local_err_l1);
    // Réduction parallèle MPI globale de la somme de l'erreur L2 au carré
    double global_err_l2 = MpiComm::global_sum(mesh, local_err_l2);
    // Réduction parallèle MPI globale de l'erreur maximale absolue Linf
    double global_err_linf = MpiComm::global_max(mesh, local_err_linf);
    // Réduction parallèle MPI globale de la masse totale initiale
    double global_mass_init = MpiComm::global_sum(mesh, local_mass_init);
    // Réduction parallèle MPI globale de la masse totale finale
    double global_mass_final = MpiComm::global_sum(mesh, local_mass_final);
    // Réduction parallèle MPI globale du nombre de cellules
    double global_cells = MpiComm::global_sum(mesh, local_cells);

    // Calcul de l'erreur moyenne normalisée L1
    double err_l1 = global_err_l1 / global_cells;
    // Calcul de l'erreur quadratique moyenne normalisée L2
    double err_l2 = std::sqrt(global_err_l2 / global_cells);
    // Erreur maximale locale Linf
    double err_linf = global_err_linf;
    // Taux de conservation relative de la masse du fluide
    double mass_conservation = (global_mass_init > 1.0e-14) ? (global_mass_final / global_mass_init) : 1.0;

    // Évaluation de la condition de contrôle
    if (mesh.rank == 0) {
        // Affichage console formaté du résultat de vérification
        std::cout << "\n>>> RESULTAT DU CAS 3.19 : ADVECTION VORTEX COVOF 2D <<<\n"
                  << "  Methode Advection  : " << config.vof.advection_method << "\n"
                  << "  Reconstruction     : " << config.vof.reconstruction_method << "\n"
                  << "  Limiteur TVD/NVD   : " << config.vof.tvd_limiter << "\n"
                  << "  Maillage           : " << nx << " x " << ny << "\n"
                  << "  Erreur L1          : " << std::scientific << std::setprecision(5) << err_l1 << "\n"
                  << "  Erreur L2          : " << std::scientific << std::setprecision(5) << err_l2 << "\n"
                  << "  Erreur Linf        : " << std::scientific << std::setprecision(5) << err_linf << "\n"
                  << "  Conservation Masse : " << std::fixed << std::setprecision(5) << (mass_conservation * 100.0) << " %\n"
                  << "========================================================\n";

        // Déclaration de variable typée
        std::string err_file = config.output.output_dir + "/vortex_error.csv";
        // Ouverture du flux d'écriture fichier
        std::ofstream ofs(err_file);
        // Évaluation de la condition de contrôle
        if (ofs.is_open()) {
            // Écriture de l'en-tête CSV enrichi avec toutes les métriques et métadonnées d'advection
            ofs << "nx,ny,dt,T,advection,reconstruction,limiter,err_l1,err_l2,err_linf,mass_init,mass_final,mass_conservation\n"
                << nx << "," << ny << "," << dt << "," << T_period << ","
                << config.vof.advection_method << "," << config.vof.reconstruction_method << "," << config.vof.tvd_limiter << ","
                << err_l1 << "," << err_l2 << "," << err_linf << ","
                << global_mass_init << "," << global_mass_final << "," << mass_conservation << "\n";
            // Fermeture du fichier de sortie
            ofs.close();
        }
    }

    vtk_writer.finalize();
}

} // namespace covof

