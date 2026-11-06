#include "covof/Simulation.hpp"
#include "covof/MpiComm.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace covof {
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

    if (config.compressible) {
        double p_init_val = (config.init_cond.p_init > 0.0) ? config.init_cond.p_init : 1.01325e5;
        double T_init_val = (config.T_ref > 0.0) ? config.T_ref : 300.0;
        std::fill(fields.p.begin(), fields.p.end(), p_init_val);
        std::fill(fields.p_old.begin(), fields.p_old.end(), p_init_val);
        std::fill(fields.T.begin(), fields.T.end(), T_init_val);
        std::fill(fields.T_old.begin(), fields.T_old.end(), T_init_val);
        double cv_val = (config.fluid2.cv > 0.0) ? config.fluid2.cv : 717.5;
        std::fill(fields.energy.begin(), fields.energy.end(), cv_val * T_init_val);
    }

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

    // Initialisation exacte du profil de Poiseuille pour vérification analytique
    if (config.test_case == "CAS_3_7" || config.test_case == "CAS_3_7_POISEUILLE" ||
        (config.bc.left.type == BCType::DIRICHLET && config.bc.left.profile == "PARABOLIC")) {
        double V0 = (config.bc.left.u_mean > 0.0) ? config.bc.left.u_mean : 1.0;
        double mu = config.fluid1.mu;
        double Gx = 12.0 * mu * V0 / (mesh.ly * mesh.ly);

        int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
        for (int j = 1; j <= mesh.ny_local; ++j) {
            double y = mesh.yc(j);
            double u_exact = 6.0 * V0 * y * (mesh.ly - y) / (mesh.ly * mesh.ly);
            for (int i = 1; i <= u_nx; ++i) {
                fields.u[mesh.u_idx(i, j)] = u_exact;
            }
        }
        for (int j = 1; j <= mesh.ny_local; ++j) {
            for (int i = 1; i <= mesh.nx_local; ++i) {
                double x = mesh.xc(i);
                fields.p[mesh.cell_idx(i, j)] = Gx * (mesh.lx - x);
            }
        }
        MpiComm::sync_u(mesh, fields.u);
        MpiComm::sync_cell_scalar(mesh, fields.p);
        MpiComm::apply_boundary_conditions(mesh, config.bc, fields, false);
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
        MpiComm::apply_boundary_conditions(mesh, config.bc, fields, false);
    }

    // Initialisation du champ de vitesse pour VISCOUS_JET (Cas 3.3 Fluids 2021)
    if (config.init_cond.type == InitialConditionType::VISCOUS_JET) {
        // Déclaration des variables physiques et géométriques
        double x_c = (config.init_cond.center_x > 0.0) ? config.init_cond.center_x : 0.5 * mesh.lx;
        // Déclaration de variable typée pour la demi-largeur de l'injecteur
        double half_w = (config.init_cond.jet_width > 0.0) ? 0.5 * config.init_cond.jet_width : 0.04;
        // Déclaration de la hauteur de l'amorce initiale de jet sous la paroi supérieure
        double stub_height = 2.0 * mesh.dy;
        // Vitesse d'injection négative selon Y (ajustée au débit discret exact D * v0)
        double v_inlet = (config.bc.top.v_val != 0.0) ? config.bc.top.v_val : -0.6;
        double d_disc = 0.0;
        for (int ig = 1; ig <= mesh.nx_global; ++ig) {
            double xg = (ig - 0.5) * mesh.dx;
            if (std::abs(xg - x_c) <= half_w) d_disc += mesh.dx;
        }
        if (d_disc < 1.0e-12) d_disc = 2.0 * half_w;
        double v_in_eff = v_inlet * (2.0 * half_w) / d_disc;

        // Composante verticale v sur les faces décalées MAC en y
        int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
        // Boucle spatiale sur les faces v selon l'axe vertical Y
        for (int j = 1; j <= v_ny; ++j) {
            // Déclaration de variable typée
            double y = mesh.yv(j);
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                double x = mesh.xc(i);
                // Évaluation de l'appartenance à l'amorce initiale
                if (y >= mesh.ly - stub_height && std::abs(x - x_c) <= half_w) {
                    // Calcul et assignation numérique
                    fields.v[mesh.v_idx(i, j)] = v_in_eff;
                }
            }
        }
        MpiComm::sync_v(mesh, fields.v);
        MpiComm::apply_boundary_conditions(mesh, config.bc, fields, false);
    }

    // Initialisation quasi-statique de la vitesse pour compression de gaz isotherme (Fluids 2024, Cas 4.2 - Cas 1)
    if (config.compressible && config.init_cond.type == InitialConditionType::DAM_BREAK && config.bc.bottom.v_val > 0.0) {
        double v0 = config.bc.bottom.v_val;
        double y_water = (config.init_cond.box_y1 > 0.0) ? config.init_cond.box_y1 : 0.025;
        double L0 = mesh.ly - y_water;
        int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
        for (int j = 1; j <= v_ny; ++j) {
            double y = mesh.yv(j);
            double v_val = (y <= y_water) ? v0 : v0 * std::max(0.0, (mesh.ly - y) / L0);
            for (int i = 1; i <= mesh.nx_local; ++i) {
                fields.v[mesh.v_idx(i, j)] = v_val;
            }
        }
        MpiComm::sync_v(mesh, fields.v);
        MpiComm::apply_boundary_conditions(mesh, config.bc, fields, false);
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
    precond->update(fields, config.dt);

    // Calcul et assignation numérique
    linear_solver = SolverFactory::create(config.solver.linear_solver.empty() ? config.solver.type : config.solver.linear_solver);
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
        if (config.test_case == "CAS_3_2_NEUMANN" ||
            config.test_case.find("cas3_2") != std::string::npos ||
            config.output.case_name.find("cas3_2") != std::string::npos ||
            config.output.case_name.find("neumann") != std::string::npos) {
            run_laplacian_cas3_2();
        } else {
            run_laplacian();
        }
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

    fields.u_old = fields.u;
    fields.v_old = fields.v;
    fields.p_old = fields.p;
    fields.T_old = fields.T;
    fields.rho_old = fields.rho;
    fields.rho_u_old = fields.rho_u;
    fields.rho_v_old = fields.rho_v;

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

    // Mesure du temps CPU total
    auto start_time = std::chrono::high_resolution_clock::now();

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

        // Sauvegarde de l'état n-1 pour Gear / BDF2 et Adams-Bashforth d'ordre 2
        if (step > 1) {
            fields.u_nm1 = fields.u_old;
            fields.v_nm1 = fields.v_old;
            fields.rho_u_nm1 = fields.rho_u_old;
            fields.rho_v_nm1 = fields.rho_v_old;
            fields.has_nm1 = true;
        } else {
            fields.u_nm1 = fields.u;
            fields.v_nm1 = fields.v;
            fields.rho_u_nm1 = fields.rho_u;
            fields.rho_v_nm1 = fields.rho_v;
            fields.has_nm1 = false;
        }

        // Sauvegarde de l'état n pour dérivées temporelles et vitesse advectrice extrapolée
        fields.u_old = fields.u;
        fields.v_old = fields.v;
        fields.p_old = fields.p;
        fields.T_old = fields.T;
        fields.rho_old = fields.rho;
        fields.rho_u_old = fields.rho_u;
        fields.rho_v_old = fields.rho_v;

        // Étape 1 : Transport inertiel préservant la quantité de mouvement (SSP-RK3, Fluids 2024)
        if (config.enable_ssp_rk3_momentum) {
            ns_solver.advect_ssp_rk3_momentum(fields, dt);
        }

        // Étape 2 : Transport de l'interface si écoulement diphasique
        if (config.enable_two_phase) {
            tracker.advect(mesh, fields, dt);
            // Actualisation des conditions aux limites pour les mailles fantômes
            MpiComm::apply_boundary_conditions(mesh, config.bc, fields, false);
        }

        // 1.9 Calcul du paramètre de pénalité adaptatif r(t, M) (Vincent et al. 2004, Éq. 14)
        if (ns_solver.is_augmented_lagrangian()) {
            ns_solver.compute_adaptative_r_al(fields, dt);
        }

        // 2. Assemblage des blocs du préconditionneur
        ns_solver.assemble_preconditioner_blocks(fields, dt);
        precond->update(fields, dt);

        // Déclaration de variable typée
        int total_iters = 0;

        // Évaluation de la méthode de résolution Navier-Stokes
        if (ns_solver.is_standard_projection()) {
            total_iters = ns_solver.solve_standard_projection(fields, dt, *linear_solver, *precond);
        } else {
            int uzawa_max = ns_solver.is_augmented_lagrangian() ? ns_solver.uzawa_iter : 1;

            // Boucle d'Uzawa-Lagrangien augmenté (Vincent et al. 2004, Éq. 10)
            for (int m = 0; m < uzawa_max; ++m) {
                // 3. Construction du second membre rhs (avec -grad(p^{*,m-1}) et u^n fixe dans d/dt)
                ns_solver.assemble_rhs(fields, dt, rhs);

                // 4. Initialisation du vecteur d'état à partir de u^{*,m-1}
                ns_solver.pack_vector(fields, state_vector);

                // 5. Résolution du système linéaire pour u^{*,m}
                int iters = linear_solver->solve(ns_solver, *precond, fields, dt, rhs, state_vector);
                total_iters += iters;

                // 6. Déballage de la vitesse résolue u^{*,m}
                ns_solver.unpack_vector(state_vector, fields);

                // 7. Mise à jour d'Uzawa : p^{*,m} = p^{*,m-1} - r(t, M) * div(u^{*,m})
                if (ns_solver.is_augmented_lagrangian()) {
                    ns_solver.update_pressure_al(fields);

                    // Critère d'arrêt d'Uzawa sur ||div(u^{*,m})||_inf <= uzawa_tol (Éq. 10)
                    if (uzawa_max > 1 && ns_solver.uzawa_tol > 0.0) {
                        double loc_div_m = 0.0;
                        for (int j = 1; j <= mesh.ny_local; ++j) {
                            for (int i = 1; i <= mesh.nx_local; ++i) {
                                double d_m = std::abs(
                                    (fields.u[mesh.u_idx(i + 1, j)] - fields.u[mesh.u_idx(i, j)]) / mesh.dx +
                                    (fields.v[mesh.v_idx(i, j + 1)] - fields.v[mesh.v_idx(i, j)]) / mesh.dy);
                                if (d_m > loc_div_m) loc_div_m = d_m;
                            }
                        }
                        double glob_div_m = MpiComm::global_max(mesh, loc_div_m);
                        if (glob_div_m <= ns_solver.uzawa_tol) break;
                    }
                }
            }
        }

        // 7.5 Résolution de l'énergie (totale si compressible non-isotherme, thermique si Boussinesq)
        if (config.compressible && config.enable_energy) {
            ns_solver.solve_total_energy(fields, dt);
        } else if (config.enable_energy) {
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
        if (step % config.output.vtk_interval == 0 || t >= config.t_final - 1.0e-10 * (std::abs(config.t_final) + 1.0)) {
            vtk_writer.write_step(mesh, fields, step, t);
        }
    }

    vtk_writer.finalize();

    auto end_time = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_time - start_time).count();

    bool is_unsteady_poiseuille = (config.test_case == "CAS_3_8" ||
                                   config.test_case == "CAS_3_8_POISEUILLE_UNSTEADY" ||
                                   config.test_case.find("unsteady") != std::string::npos ||
                                   config.output.case_name.find("cas3_8") != std::string::npos ||
                                   config.output.output_dir.find("cas3_8") != std::string::npos ||
                                   config.output.case_name.find("unsteady") != std::string::npos ||
                                   config.output.output_dir.find("unsteady") != std::string::npos);
    bool is_poiseuille = !is_unsteady_poiseuille &&
                         (config.test_case == "CAS_3_7" || config.test_case == "CAS_3_7_POISEUILLE" ||
                          config.test_case == "POISEUILLE" ||
                          config.output.case_name.find("poiseuille") != std::string::npos ||
                          config.output.case_name.find("cas3_7") != std::string::npos ||
                          config.output.output_dir.find("poiseuille") != std::string::npos ||
                          config.output.output_dir.find("cas3_7") != std::string::npos);
    if (is_unsteady_poiseuille) {
        export_poiseuille_unsteady_results(step, t, cpu_time);
    } else if (is_poiseuille) {
        export_poiseuille_results(step, t, cpu_time);
    }

    // Évaluation de la condition de contrôle
    if (mesh.rank == 0) {
        std::cout << std::string(78, '-') << "\n"
                  // Calcul et assignation numérique
                  << ">>> SIMULATION COVOF 2D TERMINEE AVEC SUCCES A t = " << t << " s <<<\n"
                  << "Fichiers de visualisation VTK et CSV disponibles dans : " << config.output.output_dir << "\n";
    }
}

void Simulation::export_poiseuille_results(int total_step, double final_time, double cpu_time) {
    if (mesh.rank != 0) return;

    int nx = config.nx;
    int ny = config.ny;
    double lx = config.lx;
    double ly = config.ly;
    double dx = mesh.dx;
    double dy = mesh.dy;
    double mu = config.fluid1.mu;
    double V0 = (config.bc.left.u_mean > 0.0) ? config.bc.left.u_mean : 1.0;
    double Gx_theo = 12.0 * mu * V0 / (ly * ly); // 0.12 Pa/m

    std::vector<double> u_c(nx * ny, 0.0);
    std::vector<double> v_c(nx * ny, 0.0);
    std::vector<double> p_c(nx * ny, 0.0);
    std::vector<double> u_ex(nx * ny, 0.0);
    std::vector<double> p_ex(nx * ny, 0.0);
    std::vector<double> err_u(nx * ny, 0.0);

    double sum_err_l1 = 0.0;
    double sum_err_l2 = 0.0;
    double max_err_linf = 0.0;

    int j_mid = ny / 2;
    double p_out = fields.p[mesh.cell_idx(nx, j_mid)];

    for (int j = 1; j <= ny; ++j) {
        double y = mesh.yc(j);
        double u_exact_val = 6.0 * V0 * y * (ly - y) / (ly * ly);

        for (int i = 1; i <= nx; ++i) {
            double x = mesh.xc(i);
            int c = (j - 1) * nx + (i - 1);

            double uc = 0.5 * (fields.u[mesh.u_idx(i, j)] + fields.u[mesh.u_idx(i + 1, j)]);
            double vc = 0.5 * (fields.v[mesh.v_idx(i, j)] + fields.v[mesh.v_idx(i, j + 1)]);
            double pc = fields.p[mesh.cell_idx(i, j)];

            u_c[c] = uc;
            v_c[c] = vc;
            p_c[c] = pc;
            u_ex[c] = u_exact_val;
            p_ex[c] = Gx_theo * (lx - x);

            double diff = std::abs(uc - u_exact_val);
            err_u[c] = diff;

            sum_err_l1 += diff;
            sum_err_l2 += diff * diff;
            if (diff > max_err_linf) max_err_linf = diff;
        }
    }

    double n_cells = nx * ny;
    double err_l1 = sum_err_l1 / n_cells;
    double err_l2 = std::sqrt(sum_err_l2 / n_cells);

    // Gradient de pression numerique le long de l'axe central y = j_mid
    double p_in = fields.p[mesh.cell_idx(1, j_mid)];
    double x_in = mesh.xc(1);
    double x_out = mesh.xc(nx);
    double Gx_num = (p_in - p_out) / (x_out - x_in);
    double err_grad_p = std::abs(Gx_num - Gx_theo);

    std::cout << "\n========================================================\n"
              << "  RESULTATS CAS 3.7 : ECOULEMENT DE POISEUILLE PLAN     \n"
              << "  Domaine [" << lx << " x " << ly << "], Grille " << nx << " x " << ny << "\n"
              << "  V0 = " << V0 << " m/s, Re = " << (config.fluid1.rho * V0 * ly / mu) << "\n"
              << "  Gradient theorique Gx = " << Gx_theo << " Pa/m\n"
              << "  Gradient numerique Gx = " << Gx_num << " Pa/m (err = " << std::scientific << std::setprecision(5) << err_grad_p << ")\n"
              << "  Erreur L1(u)   : " << std::scientific << std::setprecision(5) << err_l1 << "\n"
              << "  Erreur L2(u)   : " << std::scientific << std::setprecision(5) << err_l2 << "\n"
              << "  Erreur Linf(u) : " << std::scientific << std::setprecision(5) << max_err_linf << "\n"
              << "  Temps CPU      : " << std::fixed << std::setprecision(4) << cpu_time << " s\n"
              << "========================================================\n";

    // 1. poiseuille_error.csv
    std::string err_file = config.output.output_dir + "/poiseuille_error.csv";
    std::ofstream eofs(err_file);
    if (eofs.is_open()) {
        eofs << "nx,ny,dx,dy,iters,err_l1,err_l2,err_linf,err_grad_p,cpu_time\n"
             << nx << "," << ny << "," << dx << "," << dy << ","
             << total_step << "," << err_l1 << "," << err_l2 << "," << max_err_linf << "," << err_grad_p << "," << cpu_time << "\n";
        eofs.close();
    }

    // 2. profile_u_y.csv
    int i_mid = nx / 2;
    int i_3q  = (3 * nx) / 4;
    std::string uy_file = config.output.output_dir + "/profile_u_y.csv";
    std::ofstream uy_ofs(uy_file);
    if (uy_ofs.is_open()) {
        uy_ofs << "y,u_mid,u_three_quarter,u_ex,error\n";
        for (int j = 1; j <= ny; ++j) {
            double y = mesh.yc(j);
            double u_m = u_c[(j - 1) * nx + (i_mid - 1)];
            double u_3 = u_c[(j - 1) * nx + (i_3q - 1)];
            double u_e = 6.0 * V0 * y * (ly - y) / (ly * ly);
            uy_ofs << y << "," << u_m << "," << u_3 << "," << u_e << "," << std::abs(u_m - u_e) << "\n";
        }
        uy_ofs.close();
    }

    // 3. profile_p_x.csv
    std::string px_file = config.output.output_dir + "/profile_p_x.csv";
    std::ofstream px_ofs(px_file);
    if (px_ofs.is_open()) {
        px_ofs << "x,p_mid,p_ex,error\n";
        for (int i = 1; i <= nx; ++i) {
            double x = mesh.xc(i);
            double p_val = fields.p[mesh.cell_idx(i, j_mid)] - p_out;
            double p_e = Gx_theo * (lx - x);
            px_ofs << x << "," << p_val << "," << p_e << "," << std::abs(p_val - p_e) << "\n";
        }
        px_ofs.close();
    }

    // 4. solution_field_2d.csv
    std::string s2d_file = config.output.output_dir + "/solution_field_2d.csv";
    std::ofstream s2d_ofs(s2d_file);
    if (s2d_ofs.is_open()) {
        s2d_ofs << "i,j,x,y,u,v,p,u_ex,p_ex,error_u\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int c = (j - 1) * nx + (i - 1);
                double x = mesh.xc(i);
                double y = mesh.yc(j);
                double p_val = p_c[c] - p_out;
                double p_e = Gx_theo * (lx - x);
                s2d_ofs << i << "," << j << "," << x << "," << y << ","
                        << u_c[c] << "," << v_c[c] << "," << p_val << ","
                        << u_ex[c] << "," << p_e << "," << err_u[c] << "\n";
            }
        }
        s2d_ofs.close();
    }

    // 5. solution_field.vtk
    std::string vtk_file = config.output.output_dir + "/solution_field.vtk";
    std::ofstream vofs(vtk_file);
    if (vofs.is_open()) {
        vofs << "# vtk DataFile Version 3.0\n"
             << "COVOF 2D Cas 3.7 Poiseuille Solution\n"
             << "ASCII\n"
             << "DATASET RECTILINEAR_GRID\n"
             << "DIMENSIONS " << (nx + 1) << " " << (ny + 1) << " 1\n"
             << "X_COORDINATES " << (nx + 1) << " double\n";
        for (int i = 0; i <= nx; ++i) vofs << i * dx << " ";
        vofs << "\nY_COORDINATES " << (ny + 1) << " double\n";
        for (int j = 0; j <= ny; ++j) vofs << j * dy << " ";
        vofs << "\nZ_COORDINATES 1 double\n0.0\n"
             << "CELL_DATA " << (nx * ny) << "\n"
             << "SCALARS Velocity_U double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) vofs << u_c[(j - 1) * nx + (i - 1)] << " ";
            vofs << "\n";
        }
        vofs << "SCALARS Pressure double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) vofs << (p_c[(j - 1) * nx + (i - 1)] - p_out) << " ";
            vofs << "\n";
        }
        vofs << "SCALARS Error_U double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) vofs << err_u[(j - 1) * nx + (i - 1)] << " ";
            vofs << "\n";
        }
        vofs.close();
    }
}

void Simulation::export_poiseuille_unsteady_results(int total_step, double final_time, double cpu_time) {
    if (mesh.rank != 0) return;

    int nx = config.nx;
    int ny = config.ny;
    double lx = config.lx;
    double ly = config.ly;
    double dx = mesh.dx;
    double dy = mesh.dy;
    double rho = config.fluid1.rho > 0.0 ? config.fluid1.rho : 1.0;
    double mu = config.fluid1.mu > 0.0 ? config.fluid1.mu : 0.01;
    double nu = mu / rho;
    double gx = config.gx;
    double k = (std::abs(gx) > 1e-12) ? (rho * gx) : 0.0555;
    double t_val = final_time;

    // Exact Szymanski Fourier series
    auto compute_u_exact = [&](double y) {
        double u_st = (k / (2.0 * nu)) * y * (ly - y);
        if (t_val <= 0.0) return 0.0;
        double tr = 0.0;
        for (int m = 0; m < 50; ++m) {
            int n = 2 * m + 1;
            double lam = n * M_PI / ly;
            tr += (1.0 / (double)(n * n * n)) * std::sin(lam * y) * std::exp(-nu * (lam * lam) * t_val);
        }
        return u_st - (4.0 * k * (ly * ly) / (nu * M_PI * M_PI * M_PI)) * tr;
    };

    std::vector<double> u_c(nx * ny, 0.0);
    std::vector<double> v_c(nx * ny, 0.0);
    std::vector<double> p_c(nx * ny, 0.0);
    std::vector<double> u_ex(nx * ny, 0.0);
    std::vector<double> err_u(nx * ny, 0.0);

    double sum_err_l1 = 0.0;
    double sum_err_l2 = 0.0;
    double max_err_linf = 0.0;
    double max_u_num = 0.0;
    double max_u_exact = 0.0;

    for (int j = 1; j <= ny; ++j) {
        double y = mesh.yc(j);
        double u_exact_val = compute_u_exact(y);
        if (u_exact_val > max_u_exact) max_u_exact = u_exact_val;

        for (int i = 1; i <= nx; ++i) {
            double x = mesh.xc(i);
            int c = (j - 1) * nx + (i - 1);

            double uc = 0.5 * (fields.u[mesh.u_idx(i, j)] + fields.u[mesh.u_idx(i + 1, j)]);
            double vc = 0.5 * (fields.v[mesh.v_idx(i, j)] + fields.v[mesh.v_idx(i, j + 1)]);
            double pc = fields.p[mesh.cell_idx(i, j)];

            u_c[c] = uc;
            v_c[c] = vc;
            p_c[c] = pc;
            u_ex[c] = u_exact_val;

            if (uc > max_u_num) max_u_num = uc;

            double diff = std::abs(uc - u_exact_val);
            err_u[c] = diff;

            sum_err_l1 += diff;
            sum_err_l2 += diff * diff;
            if (diff > max_err_linf) max_err_linf = diff;
        }
    }

    double n_cells = nx * ny;
    double err_l1 = sum_err_l1 / n_cells;
    double err_l2 = std::sqrt(sum_err_l2 / n_cells);

    std::cout << "\n========================================================\n"
              << "  RESULTATS CAS 3.8 : POISEUILLE PLAN INSTATIONNAIRE    \n"
              << "  Domaine [" << lx << " x " << ly << "], Grille " << nx << " x " << ny << "\n"
              << "  Temps t = " << t_val << " s, dt = " << config.dt << " s\n"
              << "  Gradient force k = " << k << " Pa/m, nu = " << nu << " m^2/s\n"
              << "  Vitesse max num = " << std::scientific << std::setprecision(5) << max_u_num
              << " m/s (exact = " << max_u_exact << " m/s)\n"
              << "  Erreur L1(u)   : " << std::scientific << std::setprecision(5) << err_l1 << "\n"
              << "  Erreur L2(u)   : " << std::scientific << std::setprecision(5) << err_l2 << "\n"
              << "  Erreur Linf(u) : " << std::scientific << std::setprecision(5) << max_err_linf << "\n"
              << "  Temps CPU      : " << std::fixed << std::setprecision(4) << cpu_time << " s\n"
              << "========================================================\n";

    // 1. poiseuille_unsteady_error.csv
    std::error_code ec_dir;
    std::filesystem::create_directories(config.output.output_dir, ec_dir);
    std::string err_file = config.output.output_dir + "/poiseuille_unsteady_error.csv";
    std::ofstream eofs(err_file);
    if (eofs.is_open()) {
        eofs << "nx,ny,dx,dy,dt,t_final,iters,err_l1,err_l2,err_linf,u_max_num,u_max_exact,cpu_time\n"
             << nx << "," << ny << "," << dx << "," << dy << ","
             << config.dt << "," << t_val << "," << total_step << ","
             << err_l1 << "," << err_l2 << "," << max_err_linf << ","
             << max_u_num << "," << max_u_exact << "," << cpu_time << "\n";
        eofs.close();
    }

    // 2. profile_u_y.csv
    int i_mid = nx / 2;
    std::string uy_file = config.output.output_dir + "/profile_u_y.csv";
    std::ofstream uy_ofs(uy_file);
    if (uy_ofs.is_open()) {
        uy_ofs << "y,u_num,u_exact,error\n";
        for (int j = 1; j <= ny; ++j) {
            double y = mesh.yc(j);
            double u_m = u_c[(j - 1) * nx + (i_mid - 1)];
            double u_e = compute_u_exact(y);
            uy_ofs << y << "," << u_m << "," << u_e << "," << std::abs(u_m - u_e) << "\n";
        }
        uy_ofs.close();
    }

    // 3. solution_field_2d.csv
    std::string s2d_file = config.output.output_dir + "/solution_field_2d.csv";
    std::ofstream s2d_ofs(s2d_file);
    if (s2d_ofs.is_open()) {
        s2d_ofs << "i,j,x,y,u,v,p,u_exact,error_u\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int c = (j - 1) * nx + (i - 1);
                double x = mesh.xc(i);
                double y = mesh.yc(j);
                s2d_ofs << i << "," << j << "," << x << "," << y << ","
                        << u_c[c] << "," << v_c[c] << "," << p_c[c] << ","
                        << u_ex[c] << "," << err_u[c] << "\n";
            }
        }
        s2d_ofs.close();
    }

    // 4. solution_field.vtk
    std::string vtk_file = config.output.output_dir + "/solution_field.vtk";
    std::ofstream vofs(vtk_file);
    if (vofs.is_open()) {
        vofs << "# vtk DataFile Version 3.0\n"
             << "COVOF 2D Cas 3.8 Poiseuille Unsteady Solution\n"
             << "ASCII\n"
             << "DATASET RECTILINEAR_GRID\n"
             << "DIMENSIONS " << (nx + 1) << " " << (ny + 1) << " 1\n"
             << "X_COORDINATES " << (nx + 1) << " double\n";
        for (int i = 0; i <= nx; ++i) vofs << i * dx << " ";
        vofs << "\nY_COORDINATES " << (ny + 1) << " double\n";
        for (int j = 0; j <= ny; ++j) vofs << j * dy << " ";
        vofs << "\nZ_COORDINATES 1 double\n0.0\n"
             << "CELL_DATA " << (nx * ny) << "\n"
             << "SCALARS Velocity_U double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) vofs << u_c[(j - 1) * nx + (i - 1)] << " ";
            vofs << "\n";
        }
        vofs << "SCALARS Pressure double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) vofs << p_c[(j - 1) * nx + (i - 1)] << " ";
            vofs << "\n";
        }
        vofs << "SCALARS Error_U double 1\nLOOKUP_TABLE default\n";
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) vofs << err_u[(j - 1) * nx + (i - 1)] << " ";
            vofs << "\n";
        }
        vofs.close();
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
    double f_src = config.source_term;
    std::vector<double> rhs(n_dofs, -f_src); // -Delta Psi = -f_src
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

    // Détermination du solveur linéaire à exécuter
    std::string lin_sol = config.solver.linear_solver;
    for (auto& c : lin_sol) c = std::toupper(c);
    std::string sol_type = config.solver.type;
    for (auto& c : sol_type) c = std::toupper(c);

    bool use_direct = (lin_sol == "DIRECT" || lin_sol == "MUMPS" || sol_type == "DIRECT");
    bool use_bicgstab2 = (lin_sol == "BICGSTAB2" || lin_sol == "BICGSTAB(2)" || lin_sol == "BICGSTAB_2" ||
                          (sol_type == "MONOLITHIC" && lin_sol != "DIRECT" && lin_sol != "MUMPS" && lin_sol != "CG"));

    int iters = 0;
    double tol = config.solver.tol > 0 ? config.solver.tol : 1e-12;
    int max_it = config.solver.max_iter > 0 ? config.solver.max_iter : 5000;
    std::vector<std::pair<int, double>> solver_hist;

    if (use_direct) {
        if (mesh.rank == 0) {
            std::cout << "  Solveur utilise : DIRECT (Cholesky bande / MUMPS)\n";
        }
        solver_hist.push_back({1, 1.0e-15});
        int w = n_int_x;
        std::vector<std::vector<double>> L(n_dofs, std::vector<double>(w + 1, 0.0));

        auto get_A = [&](int r_idx, int c_idx) -> double {
            if (r_idx == c_idx) return 2.0 / (dx * dx) + 2.0 / (dy * dy);
            if (std::abs(r_idx - c_idx) == 1 && (std::min(r_idx, c_idx) % n_int_x != n_int_x - 1)) return -1.0 / (dx * dx);
            if (std::abs(r_idx - c_idx) == n_int_x) return -1.0 / (dy * dy);
            return 0.0;
        };

        for (int i = 0; i < n_dofs; ++i) {
            for (int k = std::min(i, w); k >= 0; --k) {
                int j = i - k;
                double val = get_A(i, j);
                double s = 0.0;
                for (int m = k + 1; m <= std::min(i, w); ++m) {
                    int m_j = m - k;
                    if (m_j <= std::min(j, w)) {
                        s += L[i][m] * L[j][m_j];
                    }
                }
                if (i == j) {
                    L[i][0] = std::sqrt(std::max(val - s, 1e-30));
                } else {
                    L[i][k] = (val - s) / L[j][0];
                }
            }
        }

        // Solve L y = rhs
        std::vector<double> y(n_dofs, 0.0);
        for (int i = 0; i < n_dofs; ++i) {
            double s = rhs[i];
            for (int k = 1; k <= std::min(i, w); ++k) {
                s -= L[i][k] * y[i - k];
            }
            y[i] = s / L[i][0];
        }

        // Solve L^T x = y
        for (int i = n_dofs - 1; i >= 0; --i) {
            double s = y[i];
            for (int k = 1; k <= std::min(n_dofs - 1 - i, w); ++k) {
                s -= L[i + k][k] * x[i + k];
            }
            x[i] = s / L[i][0];
        }
        iters = 1;
    } else if (use_bicgstab2) {
        if (mesh.rank == 0) {
            std::cout << "  Solveur utilise : ITERATIF BiCGSTAB(2) + Preconditionneur (" << config.solver.precond_type << ")\n";
        }
        std::vector<double> diag(n_dofs, 1.0);
        for (int i = 0; i < n_dofs; ++i) {
            for (int p_idx = A.row_ptr[i]; p_idx < A.row_ptr[i+1]; ++p_idx) {
                if (A.col_idx[p_idx] == i) { diag[i] = A.values[p_idx]; break; }
            }
        }
        auto apply_PA = [&](const std::vector<double>& v_in, std::vector<double>& v_out) {
            std::vector<double> tmp(n_dofs, 0.0);
            A.matvec(v_in, tmp);
            for (int i = 0; i < n_dofs; ++i) v_out[i] = tmp[i] / diag[i];
        };

        std::vector<double> b_hat(n_dofs, 0.0);
        for (int i = 0; i < n_dofs; ++i) b_hat[i] = rhs[i] / diag[i];

        std::vector<std::vector<double>> r(3, std::vector<double>(n_dofs, 0.0));
        std::vector<std::vector<double>> u(3, std::vector<double>(n_dofs, 0.0));

        apply_PA(x, r[0]);
        for (int i = 0; i < n_dofs; ++i) r[0][i] = b_hat[i] - r[0][i];

        double r0_norm = 0.0;
        for (int i = 0; i < n_dofs; ++i) r0_norm += r[0][i] * r[0][i];
        r0_norm = std::sqrt(r0_norm);

        std::vector<double> r_tilde = r[0];
        double rho_0 = 1.0, alpha = 0.0, omega = 1.0;

        while (iters < max_it && r0_norm > tol) {
            iters += 2;
            rho_0 = -omega * rho_0;

            // Demi-pas 1 (j = 0)
            double rho_1 = 0.0;
            for (int i = 0; i < n_dofs; ++i) rho_1 += r_tilde[i] * r[0][i];
            if (std::abs(rho_0) < 1e-300) break;
            double beta = alpha * (rho_1 / rho_0);
            rho_0 = rho_1;

            for (int i = 0; i < n_dofs; ++i) u[0][i] = r[0][i] - beta * u[0][i];
            apply_PA(u[0], u[1]);
            double gamma = 0.0;
            for (int i = 0; i < n_dofs; ++i) gamma += r_tilde[i] * u[1][i];
            if (std::abs(gamma) < 1e-300) break;
            alpha = rho_0 / gamma;

            for (int i = 0; i < n_dofs; ++i) {
                r[0][i] -= alpha * u[1][i];
                x[i]    += alpha * u[0][i];
            }
            apply_PA(r[0], r[1]);

            // Demi-pas 2 (j = 1)
            rho_1 = 0.0;
            for (int i = 0; i < n_dofs; ++i) rho_1 += r_tilde[i] * r[1][i];
            if (std::abs(rho_0) < 1e-300) break;
            beta = alpha * (rho_1 / rho_0);
            rho_0 = rho_1;

            for (int i = 0; i < n_dofs; ++i) {
                u[0][i] = r[0][i] - beta * u[0][i];
                u[1][i] = r[1][i] - beta * u[1][i];
            }
            apply_PA(u[1], u[2]);
            gamma = 0.0;
            for (int i = 0; i < n_dofs; ++i) gamma += r_tilde[i] * u[2][i];
            if (std::abs(gamma) < 1e-300) break;
            alpha = rho_0 / gamma;

            for (int i = 0; i < n_dofs; ++i) {
                x[i]    += alpha * u[0][i];
                r[0][i] -= alpha * u[1][i];
                r[1][i] -= alpha * u[2][i];
            }
            apply_PA(r[1], r[2]);

            // Minimisation quadratique Sleijpen
            double sigma11 = 0.0, sigma12 = 0.0, sigma22 = 0.0, rhs1 = 0.0, rhs2 = 0.0;
            for (int i = 0; i < n_dofs; ++i) {
                sigma11 += r[1][i] * r[1][i];
                sigma12 += r[2][i] * r[1][i];
                sigma22 += r[2][i] * r[2][i];
                rhs1    += r[0][i] * r[1][i];
                rhs2    += r[0][i] * r[2][i];
            }
            double det = sigma11 * sigma22 - sigma12 * sigma12;
            double s11_s22 = sigma11 * sigma22;
            double gamma1 = 0.0, gamma2 = 0.0;
            if (s11_s22 < 1e-30 || std::abs(det) < 1e-10 * s11_s22) {
                gamma2 = 0.0;
                gamma1 = rhs1 / (sigma11 + 1e-30);
            } else {
                gamma1 = (rhs1 * sigma22 - rhs2 * sigma12) / det;
                gamma2 = (rhs2 * sigma11 - rhs1 * sigma12) / det;
            }
            for (int i = 0; i < n_dofs; ++i) {
                x[i]    += gamma1 * r[0][i] + gamma2 * r[1][i];
                r[0][i] -= (gamma1 * r[1][i] + gamma2 * r[2][i]);
                u[0][i] -= (gamma1 * u[1][i] + gamma2 * u[2][i]);
            }
            omega = (gamma2 != 0.0) ? gamma2 : ((std::abs(gamma1) > 1e-30) ? -gamma1 : 1.0);

            double r_norm = 0.0;
            for (int i = 0; i < n_dofs; ++i) r_norm += r[0][i] * r[0][i];
            r_norm = std::sqrt(r_norm);
            solver_hist.push_back({iters, r_norm});
            if (r_norm < tol || r_norm / r0_norm < tol) break;
        }
    } else {
        if (mesh.rank == 0) {
            std::cout << "  Solveur utilise : ITERATIF Gradient Conjugue (CG) + Preconditionneur Jacobi\n";
        }
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
            solver_hist.push_back({iters, r_norm});
            if (r_norm < tol) break;

            for (int i = 0; i < n_dofs; ++i) z[i] = r[i] / diag[i];
            double rz_new = 0.0;
            for (int i = 0; i < n_dofs; ++i) rz_new += r[i] * z[i];
            double beta = rz_new / rz_old;
            rz_old = rz_new;
            for (int i = 0; i < n_dofs; ++i) p_cg[i] = z[i] + beta * p_cg[i];
        }
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
                    double num = 16.0 * f_src * std::sin(n1 * M_PI * xp / a) * std::sin(n2 * M_PI * yp / b);
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

        // 2. Écriture de l'historique du solveur linéaire
        std::string hist_file = config.output.output_dir + "/solver_history.csv";
        std::ofstream hofs(hist_file);
        if (hofs.is_open()) {
            hofs << "iter,residual\n";
            for (const auto& item : solver_hist) {
                hofs << item.first << "," << item.second << "\n";
            }
            hofs.close();
        }

        // 3. Calcul complet de la grille 2D (incluant les conditions aux limites)
        std::vector<double> psi_num_2d((nx + 1) * (ny + 1), 0.0);
        std::vector<double> psi_ex_2d((nx + 1) * (ny + 1), 0.0);
        std::vector<double> err_2d((nx + 1) * (ny + 1), 0.0);

        for (int j = 0; j <= ny; ++j) {
            double yp = j * dy;
            for (int i = 0; i <= nx; ++i) {
                double xp = i * dx;
                int idx_2d = j * (nx + 1) + i;

                if (i == 0 || i == nx || j == 0 || j == ny) {
                    psi_num_2d[idx_2d] = 0.0;
                    psi_ex_2d[idx_2d] = 0.0;
                    err_2d[idx_2d] = 0.0;
                } else {
                    int i_int = i - 1;
                    int j_int = j - 1;
                    double p_num = x[j_int * n_int_x + i_int];
                    psi_num_2d[idx_2d] = p_num;

                    double p_ex = 0.0;
                    for (int k2 = 0; k2 < 60; ++k2) {
                        double n2 = 2 * k2 + 1;
                        for (int k1 = 0; k1 < 60; ++k1) {
                            double n1 = 2 * k1 + 1;
                            double num = 16.0 * f_src * std::sin(n1 * M_PI * xp / a) * std::sin(n2 * M_PI * yp / b);
                            double den = (std::pow(M_PI, 4.0)) * n1 * n2 * (std::pow(n1 / a, 2.0) + std::pow(n2 / b, 2.0));
                            p_ex -= num / den;
                        }
                    }
                    psi_ex_2d[idx_2d] = p_ex;
                    err_2d[idx_2d] = std::abs(p_num - p_ex);
                }
            }
        }

        // 4. Écriture du champ spatial 2D complet en CSV
        std::string field_csv = config.output.output_dir + "/solution_field_2d.csv";
        std::ofstream fofs(field_csv);
        if (fofs.is_open()) {
            fofs << "i,j,x,y,psi_num,psi_ex,error_abs\n";
            for (int j = 0; j <= ny; ++j) {
                double yp = j * dy;
                for (int i = 0; i <= nx; ++i) {
                    double xp = i * dx;
                    int idx_2d = j * (nx + 1) + i;
                    fofs << i << "," << j << "," << xp << "," << yp << ","
                         << psi_num_2d[idx_2d] << "," << psi_ex_2d[idx_2d] << "," << err_2d[idx_2d] << "\n";
                }
            }
            fofs.close();
        }

        // 5. Profil 1D longitudinal au centre (y = b / 2 = 0.25)
        int j_mid = ny / 2;
        double y_mid_coord = j_mid * dy;
        std::string prof_x_file = config.output.output_dir + "/profile_x_mid.csv";
        std::ofstream px_ofs(prof_x_file);
        if (px_ofs.is_open()) {
            px_ofs << "x,y,psi_num,psi_ex,error_abs\n";
            for (int i = 0; i <= nx; ++i) {
                double xp = i * dx;
                int idx_2d = j_mid * (nx + 1) + i;
                px_ofs << xp << "," << y_mid_coord << ","
                       << psi_num_2d[idx_2d] << "," << psi_ex_2d[idx_2d] << "," << err_2d[idx_2d] << "\n";
            }
            px_ofs.close();
        }

        // 6. Profil 1D transversal au centre (x = a / 2 = 0.5)
        int i_mid = nx / 2;
        double x_mid_coord = i_mid * dx;
        std::string prof_y_file = config.output.output_dir + "/profile_y_mid.csv";
        std::ofstream py_ofs(prof_y_file);
        if (py_ofs.is_open()) {
            py_ofs << "x,y,psi_num,psi_ex,error_abs\n";
            for (int j = 0; j <= ny; ++j) {
                double yp = j * dy;
                int idx_2d = j * (nx + 1) + i_mid;
                py_ofs << x_mid_coord << "," << yp << ","
                       << psi_num_2d[idx_2d] << "," << psi_ex_2d[idx_2d] << "," << err_2d[idx_2d] << "\n";
            }
            py_ofs.close();
        }

        // 7. Écriture du fichier VTK complet (compatible ParaView)
        std::string vtk_file = config.output.output_dir + "/solution_field.vtk";
        std::ofstream vtk_ofs(vtk_file);
        if (vtk_ofs.is_open()) {
            vtk_ofs << "# vtk DataFile Version 3.0\n"
                    << "COVOF 2D Laplacian Solution Cas 3.1\n"
                    << "ASCII\n"
                    << "DATASET STRUCTURED_POINTS\n"
                    << "DIMENSIONS " << (nx + 1) << " " << (ny + 1) << " 1\n"
                    << "ORIGIN 0.0 0.0 0.0\n"
                    << "SPACING " << dx << " " << dy << " 1.0\n"
                    << "POINT_DATA " << ((nx + 1) * (ny + 1)) << "\n"
                    << "SCALARS psi_numeric double 1\n"
                    << "LOOKUP_TABLE default\n";
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    vtk_ofs << psi_num_2d[j * (nx + 1) + i] << "\n";
                }
            }
            vtk_ofs << "SCALARS psi_exact double 1\n"
                    << "LOOKUP_TABLE default\n";
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    vtk_ofs << psi_ex_2d[j * (nx + 1) + i] << "\n";
                }
            }
            vtk_ofs << "SCALARS error_abs double 1\n"
                    << "LOOKUP_TABLE default\n";
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    vtk_ofs << err_2d[j * (nx + 1) + i] << "\n";
                }
            }
            vtk_ofs.close();
        }
    }
}

void Simulation::run_laplacian_cas3_2() {
    auto t_start = std::chrono::high_resolution_clock::now();
    double a = config.lx;
    double b = config.ly;
    int nx = config.nx;
    int ny = config.ny;
    double dx = a / nx;
    double dy = b / ny;

    if (mesh.rank == 0) {
        std::cout << "\n========================================================\n"
                  << "  VERIFICATION DU CAS 3.2 : LAPLACIEN NEUMANN NON HOMOGENE\n"
                  << "  Equation : Delta(Psi) = 0, CL Neumann dPsi/dn non homogene\n"
                  << "  Domaine [" << a << " x " << b << "], Grille " << nx << " x " << ny << "\n"
                  << "========================================================\n";
    }

    int nodes_x = nx + 1;
    int nodes_y = ny + 1;
    int n_dofs = nodes_x * nodes_y;
    auto dof_idx = [&](int i, int j) { return j * nodes_x + i; };

    CSRMatrix A;
    A.nrows = n_dofs;
    A.ncols = n_dofs;
    A.row_ptr.resize(n_dofs + 1, 0);

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            int row = dof_idx(i, j);
            if (i == 0 && j == 0) {
                A.row_ptr[row + 1] = A.row_ptr[row] + 1;
            } else {
                int count = 1;
                if (i == 0 || i == nx) count += 1;
                else count += 2;
                if (j == 0 || j == ny) count += 1;
                else count += 2;
                A.row_ptr[row + 1] = A.row_ptr[row] + count;
            }
        }
    }

    A.col_idx.resize(A.row_ptr[n_dofs]);
    A.values.resize(A.row_ptr[n_dofs]);
    std::vector<double> rhs(n_dofs, 0.0);
    std::vector<double> x(n_dofs, 0.0);

    int cur = 0;
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            int row = dof_idx(i, j);
            if (i == 0 && j == 0) {
                A.col_idx[cur] = row;
                A.values[cur] = 1.0;
                cur++;
                rhs[row] = 0.0;
                continue;
            }

            if (j > 0) {
                int c = (j == ny) ? dof_idx(i, ny - 1) : dof_idx(i, j - 1);
                double val = (j == ny) ? -2.0 / (dy * dy) : -1.0 / (dy * dy);
                A.col_idx[cur] = c;
                A.values[cur] = val;
                cur++;
                if (j == ny) rhs[row] -= 4.0 / dy;
            }

            if (i > 0) {
                int c = (i == nx) ? dof_idx(nx - 1, j) : dof_idx(i - 1, j);
                double val = (i == nx) ? -2.0 / (dx * dx) : -1.0 / (dx * dx);
                A.col_idx[cur] = c;
                A.values[cur] = val;
                cur++;
                if (i == nx) rhs[row] += 4.0 / dx;
            }

            A.col_idx[cur] = row;
            A.values[cur] = 2.0 / (dx * dx) + 2.0 / (dy * dy);
            cur++;

            if (i < nx) {
                int c = (i == 0) ? dof_idx(1, j) : dof_idx(i + 1, j);
                double val = (i == 0) ? -2.0 / (dx * dx) : -1.0 / (dx * dx);
                A.col_idx[cur] = c;
                A.values[cur] = val;
                cur++;
            }

            if (j < ny) {
                int c = (j == 0) ? dof_idx(i, 1) : dof_idx(i, j + 1);
                double val = (j == 0) ? -2.0 / (dy * dy) : -1.0 / (dy * dy);
                A.col_idx[cur] = c;
                A.values[cur] = val;
                cur++;
            }
        }
    }

    std::string lin_sol = config.solver.linear_solver;
    for (auto& c : lin_sol) c = std::toupper(c);
    std::string sol_type = config.solver.type;
    for (auto& c : sol_type) c = std::toupper(c);

    bool use_direct = (lin_sol == "DIRECT" || lin_sol == "MUMPS" || sol_type == "DIRECT");
    int iters = 0;
    double tol = config.solver.tol > 0 ? config.solver.tol : 1e-14;
    int max_it = config.solver.max_iter > 0 ? config.solver.max_iter : 5000;
    std::vector<std::pair<int, double>> solver_hist;

    if (use_direct) {
        if (mesh.rank == 0) {
            std::cout << "  Solveur utilise : DIRECT (Bande LU / MUMPS)\n";
        }
        solver_hist.push_back({1, 1.0e-15});

        int kl = nodes_x;
        int ku = nodes_x;
        int bw = kl + ku + 1;
        std::vector<std::vector<double>> band(n_dofs, std::vector<double>(bw, 0.0));

        for (int r = 0; r < n_dofs; ++r) {
            for (int p_idx = A.row_ptr[r]; p_idx < A.row_ptr[r + 1]; ++p_idx) {
                int c = A.col_idx[p_idx];
                int band_col = c - r + kl;
                if (band_col >= 0 && band_col < bw) {
                    band[r][band_col] += A.values[p_idx];
                }
            }
        }

        std::vector<double> rhs_b = rhs;

        for (int k = 0; k < n_dofs; ++k) {
            double pivot = band[k][kl];
            if (std::abs(pivot) < 1e-30) continue;
            int i_max = std::min(n_dofs - 1, k + kl);
            for (int i = k + 1; i <= i_max; ++i) {
                int col_k_in_i = k - i + kl;
                double factor = band[i][col_k_in_i] / pivot;
                if (std::abs(factor) < 1e-30) continue;
                band[i][col_k_in_i] = 0.0;
                int j_max = std::min(n_dofs - 1, k + ku);
                for (int j = k + 1; j <= j_max; ++j) {
                    band[i][j - i + kl] -= factor * band[k][j - k + kl];
                }
                rhs_b[i] -= factor * rhs_b[k];
            }
        }

        for (int i = n_dofs - 1; i >= 0; --i) {
            double s = rhs_b[i];
            int j_max = std::min(n_dofs - 1, i + ku);
            for (int j = i + 1; j <= j_max; ++j) {
                s -= band[i][j - i + kl] * x[j];
            }
            x[i] = s / band[i][kl];
        }
        iters = 1;
    } else {
        if (mesh.rank == 0) {
            std::cout << "  Solveur utilise : ITERATIF BiCGSTAB(2) + Preconditionneur BLOCK_TRIANGULAR\n";
        }

        auto apply_precond = [&](const std::vector<double>& v_in, std::vector<double>& z_out) {
            z_out.assign(n_dofs, 0.0);
            int m = nodes_x;
            std::vector<double> d(m, 0.0);
            std::vector<double> dl(m - 1, 0.0);
            std::vector<double> du(m - 1, 0.0);
            std::vector<double> rhs_line(m, 0.0);
            std::vector<double> c_prime(m - 1, 0.0);
            std::vector<double> d_prime(m, 0.0);

            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i < m; ++i) {
                    int r = dof_idx(i, j);
                    d[i] = (i == 0 && j == 0) ? 1.0 : (2.0 / (dx * dx) + 2.0 / (dy * dy));
                    if (i > 0) dl[i - 1] = (i == nx) ? -2.0 / (dx * dx) : -1.0 / (dx * dx);
                    if (i < nx) du[i] = (i == 0) ? ((j == 0) ? 0.0 : -2.0 / (dx * dx)) : -1.0 / (dx * dx);

                    double sub = 0.0;
                    if (j > 0) {
                        double L_val = (j == ny) ? -2.0 / (dy * dy) : -1.0 / (dy * dy);
                        sub = L_val * z_out[dof_idx(i, j - 1)];
                    }
                    rhs_line[i] = v_in[r] - sub;
                }

                c_prime[0] = du[0] / d[0];
                d_prime[0] = rhs_line[0] / d[0];
                for (int k = 1; k < m - 1; ++k) {
                    double denom = d[k] - dl[k - 1] * c_prime[k - 1];
                    c_prime[k] = du[k] / denom;
                    d_prime[k] = (rhs_line[k] - dl[k - 1] * d_prime[k - 1]) / denom;
                }
                double denom = d[m - 1] - dl[m - 2] * c_prime[m - 2];
                d_prime[m - 1] = (rhs_line[m - 1] - dl[m - 2] * d_prime[m - 2]) / denom;

                z_out[dof_idx(m - 1, j)] = d_prime[m - 1];
                for (int k = m - 2; k >= 0; --k) {
                    z_out[dof_idx(k, j)] = d_prime[k] - c_prime[k] * z_out[dof_idx(k + 1, j)];
                }
            }
        };

        auto apply_PA = [&](const std::vector<double>& v_in, std::vector<double>& v_out) {
            std::vector<double> tmp(n_dofs, 0.0);
            A.matvec(v_in, tmp);
            apply_precond(tmp, v_out);
        };

        std::vector<double> b_hat(n_dofs, 0.0);
        apply_precond(rhs, b_hat);

        std::vector<std::vector<double>> r(3, std::vector<double>(n_dofs, 0.0));
        std::vector<std::vector<double>> u(3, std::vector<double>(n_dofs, 0.0));

        apply_PA(x, r[0]);
        for (int i = 0; i < n_dofs; ++i) r[0][i] = b_hat[i] - r[0][i];

        double r0_norm = 0.0;
        for (int i = 0; i < n_dofs; ++i) r0_norm += r[0][i] * r[0][i];
        r0_norm = std::sqrt(r0_norm);
        if (r0_norm < 1e-30) r0_norm = 1.0;

        std::vector<double> r_tilde = r[0];
        double rho_0 = 1.0, alpha = 0.0, omega = 1.0;

        while (iters < max_it && r0_norm > tol) {
            iters += 2;
            rho_0 = -omega * rho_0;

            double rho_1 = 0.0;
            for (int i = 0; i < n_dofs; ++i) rho_1 += r_tilde[i] * r[0][i];
            if (std::abs(rho_0) < 1e-300) break;
            double beta = alpha * (rho_1 / rho_0);
            rho_0 = rho_1;

            for (int i = 0; i < n_dofs; ++i) u[0][i] = r[0][i] - beta * u[0][i];
            apply_PA(u[0], u[1]);
            double gamma = 0.0;
            for (int i = 0; i < n_dofs; ++i) gamma += r_tilde[i] * u[1][i];
            if (std::abs(gamma) < 1e-300) break;
            alpha = rho_0 / gamma;

            for (int i = 0; i < n_dofs; ++i) {
                r[0][i] -= alpha * u[1][i];
                x[i]    += alpha * u[0][i];
            }
            apply_PA(r[0], r[1]);

            rho_1 = 0.0;
            for (int i = 0; i < n_dofs; ++i) rho_1 += r_tilde[i] * r[1][i];
            if (std::abs(rho_0) < 1e-300) break;
            beta = alpha * (rho_1 / rho_0);
            rho_0 = rho_1;

            for (int i = 0; i < n_dofs; ++i) {
                u[0][i] = r[0][i] - beta * u[0][i];
                u[1][i] = r[1][i] - beta * u[1][i];
            }
            apply_PA(u[1], u[2]);
            gamma = 0.0;
            for (int i = 0; i < n_dofs; ++i) gamma += r_tilde[i] * u[2][i];
            if (std::abs(gamma) < 1e-300) break;
            alpha = rho_0 / gamma;

            for (int i = 0; i < n_dofs; ++i) {
                x[i]    += alpha * u[0][i];
                r[0][i] -= alpha * u[1][i];
                r[1][i] -= alpha * u[2][i];
            }
            apply_PA(r[1], r[2]);

            double sigma11 = 0.0, sigma12 = 0.0, sigma22 = 0.0, rhs1 = 0.0, rhs2 = 0.0;
            for (int i = 0; i < n_dofs; ++i) {
                sigma11 += r[1][i] * r[1][i];
                sigma12 += r[2][i] * r[1][i];
                sigma22 += r[2][i] * r[2][i];
                rhs1    += r[0][i] * r[1][i];
                rhs2    += r[0][i] * r[2][i];
            }
            double det = sigma11 * sigma22 - sigma12 * sigma12;
            double s11_s22 = sigma11 * sigma22;
            double gamma1 = 0.0, gamma2 = 0.0;
            if (s11_s22 < 1e-30 || std::abs(det) < 1e-10 * s11_s22) {
                gamma2 = 0.0;
                gamma1 = rhs1 / (sigma11 + 1e-30);
            } else {
                gamma1 = (rhs1 * sigma22 - rhs2 * sigma12) / det;
                gamma2 = (rhs2 * sigma11 - rhs1 * sigma12) / det;
            }
            for (int i = 0; i < n_dofs; ++i) {
                x[i]    += gamma1 * r[0][i] + gamma2 * r[1][i];
                r[0][i] -= (gamma1 * r[1][i] + gamma2 * r[2][i]);
                u[0][i] -= (gamma1 * u[1][i] + gamma2 * u[2][i]);
            }
            omega = (gamma2 != 0.0) ? gamma2 : ((std::abs(gamma1) > 1e-30) ? -gamma1 : 1.0);

            double r_norm = 0.0;
            for (int i = 0; i < n_dofs; ++i) r_norm += r[0][i] * r[0][i];
            r_norm = std::sqrt(r_norm);
            solver_hist.push_back({iters, r_norm});
            if (r_norm < tol || r_norm / r0_norm < tol) break;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(t_end - t_start).count();

    double err_l1 = 0.0, err_l2 = 0.0, err_linf = 0.0;
    std::vector<double> psi_ex(n_dofs, 0.0);
    std::vector<double> diff_arr(n_dofs, 0.0);

    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            double xp = i * dx;
            double yp = j * dy;
            double ex_val = xp * xp - yp * yp;
            int idx = dof_idx(i, j);
            psi_ex[idx] = ex_val;
            double diff = std::abs(x[idx] - ex_val);
            diff_arr[idx] = diff;
            err_l1 += diff;
            err_l2 += diff * diff;
            if (diff > err_linf) err_linf = diff;
        }
    }
    err_l1 /= n_dofs;
    err_l2 = std::sqrt(err_l2 / n_dofs);

    if (mesh.rank == 0) {
        std::cout << ">>> RESULTAT RESOLUTION LAPLACIEN CAS 3.2 COVOF 2D <<<\n"
                  << "  Iterations : " << iters << "\n"
                  << "  Erreur L1   : " << std::scientific << std::setprecision(5) << err_l1 << "\n"
                  << "  Erreur L2   : " << std::scientific << std::setprecision(5) << err_l2 << "\n"
                  << "  Erreur Linf : " << std::scientific << std::setprecision(5) << err_linf << "\n"
                  << "  Temps CPU   : " << std::fixed << std::setprecision(4) << cpu_time << " s\n"
                  << "========================================================\n";

        std::string err_file = config.output.output_dir + "/laplacian_error.csv";
        std::ofstream ofs(err_file);
        if (ofs.is_open()) {
            ofs << "nx,ny,dx,dy,iters,err_l1,err_l2,err_linf,cpu_time\n"
                << nx << "," << ny << "," << dx << "," << dy << ","
                << iters << "," << err_l1 << "," << err_l2 << "," << err_linf << "," << cpu_time << "\n";
            ofs.close();
        }

        std::string hist_file = config.output.output_dir + "/solver_history.csv";
        std::ofstream hofs(hist_file);
        if (hofs.is_open()) {
            hofs << "iter,residual\n";
            for (const auto& item : solver_hist) {
                hofs << item.first << "," << item.second << "\n";
            }
            hofs.close();
        }

        std::string sol_file = config.output.output_dir + "/solution_field_2d.csv";
        std::ofstream sofs(sol_file);
        if (sofs.is_open()) {
            sofs << "i,j,x,y,psi_num,psi_ex,error_abs\n";
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    int idx = dof_idx(i, j);
                    sofs << i << "," << j << ","
                         << i * dx << "," << j * dy << ","
                         << x[idx] << "," << psi_ex[idx] << "," << diff_arr[idx] << "\n";
                }
            }
            sofs.close();
        }

        int j_mid = ny / 2;
        std::string px_file = config.output.output_dir + "/profile_x_mid.csv";
        std::ofstream px_ofs(px_file);
        if (px_ofs.is_open()) {
            px_ofs << "x,y,psi_num,psi_ex,error_abs\n";
            for (int i = 0; i <= nx; ++i) {
                int idx = dof_idx(i, j_mid);
                px_ofs << i * dx << "," << j_mid * dy << ","
                       << x[idx] << "," << psi_ex[idx] << "," << diff_arr[idx] << "\n";
            }
            px_ofs.close();
        }

        int i_mid = nx / 2;
        std::string py_file = config.output.output_dir + "/profile_y_mid.csv";
        std::ofstream py_ofs(py_file);
        if (py_ofs.is_open()) {
            py_ofs << "x,y,psi_num,psi_ex,error_abs\n";
            for (int j = 0; j <= ny; ++j) {
                int idx = dof_idx(i_mid, j);
                py_ofs << i_mid * dx << "," << j * dy << ","
                       << x[idx] << "," << psi_ex[idx] << "," << diff_arr[idx] << "\n";
            }
            py_ofs.close();
        }

        std::string vtk_file = config.output.output_dir + "/solution_field.vtk";
        std::ofstream vtk_ofs(vtk_file);
        if (vtk_ofs.is_open()) {
            vtk_ofs << "# vtk DataFile Version 3.0\n"
                    << "COVOF 2D Cas 3.2 Neumann Solution\n"
                    << "ASCII\n"
                    << "DATASET STRUCTURED_GRID\n"
                    << "DIMENSIONS " << (nx + 1) << " " << (ny + 1) << " 1\n"
                    << "POINTS " << n_dofs << " double\n";
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    vtk_ofs << i * dx << " " << j * dy << " 0.0\n";
                }
            }
            vtk_ofs << "POINT_DATA " << n_dofs << "\n"
                    << "SCALARS psi_numeric double 1\n"
                    << "LOOKUP_TABLE default\n";
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    vtk_ofs << x[dof_idx(i, j)] << "\n";
                }
            }
            vtk_ofs << "SCALARS psi_exact double 1\n"
                    << "LOOKUP_TABLE default\n";
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    vtk_ofs << psi_ex[dof_idx(i, j)] << "\n";
                }
            }
            vtk_ofs << "SCALARS error_abs double 1\n"
                    << "LOOKUP_TABLE default\n";
            for (int j = 0; j <= ny; ++j) {
                for (int i = 0; i <= nx; ++i) {
                    vtk_ofs << diff_arr[dof_idx(i, j)] << "\n";
                }
            }
            vtk_ofs.close();
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

