#include "covof/NavierStokesSolver.hpp"
#include "covof/MpiComm.hpp"
#include "covof/LinearSolver.hpp"
#include "covof/Preconditioner.hpp"
#include "covof/EOS.hpp"
#include <cmath>
#include <iostream>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

void NavierStokesSolver::init(const Mesh2D& m, const Config& cfg) {
    // Calcul et assignation numérique
    mesh = m;
    // Calcul et assignation numérique
    bc = cfg.bc;
    // Calcul et assignation numérique
    enable_convection = cfg.enable_convection;
    // Calcul et assignation numérique
    enable_energy = cfg.enable_energy;
    // Calcul et assignation numérique
    gx = cfg.gx;
    // Calcul et assignation numérique
    gy = cfg.gy;
    // Calcul et assignation numérique
    beta = cfg.beta;
    // Calcul et assignation numérique
    T_ref = cfg.T_ref;
    // Calcul et assignation numérique
    Cp = cfg.fluid1.Cp;
    // Calcul et assignation numérique
    k_thermal = cfg.fluid1.k_thermal;
    // Calcul et assignation numérique
    Ra = cfg.Ra;
    // Calcul et assignation numérique
    Pr = cfg.Pr;
    // Calcul et assignation numérique
    solver_type = cfg.solver.type;
    // Calcul et assignation numérique
    r_al = cfg.solver.r_al;
    // Calcul et assignation numérique
    uzawa_iter = cfg.solver.uzawa_iter;
    compressible = cfg.compressible;
    enable_ssp_rk3_momentum = cfg.enable_ssp_rk3_momentum;
    // Calcul et assignation numérique
    obstacles = cfg.obstacles;

    // Déclaration de variable typée
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    // Calcul et assignation numérique
    n_owned_u = u_nx * mesh.ny_local;

    // Déclaration de variable typée
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
    // Calcul et assignation numérique
    n_owned_v = mesh.nx_local * v_ny;

    // Calcul et assignation numérique
    n_owned_p = mesh.nx_local * mesh.ny_local;

    // Évaluation de la condition de contrôle
    if (is_augmented_lagrangian() || is_standard_projection()) {
        // Calcul et assignation numérique
        n_owned_total = n_owned_u + n_owned_v;
    } else {
        // Calcul et assignation numérique
        n_owned_total = n_owned_u + n_owned_v + n_owned_p;
    }
}

void NavierStokesSolver::pack_vector(const FlowFields& fields, std::vector<double>& vec) const {
    vec.assign(n_owned_total, 0.0);

    // 1. Pack u
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= mesh.ny_local; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= u_nx; ++i) {
            // Déclaration de variable typée
            int dof = (j - 1) * u_nx + (i - 1);
            // Calcul et assignation numérique
            vec[dof] = fields.u[mesh.u_idx(i, j)];
        }
    }

    // 2. Pack v
    int v_offset = n_owned_u;
    // Déclaration de variable typée
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= v_ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= mesh.nx_local; ++i) {
            // Déclaration de variable typée
            int dof = (j - 1) * mesh.nx_local + (i - 1);
            // Calcul et assignation numérique
            vec[v_offset + dof] = fields.v[mesh.v_idx(i, j)];
        }
    }

    // 3. Pack p (si monolithique)
    if (is_monolithic()) {
        // Déclaration de variable typée
        int p_offset = n_owned_u + n_owned_v;
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                int dof = (j - 1) * mesh.nx_local + (i - 1);
                // Calcul et assignation numérique
                vec[p_offset + dof] = fields.p[mesh.cell_idx(i, j)];
            }
        }
    }
}

void NavierStokesSolver::unpack_vector(const std::vector<double>& vec, FlowFields& fields, bool homogeneous) const {
    // 1. Unpack u
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= mesh.ny_local; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= u_nx; ++i) {
            // Déclaration de variable typée
            int dof = (j - 1) * u_nx + (i - 1);
            // Calcul de la résolution géométrique de cellule
            fields.u[mesh.u_idx(i, j)] = vec[dof];
        }
    }

    // 2. Unpack v
    int v_offset = n_owned_u;
    // Déclaration de variable typée
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= v_ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= mesh.nx_local; ++i) {
            // Déclaration de variable typée
            int dof = (j - 1) * mesh.nx_local + (i - 1);
            // Calcul de la résolution géométrique de cellule
            fields.v[mesh.v_idx(i, j)] = vec[v_offset + dof];
        }
    }

    // 3. Unpack p (si monolithique)
    if (is_monolithic()) {
        // Déclaration de variable typée
        int p_offset = n_owned_u + n_owned_v;
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                int dof = (j - 1) * mesh.nx_local + (i - 1);
                // Calcul de la résolution géométrique de cellule
                fields.p[mesh.cell_idx(i, j)] = vec[p_offset + dof];
            }
        }
        MpiComm::sync_cell_scalar(mesh, fields.p);
    }

    // Synchronisation des mailles fantômes pour calculs ultérieurs
    MpiComm::sync_u(mesh, fields.u);
    MpiComm::sync_v(mesh, fields.v);
    MpiComm::apply_boundary_conditions(mesh, bc, fields, homogeneous);
}

void NavierStokesSolver::apply_system_matrix(const FlowFields& fields_old, double dt,
                                            const std::vector<double>& x, std::vector<double>& y) const {
    y.assign(n_owned_total, 0.0);

    FlowFields tf;
    tf.allocate(mesh);
    // Calcul et assignation numérique
    unpack_vector(x, tf, true); // true = CL homogènes pour garantir la stricte linéarité de A*x

    // Déclaration de variable typée
    double dx = mesh.dx;
    // Déclaration de variable typée
    double dy = mesh.dy;
    // Déclaration de variable typée
    double dvol = dx * dy;

    // Déclaration de variable typée
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    // Déclaration de variable typée
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);

    // Précalcul de la divergence sur chaque cellule locale + halo
    std::vector<double> div_cell((mesh.nx_local + 2) * (mesh.ny_local + 2), 0.0);
    // Calcul de la résolution géométrique de cellule
    auto c_idx = [&](int i, int j) { return j * (mesh.nx_local + 2) + i; };

    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 0; j <= mesh.ny_local + 1; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i <= mesh.nx_local + 1; ++i) {
            // Déclaration de variable typée
            double d = (tf.u[mesh.u_idx(i + 1, j)] - tf.u[mesh.u_idx(i, j)]) / dx +
                       (tf.v[mesh.v_idx(i, j + 1)] - tf.v[mesh.v_idx(i, j)]) / dy;
            // Calcul de la résolution géométrique de cellule
            div_cell[c_idx(i, j)] = d;
        }
    }

    // =========================================================================
    // 1. Équation de Quantité de Mouvement en u (Faces verticales)
    // =========================================================================
    for (int j = 1; j <= mesh.ny_local; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= u_nx; ++i) {
            // Déclaration de variable typée
            int dof = (j - 1) * u_nx + (i - 1);

            // Condition aux limites physique sur les parois gauche / droite ou obstacle solide
            if (i == 1 && mesh.is_boundary_left) {
                // Déclaration de variable typée
                double rho_u = fields_old.rho_u[mesh.u_idx(i, j)];
                // Déclaration de variable typée
                double diag_bc = (rho_u / dt) * dvol;
                // Évaluation de la condition de contrôle
                if (bc.left.type == BCType::NEUMANN) {
                    // Calcul et assignation numérique
                    y[dof] = diag_bc * (tf.u[mesh.u_idx(1, j)] - tf.u[mesh.u_idx(2, j)]);
                } else {
                    // Calcul et assignation numérique
                    y[dof] = diag_bc * tf.u[mesh.u_idx(1, j)];
                }
                continue;
            }
            // Évaluation de la condition de contrôle
            if (i == u_nx && mesh.is_boundary_right) {
                // Déclaration de variable typée
                double rho_u = fields_old.rho_u[mesh.u_idx(i, j)];
                // Déclaration de variable typée
                double diag_bc = (rho_u / dt) * dvol;
                // Évaluation de la condition de contrôle
                if (bc.right.type == BCType::NEUMANN) {
                    // Calcul et assignation numérique
                    y[dof] = diag_bc * (tf.u[mesh.u_idx(u_nx, j)] - tf.u[mesh.u_idx(u_nx - 1, j)]);
                } else {
                    // Calcul et assignation numérique
                    y[dof] = diag_bc * tf.u[mesh.u_idx(u_nx, j)];
                }
                continue;
            }
            // Évaluation de la condition de contrôle
            if (is_u_face_in_obstacle(i, j)) {
                // Déclaration de variable typée
                double rho_u = fields_old.rho_u[mesh.u_idx(i, j)];
                // Déclaration de variable typée
                double diag_bc = (rho_u / dt) * dvol;
                // Calcul et assignation numérique
                y[dof] = diag_bc * tf.u[mesh.u_idx(i, j)];
                continue;
            }

            // Déclaration de variable typée
            double rho_u = fields_old.rho_u[mesh.u_idx(i, j)];

            // Terme instationnaire
            double val = (rho_u / dt) * tf.u[mesh.u_idx(i, j)] * dvol;

            // Terme convectif semi-implicite F_uu (forme advective non-conservative a densite harmonique)
            if (enable_convection) {
                double u_n = fields_old.u[mesh.u_idx(i, j)];
                double u_n_E = fields_old.u[mesh.u_idx(i + 1, j)];
                double u_n_W = fields_old.u[mesh.u_idx(i - 1, j)];

                double v_n_N = 0.5 * (fields_old.v[mesh.v_idx(i - 1, j + 1)] + fields_old.v[mesh.v_idx(i, j + 1)]);
                double v_n_S = 0.5 * (fields_old.v[mesh.v_idx(i - 1, j)] + fields_old.v[mesh.v_idx(i, j)]);

                double rho_u_E = fields_old.rho_u[mesh.u_idx(i + 1, j)];
                double rho_u_W = fields_old.rho_u[mesh.u_idx(i - 1, j)];
                double rho_u_N = (j < mesh.ny_local) ? fields_old.rho_u[mesh.u_idx(i, j + 1)] : rho_u;
                double rho_u_S = (j > 1) ? fields_old.rho_u[mesh.u_idx(i, j - 1)] : rho_u;

                double rho_E = (2.0 * rho_u * rho_u_E) / std::max(1.0e-12, rho_u + rho_u_E);
                double rho_W = (2.0 * rho_u * rho_u_W) / std::max(1.0e-12, rho_u + rho_u_W);
                double rho_N = (2.0 * rho_u * rho_u_N) / std::max(1.0e-12, rho_u + rho_u_N);
                double rho_S = (2.0 * rho_u * rho_u_S) / std::max(1.0e-12, rho_u + rho_u_S);

                double du_E = 0.5 * (u_n + u_n_E) * 0.5 * (tf.u[mesh.u_idx(i + 1, j)] - tf.u[mesh.u_idx(i, j)]) * dy;
                double du_W = 0.5 * (u_n_W + u_n) * 0.5 * (tf.u[mesh.u_idx(i, j)] - tf.u[mesh.u_idx(i - 1, j)]) * dy;
                double du_N = v_n_N * 0.5 * (tf.u[mesh.u_idx(i, j + 1)] - tf.u[mesh.u_idx(i, j)]) * dx;
                double du_S = v_n_S * 0.5 * (tf.u[mesh.u_idx(i, j)] - tf.u[mesh.u_idx(i, j - 1)]) * dx;

                val += (rho_E * du_E + rho_W * du_W + rho_N * du_N + rho_S * du_S);
            }

            // Termes visqueux implicites d'ordre 2 (Caltagirone et al. - Annexe A)
            // (a) Viscosité d'élongation 2*mu d2u/dx2
            double mu_cell_E = fields_old.mu[mesh.cell_idx(i, j)];
            // Déclaration de variable typée
            double mu_cell_W = fields_old.mu[mesh.cell_idx(i - 1, j)];
            // Déclaration de variable typée
            double diff_elong_E = 2.0 * mu_cell_E * (tf.u[mesh.u_idx(i + 1, j)] - tf.u[mesh.u_idx(i, j)]) / dx * dy;
            // Déclaration de variable typée
            double diff_elong_W = 2.0 * mu_cell_W * (tf.u[mesh.u_idx(i, j)] - tf.u[mesh.u_idx(i - 1, j)]) / dx * dy;
            // Calcul et assignation numérique
            val -= (diff_elong_E - diff_elong_W);

            // (b) Viscosité de cisaillement et rotation mu (d2u/dy2 + d2v/dxdy)
            double mu_node_N = fields_old.mu_node[mesh.node_idx(i, j + 1)];
            // Déclaration de variable typée
            double mu_node_S = fields_old.mu_node[mesh.node_idx(i, j)];

            // Déclaration de variable typée
            double shear_N = mu_node_N * ((tf.u[mesh.u_idx(i, j + 1)] - tf.u[mesh.u_idx(i, j)]) / dy +
                                          (tf.v[mesh.v_idx(i, j + 1)] - tf.v[mesh.v_idx(i - 1, j + 1)]) / dx) * dx;
            // Déclaration de variable typée
            double shear_S = mu_node_S * ((tf.u[mesh.u_idx(i, j)] - tf.u[mesh.u_idx(i, j - 1)]) / dy +
                                          (tf.v[mesh.v_idx(i, j)] - tf.v[mesh.v_idx(i - 1, j)]) / dx) * dx;
            // Calcul et assignation numérique
            val -= (shear_N - shear_S);

            // Terme de pénalité de Lagrangien augmenté : - r * d(div)/dx * dvol = - r * (div_E - div_W) * dy
            if (is_augmented_lagrangian()) {
                double grad_div_x = (div_cell[c_idx(i, j)] - div_cell[c_idx(i - 1, j)]) * dy;
                val -= r_al * grad_div_x;
            }

            // Gradient de pression discret B_u^T (uniquement si monolithique)
            if (is_monolithic()) {
                // Déclaration de variable typée
                double grad_p = (tf.p[mesh.cell_idx(i, j)] - tf.p[mesh.cell_idx(i - 1, j)]) * dy;
                // Calcul et assignation numérique
                val += grad_p;
            }

            // Calcul et assignation numérique
            y[dof] = val;
        }
    }

    // =========================================================================
    // 2. Équation de Quantité de Mouvement en v (Faces horizontales)
    // =========================================================================
    int v_offset = n_owned_u;
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= v_ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= mesh.nx_local; ++i) {
            // Déclaration de variable typée
            int dof = v_offset + (j - 1) * mesh.nx_local + (i - 1);

            // Condition aux limites physique Dirichlet ou Free-Slip sur les parois basse / haute ou obstacle solide
            bool is_wall_bot = (j == 1 && mesh.is_boundary_bottom && (bc.bottom.type == BCType::DIRICHLET || bc.bottom.type == BCType::FREE_SLIP));
            // Déclaration de variable typée
            bool is_wall_top = (j == v_ny && mesh.is_boundary_top && (bc.top.type == BCType::DIRICHLET || bc.top.type == BCType::FREE_SLIP));
            // Déclaration de variable typée
            bool in_obs = is_v_face_in_obstacle(i, j);
            // Évaluation de la condition de contrôle
            if (is_wall_bot || is_wall_top || in_obs) {
                // Déclaration de variable typée
                double rho_v = fields_old.rho_v[mesh.v_idx(i, j)];
                // Déclaration de variable typée
                double diag_bc = (rho_v / dt) * dvol;
                // Calcul et assignation numérique
                y[dof] = diag_bc * tf.v[mesh.v_idx(i, j)];
                continue;
            }

            // Déclaration de variable typée
            double rho_v = fields_old.rho_v[mesh.v_idx(i, j)];

            // Terme instationnaire
            double val = (rho_v / dt) * tf.v[mesh.v_idx(i, j)] * dvol;

            // Terme convectif semi-implicite F_vv (forme advective non-conservative a densite harmonique)
            if (enable_convection) {
                double v_n = fields_old.v[mesh.v_idx(i, j)];
                double v_n_N = fields_old.v[mesh.v_idx(i, j + 1)];
                double v_n_S = fields_old.v[mesh.v_idx(i, j - 1)];

                double u_n_E = 0.5 * (fields_old.u[mesh.u_idx(i + 1, j - 1)] + fields_old.u[mesh.u_idx(i + 1, j)]);
                double u_n_W = 0.5 * (fields_old.u[mesh.u_idx(i, j - 1)] + fields_old.u[mesh.u_idx(i, j)]);

                double rho_v_E = (i < mesh.nx_local) ? fields_old.rho_v[mesh.v_idx(i + 1, j)] : rho_v;
                double rho_v_W = (i > 1) ? fields_old.rho_v[mesh.v_idx(i - 1, j)] : rho_v;
                double rho_v_N = fields_old.rho_v[mesh.v_idx(i, j + 1)];
                double rho_v_S = fields_old.rho_v[mesh.v_idx(i, j - 1)];

                double rho_E = (2.0 * rho_v * rho_v_E) / std::max(1.0e-12, rho_v + rho_v_E);
                double rho_W = (2.0 * rho_v * rho_v_W) / std::max(1.0e-12, rho_v + rho_v_W);
                double rho_N = (2.0 * rho_v * rho_v_N) / std::max(1.0e-12, rho_v + rho_v_N);
                double rho_S = (2.0 * rho_v * rho_v_S) / std::max(1.0e-12, rho_v + rho_v_S);

                double dv_E = u_n_E * 0.5 * (tf.v[mesh.v_idx(i + 1, j)] - tf.v[mesh.v_idx(i, j)]) * dy;
                double dv_W = u_n_W * 0.5 * (tf.v[mesh.v_idx(i, j)] - tf.v[mesh.v_idx(i - 1, j)]) * dy;
                double dv_N = 0.5 * (v_n + v_n_N) * 0.5 * (tf.v[mesh.v_idx(i, j + 1)] - tf.v[mesh.v_idx(i, j)]) * dx;
                double dv_S = 0.5 * (v_n_S + v_n) * 0.5 * (tf.v[mesh.v_idx(i, j)] - tf.v[mesh.v_idx(i, j - 1)]) * dx;

                val += (rho_E * dv_E + rho_W * dv_W + rho_N * dv_N + rho_S * dv_S);
            }

            // Termes visqueux implicites d'ordre 2
            // (a) Viscosité d'élongation 2*mu d2v/dy2
            double mu_cell_N = fields_old.mu[mesh.cell_idx(i, j)];
            // Déclaration de variable typée
            double mu_cell_S = fields_old.mu[mesh.cell_idx(i, j - 1)];
            // Déclaration de variable typée
            double diff_elong_N = 2.0 * mu_cell_N * (tf.v[mesh.v_idx(i, j + 1)] - tf.v[mesh.v_idx(i, j)]) / dy * dx;
            // Déclaration de variable typée
            double diff_elong_S = 2.0 * mu_cell_S * (tf.v[mesh.v_idx(i, j)] - tf.v[mesh.v_idx(i, j - 1)]) / dy * dx;
            // Calcul et assignation numérique
            val -= (diff_elong_N - diff_elong_S);

            // (b) Viscosité de cisaillement et rotation mu (d2v/dx2 + d2u/dxdy)
            double mu_node_E = fields_old.mu_node[mesh.node_idx(i + 1, j)];
            // Déclaration de variable typée
            double mu_node_W = fields_old.mu_node[mesh.node_idx(i, j)];

            // Déclaration de variable typée
            double shear_E = mu_node_E * ((tf.v[mesh.v_idx(i + 1, j)] - tf.v[mesh.v_idx(i, j)]) / dx +
                                          (tf.u[mesh.u_idx(i + 1, j)] - tf.u[mesh.u_idx(i + 1, j - 1)]) / dy) * dy;
            // Déclaration de variable typée
            double shear_W = mu_node_W * ((tf.v[mesh.v_idx(i, j)] - tf.v[mesh.v_idx(i - 1, j)]) / dx +
                                          (tf.u[mesh.u_idx(i, j)] - tf.u[mesh.u_idx(i, j - 1)]) / dy) * dy;
            // Calcul et assignation numérique
            val -= (shear_E - shear_W);

            // Terme de pénalité de Lagrangien augmenté : - r * d(div)/dy * dvol = - r * (div_N - div_S) * dx
            if (is_augmented_lagrangian()) {
                double grad_div_y = (div_cell[c_idx(i, j)] - div_cell[c_idx(i, j - 1)]) * dx;
                val -= r_al * grad_div_y;
            }

            // Gradient de pression discret B_v^T (uniquement si monolithique)
            if (is_monolithic()) {
                // Déclaration de variable typée
                double grad_p = (tf.p[mesh.cell_idx(i, j)] - tf.p[mesh.cell_idx(i, j - 1)]) * dx;
                // Calcul et assignation numérique
                val += grad_p;
            }

            // Calcul et assignation numérique
            y[dof] = val;
        }
    }

    // =========================================================================
    // 3. Équation de Continuité (si monolithique)
    // =========================================================================
    if (is_monolithic()) {
        // Déclaration de variable typée
        int p_offset = n_owned_u + n_owned_v;
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                int dof = p_offset + (j - 1) * mesh.nx_local + (i - 1);
                // Déclaration de variable typée
                double div = div_cell[c_idx(i, j)] * dvol;
                double val_p = div;

                // Formulation Compressible / All-Mach (Fluids 2024, Éq. 12a)
                if (compressible) {
                    int c = mesh.cell_idx(i, j);
                    double chi = (fields_old.chi_T.size() > (size_t)c) ? fields_old.chi_T[c] : 0.0;
                    // Terme temporel implicite de pression (chi / dt) * p^{n+1}
                    val_p += (chi / dt) * tf.p[c] * dvol;
                    // Terme convectif chi * (v^{n+1} . grad(p^n))
                    double u_c = 0.5 * (tf.u[mesh.u_idx(i, j)] + tf.u[mesh.u_idx(i + 1, j)]);
                    double v_c = 0.5 * (tf.v[mesh.v_idx(i, j)] + tf.v[mesh.v_idx(i, j + 1)]);
                    double dp_dx = (fields_old.p[mesh.cell_idx(i + 1, j)] - fields_old.p[mesh.cell_idx(i - 1, j)]) / (2.0 * dx);
                    double dp_dy = (fields_old.p[mesh.cell_idx(i, j + 1)] - fields_old.p[mesh.cell_idx(i, j - 1)]) / (2.0 * dy);
                    val_p += chi * (u_c * dp_dx + v_c * dp_dy) * dvol;
                }

                // Calcul et assignation numérique
                y[dof] = val_p;
            }
        }
    }
}

void NavierStokesSolver::assemble_rhs(const FlowFields& fields, double dt, std::vector<double>& rhs) const {
    rhs.assign(n_owned_total, 0.0);

    // Déclaration de variable typée
    double dx = mesh.dx;
    // Déclaration de variable typée
    double dy = mesh.dy;
    // Déclaration de variable typée
    double dvol = dx * dy;

    // Déclaration de variable typée
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    // Déclaration de variable typée
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);

    // RHS pour u
    for (int j = 1; j <= mesh.ny_local; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= u_nx; ++i) {
            // Déclaration de variable typée
            int dof = (j - 1) * u_nx + (i - 1);
            // Déclaration de variable typée
            double rho_u = fields.rho_u[mesh.u_idx(i, j)];
            // Déclaration de variable typée
            double diag_bc = (rho_u / dt) * dvol;

            // Si face dans un obstacle solide (adhérence nulle)
            if (is_u_face_in_obstacle(i, j)) {
                // Calcul et assignation numérique
                rhs[dof] = 0.0;
                continue;
            }

            // Si paroi physique gauche
            if (i == 1 && mesh.is_boundary_left) {
                // Déclaration de variable typée
                double u_bc = 0.0;
                // Évaluation de la condition de contrôle
                if (bc.left.type == BCType::DIRICHLET) {
                    // Évaluation de la condition de contrôle
                    if (bc.left.profile == "POISEUILLE_STEP") {
                        // Déclaration de variable typée
                        double y = mesh.yc(j);
                        // Évaluation de la condition de contrôle
                        if (y >= bc.left.y_step) {
                            // Déclaration de variable typée
                            double h = mesh.ly - bc.left.y_step;
                            // Calcul et assignation numérique
                            u_bc = 6.0 * bc.left.u_mean * (y - bc.left.y_step) * (mesh.ly - y) / (h * h);
                        }
                    // Calcul et assignation numérique
                    } else if (bc.left.profile == "PARABOLIC") {
                        // Déclaration de variable typée
                        double y = mesh.yc(j);
                        // Déclaration de variable typée
                        double h = mesh.ly;
                        // Calcul et assignation numérique
                        u_bc = 6.0 * bc.left.u_mean * y * (h - y) / (h * h);
                    } else {
                        // Calcul et assignation numérique
                        u_bc = bc.left.u_val;
                    }
                // Calcul et assignation numérique
                } else if (bc.left.type == BCType::NEUMANN) {
                    // Calcul et assignation numérique
                    u_bc = 0.0;
                }
                // Calcul et assignation numérique
                rhs[dof] = diag_bc * u_bc;
                continue;
            }
            // Si paroi physique droite
            if (i == u_nx && mesh.is_boundary_right) {
                // Déclaration de variable typée
                double u_bc = 0.0;
                // Évaluation de la condition de contrôle
                if (bc.right.type == BCType::DIRICHLET) {
                    // Calcul et assignation numérique
                    u_bc = bc.right.u_val;
                // Calcul et assignation numérique
                } else if (bc.right.type == BCType::NEUMANN) {
                    // Calcul et assignation numérique
                    u_bc = 0.0;
                }
                // Calcul et assignation numérique
                rhs[dof] = diag_bc * u_bc;
                continue;
            }

            // Déclaration de variable typée
            double u_inertial = (enable_ssp_rk3_momentum && fields.rhou_star.size() > (size_t)mesh.u_idx(i, j))
                               ? fields.rhou_star[mesh.u_idx(i, j)]
                               : (rho_u * fields.u[mesh.u_idx(i, j)]);
            double val = (u_inertial / dt) * dvol;
            // Calcul et assignation numérique
            val += rho_u * gx * dvol;
            // Calcul et assignation numérique
            val += fields.F_sigma_x[mesh.u_idx(i, j)] * dvol;

            // Pour Lagrangien augmenté standard : le gradient de pression -grad(p^n) est explicite au second membre
            if (is_augmented_lagrangian()) {
                // Déclaration de variable typée
                double grad_p = (fields.p[mesh.cell_idx(i, j)] - fields.p[mesh.cell_idx(i - 1, j)]) * dy;
                // Calcul et assignation numérique
                val -= grad_p;
            }

            // Contribution de contrainte tangentielle à la paroi pour u (haut / bas)
            if (j == mesh.ny_local && mesh.is_boundary_top && bc.top.type == BCType::DIRICHLET) {
                // Déclaration de variable typée
                double mu_N = fields.mu_node[mesh.node_idx(i, j + 1)];
                // Calcul et assignation numérique
                val += mu_N * (2.0 * bc.top.u_val / dy) * dx;
            }
            // Évaluation de la condition de contrôle
            if (j == 1 && mesh.is_boundary_bottom && bc.bottom.type == BCType::DIRICHLET) {
                // Déclaration de variable typée
                double mu_S = fields.mu_node[mesh.node_idx(i, j)];
                // Calcul et assignation numérique
                val += mu_S * (2.0 * bc.bottom.u_val / dy) * dx;
            }

            // Calcul et assignation numérique
            rhs[dof] = val;
        }
    }

    // RHS pour v
    int v_offset = n_owned_u;
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= v_ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= mesh.nx_local; ++i) {
            // Déclaration de variable typée
            int dof = v_offset + (j - 1) * mesh.nx_local + (i - 1);
            // Déclaration de variable typée
            double rho_v = fields.rho_v[mesh.v_idx(i, j)];
            // Déclaration de variable typée
            double diag_bc = (rho_v / dt) * dvol;

            // Si face dans un obstacle solide (adhérence nulle)
            if (is_v_face_in_obstacle(i, j)) {
                // Calcul et assignation numérique
                rhs[dof] = 0.0;
                continue;
            }

            // Évaluation de la condition de contrôle
            if (j == 1 && mesh.is_boundary_bottom && (bc.bottom.type == BCType::DIRICHLET || bc.bottom.type == BCType::FREE_SLIP)) {
                // Déclaration de variable typée
                double v_bc = (bc.bottom.type == BCType::DIRICHLET) ? bc.bottom.v_val : 0.0;
                // Calcul et assignation numérique
                rhs[dof] = diag_bc * v_bc;
                continue;
            }
            // Évaluation de la condition de contrôle
            if (j == v_ny && mesh.is_boundary_top && (bc.top.type == BCType::DIRICHLET || bc.top.type == BCType::FREE_SLIP)) {
                // Déclaration de variable typée
                double v_bc = 0.0;
                // Évaluation de la condition d'injecteur (Cas 3.3 Fluids 2021)
                if (bc.top.profile == "JET_INLET") {
                    // Déclaration de variable typée
                    double x = mesh.xc(i);
                    // Déclaration de variable typée pour le centre et demi-largeur
                    double xc = (bc.top.jet_center > 0.0) ? bc.top.jet_center : 0.5 * mesh.lx;
                    // Calcul de la demi-largeur de l'injecteur
                    double half_w = (bc.top.jet_width > 0.0) ? 0.5 * bc.top.jet_width : 0.04;
                    double d_disc = 0.0;
                    for (int ig = 1; ig <= mesh.nx_global; ++ig) {
                        double xg = (ig - 0.5) * mesh.dx;
                        if (std::abs(xg - xc) <= half_w) d_disc += mesh.dx;
                    }
                    if (d_disc < 1.0e-12) d_disc = 2.0 * half_w;
                    // Évaluation de l'appartenance à l'injecteur
                    if (std::abs(x - xc) <= half_w) {
                        // Vitesse d'injection négative imposée (conservant le débit D * v0 exact)
                        v_bc = bc.top.v_val * (2.0 * half_w) / d_disc;
                    } else {
                        // Vitesse positive d'échappement compensatoire du gaz
                        v_bc = -bc.top.v_val * (2.0 * half_w) / (mesh.lx - d_disc);
                    }
                } else if (bc.top.type == BCType::DIRICHLET) {
                    // Calcul et assignation numérique standard
                    v_bc = bc.top.v_val;
                }
                // Calcul et assignation numérique
                rhs[dof] = diag_bc * v_bc;
                continue;
            }

            // Déclaration de variable typée
            double v_inertial = (enable_ssp_rk3_momentum && fields.rhov_star.size() > (size_t)mesh.v_idx(i, j))
                               ? fields.rhov_star[mesh.v_idx(i, j)]
                               : (rho_v * fields.v[mesh.v_idx(i, j)]);
            double val = (v_inertial / dt) * dvol;
            // Calcul et assignation numérique
            val += rho_v * gy * dvol;
            // Évaluation de la condition de contrôle
            if (enable_energy) {
                // Déclaration de variable typée
                double T_v = 0.5 * (fields.T[mesh.cell_idx(i, j)] + fields.T[mesh.cell_idx(i, j - 1)]);
                // Calcul et assignation numérique
                val += rho_v * beta * (T_v - T_ref) * gy * dvol;
            }
            // Calcul et assignation numérique
            val += fields.F_sigma_y[mesh.v_idx(i, j)] * dvol;

            // Pour Lagrangien augmenté standard : -grad(p^n)
            if (is_augmented_lagrangian()) {
                // Déclaration de variable typée
                double grad_p = (fields.p[mesh.cell_idx(i, j)] - fields.p[mesh.cell_idx(i, j - 1)]) * dx;
                // Calcul et assignation numérique
                val -= grad_p;
            }

            // Contribution de contrainte tangentielle à la paroi pour v (gauche / droite)
            if (i == 1 && mesh.is_boundary_left && bc.left.type == BCType::DIRICHLET) {
                // Déclaration de variable typée
                double mu_W = fields.mu_node[mesh.node_idx(i, j)];
                // Calcul et assignation numérique
                val += mu_W * (2.0 * bc.left.v_val / dx) * dy;
            }
            // Évaluation de la condition de contrôle
            if (i == mesh.nx_local && mesh.is_boundary_right && bc.right.type == BCType::DIRICHLET) {
                // Déclaration de variable typée
                double mu_E = fields.mu_node[mesh.node_idx(i + 1, j)];
                // Calcul et assignation numérique
                val += mu_E * (2.0 * bc.right.v_val / dx) * dy;
            }

            // Calcul et assignation numérique
            rhs[dof] = val;
        }
    }

    // RHS pour p (si monolithique régularisé ou compressible)
    if (is_monolithic()) {
        // Déclaration de variable typée
        int p_offset = n_owned_u + n_owned_v;
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                int dof = p_offset + (j - 1) * mesh.nx_local + (i - 1);
                double val_p = 0.0;
                if (compressible) {
                    int c = mesh.cell_idx(i, j);
                    double chi = (fields.chi_T.size() > (size_t)c) ? fields.chi_T[c] : 0.0;
                    // Éq. 12a RHS : (chi / dt) * p^n * dvol
                    val_p = (chi / dt) * fields.p[c] * dvol;
                }
                // Calcul et assignation numérique
                rhs[dof] = val_p;
            }
        }
    }
}

void NavierStokesSolver::update_pressure_al(FlowFields& fields) const {
    // Déclaration de variable typée
    double dx = mesh.dx;
    // Déclaration de variable typée
    double dy = mesh.dy;

    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= mesh.ny_local; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= mesh.nx_local; ++i) {
            // Déclaration de variable typée
            double div = (fields.u[mesh.u_idx(i + 1, j)] - fields.u[mesh.u_idx(i, j)]) / dx +
                         (fields.v[mesh.v_idx(i, j + 1)] - fields.v[mesh.v_idx(i, j)]) / dy;
            // Calcul de la résolution géométrique de cellule
            fields.p[mesh.cell_idx(i, j)] -= r_al * div;
        }
    }

    MpiComm::sync_cell_scalar(mesh, fields.p);
}

int NavierStokesSolver::solve_standard_projection(FlowFields& fields, double dt,
                                                 const LinearSolverBase& linear_solver,
                                                 const PreconditionerBase& precond) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;
    double dvol = dx * dy;
    int np = nx * ny;

    // =========================================================================
    // 1. Étape de Prédiction : Résolution de la quantité de mouvement pour u* et v*
    //    sans gradient de pression et sans pénalité de Lagrangien augmenté
    // =========================================================================
    assemble_preconditioner_blocks(fields, dt);

    std::vector<double> rhs(n_owned_total, 0.0);
    assemble_rhs(fields, dt, rhs);

    std::vector<double> state_vec(n_owned_total, 0.0);
    pack_vector(fields, state_vec);

    int iters_u = linear_solver.solve(*this, precond, fields, dt, rhs, state_vec);
    unpack_vector(state_vec, fields);

    // =========================================================================
    // 2. Étape de Poisson : Résolution de div( (dt/rho) * grad(p) ) = div(u*)
    // =========================================================================
    std::vector<double> a_W(np, 0.0);
    std::vector<double> a_E(np, 0.0);
    std::vector<double> a_S(np, 0.0);
    std::vector<double> a_N(np, 0.0);
    std::vector<double> a_P(np, 0.0);
    std::vector<double> b_p(np, 0.0);

    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int c = (j - 1) * nx + (i - 1);

            // Second membre : divergence de vitesse prédite intégrée sur la cellule (négative)
            double div_u_star = (fields.u[mesh.u_idx(i + 1, j)] - fields.u[mesh.u_idx(i, j)]) * dy +
                                (fields.v[mesh.v_idx(i, j + 1)] - fields.v[mesh.v_idx(i, j)]) * dx;
            b_p[c] = -div_u_star;

            // Face Ouest (u_i, j)
            bool is_wall_W = (i == 1 && mesh.is_boundary_left &&
                             (bc.left.type == BCType::DIRICHLET || bc.left.type == BCType::FREE_SLIP)) ||
                             is_u_face_in_obstacle(i, j);
            if (!is_wall_W) {
                double rho_u = fields.rho_u[mesh.u_idx(i, j)];
                if (rho_u < 1.0e-12) rho_u = 1.0;
                a_W[c] = (dt * dy) / (rho_u * dx);
            }

            // Face Est (u_i+1, j)
            bool is_wall_E = (i == nx && mesh.is_boundary_right &&
                             (bc.right.type == BCType::DIRICHLET || bc.right.type == BCType::FREE_SLIP)) ||
                             is_u_face_in_obstacle(i + 1, j);
            if (!is_wall_E) {
                double rho_u = fields.rho_u[mesh.u_idx(i + 1, j)];
                if (rho_u < 1.0e-12) rho_u = 1.0;
                a_E[c] = (dt * dy) / (rho_u * dx);
            }

            // Face Sud (v_i, j)
            bool is_wall_S = (j == 1 && mesh.is_boundary_bottom &&
                             (bc.bottom.type == BCType::DIRICHLET || bc.bottom.type == BCType::FREE_SLIP)) ||
                             is_v_face_in_obstacle(i, j);
            if (!is_wall_S) {
                double rho_v = fields.rho_v[mesh.v_idx(i, j)];
                if (rho_v < 1.0e-12) rho_v = 1.0;
                a_S[c] = (dt * dx) / (rho_v * dy);
            }

            // Face Nord (v_i, j+1)
            bool is_wall_N = (j == ny && mesh.is_boundary_top &&
                             (bc.top.type == BCType::DIRICHLET || bc.top.type == BCType::FREE_SLIP)) ||
                             is_v_face_in_obstacle(i, j + 1);
            if (!is_wall_N) {
                double rho_v = fields.rho_v[mesh.v_idx(i, j + 1)];
                if (rho_v < 1.0e-12) rho_v = 1.0;
                a_N[c] = (dt * dx) / (rho_v * dy);
            }

            // Diagonale avec régularisation infinitésimale pour fixer la jauge de pression
            a_P[c] = a_W[c] + a_E[c] + a_S[c] + a_N[c] + 1.0e-12 * dvol;
        }
    }

    // Résolution de A_p * p = b_p par Gradient Conjugué Préconditionné SSOR (PCG)
    std::vector<double> p_sol(np, 0.0);
    // Initialisation avec la pression précédente
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            p_sol[(j - 1) * nx + (i - 1)] = fields.p[mesh.cell_idx(i, j)];
        }
    }

    auto matvec_p = [&](const std::vector<double>& x, std::vector<double>& y) {
        y.assign(np, 0.0);
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int c = (j - 1) * nx + (i - 1);
                double val = a_P[c] * x[c];
                if (i > 1) val -= a_W[c] * x[c - 1];
                else if (bc.left.type == BCType::PERIODIC) val -= a_W[c] * x[(j - 1) * nx + (nx - 1)];

                if (i < nx) val -= a_E[c] * x[c + 1];
                else if (bc.right.type == BCType::PERIODIC) val -= a_E[c] * x[(j - 1) * nx];

                if (j > 1) val -= a_S[c] * x[c - nx];
                else if (bc.bottom.type == BCType::PERIODIC) val -= a_S[c] * x[(ny - 1) * nx + (i - 1)];

                if (j < ny) val -= a_N[c] * x[c + nx];
                else if (bc.top.type == BCType::PERIODIC) val -= a_N[c] * x[i - 1];

                y[c] = val;
            }
        }
    };

    auto apply_ssor_p = [&](const std::vector<double>& r, std::vector<double>& z) {
        z.assign(np, 0.0);
        // Descente avant (Forward sweep)
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int c = (j - 1) * nx + (i - 1);
                double val = r[c];
                if (i > 1) val += a_W[c] * z[c - 1];
                else if (bc.left.type == BCType::PERIODIC) val += a_W[c] * z[(j - 1) * nx + (nx - 1)];

                if (j > 1) val += a_S[c] * z[c - nx];
                else if (bc.bottom.type == BCType::PERIODIC) val += a_S[c] * z[(ny - 1) * nx + (i - 1)];

                z[c] = val / a_P[c];
            }
        }
        // Remontée arrière (Backward sweep)
        for (int j = ny; j >= 1; --j) {
            for (int i = nx; i >= 1; --i) {
                int c = (j - 1) * nx + (i - 1);
                double val = a_P[c] * z[c];
                if (i < nx) val += a_E[c] * z[c + 1];
                else if (bc.right.type == BCType::PERIODIC) val += a_E[c] * z[(j - 1) * nx];

                if (j < ny) val += a_N[c] * z[c + nx];
                else if (bc.top.type == BCType::PERIODIC) val += a_N[c] * z[i - 1];

                z[c] = val / a_P[c];
            }
        }
    };

    // Algorithme PCG
    std::vector<double> r_p(np, 0.0), Ap_p(np, 0.0);
    matvec_p(p_sol, Ap_p);

    for (int c = 0; c < np; ++c) {
        r_p[c] = b_p[c] - Ap_p[c];
    }
    double b_norm = MpiComm::global_norm2(mesh, b_p, np);
    if (b_norm < 1.0e-30) b_norm = 1.0;

    double r_norm = MpiComm::global_norm2(mesh, r_p, np);

    int iters_p = 0;
    const double tol_p = 1.0e-7;
    const int max_iter_p = 1000;

    if (r_norm / b_norm > tol_p) {
        std::vector<double> z_p(np, 0.0);
        apply_ssor_p(r_p, z_p);
        std::vector<double> d_p = z_p;

        double rz_old = MpiComm::global_dot_product(mesh, r_p, z_p, np);

        for (int iter = 0; iter < max_iter_p; ++iter) {
            iters_p++;
            std::vector<double> Ad_p(np, 0.0);
            matvec_p(d_p, Ad_p);

            double dAd = MpiComm::global_dot_product(mesh, d_p, Ad_p, np);
            if (std::abs(dAd) < 1.0e-35) break;

            double alpha = rz_old / dAd;

            for (int c = 0; c < np; ++c) {
                p_sol[c] += alpha * d_p[c];
                r_p[c]   -= alpha * Ad_p[c];
            }
            r_norm = MpiComm::global_norm2(mesh, r_p, np);

            if (r_norm / b_norm <= tol_p) break;

            apply_ssor_p(r_p, z_p);

            double rz_new = MpiComm::global_dot_product(mesh, r_p, z_p, np);

            if (std::abs(rz_old) < 1.0e-35) break;
            double beta = rz_new / rz_old;
            rz_old = rz_new;

            for (int c = 0; c < np; ++c) {
                d_p[c] = z_p[c] + beta * d_p[c];
            }
        }
    }

    // Référence de pression à moyenne nulle
    double local_p_sum = 0.0;
    for (int c = 0; c < np; ++c) local_p_sum += p_sol[c];
    double p_total_sum = MpiComm::global_sum(mesh, local_p_sum);
    double p_mean = p_total_sum / (mesh.nx_global * mesh.ny_global);
    for (int c = 0; c < np; ++c) p_sol[c] -= p_mean;

    // Mise à jour de fields.p
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            fields.p[mesh.cell_idx(i, j)] = p_sol[(j - 1) * nx + (i - 1)];
        }
    }

    // =========================================================================
    // 3. Étape de Correction / Projection de Vitesse : u = u* - (dt/rho) * grad(p)
    // =========================================================================
    int u_nx = nx + (mesh.is_boundary_right ? 1 : 0);
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= u_nx; ++i) {
            bool is_wall = (i == 1 && mesh.is_boundary_left && (bc.left.type == BCType::DIRICHLET || bc.left.type == BCType::FREE_SLIP)) ||
                           (i == u_nx && mesh.is_boundary_right && (bc.right.type == BCType::DIRICHLET || bc.right.type == BCType::FREE_SLIP)) ||
                           is_u_face_in_obstacle(i, j);
            if (is_wall) continue;

            double rho_u = fields.rho_u[mesh.u_idx(i, j)];
            if (rho_u < 1.0e-12) rho_u = 1.0;

            double p_w = (i > 1) ? fields.p[mesh.cell_idx(i - 1, j)] :
                         (bc.left.type == BCType::PERIODIC ? fields.p[mesh.cell_idx(nx, j)] : fields.p[mesh.cell_idx(1, j)]);
            double p_e = (i <= nx) ? fields.p[mesh.cell_idx(i, j)] :
                         (bc.right.type == BCType::PERIODIC ? fields.p[mesh.cell_idx(1, j)] : fields.p[mesh.cell_idx(nx, j)]);

            double grad_p_x = (p_e - p_w) / dx;
            fields.u[mesh.u_idx(i, j)] -= (dt / rho_u) * grad_p_x;
        }
    }

    int v_ny = ny + (mesh.is_boundary_top ? 1 : 0);
    for (int j = 1; j <= v_ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            bool is_wall = (j == 1 && mesh.is_boundary_bottom && (bc.bottom.type == BCType::DIRICHLET || bc.bottom.type == BCType::FREE_SLIP)) ||
                           (j == v_ny && mesh.is_boundary_top && (bc.top.type == BCType::DIRICHLET || bc.top.type == BCType::FREE_SLIP)) ||
                           is_v_face_in_obstacle(i, j);
            if (is_wall) continue;

            double rho_v = fields.rho_v[mesh.v_idx(i, j)];
            if (rho_v < 1.0e-12) rho_v = 1.0;

            double p_s = (j > 1) ? fields.p[mesh.cell_idx(i, j - 1)] :
                         (bc.bottom.type == BCType::PERIODIC ? fields.p[mesh.cell_idx(i, ny)] : fields.p[mesh.cell_idx(i, 1)]);
            double p_n = (j <= ny) ? fields.p[mesh.cell_idx(i, j)] :
                         (bc.top.type == BCType::PERIODIC ? fields.p[mesh.cell_idx(i, 1)] : fields.p[mesh.cell_idx(nx, j)]);

            double grad_p_y = (p_n - p_s) / dy;
            fields.v[mesh.v_idx(i, j)] -= (dt / rho_v) * grad_p_y;
        }
    }

    // Synchronisation des mailles fantômes et application des CL
    MpiComm::sync_u(mesh, fields.u);
    MpiComm::sync_v(mesh, fields.v);
    MpiComm::sync_cell_scalar(mesh, fields.p);
    MpiComm::apply_boundary_conditions(mesh, bc, fields, false);

    return iters_u + iters_p;
}

void NavierStokesSolver::assemble_preconditioner_blocks(const FlowFields& fields, double dt) {
    // Déclaration de variable typée
    double dx = mesh.dx;
    // Déclaration de variable typée
    double dy = mesh.dy;
    // Déclaration de variable typée
    double dvol = dx * dy;

    // Calcul des diagonales effectives de Fuu et Fvv pour le préconditionnement Jacobi
    diag_u.assign(n_owned_u, 1.0);
    // Déclaration de variable typée
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= mesh.ny_local; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= u_nx; ++i) {
            // Déclaration de variable typée
            int dof = (j - 1) * u_nx + (i - 1);
            // Déclaration de variable typée
            double rho_u = fields.rho_u[mesh.u_idx(i, j)];
            // Déclaration de variable typée
            bool is_wall = (i == 1 && mesh.is_boundary_left) ||
                           // Calcul et assignation numérique
                           (i == u_nx && mesh.is_boundary_right) ||
                           is_u_face_in_obstacle(i, j);
            // Évaluation de la condition de contrôle
            if (is_wall) {
                // Calcul et assignation numérique
                diag_u[dof] = (rho_u / dt) * dvol;
                continue;
            }
            // Déclaration de variable typée
            double mu_c = fields.mu[mesh.cell_idx(i, j)];
            double al_u = is_augmented_lagrangian() ? 2.0 * r_al * (dy / dx) : 0.0;
            double d = (rho_u / dt) * dvol + 4.0 * mu_c * (dy / dx) + 2.0 * mu_c * (dx / dy) + al_u;
            diag_u[dof] = (d > 1.0e-15) ? d : 1.0;
        }
    }

    diag_v.assign(n_owned_v, 1.0);
    // Déclaration de variable typée
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= v_ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= mesh.nx_local; ++i) {
            // Déclaration de variable typée
            int dof = (j - 1) * mesh.nx_local + (i - 1);
            // Déclaration de variable typée
            double rho_v = fields.rho_v[mesh.v_idx(i, j)];
            // Déclaration de variable typée
            bool is_wall = (j == 1 && mesh.is_boundary_bottom && (bc.bottom.type == BCType::DIRICHLET || bc.bottom.type == BCType::FREE_SLIP)) ||
                           // Calcul et assignation numérique
                           (j == v_ny && mesh.is_boundary_top && (bc.top.type == BCType::DIRICHLET || bc.top.type == BCType::FREE_SLIP)) ||
                           is_v_face_in_obstacle(i, j);
            // Évaluation de la condition de contrôle
            if (is_wall) {
                // Calcul et assignation numérique
                diag_v[dof] = (rho_v / dt) * dvol;
                continue;
            }
            // Déclaration de variable typée
            double mu_c = fields.mu[mesh.cell_idx(i, j)];
            double al_v = is_augmented_lagrangian() ? 2.0 * r_al * (dx / dy) : 0.0;
            double d = (rho_v / dt) * dvol + 4.0 * mu_c * (dx / dy) + 2.0 * mu_c * (dy / dx) + al_v;
            diag_v[dof] = (d > 1.0e-15) ? d : 1.0;
        }
    }

    // Évaluation de la condition de contrôle
    if (is_monolithic()) {
        diag_p.assign(n_owned_p, 1.0);
    }
}

void NavierStokesSolver::solve_energy(FlowFields& fields, double dt) const {
    // Évaluation de la condition de contrôle
    if (!enable_energy) return;

    // Déclaration de variable typée
    int nx = mesh.nx_local;
    // Déclaration de variable typée
    int ny = mesh.ny_local;
    // Déclaration de variable typée
    int n_cells = nx * ny;

    // Déclaration de variable typée
    double dx = mesh.dx;
    // Déclaration de variable typée
    double dy = mesh.dy;
    // Déclaration de variable typée
    double dvol = dx * dy;
    // Déclaration de variable typée
    double rho = fields.rho[mesh.cell_idx(1, 1)];

    // Calcul et assignation numérique
    auto cell_dof = [&](int i, int j) {
        // Retour de fonction
        return (j - 1) * nx + (i - 1);
    };

    // Déclaration de variable typée
    std::vector<int> rows;
    // Déclaration de variable typée
    std::vector<int> cols;
    // Déclaration de variable typée
    std::vector<double> vals;
    // Déclaration de variable typée
    std::vector<double> rhs(n_cells, 0.0);
    // Déclaration de variable typée
    std::vector<double> T_sol(n_cells, 0.0);

    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Déclaration de variable typée
            int c_dof = cell_dof(i, j);
            // Calcul et assignation numérique
            T_sol[c_dof] = fields.T[mesh.cell_idx(i, j)];

            // Déclaration de variable typée
            double a_c = (rho * Cp / dt) * dvol;
            // Calcul et assignation numérique
            rhs[c_dof] = (rho * Cp / dt) * dvol * fields.T[mesh.cell_idx(i, j)];

            // Flux x-Est
            double u_e = fields.u[mesh.u_idx(i + 1, j)];
            // Déclaration de variable typée
            double d_e = k_thermal * dy / dx;
            // Déclaration de variable typée
            double conv_e = rho * Cp * u_e * dy * 0.5;

            // Flux x-Ouest
            double u_w = fields.u[mesh.u_idx(i, j)];
            // Déclaration de variable typée
            double d_w = k_thermal * dy / dx;
            // Déclaration de variable typée
            double conv_w = rho * Cp * u_w * dy * 0.5;

            // Flux y-Nord
            double v_n = fields.v[mesh.v_idx(i, j + 1)];
            // Déclaration de variable typée
            double d_n = k_thermal * dx / dy;
            // Déclaration de variable typée
            double conv_n = rho * Cp * v_n * dx * 0.5;

            // Flux y-Sud
            double v_s = fields.v[mesh.v_idx(i, j)];
            // Déclaration de variable typée
            double d_s = k_thermal * dx / dy;
            // Déclaration de variable typée
            double conv_s = rho * Cp * v_s * dx * 0.5;

            // Voisin Est
            if (i < nx) {
                // Calcul et assignation numérique
                a_c += d_e + conv_e;
                rows.push_back(c_dof);
                cols.push_back(cell_dof(i + 1, j));
                vals.push_back(-d_e + conv_e);
            } else if (mesh.is_boundary_right) {
                // Évaluation de la condition de contrôle
                if (bc.right.T_type == BCType::DIRICHLET) {
                    // Calcul et assignation numérique
                    a_c += 2.0 * d_e;
                    // Calcul et assignation numérique
                    rhs[c_dof] += 2.0 * d_e * bc.right.T_val;
                }
            }

            // Voisin Ouest
            if (i > 1) {
                // Calcul et assignation numérique
                a_c += d_w - conv_w;
                rows.push_back(c_dof);
                cols.push_back(cell_dof(i - 1, j));
                vals.push_back(-d_w - conv_w);
            } else if (mesh.is_boundary_left) {
                // Évaluation de la condition de contrôle
                if (bc.left.T_type == BCType::DIRICHLET) {
                    // Calcul et assignation numérique
                    a_c += 2.0 * d_w;
                    // Calcul et assignation numérique
                    rhs[c_dof] += 2.0 * d_w * bc.left.T_val;
                }
            }

            // Voisin Nord
            if (j < ny) {
                // Calcul et assignation numérique
                a_c += d_n + conv_n;
                rows.push_back(c_dof);
                cols.push_back(cell_dof(i, j + 1));
                vals.push_back(-d_n + conv_n);
            } else if (mesh.is_boundary_top) {
                // Évaluation de la condition de contrôle
                if (bc.top.T_type == BCType::DIRICHLET) {
                    // Calcul et assignation numérique
                    a_c += 2.0 * d_n;
                    // Calcul et assignation numérique
                    rhs[c_dof] += 2.0 * d_n * bc.top.T_val;
                }
            }

            // Voisin Sud
            if (j > 1) {
                // Calcul et assignation numérique
                a_c += d_s - conv_s;
                rows.push_back(c_dof);
                cols.push_back(cell_dof(i, j - 1));
                vals.push_back(-d_s - conv_s);
            } else if (mesh.is_boundary_bottom) {
                // Évaluation de la condition de contrôle
                if (bc.bottom.T_type == BCType::DIRICHLET) {
                    // Calcul et assignation numérique
                    a_c += 2.0 * d_s;
                    // Calcul et assignation numérique
                    rhs[c_dof] += 2.0 * d_s * bc.bottom.T_val;
                }
            }

            rows.push_back(c_dof);
            cols.push_back(c_dof);
            vals.push_back(a_c);
        }
    }

    CSRMatrix A;
    A.set_from_triplets(n_cells, n_cells, rows, cols, vals);

    // Solveur BiCGSTAB avec préconditionnement Jacobi
    std::vector<double> r = rhs;
    // Déclaration de variable typée
    std::vector<double> Ax(n_cells, 0.0);
    A.matvec(T_sol, Ax);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n_cells; ++i) r[i] -= Ax[i];

    // Déclaration de variable typée
    std::vector<double> r0_hat = r;
    // Déclaration de variable typée
    std::vector<double> p = r;
    // Déclaration de variable typée
    std::vector<double> v(n_cells, 0.0);
    // Déclaration de variable typée
    std::vector<double> s(n_cells, 0.0);
    // Déclaration de variable typée
    std::vector<double> t(n_cells, 0.0);
    // Déclaration de variable typée
    std::vector<double> y_prec(n_cells, 0.0);
    // Déclaration de variable typée
    std::vector<double> z_prec(n_cells, 0.0);

    // Déclaration de variable typée
    double rho_old = 1.0;
    // Déclaration de variable typée
    double alpha = 1.0;
    // Déclaration de variable typée
    double omega = 1.0;

    // Déclaration de variable typée
    double res_init = 0.0;
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n_cells; ++i) res_init += r[i] * r[i];
    // Calcul de norme euclidienne ou résidu de convergence
    res_init = std::sqrt(res_init);
    // Évaluation de la condition de contrôle
    if (res_init > 1e-13) {
        // Déclaration de variable typée
        int max_it = 500;
        // Boucle d'itération algorithmique
        for (int iter = 0; iter < max_it; ++iter) {
            // Déclaration de variable typée
            double rho_new = 0.0;
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n_cells; ++i) rho_new += r0_hat[i] * r[i];
            // Évaluation de la condition de contrôle
            if (std::abs(rho_new) < 1e-30) break;

            // Évaluation de la condition de contrôle
            if (iter > 0) {
                // Déclaration de variable typée
                double beta_param = (rho_new / rho_old) * (alpha / omega);
                // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
                for (int i = 0; i < n_cells; ++i) {
                    // Calcul et assignation numérique
                    p[i] = r[i] + beta_param * (p[i] - omega * v[i]);
                }
            } else {
                // Calcul et assignation numérique
                p = r;
            }

            // Préconditionnement Jacobi : y = D^{-1} p
            for (int i = 0; i < n_cells; ++i) y_prec[i] = p[i] / A.diag[i];
            A.matvec(y_prec, v);

            // Déclaration de variable typée
            double r0v = 0.0;
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n_cells; ++i) r0v += r0_hat[i] * v[i];
            // Évaluation de la condition de contrôle
            if (std::abs(r0v) < 1e-30) break;
            // Calcul et assignation numérique
            alpha = rho_new / r0v;

            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n_cells; ++i) s[i] = r[i] - alpha * v[i];
            // Déclaration de variable typée
            double s_norm = 0.0;
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n_cells; ++i) s_norm += s[i] * s[i];
            // Évaluation de la condition de contrôle
            if (std::sqrt(s_norm) / res_init < 1e-8) {
                // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
                for (int i = 0; i < n_cells; ++i) T_sol[i] += alpha * y_prec[i];
                break;
            }

            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n_cells; ++i) z_prec[i] = s[i] / A.diag[i];
            A.matvec(z_prec, t);

            // Déclaration de variable typée
            double tt = 0.0, ts = 0.0;
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n_cells; ++i) {
                // Calcul et assignation numérique
                tt += t[i] * t[i];
                // Calcul et assignation numérique
                ts += t[i] * s[i];
            }
            // Calcul et assignation numérique
            omega = (tt > 1e-30) ? (ts / tt) : 0.0;

            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n_cells; ++i) {
                // Calcul et assignation numérique
                T_sol[i] += alpha * y_prec[i] + omega * z_prec[i];
                // Calcul et assignation numérique
                r[i] = s[i] - omega * t[i];
            }

            // Calcul et assignation numérique
            rho_old = rho_new;

            // Déclaration de variable typée
            double r_norm = 0.0;
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n_cells; ++i) r_norm += r[i] * r[i];
            // Évaluation de la condition de contrôle
            if (std::sqrt(r_norm) / res_init < 1e-8) break;
        }
    }

    // Réinjecter T_sol dans fields.T
    for (int j = 1; j <= ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Calcul de la résolution géométrique de cellule
            fields.T[mesh.cell_idx(i, j)] = T_sol[cell_dof(i, j)];
        }
    }

    MpiComm::sync_cell_scalar(mesh, fields.T);
    MpiComm::apply_thermal_boundary_conditions(mesh, bc, fields);
}

// Déclaration de variable typée
double NavierStokesSolver::compute_mean_nusselt_hot_wall(const FlowFields& fields) const {
    // Déclaration de variable typée
    double local_nu_sum = 0.0;
    // Déclaration de variable typée
    int count = 0;
    // Évaluation de la condition de contrôle
    if (mesh.is_boundary_left) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Déclaration de variable typée
            double T_w = bc.left.T_val;
            // Déclaration de variable typée
            double T_1 = fields.T[mesh.cell_idx(1, j)];
            // Déclaration de variable typée
            double delta_T = std::abs(bc.left.T_val - bc.right.T_val);
            // Évaluation de la condition de contrôle
            if (delta_T < 1e-12) delta_T = 1.0;
            // Déclaration de variable typée
            double nu_local = 2.0 * (T_w - T_1) / mesh.dx * (mesh.lx / delta_T);
            // Calcul et assignation numérique
            local_nu_sum += nu_local;
            count++;
        }
    }
    // Déclaration de variable typée
    double global_nu_sum = MpiComm::global_sum(mesh, local_nu_sum);
    // Déclaration de variable typée
    double global_count = MpiComm::global_sum(mesh, static_cast<double>(count));
    // Évaluation de la condition de contrôle
    if (global_count > 0.0) {
        // Retour de fonction
        return global_nu_sum / global_count;
    }
    // Retour de fonction
    return 1.0;
}

void NavierStokesSolver::advect_ssp_rk3_momentum(FlowFields& fields, double dt) const {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    std::vector<double> u_adv1 = fields.u;
    std::vector<double> v_adv1 = fields.v;
    std::vector<double> u_adv2 = fields.u;
    std::vector<double> v_adv2 = fields.v;

    bool has_old_v = (!fields.u_old.empty() && fields.u_old.size() == fields.u.size());
    if (has_old_v) {
        for (size_t i = 0; i < fields.u.size(); ++i) {
            u_adv1[i] = 2.0 * fields.u[i] - fields.u_old[i];
            u_adv2[i] = 1.5 * fields.u[i] - 0.5 * fields.u_old[i];
        }
        for (size_t i = 0; i < fields.v.size(); ++i) {
            v_adv1[i] = 2.0 * fields.v[i] - fields.v_old[i];
            v_adv2[i] = 1.5 * fields.v[i] - 0.5 * fields.v_old[i];
        }
    }

    auto div_flux_cell = [&](const std::vector<double>& phi, const std::vector<double>& u_adv, const std::vector<double>& v_adv, std::vector<double>& div_f) {
        div_f.assign(phi.size(), 0.0);
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int c = mesh.cell_idx(i, j);
                double u_e = u_adv[mesh.u_idx(i + 1, j)];
                double u_w = u_adv[mesh.u_idx(i, j)];
                double v_n = v_adv[mesh.v_idx(i, j + 1)];
                double v_s = v_adv[mesh.v_idx(i, j)];

                double phi_e = (u_e >= 0.0) ? phi[c] : phi[mesh.cell_idx(i + 1, j)];
                double phi_w = (u_w >= 0.0) ? phi[mesh.cell_idx(i - 1, j)] : phi[c];
                double phi_n = (v_n >= 0.0) ? phi[c] : phi[mesh.cell_idx(i, j + 1)];
                double phi_s = (v_s >= 0.0) ? phi[mesh.cell_idx(i, j - 1)] : phi[c];

                double fx_e = u_e * phi_e;
                double fx_w = u_w * phi_w;
                double fy_n = v_n * phi_n;
                double fy_s = v_s * phi_s;

                div_f[c] = (fx_e - fx_w) / dx + (fy_n - fy_s) / dy;
            }
        }
    };

    auto div_flux_u = [&](const std::vector<double>& rhou, const std::vector<double>& u_adv, const std::vector<double>& v_adv, std::vector<double>& div_fu) {
        div_fu.assign(rhou.size(), 0.0);
        int u_nx = nx + (mesh.is_boundary_right ? 1 : 0);
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= u_nx; ++i) {
                int idx = mesh.u_idx(i, j);
                double u_c = 0.5 * (u_adv[mesh.u_idx(i, j)] + ((i < nx + 1) ? u_adv[mesh.u_idx(i + 1, j)] : u_adv[mesh.u_idx(i, j)]));
                double u_w = 0.5 * (u_adv[mesh.u_idx(i, j)] + ((i > 1) ? u_adv[mesh.u_idx(i - 1, j)] : u_adv[mesh.u_idx(i, j)]));

                double v_node_n = 0.5 * (v_adv[mesh.v_idx(std::min(nx, i), j + 1)] + v_adv[mesh.v_idx(std::max(1, i - 1), j + 1)]);
                double v_node_s = 0.5 * (v_adv[mesh.v_idx(std::min(nx, i), j)] + v_adv[mesh.v_idx(std::max(1, i - 1), j)]);

                double val_e = (u_c >= 0.0) ? rhou[idx] : rhou[mesh.u_idx(std::min(nx + 1, i + 1), j)];
                double val_w = (u_w >= 0.0) ? rhou[mesh.u_idx(std::max(1, i - 1), j)] : rhou[idx];
                double val_n = (v_node_n >= 0.0) ? rhou[idx] : rhou[mesh.u_idx(i, std::min(ny, j + 1))];
                double val_s = (v_node_s >= 0.0) ? rhou[mesh.u_idx(i, std::max(1, j - 1))] : rhou[idx];

                div_fu[idx] = (u_c * val_e - u_w * val_w) / dx + (v_node_n * val_n - v_node_s * val_s) / dy;
            }
        }
    };

    auto div_flux_v = [&](const std::vector<double>& rhov, const std::vector<double>& u_adv, const std::vector<double>& v_adv, std::vector<double>& div_fv) {
        div_fv.assign(rhov.size(), 0.0);
        int v_ny = ny + (mesh.is_boundary_top ? 1 : 0);
        for (int j = 1; j <= v_ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int idx = mesh.v_idx(i, j);
                double v_c = 0.5 * (v_adv[mesh.v_idx(i, j)] + ((j < ny + 1) ? v_adv[mesh.v_idx(i, j + 1)] : v_adv[mesh.v_idx(i, j)]));
                double v_s = 0.5 * (v_adv[mesh.v_idx(i, j)] + ((j > 1) ? v_adv[mesh.v_idx(i, j - 1)] : v_adv[mesh.v_idx(i, j)]));

                double u_node_e = 0.5 * (u_adv[mesh.u_idx(i + 1, std::min(ny, j))] + u_adv[mesh.u_idx(i + 1, std::max(1, j - 1))]);
                double u_node_w = 0.5 * (u_adv[mesh.u_idx(i, std::min(ny, j))] + u_adv[mesh.u_idx(i, std::max(1, j - 1))]);

                double val_e = (u_node_e >= 0.0) ? rhov[mesh.v_idx(std::max(1, i - 1), j)] : rhov[idx];
                double val_w = (u_node_w >= 0.0) ? rhov[idx] : rhov[mesh.v_idx(std::min(nx, i + 1), j)];
                double val_n = (v_c >= 0.0) ? rhov[idx] : rhov[mesh.v_idx(i, std::min(ny + 1, j + 1))];
                double val_s = (v_s >= 0.0) ? rhov[mesh.v_idx(i, std::max(1, j - 1))] : rhov[idx];

                div_fv[idx] = (u_node_e * val_w - u_node_w * val_e) / dx + (v_c * val_n - v_s * val_s) / dy;
            }
        }
    };

    std::vector<double> rho_n = fields.rho;
    std::vector<double> div_rho;
    div_flux_cell(rho_n, fields.u, fields.v, div_rho);
    std::vector<double> rho_1(rho_n.size(), 0.0);
    for (size_t c = 0; c < rho_n.size(); ++c) {
        rho_1[c] = rho_n[c] - dt * div_rho[c];
    }
    div_flux_cell(rho_1, u_adv1, v_adv1, div_rho);
    std::vector<double> rho_2(rho_n.size(), 0.0);
    for (size_t c = 0; c < rho_n.size(); ++c) {
        rho_2[c] = 0.75 * rho_n[c] + 0.25 * rho_1[c] - 0.25 * dt * div_rho[c];
    }
    div_flux_cell(rho_2, u_adv2, v_adv2, div_rho);
    std::vector<double> rho_star(rho_n.size(), 0.0);
    for (size_t c = 0; c < rho_n.size(); ++c) {
        rho_star[c] = (1.0 / 3.0) * rho_n[c] + (2.0 / 3.0) * rho_2[c] - (2.0 / 3.0) * dt * div_rho[c];
    }
    fields.rho = rho_star;
    MpiComm::sync_cell_scalar(mesh, fields.rho);

    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx + 1; ++i) {
            fields.rho_u[mesh.u_idx(i, j)] = 0.5 * (fields.rho[mesh.cell_idx(i - 1, j)] + fields.rho[mesh.cell_idx(i, j)]);
        }
    }
    for (int j = 1; j <= ny + 1; ++j) {
        for (int i = 1; i <= nx; ++i) {
            fields.rho_v[mesh.v_idx(i, j)] = 0.5 * (fields.rho[mesh.cell_idx(i, j - 1)] + fields.rho[mesh.cell_idx(i, j)]);
        }
    }

    std::vector<double> rhou_n(fields.u.size(), 0.0);
    for (size_t k = 0; k < fields.u.size(); ++k) {
        rhou_n[k] = fields.rho_u[k] * fields.u[k];
    }
    std::vector<double> div_u_k;
    div_flux_u(rhou_n, fields.u, fields.v, div_u_k);
    std::vector<double> rhou_1(rhou_n.size(), 0.0);
    for (size_t k = 0; k < rhou_n.size(); ++k) {
        rhou_1[k] = rhou_n[k] - dt * div_u_k[k];
    }
    div_flux_u(rhou_1, u_adv1, v_adv1, div_u_k);
    std::vector<double> rhou_2(rhou_n.size(), 0.0);
    for (size_t k = 0; k < rhou_n.size(); ++k) {
        rhou_2[k] = 0.75 * rhou_n[k] + 0.25 * rhou_1[k] - 0.25 * dt * div_u_k[k];
    }
    div_flux_u(rhou_2, u_adv2, v_adv2, div_u_k);
    fields.rhou_star.resize(rhou_n.size(), 0.0);
    for (size_t k = 0; k < rhou_n.size(); ++k) {
        fields.rhou_star[k] = (1.0 / 3.0) * rhou_n[k] + (2.0 / 3.0) * rhou_2[k] - (2.0 / 3.0) * dt * div_u_k[k];
    }

    std::vector<double> rhov_n(fields.v.size(), 0.0);
    for (size_t k = 0; k < fields.v.size(); ++k) {
        rhov_n[k] = fields.rho_v[k] * fields.v[k];
    }
    std::vector<double> div_v_k;
    div_flux_v(rhov_n, fields.u, fields.v, div_v_k);
    std::vector<double> rhov_1(rhov_n.size(), 0.0);
    for (size_t k = 0; k < rhov_n.size(); ++k) {
        rhov_1[k] = rhov_n[k] - dt * div_v_k[k];
    }
    div_flux_v(rhov_1, u_adv1, v_adv1, div_v_k);
    std::vector<double> rhov_2(rhov_n.size(), 0.0);
    for (size_t k = 0; k < rhov_n.size(); ++k) {
        rhov_2[k] = 0.75 * rhov_n[k] + 0.25 * rhov_1[k] - 0.25 * dt * div_v_k[k];
    }
    div_flux_v(rhov_2, u_adv2, v_adv2, div_v_k);
    fields.rhov_star.resize(rhov_n.size(), 0.0);
    for (size_t k = 0; k < rhov_n.size(); ++k) {
        fields.rhov_star[k] = (1.0 / 3.0) * rhov_n[k] + (2.0 / 3.0) * rhov_2[k] - (2.0 / 3.0) * dt * div_v_k[k];
    }

    if (compressible) {
        std::vector<double> rhoe_n(fields.rho.size(), 0.0);
        for (size_t c = 0; c < fields.rho.size(); ++c) {
            rhoe_n[c] = fields.rho[c] * fields.energy[c];
        }
        std::vector<double> div_e;
        div_flux_cell(rhoe_n, fields.u, fields.v, div_e);
        std::vector<double> rhoe_1(rhoe_n.size(), 0.0);
        for (size_t c = 0; c < rhoe_n.size(); ++c) {
            rhoe_1[c] = rhoe_n[c] - dt * div_e[c];
        }
        div_flux_cell(rhoe_1, u_adv1, v_adv1, div_e);
        std::vector<double> rhoe_2(rhoe_n.size(), 0.0);
        for (size_t c = 0; c < rhoe_n.size(); ++c) {
            rhoe_2[c] = 0.75 * rhoe_n[c] + 0.25 * rhoe_1[c] - 0.25 * dt * div_e[c];
        }
        div_flux_cell(rhoe_2, u_adv2, v_adv2, div_e);
        fields.rhoe_star.resize(rhoe_n.size(), 0.0);
        for (size_t c = 0; c < rhoe_n.size(); ++c) {
            fields.rhoe_star[c] = (1.0 / 3.0) * rhoe_n[c] + (2.0 / 3.0) * rhoe_2[c] - (2.0 / 3.0) * dt * div_e[c];
        }
    }
}

void NavierStokesSolver::solve_total_energy(FlowFields& fields, double dt) const {
    if (!compressible) return;

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int c = mesh.cell_idx(i, j);

            double u_e = fields.u[mesh.u_idx(i + 1, j)];
            double u_w = fields.u[mesh.u_idx(i, j)];
            double v_n = fields.v[mesh.v_idx(i, j + 1)];
            double v_s = fields.v[mesh.v_idx(i, j)];

            double p_e = 0.5 * (fields.p[c] + fields.p[mesh.cell_idx(i + 1, j)]);
            double p_w = 0.5 * (fields.p[c] + fields.p[mesh.cell_idx(i - 1, j)]);
            double p_n = 0.5 * (fields.p[c] + fields.p[mesh.cell_idx(i, j + 1)]);
            double p_s = 0.5 * (fields.p[c] + fields.p[mesh.cell_idx(i, j - 1)]);

            double div_vp = (u_e * p_e - u_w * p_w) / dx + (v_n * p_n - v_s * p_s) / dy;

            double uc = 0.5 * (u_e + u_w);
            double vc = 0.5 * (v_n + v_s);
            double v_sq = uc * uc + vc * vc;

            double grav_work = fields.rho[c] * (gx * uc + gy * vc);

            double rhoe_star_val = (fields.rhoe_star.size() > (size_t)c) ? fields.rhoe_star[c] : (fields.rho[c] * fields.energy[c]);
            double rhoe_new = rhoe_star_val + dt * (-div_vp + grav_work);

            double rho_new = std::max(1.0e-6, fields.rho[c]);
            double e_new = rhoe_new / rho_new;

            fields.rho_e[c] = rhoe_new;
            fields.energy[c] = e_new;

            double u_int = std::max(1.0, e_new - 0.5 * v_sq);
            double gamma_eff = 1.4;
            double r_eff = 287.05;
            fields.T[c] = (gamma_eff - 1.0) / r_eff * u_int;
        }
    }
    MpiComm::sync_cell_scalar(mesh, fields.energy);
    MpiComm::sync_cell_scalar(mesh, fields.T);
}

} // namespace covof

