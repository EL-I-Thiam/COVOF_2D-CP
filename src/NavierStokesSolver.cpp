#include "fugu/NavierStokesSolver.hpp"
#include "fugu/MpiComm.hpp"
#include <cmath>
#include <iostream>

namespace fugu {

void NavierStokesSolver::init(const Mesh2D& m, const Config& cfg) {
    mesh = m;
    bc = cfg.bc;
    enable_convection = cfg.enable_convection;
    enable_energy = cfg.enable_energy;
    gx = cfg.gx;
    gy = cfg.gy;
    beta = cfg.beta;
    T_ref = cfg.T_ref;
    Cp = cfg.fluid1.Cp;
    k_thermal = cfg.fluid1.k_thermal;
    Ra = cfg.Ra;
    Pr = cfg.Pr;
    solver_type = cfg.solver.type;
    r_al = cfg.solver.r_al;
    uzawa_iter = cfg.solver.uzawa_iter;
    obstacles = cfg.obstacles;

    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    n_owned_u = u_nx * mesh.ny_local;

    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
    n_owned_v = mesh.nx_local * v_ny;

    n_owned_p = mesh.nx_local * mesh.ny_local;

    if (is_augmented_lagrangian()) {
        n_owned_total = n_owned_u + n_owned_v;
    } else {
        n_owned_total = n_owned_u + n_owned_v + n_owned_p;
    }
}

void NavierStokesSolver::pack_vector(const FlowFields& fields, std::vector<double>& vec) const {
    vec.assign(n_owned_total, 0.0);

    // 1. Pack u
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    for (int j = 1; j <= mesh.ny_local; ++j) {
        for (int i = 1; i <= u_nx; ++i) {
            int dof = (j - 1) * u_nx + (i - 1);
            vec[dof] = fields.u[mesh.u_idx(i, j)];
        }
    }

    // 2. Pack v
    int v_offset = n_owned_u;
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
    for (int j = 1; j <= v_ny; ++j) {
        for (int i = 1; i <= mesh.nx_local; ++i) {
            int dof = (j - 1) * mesh.nx_local + (i - 1);
            vec[v_offset + dof] = fields.v[mesh.v_idx(i, j)];
        }
    }

    // 3. Pack p (si monolithique)
    if (is_monolithic()) {
        int p_offset = n_owned_u + n_owned_v;
        for (int j = 1; j <= mesh.ny_local; ++j) {
            for (int i = 1; i <= mesh.nx_local; ++i) {
                int dof = (j - 1) * mesh.nx_local + (i - 1);
                vec[p_offset + dof] = fields.p[mesh.cell_idx(i, j)];
            }
        }
    }
}

void NavierStokesSolver::unpack_vector(const std::vector<double>& vec, FlowFields& fields, bool homogeneous) const {
    // 1. Unpack u
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    for (int j = 1; j <= mesh.ny_local; ++j) {
        for (int i = 1; i <= u_nx; ++i) {
            int dof = (j - 1) * u_nx + (i - 1);
            fields.u[mesh.u_idx(i, j)] = vec[dof];
        }
    }

    // 2. Unpack v
    int v_offset = n_owned_u;
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
    for (int j = 1; j <= v_ny; ++j) {
        for (int i = 1; i <= mesh.nx_local; ++i) {
            int dof = (j - 1) * mesh.nx_local + (i - 1);
            fields.v[mesh.v_idx(i, j)] = vec[v_offset + dof];
        }
    }

    // 3. Unpack p (si monolithique)
    if (is_monolithic()) {
        int p_offset = n_owned_u + n_owned_v;
        for (int j = 1; j <= mesh.ny_local; ++j) {
            for (int i = 1; i <= mesh.nx_local; ++i) {
                int dof = (j - 1) * mesh.nx_local + (i - 1);
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
    unpack_vector(x, tf, true); // true = CL homogènes pour garantir la stricte linéarité de A*x

    double dx = mesh.dx;
    double dy = mesh.dy;
    double dvol = dx * dy;

    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);

    // Précalcul de la divergence sur chaque cellule locale + halo
    std::vector<double> div_cell((mesh.nx_local + 2) * (mesh.ny_local + 2), 0.0);
    auto c_idx = [&](int i, int j) { return j * (mesh.nx_local + 2) + i; };

    for (int j = 0; j <= mesh.ny_local + 1; ++j) {
        for (int i = 0; i <= mesh.nx_local + 1; ++i) {
            double d = (tf.u[mesh.u_idx(i + 1, j)] - tf.u[mesh.u_idx(i, j)]) / dx +
                       (tf.v[mesh.v_idx(i, j + 1)] - tf.v[mesh.v_idx(i, j)]) / dy;
            div_cell[c_idx(i, j)] = d;
        }
    }

    // =========================================================================
    // 1. Équation de Quantité de Mouvement en u (Faces verticales)
    // =========================================================================
    for (int j = 1; j <= mesh.ny_local; ++j) {
        for (int i = 1; i <= u_nx; ++i) {
            int dof = (j - 1) * u_nx + (i - 1);

            // Condition aux limites physique sur les parois gauche / droite ou obstacle solide
            if (i == 1 && mesh.is_boundary_left) {
                double rho_u = fields_old.rho_u[mesh.u_idx(i, j)];
                double diag_bc = (rho_u / dt) * dvol;
                if (bc.left.type == BCType::NEUMANN) {
                    y[dof] = diag_bc * (tf.u[mesh.u_idx(1, j)] - tf.u[mesh.u_idx(2, j)]);
                } else {
                    y[dof] = diag_bc * tf.u[mesh.u_idx(1, j)];
                }
                continue;
            }
            if (i == u_nx && mesh.is_boundary_right) {
                double rho_u = fields_old.rho_u[mesh.u_idx(i, j)];
                double diag_bc = (rho_u / dt) * dvol;
                if (bc.right.type == BCType::NEUMANN) {
                    y[dof] = diag_bc * (tf.u[mesh.u_idx(u_nx, j)] - tf.u[mesh.u_idx(u_nx - 1, j)]);
                } else {
                    y[dof] = diag_bc * tf.u[mesh.u_idx(u_nx, j)];
                }
                continue;
            }
            if (is_u_face_in_obstacle(i, j)) {
                double rho_u = fields_old.rho_u[mesh.u_idx(i, j)];
                double diag_bc = (rho_u / dt) * dvol;
                y[dof] = diag_bc * tf.u[mesh.u_idx(i, j)];
                continue;
            }

            double rho_u = fields_old.rho_u[mesh.u_idx(i, j)];

            // Terme instationnaire
            double val = (rho_u / dt) * tf.u[mesh.u_idx(i, j)] * dvol;

            // Terme convectif semi-implicite F_uu
            if (enable_convection) {
                double u_n = fields_old.u[mesh.u_idx(i, j)];
                double u_n_E = fields_old.u[mesh.u_idx(i + 1, j)];
                double u_n_W = fields_old.u[mesh.u_idx(i - 1, j)];

                double v_n_N = 0.5 * (fields_old.v[mesh.v_idx(i - 1, j + 1)] + fields_old.v[mesh.v_idx(i, j + 1)]);
                double v_n_S = 0.5 * (fields_old.v[mesh.v_idx(i - 1, j)] + fields_old.v[mesh.v_idx(i, j)]);

                double conv_E = 0.5 * (u_n + u_n_E) * 0.5 * (tf.u[mesh.u_idx(i, j)] + tf.u[mesh.u_idx(i + 1, j)]) * dy;
                double conv_W = 0.5 * (u_n_W + u_n) * 0.5 * (tf.u[mesh.u_idx(i - 1, j)] + tf.u[mesh.u_idx(i, j)]) * dy;
                double conv_N = v_n_N * 0.5 * (tf.u[mesh.u_idx(i, j)] + tf.u[mesh.u_idx(i, j + 1)]) * dx;
                double conv_S = v_n_S * 0.5 * (tf.u[mesh.u_idx(i, j - 1)] + tf.u[mesh.u_idx(i, j)]) * dx;

                val += rho_u * (conv_E - conv_W + conv_N - conv_S);
            }

            // Termes visqueux implicites d'ordre 2 (Caltagirone et al. - Annexe A)
            // (a) Viscosité d'élongation 2*mu d2u/dx2
            double mu_cell_E = fields_old.mu[mesh.cell_idx(i, j)];
            double mu_cell_W = fields_old.mu[mesh.cell_idx(i - 1, j)];
            double diff_elong_E = 2.0 * mu_cell_E * (tf.u[mesh.u_idx(i + 1, j)] - tf.u[mesh.u_idx(i, j)]) / dx * dy;
            double diff_elong_W = 2.0 * mu_cell_W * (tf.u[mesh.u_idx(i, j)] - tf.u[mesh.u_idx(i - 1, j)]) / dx * dy;
            val -= (diff_elong_E - diff_elong_W);

            // (b) Viscosité de cisaillement et rotation mu (d2u/dy2 + d2v/dxdy)
            double mu_node_N = fields_old.mu_node[mesh.node_idx(i, j + 1)];
            double mu_node_S = fields_old.mu_node[mesh.node_idx(i, j)];

            double shear_N = mu_node_N * ((tf.u[mesh.u_idx(i, j + 1)] - tf.u[mesh.u_idx(i, j)]) / dy +
                                          (tf.v[mesh.v_idx(i, j + 1)] - tf.v[mesh.v_idx(i - 1, j + 1)]) / dx) * dx;
            double shear_S = mu_node_S * ((tf.u[mesh.u_idx(i, j)] - tf.u[mesh.u_idx(i, j - 1)]) / dy +
                                          (tf.v[mesh.v_idx(i, j)] - tf.v[mesh.v_idx(i - 1, j)]) / dx) * dx;
            val -= (shear_N - shear_S);

            // Terme de pénalité de Lagrangien augmenté : - r * d(div)/dx * dvol = - r * (div_E - div_W) * dy
            double grad_div_x = (div_cell[c_idx(i, j)] - div_cell[c_idx(i - 1, j)]) * dy;
            val -= r_al * grad_div_x;

            // Gradient de pression discret B_u^T (uniquement si monolithique)
            if (is_monolithic()) {
                double grad_p = (tf.p[mesh.cell_idx(i, j)] - tf.p[mesh.cell_idx(i - 1, j)]) * dy;
                val += grad_p;
            }

            y[dof] = val;
        }
    }

    // =========================================================================
    // 2. Équation de Quantité de Mouvement en v (Faces horizontales)
    // =========================================================================
    int v_offset = n_owned_u;
    for (int j = 1; j <= v_ny; ++j) {
        for (int i = 1; i <= mesh.nx_local; ++i) {
            int dof = v_offset + (j - 1) * mesh.nx_local + (i - 1);

            // Condition aux limites physique Dirichlet ou Free-Slip sur les parois basse / haute ou obstacle solide
            bool is_wall_bot = (j == 1 && mesh.is_boundary_bottom && (bc.bottom.type == BCType::DIRICHLET || bc.bottom.type == BCType::FREE_SLIP));
            bool is_wall_top = (j == v_ny && mesh.is_boundary_top && (bc.top.type == BCType::DIRICHLET || bc.top.type == BCType::FREE_SLIP));
            bool in_obs = is_v_face_in_obstacle(i, j);
            if (is_wall_bot || is_wall_top || in_obs) {
                double rho_v = fields_old.rho_v[mesh.v_idx(i, j)];
                double diag_bc = (rho_v / dt) * dvol;
                y[dof] = diag_bc * tf.v[mesh.v_idx(i, j)];
                continue;
            }

            double rho_v = fields_old.rho_v[mesh.v_idx(i, j)];

            // Terme instationnaire
            double val = (rho_v / dt) * tf.v[mesh.v_idx(i, j)] * dvol;

            // Terme convectif semi-implicite F_vv
            if (enable_convection) {
                double v_n = fields_old.v[mesh.v_idx(i, j)];
                double v_n_N = fields_old.v[mesh.v_idx(i, j + 1)];
                double v_n_S = fields_old.v[mesh.v_idx(i, j - 1)];

                double u_n_E = 0.5 * (fields_old.u[mesh.u_idx(i + 1, j - 1)] + fields_old.u[mesh.u_idx(i + 1, j)]);
                double u_n_W = 0.5 * (fields_old.u[mesh.u_idx(i, j - 1)] + fields_old.u[mesh.u_idx(i, j)]);

                double conv_E = u_n_E * 0.5 * (tf.v[mesh.v_idx(i, j)] + tf.v[mesh.v_idx(i + 1, j)]) * dy;
                double conv_W = u_n_W * 0.5 * (tf.v[mesh.v_idx(i - 1, j)] + tf.v[mesh.v_idx(i, j)]) * dy;
                double conv_N = 0.5 * (v_n + v_n_N) * 0.5 * (tf.v[mesh.v_idx(i, j)] + tf.v[mesh.v_idx(i, j + 1)]) * dx;
                double conv_S = 0.5 * (v_n_S + v_n) * 0.5 * (tf.v[mesh.v_idx(i, j - 1)] + tf.v[mesh.v_idx(i, j)]) * dx;

                val += rho_v * (conv_E - conv_W + conv_N - conv_S);
            }

            // Termes visqueux implicites d'ordre 2
            // (a) Viscosité d'élongation 2*mu d2v/dy2
            double mu_cell_N = fields_old.mu[mesh.cell_idx(i, j)];
            double mu_cell_S = fields_old.mu[mesh.cell_idx(i, j - 1)];
            double diff_elong_N = 2.0 * mu_cell_N * (tf.v[mesh.v_idx(i, j + 1)] - tf.v[mesh.v_idx(i, j)]) / dy * dx;
            double diff_elong_S = 2.0 * mu_cell_S * (tf.v[mesh.v_idx(i, j)] - tf.v[mesh.v_idx(i, j - 1)]) / dy * dx;
            val -= (diff_elong_N - diff_elong_S);

            // (b) Viscosité de cisaillement et rotation mu (d2v/dx2 + d2u/dxdy)
            double mu_node_E = fields_old.mu_node[mesh.node_idx(i + 1, j)];
            double mu_node_W = fields_old.mu_node[mesh.node_idx(i, j)];

            double shear_E = mu_node_E * ((tf.v[mesh.v_idx(i + 1, j)] - tf.v[mesh.v_idx(i, j)]) / dx +
                                          (tf.u[mesh.u_idx(i + 1, j)] - tf.u[mesh.u_idx(i + 1, j - 1)]) / dy) * dy;
            double shear_W = mu_node_W * ((tf.v[mesh.v_idx(i, j)] - tf.v[mesh.v_idx(i - 1, j)]) / dx +
                                          (tf.u[mesh.u_idx(i, j)] - tf.u[mesh.u_idx(i, j - 1)]) / dy) * dy;
            val -= (shear_E - shear_W);

            // Terme de pénalité de Lagrangien augmenté : - r * d(div)/dy * dvol = - r * (div_N - div_S) * dx
            double grad_div_y = (div_cell[c_idx(i, j)] - div_cell[c_idx(i, j - 1)]) * dx;
            val -= r_al * grad_div_y;

            // Gradient de pression discret B_v^T (uniquement si monolithique)
            if (is_monolithic()) {
                double grad_p = (tf.p[mesh.cell_idx(i, j)] - tf.p[mesh.cell_idx(i, j - 1)]) * dx;
                val += grad_p;
            }

            y[dof] = val;
        }
    }

    // =========================================================================
    // 3. Équation de Continuité (si monolithique)
    // =========================================================================
    if (is_monolithic()) {
        int p_offset = n_owned_u + n_owned_v;
        for (int j = 1; j <= mesh.ny_local; ++j) {
            for (int i = 1; i <= mesh.nx_local; ++i) {
                int dof = p_offset + (j - 1) * mesh.nx_local + (i - 1);
                double div = div_cell[c_idx(i, j)] * dvol;
                double diag_p_term = (dvol / r_al) * tf.p[mesh.cell_idx(i, j)];
                y[dof] = div - diag_p_term;
            }
        }
    }
}

void NavierStokesSolver::assemble_rhs(const FlowFields& fields, double dt, std::vector<double>& rhs) const {
    rhs.assign(n_owned_total, 0.0);

    double dx = mesh.dx;
    double dy = mesh.dy;
    double dvol = dx * dy;

    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);

    // RHS pour u
    for (int j = 1; j <= mesh.ny_local; ++j) {
        for (int i = 1; i <= u_nx; ++i) {
            int dof = (j - 1) * u_nx + (i - 1);
            double rho_u = fields.rho_u[mesh.u_idx(i, j)];
            double diag_bc = (rho_u / dt) * dvol;

            // Si face dans un obstacle solide (adhérence nulle)
            if (is_u_face_in_obstacle(i, j)) {
                rhs[dof] = 0.0;
                continue;
            }

            // Si paroi physique gauche
            if (i == 1 && mesh.is_boundary_left) {
                double u_bc = 0.0;
                if (bc.left.type == BCType::DIRICHLET) {
                    if (bc.left.profile == "POISEUILLE_STEP") {
                        double y = mesh.yc(j);
                        if (y >= bc.left.y_step) {
                            double h = mesh.ly - bc.left.y_step;
                            u_bc = 6.0 * bc.left.u_mean * (y - bc.left.y_step) * (mesh.ly - y) / (h * h);
                        }
                    } else if (bc.left.profile == "PARABOLIC") {
                        double y = mesh.yc(j);
                        double h = mesh.ly;
                        u_bc = 6.0 * bc.left.u_mean * y * (h - y) / (h * h);
                    } else {
                        u_bc = bc.left.u_val;
                    }
                } else if (bc.left.type == BCType::NEUMANN) {
                    u_bc = 0.0;
                }
                rhs[dof] = diag_bc * u_bc;
                continue;
            }
            // Si paroi physique droite
            if (i == u_nx && mesh.is_boundary_right) {
                double u_bc = 0.0;
                if (bc.right.type == BCType::DIRICHLET) {
                    u_bc = bc.right.u_val;
                } else if (bc.right.type == BCType::NEUMANN) {
                    u_bc = 0.0;
                }
                rhs[dof] = diag_bc * u_bc;
                continue;
            }

            double val = (rho_u / dt) * fields.u[mesh.u_idx(i, j)] * dvol;
            val += rho_u * gx * dvol;
            val += fields.F_sigma_x[mesh.u_idx(i, j)] * dvol;

            // Pour Lagrangien augmenté standard : le gradient de pression -grad(p^n) est explicite au second membre
            if (is_augmented_lagrangian()) {
                double grad_p = (fields.p[mesh.cell_idx(i, j)] - fields.p[mesh.cell_idx(i - 1, j)]) * dy;
                val -= grad_p;
            }

            // Contribution de contrainte tangentielle à la paroi pour u (haut / bas)
            if (j == mesh.ny_local && mesh.is_boundary_top && bc.top.type == BCType::DIRICHLET) {
                double mu_N = fields.mu_node[mesh.node_idx(i, j + 1)];
                val += mu_N * (2.0 * bc.top.u_val / dy) * dx;
            }
            if (j == 1 && mesh.is_boundary_bottom && bc.bottom.type == BCType::DIRICHLET) {
                double mu_S = fields.mu_node[mesh.node_idx(i, j)];
                val += mu_S * (2.0 * bc.bottom.u_val / dy) * dx;
            }

            rhs[dof] = val;
        }
    }

    // RHS pour v
    int v_offset = n_owned_u;
    for (int j = 1; j <= v_ny; ++j) {
        for (int i = 1; i <= mesh.nx_local; ++i) {
            int dof = v_offset + (j - 1) * mesh.nx_local + (i - 1);
            double rho_v = fields.rho_v[mesh.v_idx(i, j)];
            double diag_bc = (rho_v / dt) * dvol;

            // Si face dans un obstacle solide (adhérence nulle)
            if (is_v_face_in_obstacle(i, j)) {
                rhs[dof] = 0.0;
                continue;
            }

            if (j == 1 && mesh.is_boundary_bottom && (bc.bottom.type == BCType::DIRICHLET || bc.bottom.type == BCType::FREE_SLIP)) {
                double v_bc = (bc.bottom.type == BCType::DIRICHLET) ? bc.bottom.v_val : 0.0;
                rhs[dof] = diag_bc * v_bc;
                continue;
            }
            if (j == v_ny && mesh.is_boundary_top && (bc.top.type == BCType::DIRICHLET || bc.top.type == BCType::FREE_SLIP)) {
                double v_bc = (bc.top.type == BCType::DIRICHLET) ? bc.top.v_val : 0.0;
                rhs[dof] = diag_bc * v_bc;
                continue;
            }

            double val = (rho_v / dt) * fields.v[mesh.v_idx(i, j)] * dvol;
            val += rho_v * gy * dvol;
            if (enable_energy) {
                double T_v = 0.5 * (fields.T[mesh.cell_idx(i, j)] + fields.T[mesh.cell_idx(i, j - 1)]);
                val += rho_v * beta * (T_v - T_ref) * gy * dvol;
            }
            val += fields.F_sigma_y[mesh.v_idx(i, j)] * dvol;

            // Pour Lagrangien augmenté standard : -grad(p^n)
            if (is_augmented_lagrangian()) {
                double grad_p = (fields.p[mesh.cell_idx(i, j)] - fields.p[mesh.cell_idx(i, j - 1)]) * dx;
                val -= grad_p;
            }

            // Contribution de contrainte tangentielle à la paroi pour v (gauche / droite)
            if (i == 1 && mesh.is_boundary_left && bc.left.type == BCType::DIRICHLET) {
                double mu_W = fields.mu_node[mesh.node_idx(i, j)];
                val += mu_W * (2.0 * bc.left.v_val / dx) * dy;
            }
            if (i == mesh.nx_local && mesh.is_boundary_right && bc.right.type == BCType::DIRICHLET) {
                double mu_E = fields.mu_node[mesh.node_idx(i + 1, j)];
                val += mu_E * (2.0 * bc.right.v_val / dx) * dy;
            }

            rhs[dof] = val;
        }
    }

    // RHS pour p (si monolithique régularisé)
    if (is_monolithic()) {
        int p_offset = n_owned_u + n_owned_v;
        for (int j = 1; j <= mesh.ny_local; ++j) {
            for (int i = 1; i <= mesh.nx_local; ++i) {
                int dof = p_offset + (j - 1) * mesh.nx_local + (i - 1);
                rhs[dof] = - (dvol / r_al) * fields.p[mesh.cell_idx(i, j)];
            }
        }
    }
}

void NavierStokesSolver::update_pressure_al(FlowFields& fields) const {
    double dx = mesh.dx;
    double dy = mesh.dy;

    for (int j = 1; j <= mesh.ny_local; ++j) {
        for (int i = 1; i <= mesh.nx_local; ++i) {
            double div = (fields.u[mesh.u_idx(i + 1, j)] - fields.u[mesh.u_idx(i, j)]) / dx +
                         (fields.v[mesh.v_idx(i, j + 1)] - fields.v[mesh.v_idx(i, j)]) / dy;
            fields.p[mesh.cell_idx(i, j)] -= r_al * div;
        }
    }

    MpiComm::sync_cell_scalar(mesh, fields.p);
}

void NavierStokesSolver::assemble_preconditioner_blocks(const FlowFields& fields, double dt) {
    double dx = mesh.dx;
    double dy = mesh.dy;
    double dvol = dx * dy;

    // Calcul des diagonales effectives de Fuu et Fvv pour le préconditionnement Jacobi
    diag_u.assign(n_owned_u, 1.0);
    int u_nx = mesh.nx_local + (mesh.is_boundary_right ? 1 : 0);
    for (int j = 1; j <= mesh.ny_local; ++j) {
        for (int i = 1; i <= u_nx; ++i) {
            int dof = (j - 1) * u_nx + (i - 1);
            double rho_u = fields.rho_u[mesh.u_idx(i, j)];
            bool is_wall = (i == 1 && mesh.is_boundary_left) ||
                           (i == u_nx && mesh.is_boundary_right) ||
                           is_u_face_in_obstacle(i, j);
            if (is_wall) {
                diag_u[dof] = (rho_u / dt) * dvol;
                continue;
            }
            double mu_c = fields.mu[mesh.cell_idx(i, j)];
            double d = (rho_u / dt) * dvol + 4.0 * mu_c * (dy / dx) + 2.0 * mu_c * (dx / dy) + 2.0 * r_al * (dy / dx);
            diag_u[dof] = (d > 1.0e-15) ? d : 1.0;
        }
    }

    diag_v.assign(n_owned_v, 1.0);
    int v_ny = mesh.ny_local + (mesh.is_boundary_top ? 1 : 0);
    for (int j = 1; j <= v_ny; ++j) {
        for (int i = 1; i <= mesh.nx_local; ++i) {
            int dof = (j - 1) * mesh.nx_local + (i - 1);
            double rho_v = fields.rho_v[mesh.v_idx(i, j)];
            bool is_wall = (j == 1 && mesh.is_boundary_bottom && (bc.bottom.type == BCType::DIRICHLET || bc.bottom.type == BCType::FREE_SLIP)) ||
                           (j == v_ny && mesh.is_boundary_top && (bc.top.type == BCType::DIRICHLET || bc.top.type == BCType::FREE_SLIP)) ||
                           is_v_face_in_obstacle(i, j);
            if (is_wall) {
                diag_v[dof] = (rho_v / dt) * dvol;
                continue;
            }
            double mu_c = fields.mu[mesh.cell_idx(i, j)];
            double d = (rho_v / dt) * dvol + 4.0 * mu_c * (dx / dy) + 2.0 * mu_c * (dy / dx) + 2.0 * r_al * (dx / dy);
            diag_v[dof] = (d > 1.0e-15) ? d : 1.0;
        }
    }

    if (is_monolithic()) {
        diag_p.assign(n_owned_p, dvol / r_al);
    }
}

void NavierStokesSolver::solve_energy(FlowFields& fields, double dt) const {
    if (!enable_energy) return;

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    int n_cells = nx * ny;

    double dx = mesh.dx;
    double dy = mesh.dy;
    double dvol = dx * dy;
    double rho = fields.rho[mesh.cell_idx(1, 1)];

    auto cell_dof = [&](int i, int j) {
        return (j - 1) * nx + (i - 1);
    };

    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
    std::vector<double> rhs(n_cells, 0.0);
    std::vector<double> T_sol(n_cells, 0.0);

    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int c_dof = cell_dof(i, j);
            T_sol[c_dof] = fields.T[mesh.cell_idx(i, j)];

            double a_c = (rho * Cp / dt) * dvol;
            rhs[c_dof] = (rho * Cp / dt) * dvol * fields.T[mesh.cell_idx(i, j)];

            // Flux x-Est
            double u_e = fields.u[mesh.u_idx(i + 1, j)];
            double d_e = k_thermal * dy / dx;
            double conv_e = rho * Cp * u_e * dy * 0.5;

            // Flux x-Ouest
            double u_w = fields.u[mesh.u_idx(i, j)];
            double d_w = k_thermal * dy / dx;
            double conv_w = rho * Cp * u_w * dy * 0.5;

            // Flux y-Nord
            double v_n = fields.v[mesh.v_idx(i, j + 1)];
            double d_n = k_thermal * dx / dy;
            double conv_n = rho * Cp * v_n * dx * 0.5;

            // Flux y-Sud
            double v_s = fields.v[mesh.v_idx(i, j)];
            double d_s = k_thermal * dx / dy;
            double conv_s = rho * Cp * v_s * dx * 0.5;

            // Voisin Est
            if (i < nx) {
                a_c += d_e + conv_e;
                rows.push_back(c_dof);
                cols.push_back(cell_dof(i + 1, j));
                vals.push_back(-d_e + conv_e);
            } else if (mesh.is_boundary_right) {
                if (bc.right.T_type == BCType::DIRICHLET) {
                    a_c += 2.0 * d_e;
                    rhs[c_dof] += 2.0 * d_e * bc.right.T_val;
                }
            }

            // Voisin Ouest
            if (i > 1) {
                a_c += d_w - conv_w;
                rows.push_back(c_dof);
                cols.push_back(cell_dof(i - 1, j));
                vals.push_back(-d_w - conv_w);
            } else if (mesh.is_boundary_left) {
                if (bc.left.T_type == BCType::DIRICHLET) {
                    a_c += 2.0 * d_w;
                    rhs[c_dof] += 2.0 * d_w * bc.left.T_val;
                }
            }

            // Voisin Nord
            if (j < ny) {
                a_c += d_n + conv_n;
                rows.push_back(c_dof);
                cols.push_back(cell_dof(i, j + 1));
                vals.push_back(-d_n + conv_n);
            } else if (mesh.is_boundary_top) {
                if (bc.top.T_type == BCType::DIRICHLET) {
                    a_c += 2.0 * d_n;
                    rhs[c_dof] += 2.0 * d_n * bc.top.T_val;
                }
            }

            // Voisin Sud
            if (j > 1) {
                a_c += d_s - conv_s;
                rows.push_back(c_dof);
                cols.push_back(cell_dof(i, j - 1));
                vals.push_back(-d_s - conv_s);
            } else if (mesh.is_boundary_bottom) {
                if (bc.bottom.T_type == BCType::DIRICHLET) {
                    a_c += 2.0 * d_s;
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
    std::vector<double> Ax(n_cells, 0.0);
    A.matvec(T_sol, Ax);
    for (int i = 0; i < n_cells; ++i) r[i] -= Ax[i];

    std::vector<double> r0_hat = r;
    std::vector<double> p = r;
    std::vector<double> v(n_cells, 0.0);
    std::vector<double> s(n_cells, 0.0);
    std::vector<double> t(n_cells, 0.0);
    std::vector<double> y_prec(n_cells, 0.0);
    std::vector<double> z_prec(n_cells, 0.0);

    double rho_old = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    double res_init = 0.0;
    for (int i = 0; i < n_cells; ++i) res_init += r[i] * r[i];
    res_init = std::sqrt(res_init);
    if (res_init > 1e-13) {
        int max_it = 500;
        for (int iter = 0; iter < max_it; ++iter) {
            double rho_new = 0.0;
            for (int i = 0; i < n_cells; ++i) rho_new += r0_hat[i] * r[i];
            if (std::abs(rho_new) < 1e-30) break;

            if (iter > 0) {
                double beta_param = (rho_new / rho_old) * (alpha / omega);
                for (int i = 0; i < n_cells; ++i) {
                    p[i] = r[i] + beta_param * (p[i] - omega * v[i]);
                }
            } else {
                p = r;
            }

            // Préconditionnement Jacobi : y = D^{-1} p
            for (int i = 0; i < n_cells; ++i) y_prec[i] = p[i] / A.diag[i];
            A.matvec(y_prec, v);

            double r0v = 0.0;
            for (int i = 0; i < n_cells; ++i) r0v += r0_hat[i] * v[i];
            if (std::abs(r0v) < 1e-30) break;
            alpha = rho_new / r0v;

            for (int i = 0; i < n_cells; ++i) s[i] = r[i] - alpha * v[i];
            double s_norm = 0.0;
            for (int i = 0; i < n_cells; ++i) s_norm += s[i] * s[i];
            if (std::sqrt(s_norm) / res_init < 1e-8) {
                for (int i = 0; i < n_cells; ++i) T_sol[i] += alpha * y_prec[i];
                break;
            }

            for (int i = 0; i < n_cells; ++i) z_prec[i] = s[i] / A.diag[i];
            A.matvec(z_prec, t);

            double tt = 0.0, ts = 0.0;
            for (int i = 0; i < n_cells; ++i) {
                tt += t[i] * t[i];
                ts += t[i] * s[i];
            }
            omega = (tt > 1e-30) ? (ts / tt) : 0.0;

            for (int i = 0; i < n_cells; ++i) {
                T_sol[i] += alpha * y_prec[i] + omega * z_prec[i];
                r[i] = s[i] - omega * t[i];
            }

            rho_old = rho_new;

            double r_norm = 0.0;
            for (int i = 0; i < n_cells; ++i) r_norm += r[i] * r[i];
            if (std::sqrt(r_norm) / res_init < 1e-8) break;
        }
    }

    // Réinjecter T_sol dans fields.T
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            fields.T[mesh.cell_idx(i, j)] = T_sol[cell_dof(i, j)];
        }
    }

    MpiComm::sync_cell_scalar(mesh, fields.T);
    MpiComm::apply_thermal_boundary_conditions(mesh, bc, fields);
}

double NavierStokesSolver::compute_mean_nusselt_hot_wall(const FlowFields& fields) const {
    double local_nu_sum = 0.0;
    int count = 0;
    if (mesh.is_boundary_left) {
        for (int j = 1; j <= mesh.ny_local; ++j) {
            double T_w = bc.left.T_val;
            double T_1 = fields.T[mesh.cell_idx(1, j)];
            double delta_T = std::abs(bc.left.T_val - bc.right.T_val);
            if (delta_T < 1e-12) delta_T = 1.0;
            double nu_local = 2.0 * (T_w - T_1) / mesh.dx * (mesh.lx / delta_T);
            local_nu_sum += nu_local;
            count++;
        }
    }
    double global_nu_sum = MpiComm::global_sum(mesh, local_nu_sum);
    double global_count = MpiComm::global_sum(mesh, static_cast<double>(count));
    if (global_count > 0.0) {
        return global_nu_sum / global_count;
    }
    return 1.0;
}

} // namespace fugu

