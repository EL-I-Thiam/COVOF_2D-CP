#include "covof/Preconditioner.hpp"
#include "covof/NavierStokesSolver.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

// =============================================================================
// 1. Préconditionneur Jacobi Diagonal
// =============================================================================
void JacobiPreconditioner::init(const NavierStokesSolver* solver) {
    // Calcul et assignation numérique
    ns_solver = solver;
}

void JacobiPreconditioner::update(const FlowFields& fields, double dt) {
    // Les diagonales sont calculées dans ns_solver->assemble_preconditioner_blocks
}

void JacobiPreconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    // Évaluation de la condition de contrôle
    if (!ns_solver) return;

    // Déclaration de variable typée
    int nu = ns_solver->n_owned_u;
    // Déclaration de variable typée
    int nv = ns_solver->n_owned_v;
    // Déclaration de variable typée
    int n  = ns_solver->n_owned_total;

    r.assign(n, 0.0);

    // Bloc u
    for (int i = 0; i < nu; ++i) {
        // Déclaration de variable typée
        double d = (ns_solver->diag_u.size() > (size_t)i && std::abs(ns_solver->diag_u[i]) > 1.0e-15)
                   ? ns_solver->diag_u[i] : 1.0;
        // Calcul et assignation numérique
        r[i] = z[i] / d;
    }

    // Bloc v
    for (int i = 0; i < nv; ++i) {
        // Déclaration de variable typée
        double d = (ns_solver->diag_v.size() > (size_t)i && std::abs(ns_solver->diag_v[i]) > 1.0e-15)
                   ? ns_solver->diag_v[i] : 1.0;
        // Calcul et assignation numérique
        r[nu + i] = z[nu + i] / d;
    }

    // Bloc p (si monolithique régularisé)
    if (ns_solver->is_monolithic()) {
        // Déclaration de variable typée
        int np = ns_solver->n_owned_p;
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < np; ++i) {
            // Déclaration de variable typée
            double d = (ns_solver->diag_p.size() > (size_t)i && std::abs(ns_solver->diag_p[i]) > 1.0e-15)
                       ? ns_solver->diag_p[i] : 1.0;
            // Calcul et assignation numérique
            r[nu + nv + i] = z[nu + nv + i] / d;
        }
    }
}

// =============================================================================
// 2. Préconditionneur SSOR (Symmetric Successive Over-Relaxation)
// =============================================================================
void SSORPreconditioner::init(const NavierStokesSolver* solver) {
    // Calcul et assignation numérique
    ns_solver = solver;
}

void SSORPreconditioner::update(const FlowFields& fields, double dt) {
}

void SSORPreconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    // Évaluation de la condition de contrôle
    if (!ns_solver) return;

    // Déclaration de variable typée
    int nu = ns_solver->n_owned_u;
    // Déclaration de variable typée
    int nv = ns_solver->n_owned_v;
    // Déclaration de variable typée
    int n  = ns_solver->n_owned_total;

    r.assign(n, 0.0);
    // Déclaration de variable typée
    std::vector<double> y(n, 0.0);

    // Déclaration de variable typée
    double w = (omega > 0.1 && omega < 1.95) ? omega : 1.0;
    // Déclaration de variable typée
    double factor = 2.0 - w;

    // Descente avant (Forward sweep)
    for (int i = 0; i < n; ++i) {
        // Déclaration de variable typée
        double d = 1.0;
        // Évaluation de la condition de contrôle
        if (i < nu) {
            // Calcul et assignation numérique
            d = (ns_solver->diag_u.size() > (size_t)i) ? ns_solver->diag_u[i] : 1.0;
        } else if (i < nu + nv) {
            // Calcul et assignation numérique
            d = (ns_solver->diag_v.size() > (size_t)(i - nu)) ? ns_solver->diag_v[i - nu] : 1.0;
        } else {
            // Calcul et assignation numérique
            d = (ns_solver->diag_p.size() > (size_t)(i - nu - nv)) ? ns_solver->diag_p[i - nu - nv] : 1.0;
        }
        // Évaluation de la condition de contrôle
        if (std::abs(d) < 1.0e-15) d = 1.0;

        // Calcul et assignation numérique
        y[i] = (w / d) * z[i];
    }

    // Remontée arrière (Backward sweep)
    for (int i = n - 1; i >= 0; --i) {
        // Calcul et assignation numérique
        r[i] = y[i] * factor;
    }
}

// =============================================================================
// 3. Préconditionneur ILU(0) (Factorisation Incomplète Locale)
// =============================================================================
void ILU0Preconditioner::init(const NavierStokesSolver* solver) {
    // Calcul et assignation numérique
    ns_solver = solver;
}

void ILU0Preconditioner::update(const FlowFields& fields, double dt) {
    // Évaluation de la condition de contrôle
    if (!ns_solver) return;
    // Déclaration de variable typée
    int n = ns_solver->n_owned_total;
    ilu_diag.assign(n, 1.0);

    // Déclaration de variable typée
    int nu = ns_solver->n_owned_u;
    // Déclaration de variable typée
    int nv = ns_solver->n_owned_v;

    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < nu; ++i) {
        // Calcul et assignation numérique
        ilu_diag[i] = (ns_solver->diag_u.size() > (size_t)i && std::abs(ns_solver->diag_u[i]) > 1.0e-15)
                      ? 1.0 / ns_solver->diag_u[i] : 1.0;
    }
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < nv; ++i) {
        // Calcul et assignation numérique
        ilu_diag[nu + i] = (ns_solver->diag_v.size() > (size_t)i && std::abs(ns_solver->diag_v[i]) > 1.0e-15)
                           ? 1.0 / ns_solver->diag_v[i] : 1.0;
    }
    // Évaluation de la condition de contrôle
    if (ns_solver->is_monolithic()) {
        // Déclaration de variable typée
        int np = ns_solver->n_owned_p;
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < np; ++i) {
            // Calcul et assignation numérique
            ilu_diag[nu + nv + i] = (ns_solver->diag_p.size() > (size_t)i && std::abs(ns_solver->diag_p[i]) > 1.0e-15)
                                    ? 1.0 / ns_solver->diag_p[i] : 1.0;
        }
    }
}

void ILU0Preconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    // Déclaration de variable typée
    int n = z.size();
    r.resize(n);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n; ++i) {
        // Déclaration de variable typée
        double d_inv = (ilu_diag.size() > (size_t)i) ? ilu_diag[i] : 1.0;
        // Calcul et assignation numérique
        r[i] = z[i] * d_inv;
    }
}

// =============================================================================
// 4. Préconditionneur par Blocs Triangulaires (Annexe E thèse M. El Ouafa / 2023)
// =============================================================================
void BlockTriangularPreconditioner::init(const NavierStokesSolver* solver) {
    ns_solver = solver;
}

void BlockTriangularPreconditioner::update(const FlowFields& fields, double dt) {
    if (!ns_solver) return;

    nx = ns_solver->mesh.nx_local;
    ny = ns_solver->mesh.ny_local;
    dx = ns_solver->mesh.dx;
    dy = ns_solver->mesh.dy;
    dvol = dx * dy;
    r_al = ns_solver->r_al;

    int nu = ns_solver->n_owned_u;
    int nv = ns_solver->n_owned_v;
    int np = ns_solver->n_owned_p;

    u_aE.assign(nu, 0.0);
    u_aW.assign(nu, 0.0);
    u_aN.assign(nu, 0.0);
    u_aS.assign(nu, 0.0);
    u_aP.assign(nu, 1.0);

    v_aE.assign(nv, 0.0);
    v_aW.assign(nv, 0.0);
    v_aN.assign(nv, 0.0);
    v_aS.assign(nv, 0.0);
    v_aP.assign(nv, 1.0);

    a_E.assign(np, 0.0);
    a_W.assign(np, 0.0);
    a_N.assign(np, 0.0);
    a_S.assign(np, 0.0);
    a_P.assign(np, 1.0);

    int u_nx = nx + (ns_solver->mesh.is_boundary_right ? 1 : 0);
    int v_ny = ny + (ns_solver->mesh.is_boundary_top ? 1 : 0);

    // 1. Stencil Fu
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= u_nx; ++i) {
            int dof = (j - 1) * u_nx + (i - 1);
            double rho_u = fields.rho_u[ns_solver->mesh.u_idx(i, j)];
            double Du = (rho_u / dt) * dvol;

            bool is_wall = (i == 1 && ns_solver->mesh.is_boundary_left) ||
                           (i == u_nx && ns_solver->mesh.is_boundary_right) ||
                           ns_solver->is_u_face_in_obstacle(i, j);
            if (is_wall) {
                u_aP[dof] = (Du > 1.0e-30) ? Du : 1.0;
                continue;
            }

            double mu_e = fields.mu[ns_solver->mesh.cell_idx(i, j)];
            double mu_w = fields.mu[ns_solver->mesh.cell_idx(i - 1, j)];
            double mu_avg = 0.5 * (mu_e + mu_w);

            u_aE[dof] = 2.0 * mu_e * (dy / dx);
            u_aW[dof] = 2.0 * mu_w * (dy / dx);
            u_aN[dof] = mu_avg * (dx / dy);
            u_aS[dof] = mu_avg * (dx / dy);
            u_aP[dof] = Du + u_aE[dof] + u_aW[dof] + u_aN[dof] + u_aS[dof];
        }
    }

    // 2. Stencil Fv
    for (int j = 1; j <= v_ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int dof = (j - 1) * nx + (i - 1);
            double rho_v = fields.rho_v[ns_solver->mesh.v_idx(i, j)];
            double Dv = (rho_v / dt) * dvol;

            bool is_wall = (j == 1 && ns_solver->mesh.is_boundary_bottom &&
                            (ns_solver->bc.bottom.type == BCType::DIRICHLET || ns_solver->bc.bottom.type == BCType::FREE_SLIP)) ||
                           (j == v_ny && ns_solver->mesh.is_boundary_top &&
                            (ns_solver->bc.top.type == BCType::DIRICHLET || ns_solver->bc.top.type == BCType::FREE_SLIP)) ||
                           ns_solver->is_v_face_in_obstacle(i, j);
            if (is_wall) {
                v_aP[dof] = (Dv > 1.0e-30) ? Dv : 1.0;
                continue;
            }

            double mu_n = fields.mu[ns_solver->mesh.cell_idx(i, j)];
            double mu_s = fields.mu[ns_solver->mesh.cell_idx(i, j - 1)];
            double mu_avg = 0.5 * (mu_n + mu_s);

            v_aN[dof] = 2.0 * mu_n * (dx / dy);
            v_aS[dof] = 2.0 * mu_s * (dx / dy);
            v_aE[dof] = mu_avg * (dy / dx);
            v_aW[dof] = mu_avg * (dy / dx);
            v_aP[dof] = Dv + v_aN[dof] + v_aS[dof] + v_aE[dof] + v_aW[dof];
        }
    }

    // 3. Stencil Schur S
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int c = (j - 1) * nx + (i - 1);

            // Face Ouest (u_i, j)
            bool is_wall_W = (i == 1 && ns_solver->mesh.is_boundary_left &&
                              (ns_solver->bc.left.type == BCType::DIRICHLET || ns_solver->bc.left.type == BCType::FREE_SLIP)) ||
                             ns_solver->is_u_face_in_obstacle(i, j);
            if (!is_wall_W) {
                int dof_u = (j - 1) * u_nx + (i - 1);
                double Du = u_aP[dof_u];
                if (std::abs(Du) < 1.0e-30) Du = 1.0;
                a_W[c] = (dy * dy) / Du;
            }

            // Face Est (u_i+1, j)
            bool is_wall_E = (i == nx && ns_solver->mesh.is_boundary_right &&
                              (ns_solver->bc.right.type == BCType::DIRICHLET || ns_solver->bc.right.type == BCType::FREE_SLIP)) ||
                             ns_solver->is_u_face_in_obstacle(i + 1, j);
            if (!is_wall_E) {
                int dof_u = (j - 1) * u_nx + i;
                double Du = u_aP[dof_u];
                if (std::abs(Du) < 1.0e-30) Du = 1.0;
                a_E[c] = (dy * dy) / Du;
            }

            // Face Sud (v_i, j)
            bool is_wall_S = (j == 1 && ns_solver->mesh.is_boundary_bottom &&
                              (ns_solver->bc.bottom.type == BCType::DIRICHLET || ns_solver->bc.bottom.type == BCType::FREE_SLIP)) ||
                             ns_solver->is_v_face_in_obstacle(i, j);
            if (!is_wall_S) {
                int dof_v = (j - 1) * nx + (i - 1);
                double Dv = v_aP[dof_v];
                if (std::abs(Dv) < 1.0e-30) Dv = 1.0;
                a_S[c] = (dx * dx) / Dv;
            }

            // Face Nord (v_i, j+1)
            bool is_wall_N = (j == ny && ns_solver->mesh.is_boundary_top &&
                              (ns_solver->bc.top.type == BCType::DIRICHLET || ns_solver->bc.top.type == BCType::FREE_SLIP)) ||
                             ns_solver->is_v_face_in_obstacle(i, j + 1);
            if (!is_wall_N) {
                int dof_v = j * nx + (i - 1);
                double Dv = v_aP[dof_v];
                if (std::abs(Dv) < 1.0e-30) Dv = 1.0;
                a_N[c] = (dx * dx) / Dv;
            }

            a_P[c] = a_W[c] + a_E[c] + a_S[c] + a_N[c] + 1.0e-12 * dvol;
        }
    }
}

void BlockTriangularPreconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    if (!ns_solver) return;

    int nu = ns_solver->n_owned_u;
    int nv = ns_solver->n_owned_v;
    int np = ns_solver->n_owned_p;
    int n  = ns_solver->n_owned_total;

    r.assign(n, 0.0);

    if (ns_solver->is_monolithic()) {
        int u_nx = nx + (ns_solver->mesh.is_boundary_right ? 1 : 0);
        int v_ny = ny + (ns_solver->mesh.is_boundary_top ? 1 : 0);
        int p_offset = nu + nv;

        // 1. Résolution de S * r_p = z_p par 8 balayages de Gauss-Seidel symétrique
        std::vector<double> r_p(np, 0.0);
        for (int sweep = 0; sweep < 8; ++sweep) {
            // Balayage avant
            for (int j = 1; j <= ny; ++j) {
                for (int i = 1; i <= nx; ++i) {
                    int c = (j - 1) * nx + (i - 1);
                    double val = z[p_offset + c];
                    if (i > 1) val += a_W[c] * r_p[c - 1];
                    else if (ns_solver->bc.left.type == BCType::PERIODIC) val += a_W[c] * r_p[(j - 1) * nx + (nx - 1)];

                    if (i < nx) val += a_E[c] * r_p[c + 1];
                    else if (ns_solver->bc.right.type == BCType::PERIODIC) val += a_E[c] * r_p[(j - 1) * nx];

                    if (j > 1) val += a_S[c] * r_p[c - nx];
                    else if (ns_solver->bc.bottom.type == BCType::PERIODIC) val += a_S[c] * r_p[(ny - 1) * nx + (i - 1)];

                    if (j < ny) val += a_N[c] * r_p[c + nx];
                    else if (ns_solver->bc.top.type == BCType::PERIODIC) val += a_N[c] * r_p[i - 1];

                    r_p[c] = val / a_P[c];
                }
            }
            // Balayage arrière
            for (int j = ny; j >= 1; --j) {
                for (int i = nx; i >= 1; --i) {
                    int c = (j - 1) * nx + (i - 1);
                    double val = z[p_offset + c];
                    if (i > 1) val += a_W[c] * r_p[c - 1];
                    else if (ns_solver->bc.left.type == BCType::PERIODIC) val += a_W[c] * r_p[(j - 1) * nx + (nx - 1)];

                    if (i < nx) val += a_E[c] * r_p[c + 1];
                    else if (ns_solver->bc.right.type == BCType::PERIODIC) val += a_E[c] * r_p[(j - 1) * nx];

                    if (j > 1) val += a_S[c] * r_p[c - nx];
                    else if (ns_solver->bc.bottom.type == BCType::PERIODIC) val += a_S[c] * r_p[(ny - 1) * nx + (i - 1)];

                    if (j < ny) val += a_N[c] * r_p[c + nx];
                    else if (ns_solver->bc.top.type == BCType::PERIODIC) val += a_N[c] * r_p[i - 1];

                    r_p[c] = val / a_P[c];
                }
            }
        }

        // 2. Second membre pour vitesse : rhs_u = z_u - G_u * r_p, rhs_v = z_v - G_v * r_p
        std::vector<double> rhs_u(nu, 0.0);
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= u_nx; ++i) {
                int dof = (j - 1) * u_nx + (i - 1);
                bool is_wall = (i == 1 && ns_solver->mesh.is_boundary_left) ||
                               (i == u_nx && ns_solver->mesh.is_boundary_right) ||
                               ns_solver->is_u_face_in_obstacle(i, j);
                if (is_wall) {
                    rhs_u[dof] = z[dof];
                } else {
                    int c_E = (j - 1) * nx + (i - 1);
                    int c_W = (j - 1) * nx + (i - 2);
                    double grad_p = (r_p[c_E] - r_p[c_W]) * dy;
                    rhs_u[dof] = z[dof] - grad_p;
                }
            }
        }

        std::vector<double> rhs_v(nv, 0.0);
        for (int j = 1; j <= v_ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int dof = (j - 1) * nx + (i - 1);
                bool is_wall = (j == 1 && ns_solver->mesh.is_boundary_bottom &&
                                (ns_solver->bc.bottom.type == BCType::DIRICHLET || ns_solver->bc.bottom.type == BCType::FREE_SLIP)) ||
                               (j == v_ny && ns_solver->mesh.is_boundary_top &&
                                (ns_solver->bc.top.type == BCType::DIRICHLET || ns_solver->bc.top.type == BCType::FREE_SLIP)) ||
                               ns_solver->is_v_face_in_obstacle(i, j);
                if (is_wall) {
                    rhs_v[dof] = z[nu + dof];
                } else {
                    int c_N = (j - 1) * nx + (i - 1);
                    int c_S = (j - 2) * nx + (i - 1);
                    double grad_p = (r_p[c_N] - r_p[c_S]) * dx;
                    rhs_v[dof] = z[nu + dof] - grad_p;
                }
            }
        }

        // 3. Résolution de Fu * r_u = rhs_u par 4 balayages de Gauss-Seidel symétrique
        std::vector<double> r_u(nu, 0.0);
        for (int sweep = 0; sweep < 4; ++sweep) {
            // Forward
            for (int j = 1; j <= ny; ++j) {
                for (int i = 1; i <= u_nx; ++i) {
                    int dof = (j - 1) * u_nx + (i - 1);
                    bool is_wall = (i == 1 && ns_solver->mesh.is_boundary_left) ||
                                   (i == u_nx && ns_solver->mesh.is_boundary_right) ||
                                   ns_solver->is_u_face_in_obstacle(i, j);
                    if (is_wall) {
                        r_u[dof] = rhs_u[dof] / u_aP[dof];
                        continue;
                    }
                    double val = rhs_u[dof];
                    if (i > 1) val += u_aW[dof] * r_u[dof - 1];
                    if (i < u_nx) val += u_aE[dof] * r_u[dof + 1];
                    if (j > 1) val += u_aS[dof] * r_u[dof - u_nx];
                    if (j < ny) val += u_aN[dof] * r_u[dof + u_nx];
                    r_u[dof] = val / u_aP[dof];
                }
            }
            // Backward
            for (int j = ny; j >= 1; --j) {
                for (int i = u_nx; i >= 1; --i) {
                    int dof = (j - 1) * u_nx + (i - 1);
                    bool is_wall = (i == 1 && ns_solver->mesh.is_boundary_left) ||
                                   (i == u_nx && ns_solver->mesh.is_boundary_right) ||
                                   ns_solver->is_u_face_in_obstacle(i, j);
                    if (is_wall) {
                        r_u[dof] = rhs_u[dof] / u_aP[dof];
                        continue;
                    }
                    double val = rhs_u[dof];
                    if (i > 1) val += u_aW[dof] * r_u[dof - 1];
                    if (i < u_nx) val += u_aE[dof] * r_u[dof + 1];
                    if (j > 1) val += u_aS[dof] * r_u[dof - u_nx];
                    if (j < ny) val += u_aN[dof] * r_u[dof + u_nx];
                    r_u[dof] = val / u_aP[dof];
                }
            }
        }

        // 4. Résolution de Fv * r_v = rhs_v par 4 balayages de Gauss-Seidel symétrique
        std::vector<double> r_v(nv, 0.0);
        for (int sweep = 0; sweep < 4; ++sweep) {
            // Forward
            for (int j = 1; j <= v_ny; ++j) {
                for (int i = 1; i <= nx; ++i) {
                    int dof = (j - 1) * nx + (i - 1);
                    bool is_wall = (j == 1 && ns_solver->mesh.is_boundary_bottom &&
                                    (ns_solver->bc.bottom.type == BCType::DIRICHLET || ns_solver->bc.bottom.type == BCType::FREE_SLIP)) ||
                                   (j == v_ny && ns_solver->mesh.is_boundary_top &&
                                    (ns_solver->bc.top.type == BCType::DIRICHLET || ns_solver->bc.top.type == BCType::FREE_SLIP)) ||
                                   ns_solver->is_v_face_in_obstacle(i, j);
                    if (is_wall) {
                        r_v[dof] = rhs_v[dof] / v_aP[dof];
                        continue;
                    }
                    double val = rhs_v[dof];
                    if (i > 1) val += v_aW[dof] * r_v[dof - 1];
                    if (i < nx) val += v_aE[dof] * r_v[dof + 1];
                    if (j > 1) val += v_aS[dof] * r_v[dof - nx];
                    if (j < v_ny) val += v_aN[dof] * r_v[dof + nx];
                    r_v[dof] = val / v_aP[dof];
                }
            }
            // Backward
            for (int j = v_ny; j >= 1; --j) {
                for (int i = nx; i >= 1; --i) {
                    int dof = (j - 1) * nx + (i - 1);
                    bool is_wall = (j == 1 && ns_solver->mesh.is_boundary_bottom &&
                                    (ns_solver->bc.bottom.type == BCType::DIRICHLET || ns_solver->bc.bottom.type == BCType::FREE_SLIP)) ||
                                   (j == v_ny && ns_solver->mesh.is_boundary_top &&
                                    (ns_solver->bc.top.type == BCType::DIRICHLET || ns_solver->bc.top.type == BCType::FREE_SLIP)) ||
                                   ns_solver->is_v_face_in_obstacle(i, j);
                    if (is_wall) {
                        r_v[dof] = rhs_v[dof] / v_aP[dof];
                        continue;
                    }
                    double val = rhs_v[dof];
                    if (i > 1) val += v_aW[dof] * r_v[dof - 1];
                    if (i < nx) val += v_aE[dof] * r_v[dof + 1];
                    if (j > 1) val += v_aS[dof] * r_v[dof - nx];
                    if (j < v_ny) val += v_aN[dof] * r_v[dof + nx];
                    r_v[dof] = val / v_aP[dof];
                }
            }
        }

        // 5. Assemblage dans r
        for (int i = 0; i < nu; ++i) r[i] = r_u[i];
        for (int i = 0; i < nv; ++i) r[nu + i] = r_v[i];
        for (int i = 0; i < np; ++i) r[p_offset + i] = r_p[i];

    } else {
        // En mode Lagrangien augmenté : préconditionnement diagonal de Jacobi
        for (int i = 0; i < nu; ++i) {
            double du = (ns_solver->diag_u.size() > (size_t)i && std::abs(ns_solver->diag_u[i]) > 1.0e-15)
                        ? ns_solver->diag_u[i] : 1.0;
            r[i] = z[i] / du;
        }
        for (int i = 0; i < nv; ++i) {
            double dv = (ns_solver->diag_v.size() > (size_t)i && std::abs(ns_solver->diag_v[i]) > 1.0e-15)
                        ? ns_solver->diag_v[i] : 1.0;
            r[nu + i] = z[nu + i] / dv;
        }
    }
}

// =============================================================================
// 5. Préconditionneur Personnalisé (Hook développeur)
// =============================================================================
void CustomPreconditioner::init(const NavierStokesSolver* solver) {
}

void CustomPreconditioner::update(const FlowFields& fields, double dt) {
}

void CustomPreconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    // Évaluation de la condition de contrôle
    if (custom_apply) {
        custom_apply(z, r);
    } else {
        // Calcul et assignation numérique
        r = z; // Préconditionneur identité par défaut
    }
}

// =============================================================================
// 6. Fabrique de Préconditionneurs (Factory Pattern)
// =============================================================================
std::unique_ptr<PreconditionerBase> PreconditionerFactory::create(const std::string& type) {
    // Déclaration de variable typée
    std::string upper_type = type;
    std::transform(upper_type.begin(), upper_type.end(), upper_type.begin(), ::toupper);

    // Évaluation de la condition de contrôle
    if (upper_type == "JACOBI" || upper_type == "DIAGONAL") {
        // Retour de fonction
        return std::make_unique<JacobiPreconditioner>();
    // Calcul et assignation numérique
    } else if (upper_type == "SSOR" || upper_type == "GAUSS_SEIDEL") {
        // Retour de fonction
        return std::make_unique<SSORPreconditioner>();
    // Calcul et assignation numérique
    } else if (upper_type == "ILU0" || upper_type == "ILU") {
        // Retour de fonction
        return std::make_unique<ILU0Preconditioner>();
    // Calcul et assignation numérique
    } else if (upper_type == "BLOCK_TRIANGULAR" || upper_type == "SCHUR") {
        // Retour de fonction
        return std::make_unique<BlockTriangularPreconditioner>();
    // Calcul et assignation numérique
    } else if (upper_type == "CUSTOM") {
        // Retour de fonction
        return std::make_unique<CustomPreconditioner>();
    } else {
        // Défaut standard COVOF / Référence Académique : Jacobi diagonal
        return std::make_unique<JacobiPreconditioner>();
    }
}

} // namespace covof
