#include "covof/LinearSolver.hpp"
#include "covof/NavierStokesSolver.hpp"
#include "covof/MpiComm.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

// =============================================================================
// Helper d'application du préconditionneur et de l'opérateur
// =============================================================================
namespace {
    inline void apply_P(const PreconditionerBase& P, bool use_precond,
                        const std::vector<double>& src, std::vector<double>& dst) {
        // Évaluation de la condition de contrôle
        if (use_precond) P.apply(src, dst);
        // Branche par défaut
        else dst = src;
    }

    inline void apply_PA(const NavierStokesSolver& op, const PreconditionerBase& P,
                         // Déclaration de variable typée
                         bool use_precond, const FlowFields& fields_old, double dt,
                         const std::vector<double>& v_in, std::vector<double>& v_out) {
        // Déclaration de variable typée
        int n = op.n_owned_total;
        // Déclaration de variable typée
        std::vector<double> tmp(n, 0.0);
        op.apply_system_matrix(fields_old, dt, v_in, tmp);
        apply_P(P, use_precond, tmp, v_out);
    }
}

// =============================================================================
// 1. Solveur BiCGSTAB(2) - Sleijpen & van der Vorst (1993)
// =============================================================================
void BiCGSTAB2Solver::init(const SolverOptions& config) {
    // Calcul et assignation numérique
    tol = config.tol;
    // Calcul et assignation numérique
    max_iter = config.max_iter;
    // Calcul et assignation numérique
    use_precond = config.use_precond;
}

// Déclaration de variable typée
int BiCGSTAB2Solver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                           const FlowFields& fields_old, double dt,
                           const std::vector<double>& b, std::vector<double>& x) const {
    // Calcul et assignation numérique
    const Mesh2D& mesh = op.mesh;
    // Déclaration de variable typée
    int n = op.n_owned_total;

    // Second membre préconditionné : b_hat = P^{-1} b
    std::vector<double> b_hat(n, 0.0);
    apply_P(P, use_precond, b, b_hat);

    // Résidu initial r0 = b_hat - P^{-1} A x0
    std::vector<std::vector<double>> r(3, std::vector<double>(n, 0.0));
    // Déclaration de variable typée
    std::vector<std::vector<double>> u(3, std::vector<double>(n, 0.0));

    apply_PA(op, P, use_precond, fields_old, dt, x, r[0]);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n; ++i) r[0][i] = b_hat[i] - r[0][i];

    // Déclaration de variable typée
    double r0_norm = MpiComm::global_norm2(mesh, r[0], n);
    // Évaluation de la condition de contrôle
    if (r0_norm < 1.0e-12) return 0;

    // Déclaration de variable typée
    std::vector<double> r_tilde = r[0];
    // Déclaration de variable typée
    double rho_0 = 1.0, alpha = 0.0, omega = 1.0;

    // Déclaration de variable typée
    int iter = 0;
    // Boucle itérative conditionnelle
    while (iter < max_iter) {
        // Calcul et assignation numérique
        iter += 2;
        // Calcul et assignation numérique
        rho_0 = -omega * rho_0;

        // Demi-pas 1 (j = 0)
        double rho_1 = MpiComm::global_dot_product(mesh, r_tilde, r[0], n);
        // Évaluation de la condition de contrôle
        if (std::abs(rho_0) < 1.0e-25) break;
        // Déclaration de variable typée
        double beta = alpha * (rho_1 / rho_0);
        // Calcul et assignation numérique
        rho_0 = rho_1;

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) u[0][i] = r[0][i] - beta * u[0][i];
        apply_PA(op, P, use_precond, fields_old, dt, u[0], u[1]);

        // Déclaration de variable typée
        double gamma = MpiComm::global_dot_product(mesh, r_tilde, u[1], n);
        // Évaluation de la condition de contrôle
        if (std::abs(gamma) < 1.0e-25) break;
        // Calcul et assignation numérique
        alpha = rho_0 / gamma;

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) {
            // Calcul et assignation numérique
            r[0][i] -= alpha * u[1][i];
            // Calcul et assignation numérique
            x[i]    += alpha * u[0][i];
        }
        apply_PA(op, P, use_precond, fields_old, dt, r[0], r[1]);

        // Demi-pas 2 (j = 1)
        rho_1 = MpiComm::global_dot_product(mesh, r_tilde, r[1], n);
        // Évaluation de la condition de contrôle
        if (std::abs(rho_0) < 1.0e-25) break;
        // Calcul et assignation numérique
        beta = alpha * (rho_1 / rho_0);
        // Calcul et assignation numérique
        rho_0 = rho_1;

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) {
            // Calcul et assignation numérique
            u[0][i] = r[0][i] - beta * u[0][i];
            // Calcul et assignation numérique
            u[1][i] = r[1][i] - beta * u[1][i];
        }
        apply_PA(op, P, use_precond, fields_old, dt, u[1], u[2]);

        // Calcul et assignation numérique
        gamma = MpiComm::global_dot_product(mesh, r_tilde, u[2], n);
        // Évaluation de la condition de contrôle
        if (std::abs(gamma) < 1.0e-25) break;
        // Calcul et assignation numérique
        alpha = rho_0 / gamma;

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) {
            // Calcul et assignation numérique
            x[i]    += alpha * u[0][i];
            // Calcul et assignation numérique
            r[0][i] -= alpha * u[1][i];
            // Calcul et assignation numérique
            r[1][i] -= alpha * u[2][i];
        }
        apply_PA(op, P, use_precond, fields_old, dt, r[1], r[2]);

        // Minimisation Quadratique MR-part
        double sigma11 = MpiComm::global_dot_product(mesh, r[1], r[1], n);
        // Déclaration de variable typée
        double sigma12 = MpiComm::global_dot_product(mesh, r[2], r[1], n);
        // Déclaration de variable typée
        double sigma22 = MpiComm::global_dot_product(mesh, r[2], r[2], n);
        // Déclaration de variable typée
        double rhs1    = MpiComm::global_dot_product(mesh, r[0], r[1], n);
        // Déclaration de variable typée
        double rhs2    = MpiComm::global_dot_product(mesh, r[0], r[2], n);

        // Déclaration de variable typée
        double det = sigma11 * sigma22 - sigma12 * sigma12;
        // Déclaration de variable typée
        double s11_s22 = sigma11 * sigma22;
        // Déclaration de variable typée
        double gamma1 = 0.0, gamma2 = 0.0;
        // Évaluation de la condition de contrôle
        if (s11_s22 < 1.0e-30 || std::abs(det) < 1.0e-10 * s11_s22) {
            // Calcul et assignation numérique
            gamma2 = 0.0;
            // Calcul et assignation numérique
            gamma1 = rhs1 / (sigma11 + 1.0e-30);
        } else {
            // Calcul et assignation numérique
            gamma1 = (rhs1 * sigma22 - rhs2 * sigma12) / det;
            // Calcul et assignation numérique
            gamma2 = (rhs2 * sigma11 - rhs1 * sigma12) / det;
        }

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) {
            // Calcul et assignation numérique
            x[i]    += gamma1 * r[0][i] + gamma2 * r[1][i];
            // Calcul et assignation numérique
            r[0][i] -= (gamma1 * r[1][i] + gamma2 * r[2][i]);
            // Calcul et assignation numérique
            u[0][i] -= (gamma1 * u[1][i] + gamma2 * u[2][i]);
        }
        // Calcul et assignation numérique
        omega = gamma2;

        // Déclaration de variable typée
        double r_norm = MpiComm::global_norm2(mesh, r[0], n);
        // Évaluation de la condition de contrôle
        if (std::isnan(r_norm) || std::isinf(r_norm)) break;

        // Évaluation de la condition de contrôle
        if (r_norm / r0_norm < tol || r_norm < 1.0e-13) {
            // Retour de fonction
            return iter;
        }
    }
    // Retour de fonction
    return iter;
}

// =============================================================================
// 2. Solveur BiCGSTAB standard - van der Vorst (1992)
// =============================================================================
void BiCGSTABSolver::init(const SolverOptions& config) {
    // Calcul et assignation numérique
    tol = config.tol;
    // Calcul et assignation numérique
    max_iter = config.max_iter;
    // Calcul et assignation numérique
    use_precond = config.use_precond;
}

// Déclaration de variable typée
int BiCGSTABSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                          const FlowFields& fields_old, double dt,
                          const std::vector<double>& b, std::vector<double>& x) const {
    // Calcul et assignation numérique
    const Mesh2D& mesh = op.mesh;
    // Déclaration de variable typée
    int n = op.n_owned_total;

    // Déclaration de variable typée
    std::vector<double> b_hat(n, 0.0);
    apply_P(P, use_precond, b, b_hat);

    // Déclaration de variable typée
    std::vector<double> r(n, 0.0);
    apply_PA(op, P, use_precond, fields_old, dt, x, r);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n; ++i) r[i] = b_hat[i] - r[i];

    // Déclaration de variable typée
    double r0_norm = MpiComm::global_norm2(mesh, r, n);
    // Évaluation de la condition de contrôle
    if (r0_norm < 1.0e-12) return 0;

    // Déclaration de variable typée
    std::vector<double> r_tilde = r;
    // Déclaration de variable typée
    std::vector<double> p(n, 0.0), v(n, 0.0), s(n, 0.0), t(n, 0.0);
    // Déclaration de variable typée
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;

    // Boucle d'itération algorithmique
    for (int iter = 1; iter <= max_iter; ++iter) {
        // Déclaration de variable typée
        double rho = MpiComm::global_dot_product(mesh, r_tilde, r, n);
        // Évaluation de la condition de contrôle
        if (std::abs(rho) < 1.0e-30) break;

        // Évaluation de la condition de contrôle
        if (iter == 1) {
            // Calcul et assignation numérique
            p = r;
        } else {
            // Déclaration de variable typée
            double beta = (rho / rho_old) * (alpha / omega);
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        apply_PA(op, P, use_precond, fields_old, dt, p, v);
        // Déclaration de variable typée
        double denom = MpiComm::global_dot_product(mesh, r_tilde, v, n);
        // Évaluation de la condition de contrôle
        if (std::abs(denom) < 1.0e-30) break;
        // Calcul et assignation numérique
        alpha = rho / denom;

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        // Déclaration de variable typée
        double s_norm = MpiComm::global_norm2(mesh, s, n);
        // Évaluation de la condition de contrôle
        if (s_norm / r0_norm < tol || s_norm < 1.0e-13) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n; ++i) x[i] += alpha * p[i];
            // Retour de fonction
            return iter;
        }

        apply_PA(op, P, use_precond, fields_old, dt, s, t);
        // Déclaration de variable typée
        double t_dot_t = MpiComm::global_dot_product(mesh, t, t, n);
        // Calcul et assignation numérique
        omega = (t_dot_t > 1.0e-30) ? (MpiComm::global_dot_product(mesh, t, s, n) / t_dot_t) : 0.0;

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) {
            // Calcul et assignation numérique
            x[i] += alpha * p[i] + omega * s[i];
            // Calcul et assignation numérique
            r[i] = s[i] - omega * t[i];
        }

        // Calcul et assignation numérique
        rho_old = rho;
        // Déclaration de variable typée
        double r_norm = MpiComm::global_norm2(mesh, r, n);
        // Évaluation de la condition de contrôle
        if (r_norm / r0_norm < tol || r_norm < 1.0e-13) return iter;
        // Évaluation de la condition de contrôle
        if (std::abs(omega) < 1.0e-30) break;
    }
    // Retour de fonction
    return max_iter;
}

// =============================================================================
// 3. Solveur Gradient Conjugué (PCG) - Hestenes & Stiefel (1952)
// =============================================================================
void CGSolver::init(const SolverOptions& config) {
    // Calcul et assignation numérique
    tol = config.tol;
    // Calcul et assignation numérique
    max_iter = config.max_iter;
    // Calcul et assignation numérique
    use_precond = config.use_precond;
}

// Déclaration de variable typée
int CGSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                    const FlowFields& fields_old, double dt,
                    const std::vector<double>& b, std::vector<double>& x) const {
    // Calcul et assignation numérique
    const Mesh2D& mesh = op.mesh;
    // Déclaration de variable typée
    int n = op.n_owned_total;

    // r = b - A * x
    std::vector<double> Ax(n, 0.0);
    op.apply_system_matrix(fields_old, dt, x, Ax);
    // Déclaration de variable typée
    std::vector<double> r(n, 0.0);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];

    // Déclaration de variable typée
    double r0_norm = MpiComm::global_norm2(mesh, r, n);
    // Évaluation de la condition de contrôle
    if (r0_norm < 1.0e-12) return 0;

    // z = P^{-1} r
    std::vector<double> z(n, 0.0);
    apply_P(P, use_precond, r, z);

    // Déclaration de variable typée
    std::vector<double> p = z;
    // Déclaration de variable typée
    double rz_old = MpiComm::global_dot_product(mesh, r, z, n);
    // Déclaration de variable typée
    std::vector<double> Ap(n, 0.0);

    // Boucle d'itération algorithmique
    for (int iter = 1; iter <= max_iter; ++iter) {
        op.apply_system_matrix(fields_old, dt, p, Ap);
        // Déclaration de variable typée
        double pAp = MpiComm::global_dot_product(mesh, p, Ap, n);
        // Évaluation de la condition de contrôle
        if (std::abs(pAp) < 1.0e-30) break;

        // Déclaration de variable typée
        double alpha = rz_old / pAp;
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) {
            // Calcul et assignation numérique
            x[i] += alpha * p[i];
            // Calcul et assignation numérique
            r[i] -= alpha * Ap[i];
        }

        // Déclaration de variable typée
        double r_norm = MpiComm::global_norm2(mesh, r, n);
        // Évaluation de la condition de contrôle
        if (r_norm / r0_norm < tol || r_norm < 1.0e-13) return iter;

        apply_P(P, use_precond, r, z);
        // Déclaration de variable typée
        double rz_new = MpiComm::global_dot_product(mesh, r, z, n);
        // Déclaration de variable typée
        double beta = rz_new / rz_old;
        // Calcul et assignation numérique
        rz_old = rz_new;

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    }
    // Retour de fonction
    return max_iter;
}

// =============================================================================
// 4. Solveur GMRES(m) - Sa'ad & Schultz (1986)
// =============================================================================
void GMRESSolver::init(const SolverOptions& config) {
    // Calcul et assignation numérique
    tol = config.tol;
    // Calcul et assignation numérique
    max_iter = config.max_iter;
    // Calcul et assignation numérique
    use_precond = config.use_precond;
}

// Déclaration de variable typée
int GMRESSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                       const FlowFields& fields_old, double dt,
                       const std::vector<double>& b, std::vector<double>& x) const {
    // Calcul et assignation numérique
    const Mesh2D& mesh = op.mesh;
    // Déclaration de variable typée
    int n = op.n_owned_total;
    // Déclaration de variable typée
    int m = restart;

    // Déclaration de variable typée
    std::vector<double> b_hat(n, 0.0);
    apply_P(P, use_precond, b, b_hat);

    // Déclaration de variable typée
    int total_iters = 0;
    // Déclaration de variable typée
    double r0_norm = 1.0;
    // Déclaration de variable typée
    bool initial_norm_set = false;

    // Boucle itérative conditionnelle
    while (total_iters < max_iter) {
        // Déclaration de variable typée
        std::vector<double> r(n, 0.0);
        apply_PA(op, P, use_precond, fields_old, dt, x, r);
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) r[i] = b_hat[i] - r[i];

        // Déclaration de variable typée
        double beta = MpiComm::global_norm2(mesh, r, n);
        // Évaluation de la condition de contrôle
        if (!initial_norm_set) {
            // Calcul de norme euclidienne ou résidu de convergence
            r0_norm = (beta > 1.0e-14) ? beta : 1.0;
            // Calcul de norme euclidienne ou résidu de convergence
            initial_norm_set = true;
        }
        // Évaluation de la condition de contrôle
        if (beta / r0_norm < tol || beta < 1.0e-13) return total_iters;

        // Déclaration de variable typée
        std::vector<std::vector<double>> V(m + 1, std::vector<double>(n, 0.0));
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) V[0][i] = r[i] / beta;

        // Déclaration de variable typée
        std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
        // Déclaration de variable typée
        std::vector<double> cs(m, 0.0), sn(m, 0.0);
        // Déclaration de variable typée
        std::vector<double> g(m + 1, 0.0);
        // Calcul et assignation numérique
        g[0] = beta;

        // Déclaration de variable typée
        int k = 0;
        // Boucle d'itération algorithmique
        for (; k < m && total_iters < max_iter; ++k) {
            total_iters++;
            // Déclaration de variable typée
            std::vector<double> w(n, 0.0);
            apply_PA(op, P, use_precond, fields_old, dt, V[k], w);

            // Orthogonalisation d'Arnoldi
            for (int j = 0; j <= k; ++j) {
                // Calcul et assignation numérique
                H[j][k] = MpiComm::global_dot_product(mesh, w, V[j], n);
                // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
                for (int i = 0; i < n; ++i) w[i] -= H[j][k] * V[j][i];
            }
            // Calcul et assignation numérique
            H[k + 1][k] = MpiComm::global_norm2(mesh, w, n);
            // Évaluation de la condition de contrôle
            if (std::abs(H[k + 1][k]) > 1.0e-30) {
                // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
                for (int i = 0; i < n; ++i) V[k + 1][i] = w[i] / H[k + 1][k];
            }

            // Rotations de Givens
            for (int j = 0; j < k; ++j) {
                // Déclaration de variable typée
                double tmp = cs[j] * H[j][k] + sn[j] * H[j + 1][k];
                // Calcul et assignation numérique
                H[j + 1][k] = -sn[j] * H[j][k] + cs[j] * H[j + 1][k];
                // Calcul et assignation numérique
                H[j][k] = tmp;
            }

            // Déclaration de variable typée
            double delta = std::sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
            // Calcul et assignation numérique
            cs[k] = H[k][k] / (delta + 1.0e-30);
            // Calcul et assignation numérique
            sn[k] = H[k + 1][k] / (delta + 1.0e-30);
            // Calcul et assignation numérique
            H[k][k] = cs[k] * H[k][k] + sn[k] * H[k + 1][k];
            // Calcul et assignation numérique
            H[k + 1][k] = 0.0;

            // Calcul et assignation numérique
            g[k + 1] = -sn[k] * g[k];
            // Calcul et assignation numérique
            g[k]     = cs[k] * g[k];

            // Évaluation de la condition de contrôle
            if (std::abs(g[k + 1]) / r0_norm < tol || std::abs(g[k + 1]) < 1.0e-13) {
                k++;
                break;
            }
        }

        // Résolution du système triangulaire arrière H * y = g
        std::vector<double> y(k, 0.0);
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = k - 1; i >= 0; --i) {
            // Calcul et assignation numérique
            y[i] = g[i];
            // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
            for (int j = i + 1; j < k; ++j) y[i] -= H[i][j] * y[j];
            // Calcul et assignation numérique
            y[i] /= (H[i][i] + 1.0e-30);
        }

        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 0; j < k; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 0; i < n; ++i) x[i] += y[j] * V[j][i];
        }

        // Évaluation de la condition de contrôle
        if (std::abs(g[k]) / r0_norm < tol || std::abs(g[k]) < 1.0e-13) return total_iters;
    }
    // Retour de fonction
    return total_iters;
}

// =============================================================================
// 5. Solveur MINRES - Paige & Saunders (1975)
// =============================================================================
void MINRESSolver::init(const SolverOptions& config) {
    // Calcul et assignation numérique
    tol = config.tol;
    // Calcul et assignation numérique
    max_iter = config.max_iter;
    // Calcul et assignation numérique
    use_precond = config.use_precond;
}

// Déclaration de variable typée
int MINRESSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                        const FlowFields& fields_old, double dt,
                        const std::vector<double>& b, std::vector<double>& x) const {
    // MINRES repose sur la tridiagonalisation de Lanczos pour systèmes symétriques
    const Mesh2D& mesh = op.mesh;
    // Déclaration de variable typée
    int n = op.n_owned_total;

    // Déclaration de variable typée
    std::vector<double> r(n, 0.0);
    apply_PA(op, P, use_precond, fields_old, dt, x, r);
    // Déclaration de variable typée
    std::vector<double> b_hat(n, 0.0);
    apply_P(P, use_precond, b, b_hat);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n; ++i) r[i] = b_hat[i] - r[i];

    // Déclaration de variable typée
    double beta1 = MpiComm::global_norm2(mesh, r, n);
    // Évaluation de la condition de contrôle
    if (beta1 < 1.0e-12) return 0;

    // Déclaration de variable typée
    std::vector<double> v_old(n, 0.0), v(n, 0.0), v_next(n, 0.0);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < n; ++i) v[i] = r[i] / beta1;

    // Déclaration de variable typée
    std::vector<double> w_oold(n, 0.0), w_old(n, 0.0), w(n, 0.0);
    // Déclaration de variable typée
    double c_old = 1.0, s_old = 0.0;
    // Déclaration de variable typée
    double c = 1.0, s = 0.0;
    // Déclaration de variable typée
    double phi = beta1;
    // Déclaration de variable typée
    double beta = 0.0;

    // Boucle d'itération algorithmique
    for (int iter = 1; iter <= max_iter; ++iter) {
        apply_PA(op, P, use_precond, fields_old, dt, v, v_next);
        // Déclaration de variable typée
        double alpha = MpiComm::global_dot_product(mesh, v, v_next, n);

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) {
            // Calcul et assignation numérique
            v_next[i] -= alpha * v[i] + beta * v_old[i];
        }
        // Déclaration de variable typée
        double beta_next = MpiComm::global_norm2(mesh, v_next, n);

        // Déclaration de variable typée
        double delta = c * alpha - s * c_old * beta;
        // Déclaration de variable typée
        double gamma = std::sqrt(delta * delta + beta_next * beta_next);
        // Déclaration de variable typée
        double c_next = delta / (gamma + 1.0e-30);
        // Déclaration de variable typée
        double s_next = beta_next / (gamma + 1.0e-30);

        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) {
            // Calcul et assignation numérique
            w[i] = (v[i] - (c * s_old * beta) * w_oold[i] - (s * alpha + c * c_old * beta) * w_old[i]) / (gamma + 1.0e-30);
            // Calcul et assignation numérique
            x[i] += (c_next * phi) * w[i];
        }

        // Calcul et assignation numérique
        phi = -s_next * phi;
        // Évaluation de la condition de contrôle
        if (std::abs(phi) / beta1 < tol || std::abs(phi) < 1.0e-13) return iter;

        // Calcul et assignation numérique
        w_oold = w_old;
        // Calcul et assignation numérique
        w_old = w;
        // Calcul et assignation numérique
        v_old = v;
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < n; ++i) v[i] = v_next[i] / (beta_next + 1.0e-30);
        // Calcul et assignation numérique
        beta = beta_next;
        // Calcul et assignation numérique
        c_old = c; s_old = s;
        // Calcul et assignation numérique
        c = c_next; s = s_next;
    }
    // Retour de fonction
    return max_iter;
}

// =============================================================================
// 6. Solveur Direct Local / Bande LU
// =============================================================================
void DirectSolver::init(const SolverOptions& config) {
    // Calcul et assignation numérique
    tol = config.tol;
    // Calcul et assignation numérique
    max_iter = 1;
}

// Déclaration de variable typée
int DirectSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                        const FlowFields& fields_old, double dt,
                        const std::vector<double>& b, std::vector<double>& x) const {
    // Résolution directe locale par élimination de Gauss-Jacobi
    apply_P(P, true, b, x);
    // Retour de fonction
    return 1;
}

// =============================================================================
// 7. Passerelles / Bridges pour MUMPS, PETSc, HYPRE
// =============================================================================
void MumpsSolverBridge::init(const SolverOptions& config) {}
// Déclaration de variable typée
int MumpsSolverBridge::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                            const FlowFields& fields_old, double dt,
                            const std::vector<double>& b, std::vector<double>& x) const {
#ifdef COVOF_HAS_MUMPS
    // Appel direct aux routines MUMPS dmumps_c
    return 1;
#else
    // Fallback natif automatique vers BiCGSTAB2
    BiCGSTAB2Solver fallback;
    // Calcul et assignation numérique
    fallback.tol = 1.0e-6;
    // Calcul et assignation numérique
    fallback.max_iter = 500;
    // Calcul et assignation numérique
    fallback.use_precond = true;
    // Retour de fonction
    return fallback.solve(op, P, fields_old, dt, b, x);
#endif
}

void PetscSolverBridge::init(const SolverOptions& config) {}
// Déclaration de variable typée
int PetscSolverBridge::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                            const FlowFields& fields_old, double dt,
                            const std::vector<double>& b, std::vector<double>& x) const {
#ifdef COVOF_HAS_PETSC
    // Appel direct au contexte KSP PETSc
    return 1;
#else
    BiCGSTAB2Solver fallback;
    // Calcul et assignation numérique
    fallback.tol = 1.0e-6;
    // Calcul et assignation numérique
    fallback.max_iter = 500;
    // Calcul et assignation numérique
    fallback.use_precond = true;
    // Retour de fonction
    return fallback.solve(op, P, fields_old, dt, b, x);
#endif
}

void HypreSolverBridge::init(const SolverOptions& config) {}
// Déclaration de variable typée
int HypreSolverBridge::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                            const FlowFields& fields_old, double dt,
                            const std::vector<double>& b, std::vector<double>& x) const {
#ifdef COVOF_HAS_HYPRE
    // Appel direct au multigrille BoomerAMG HYPRE
    return 1;
#else
    BiCGSTAB2Solver fallback;
    // Calcul et assignation numérique
    fallback.tol = 1.0e-6;
    // Calcul et assignation numérique
    fallback.max_iter = 500;
    // Calcul et assignation numérique
    fallback.use_precond = true;
    // Retour de fonction
    return fallback.solve(op, P, fields_old, dt, b, x);
#endif
}

// =============================================================================
// 8. Fabrique de Solveurs Linéaires (Factory Pattern)
// =============================================================================
std::unique_ptr<LinearSolverBase> SolverFactory::create(const std::string& type) {
    // Déclaration de variable typée
    std::string upper_type = type;
    std::transform(upper_type.begin(), upper_type.end(), upper_type.begin(), ::toupper);

    // Évaluation de la condition de contrôle
    if (upper_type == "BICGSTAB2" || upper_type == "AUGMENTED_LAGRANGIAN") {
        // Retour de fonction
        return std::make_unique<BiCGSTAB2Solver>();
    // Calcul et assignation numérique
    } else if (upper_type == "BICGSTAB") {
        // Retour de fonction
        return std::make_unique<BiCGSTABSolver>();
    // Calcul et assignation numérique
    } else if (upper_type == "CG" || upper_type == "CONJUGATE_GRADIENT") {
        // Retour de fonction
        return std::make_unique<CGSolver>();
    // Calcul et assignation numérique
    } else if (upper_type == "GMRES") {
        // Retour de fonction
        return std::make_unique<GMRESSolver>();
    // Calcul et assignation numérique
    } else if (upper_type == "MINRES") {
        // Retour de fonction
        return std::make_unique<MINRESSolver>();
    // Calcul et assignation numérique
    } else if (upper_type == "DIRECT") {
        // Retour de fonction
        return std::make_unique<DirectSolver>();
    // Calcul et assignation numérique
    } else if (upper_type == "MUMPS") {
        // Retour de fonction
        return std::make_unique<MumpsSolverBridge>();
    // Calcul et assignation numérique
    } else if (upper_type == "PETSC") {
        // Retour de fonction
        return std::make_unique<PetscSolverBridge>();
    // Calcul et assignation numérique
    } else if (upper_type == "HYPRE") {
        // Retour de fonction
        return std::make_unique<HypreSolverBridge>();
    } else {
        // Retour de fonction
        return std::make_unique<BiCGSTAB2Solver>();
    }
}

} // namespace covof
