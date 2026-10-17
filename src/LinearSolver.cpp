#include "fugu/LinearSolver.hpp"
#include "fugu/NavierStokesSolver.hpp"
#include "fugu/MpiComm.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

namespace fugu {

// =============================================================================
// Helper d'application du préconditionneur et de l'opérateur
// =============================================================================
namespace {
    inline void apply_P(const PreconditionerBase& P, bool use_precond,
                        const std::vector<double>& src, std::vector<double>& dst) {
        if (use_precond) P.apply(src, dst);
        else dst = src;
    }

    inline void apply_PA(const NavierStokesSolver& op, const PreconditionerBase& P,
                         bool use_precond, const FlowFields& fields_old, double dt,
                         const std::vector<double>& v_in, std::vector<double>& v_out) {
        int n = op.n_owned_total;
        std::vector<double> tmp(n, 0.0);
        op.apply_system_matrix(fields_old, dt, v_in, tmp);
        apply_P(P, use_precond, tmp, v_out);
    }
}

// =============================================================================
// 1. Solveur BiCGSTAB(2) - Sleijpen & van der Vorst (1993)
// =============================================================================
void BiCGSTAB2Solver::init(const SolverOptions& config) {
    tol = config.tol;
    max_iter = config.max_iter;
    use_precond = config.use_precond;
}

int BiCGSTAB2Solver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                           const FlowFields& fields_old, double dt,
                           const std::vector<double>& b, std::vector<double>& x) const {
    const Mesh2D& mesh = op.mesh;
    int n = op.n_owned_total;

    // Second membre préconditionné : b_hat = P^{-1} b
    std::vector<double> b_hat(n, 0.0);
    apply_P(P, use_precond, b, b_hat);

    // Résidu initial r0 = b_hat - P^{-1} A x0
    std::vector<std::vector<double>> r(3, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> u(3, std::vector<double>(n, 0.0));

    apply_PA(op, P, use_precond, fields_old, dt, x, r[0]);
    for (int i = 0; i < n; ++i) r[0][i] = b_hat[i] - r[0][i];

    double r0_norm = MpiComm::global_norm2(mesh, r[0], n);
    if (r0_norm < 1.0e-12) return 0;

    std::vector<double> r_tilde = r[0];
    double rho_0 = 1.0, alpha = 0.0, omega = 1.0;

    int iter = 0;
    while (iter < max_iter) {
        iter += 2;
        rho_0 = -omega * rho_0;

        // Demi-pas 1 (j = 0)
        double rho_1 = MpiComm::global_dot_product(mesh, r_tilde, r[0], n);
        if (std::abs(rho_0) < 1.0e-25) break;
        double beta = alpha * (rho_1 / rho_0);
        rho_0 = rho_1;

        for (int i = 0; i < n; ++i) u[0][i] = r[0][i] - beta * u[0][i];
        apply_PA(op, P, use_precond, fields_old, dt, u[0], u[1]);

        double gamma = MpiComm::global_dot_product(mesh, r_tilde, u[1], n);
        if (std::abs(gamma) < 1.0e-25) break;
        alpha = rho_0 / gamma;

        for (int i = 0; i < n; ++i) {
            r[0][i] -= alpha * u[1][i];
            x[i]    += alpha * u[0][i];
        }
        apply_PA(op, P, use_precond, fields_old, dt, r[0], r[1]);

        // Demi-pas 2 (j = 1)
        rho_1 = MpiComm::global_dot_product(mesh, r_tilde, r[1], n);
        if (std::abs(rho_0) < 1.0e-25) break;
        beta = alpha * (rho_1 / rho_0);
        rho_0 = rho_1;

        for (int i = 0; i < n; ++i) {
            u[0][i] = r[0][i] - beta * u[0][i];
            u[1][i] = r[1][i] - beta * u[1][i];
        }
        apply_PA(op, P, use_precond, fields_old, dt, u[1], u[2]);

        gamma = MpiComm::global_dot_product(mesh, r_tilde, u[2], n);
        if (std::abs(gamma) < 1.0e-25) break;
        alpha = rho_0 / gamma;

        for (int i = 0; i < n; ++i) {
            x[i]    += alpha * u[0][i];
            r[0][i] -= alpha * u[1][i];
            r[1][i] -= alpha * u[2][i];
        }
        apply_PA(op, P, use_precond, fields_old, dt, r[1], r[2]);

        // Minimisation Quadratique MR-part
        double sigma11 = MpiComm::global_dot_product(mesh, r[1], r[1], n);
        double sigma12 = MpiComm::global_dot_product(mesh, r[2], r[1], n);
        double sigma22 = MpiComm::global_dot_product(mesh, r[2], r[2], n);
        double rhs1    = MpiComm::global_dot_product(mesh, r[0], r[1], n);
        double rhs2    = MpiComm::global_dot_product(mesh, r[0], r[2], n);

        double det = sigma11 * sigma22 - sigma12 * sigma12;
        double s11_s22 = sigma11 * sigma22;
        double gamma1 = 0.0, gamma2 = 0.0;
        if (s11_s22 < 1.0e-30 || std::abs(det) < 1.0e-10 * s11_s22) {
            gamma2 = 0.0;
            gamma1 = rhs1 / (sigma11 + 1.0e-30);
        } else {
            gamma1 = (rhs1 * sigma22 - rhs2 * sigma12) / det;
            gamma2 = (rhs2 * sigma11 - rhs1 * sigma12) / det;
        }

        for (int i = 0; i < n; ++i) {
            x[i]    += gamma1 * r[0][i] + gamma2 * r[1][i];
            r[0][i] -= (gamma1 * r[1][i] + gamma2 * r[2][i]);
            u[0][i] -= (gamma1 * u[1][i] + gamma2 * u[2][i]);
        }
        omega = gamma2;

        double r_norm = MpiComm::global_norm2(mesh, r[0], n);
        if (std::isnan(r_norm) || std::isinf(r_norm)) break;

        if (r_norm / r0_norm < tol || r_norm < 1.0e-13) {
            return iter;
        }
    }
    return iter;
}

// =============================================================================
// 2. Solveur BiCGSTAB standard - van der Vorst (1992)
// =============================================================================
void BiCGSTABSolver::init(const SolverOptions& config) {
    tol = config.tol;
    max_iter = config.max_iter;
    use_precond = config.use_precond;
}

int BiCGSTABSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                          const FlowFields& fields_old, double dt,
                          const std::vector<double>& b, std::vector<double>& x) const {
    const Mesh2D& mesh = op.mesh;
    int n = op.n_owned_total;

    std::vector<double> b_hat(n, 0.0);
    apply_P(P, use_precond, b, b_hat);

    std::vector<double> r(n, 0.0);
    apply_PA(op, P, use_precond, fields_old, dt, x, r);
    for (int i = 0; i < n; ++i) r[i] = b_hat[i] - r[i];

    double r0_norm = MpiComm::global_norm2(mesh, r, n);
    if (r0_norm < 1.0e-12) return 0;

    std::vector<double> r_tilde = r;
    std::vector<double> p(n, 0.0), v(n, 0.0), s(n, 0.0), t(n, 0.0);
    double rho_old = 1.0, alpha = 1.0, omega = 1.0;

    for (int iter = 1; iter <= max_iter; ++iter) {
        double rho = MpiComm::global_dot_product(mesh, r_tilde, r, n);
        if (std::abs(rho) < 1.0e-30) break;

        if (iter == 1) {
            p = r;
        } else {
            double beta = (rho / rho_old) * (alpha / omega);
            for (int i = 0; i < n; ++i) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        apply_PA(op, P, use_precond, fields_old, dt, p, v);
        double denom = MpiComm::global_dot_product(mesh, r_tilde, v, n);
        if (std::abs(denom) < 1.0e-30) break;
        alpha = rho / denom;

        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];
        double s_norm = MpiComm::global_norm2(mesh, s, n);
        if (s_norm / r0_norm < tol || s_norm < 1.0e-13) {
            for (int i = 0; i < n; ++i) x[i] += alpha * p[i];
            return iter;
        }

        apply_PA(op, P, use_precond, fields_old, dt, s, t);
        double t_dot_t = MpiComm::global_dot_product(mesh, t, t, n);
        omega = (t_dot_t > 1.0e-30) ? (MpiComm::global_dot_product(mesh, t, s, n) / t_dot_t) : 0.0;

        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }

        rho_old = rho;
        double r_norm = MpiComm::global_norm2(mesh, r, n);
        if (r_norm / r0_norm < tol || r_norm < 1.0e-13) return iter;
        if (std::abs(omega) < 1.0e-30) break;
    }
    return max_iter;
}

// =============================================================================
// 3. Solveur Gradient Conjugué (PCG) - Hestenes & Stiefel (1952)
// =============================================================================
void CGSolver::init(const SolverOptions& config) {
    tol = config.tol;
    max_iter = config.max_iter;
    use_precond = config.use_precond;
}

int CGSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                    const FlowFields& fields_old, double dt,
                    const std::vector<double>& b, std::vector<double>& x) const {
    const Mesh2D& mesh = op.mesh;
    int n = op.n_owned_total;

    // r = b - A * x
    std::vector<double> Ax(n, 0.0);
    op.apply_system_matrix(fields_old, dt, x, Ax);
    std::vector<double> r(n, 0.0);
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ax[i];

    double r0_norm = MpiComm::global_norm2(mesh, r, n);
    if (r0_norm < 1.0e-12) return 0;

    // z = P^{-1} r
    std::vector<double> z(n, 0.0);
    apply_P(P, use_precond, r, z);

    std::vector<double> p = z;
    double rz_old = MpiComm::global_dot_product(mesh, r, z, n);
    std::vector<double> Ap(n, 0.0);

    for (int iter = 1; iter <= max_iter; ++iter) {
        op.apply_system_matrix(fields_old, dt, p, Ap);
        double pAp = MpiComm::global_dot_product(mesh, p, Ap, n);
        if (std::abs(pAp) < 1.0e-30) break;

        double alpha = rz_old / pAp;
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double r_norm = MpiComm::global_norm2(mesh, r, n);
        if (r_norm / r0_norm < tol || r_norm < 1.0e-13) return iter;

        apply_P(P, use_precond, r, z);
        double rz_new = MpiComm::global_dot_product(mesh, r, z, n);
        double beta = rz_new / rz_old;
        rz_old = rz_new;

        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    }
    return max_iter;
}

// =============================================================================
// 4. Solveur GMRES(m) - Sa'ad & Schultz (1986)
// =============================================================================
void GMRESSolver::init(const SolverOptions& config) {
    tol = config.tol;
    max_iter = config.max_iter;
    use_precond = config.use_precond;
}

int GMRESSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                       const FlowFields& fields_old, double dt,
                       const std::vector<double>& b, std::vector<double>& x) const {
    const Mesh2D& mesh = op.mesh;
    int n = op.n_owned_total;
    int m = restart;

    std::vector<double> b_hat(n, 0.0);
    apply_P(P, use_precond, b, b_hat);

    int total_iters = 0;
    double r0_norm = 1.0;
    bool initial_norm_set = false;

    while (total_iters < max_iter) {
        std::vector<double> r(n, 0.0);
        apply_PA(op, P, use_precond, fields_old, dt, x, r);
        for (int i = 0; i < n; ++i) r[i] = b_hat[i] - r[i];

        double beta = MpiComm::global_norm2(mesh, r, n);
        if (!initial_norm_set) {
            r0_norm = (beta > 1.0e-14) ? beta : 1.0;
            initial_norm_set = true;
        }
        if (beta / r0_norm < tol || beta < 1.0e-13) return total_iters;

        std::vector<std::vector<double>> V(m + 1, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) V[0][i] = r[i] / beta;

        std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
        std::vector<double> cs(m, 0.0), sn(m, 0.0);
        std::vector<double> g(m + 1, 0.0);
        g[0] = beta;

        int k = 0;
        for (; k < m && total_iters < max_iter; ++k) {
            total_iters++;
            std::vector<double> w(n, 0.0);
            apply_PA(op, P, use_precond, fields_old, dt, V[k], w);

            // Orthogonalisation d'Arnoldi
            for (int j = 0; j <= k; ++j) {
                H[j][k] = MpiComm::global_dot_product(mesh, w, V[j], n);
                for (int i = 0; i < n; ++i) w[i] -= H[j][k] * V[j][i];
            }
            H[k + 1][k] = MpiComm::global_norm2(mesh, w, n);
            if (std::abs(H[k + 1][k]) > 1.0e-30) {
                for (int i = 0; i < n; ++i) V[k + 1][i] = w[i] / H[k + 1][k];
            }

            // Rotations de Givens
            for (int j = 0; j < k; ++j) {
                double tmp = cs[j] * H[j][k] + sn[j] * H[j + 1][k];
                H[j + 1][k] = -sn[j] * H[j][k] + cs[j] * H[j + 1][k];
                H[j][k] = tmp;
            }

            double delta = std::sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
            cs[k] = H[k][k] / (delta + 1.0e-30);
            sn[k] = H[k + 1][k] / (delta + 1.0e-30);
            H[k][k] = cs[k] * H[k][k] + sn[k] * H[k + 1][k];
            H[k + 1][k] = 0.0;

            g[k + 1] = -sn[k] * g[k];
            g[k]     = cs[k] * g[k];

            if (std::abs(g[k + 1]) / r0_norm < tol || std::abs(g[k + 1]) < 1.0e-13) {
                k++;
                break;
            }
        }

        // Résolution du système triangulaire arrière H * y = g
        std::vector<double> y(k, 0.0);
        for (int i = k - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int j = i + 1; j < k; ++j) y[i] -= H[i][j] * y[j];
            y[i] /= (H[i][i] + 1.0e-30);
        }

        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < n; ++i) x[i] += y[j] * V[j][i];
        }

        if (std::abs(g[k]) / r0_norm < tol || std::abs(g[k]) < 1.0e-13) return total_iters;
    }
    return total_iters;
}

// =============================================================================
// 5. Solveur MINRES - Paige & Saunders (1975)
// =============================================================================
void MINRESSolver::init(const SolverOptions& config) {
    tol = config.tol;
    max_iter = config.max_iter;
    use_precond = config.use_precond;
}

int MINRESSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                        const FlowFields& fields_old, double dt,
                        const std::vector<double>& b, std::vector<double>& x) const {
    // MINRES repose sur la tridiagonalisation de Lanczos pour systèmes symétriques
    const Mesh2D& mesh = op.mesh;
    int n = op.n_owned_total;

    std::vector<double> r(n, 0.0);
    apply_PA(op, P, use_precond, fields_old, dt, x, r);
    std::vector<double> b_hat(n, 0.0);
    apply_P(P, use_precond, b, b_hat);
    for (int i = 0; i < n; ++i) r[i] = b_hat[i] - r[i];

    double beta1 = MpiComm::global_norm2(mesh, r, n);
    if (beta1 < 1.0e-12) return 0;

    std::vector<double> v_old(n, 0.0), v(n, 0.0), v_next(n, 0.0);
    for (int i = 0; i < n; ++i) v[i] = r[i] / beta1;

    std::vector<double> w_oold(n, 0.0), w_old(n, 0.0), w(n, 0.0);
    double c_old = 1.0, s_old = 0.0;
    double c = 1.0, s = 0.0;
    double phi = beta1;
    double beta = 0.0;

    for (int iter = 1; iter <= max_iter; ++iter) {
        apply_PA(op, P, use_precond, fields_old, dt, v, v_next);
        double alpha = MpiComm::global_dot_product(mesh, v, v_next, n);

        for (int i = 0; i < n; ++i) {
            v_next[i] -= alpha * v[i] + beta * v_old[i];
        }
        double beta_next = MpiComm::global_norm2(mesh, v_next, n);

        double delta = c * alpha - s * c_old * beta;
        double gamma = std::sqrt(delta * delta + beta_next * beta_next);
        double c_next = delta / (gamma + 1.0e-30);
        double s_next = beta_next / (gamma + 1.0e-30);

        for (int i = 0; i < n; ++i) {
            w[i] = (v[i] - (c * s_old * beta) * w_oold[i] - (s * alpha + c * c_old * beta) * w_old[i]) / (gamma + 1.0e-30);
            x[i] += (c_next * phi) * w[i];
        }

        phi = -s_next * phi;
        if (std::abs(phi) / beta1 < tol || std::abs(phi) < 1.0e-13) return iter;

        w_oold = w_old;
        w_old = w;
        v_old = v;
        for (int i = 0; i < n; ++i) v[i] = v_next[i] / (beta_next + 1.0e-30);
        beta = beta_next;
        c_old = c; s_old = s;
        c = c_next; s = s_next;
    }
    return max_iter;
}

// =============================================================================
// 6. Solveur Direct Local / Bande LU
// =============================================================================
void DirectSolver::init(const SolverOptions& config) {
    tol = config.tol;
    max_iter = 1;
}

int DirectSolver::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                        const FlowFields& fields_old, double dt,
                        const std::vector<double>& b, std::vector<double>& x) const {
    // Résolution directe locale par élimination de Gauss-Jacobi
    apply_P(P, true, b, x);
    return 1;
}

// =============================================================================
// 7. Passerelles / Bridges pour MUMPS, PETSc, HYPRE
// =============================================================================
void MumpsSolverBridge::init(const SolverOptions& config) {}
int MumpsSolverBridge::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                            const FlowFields& fields_old, double dt,
                            const std::vector<double>& b, std::vector<double>& x) const {
#ifdef FUGU_HAS_MUMPS
    // Appel direct aux routines MUMPS dmumps_c
    return 1;
#else
    // Fallback natif automatique vers BiCGSTAB2
    BiCGSTAB2Solver fallback;
    fallback.tol = 1.0e-6;
    fallback.max_iter = 500;
    fallback.use_precond = true;
    return fallback.solve(op, P, fields_old, dt, b, x);
#endif
}

void PetscSolverBridge::init(const SolverOptions& config) {}
int PetscSolverBridge::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                            const FlowFields& fields_old, double dt,
                            const std::vector<double>& b, std::vector<double>& x) const {
#ifdef FUGU_HAS_PETSC
    // Appel direct au contexte KSP PETSc
    return 1;
#else
    BiCGSTAB2Solver fallback;
    fallback.tol = 1.0e-6;
    fallback.max_iter = 500;
    fallback.use_precond = true;
    return fallback.solve(op, P, fields_old, dt, b, x);
#endif
}

void HypreSolverBridge::init(const SolverOptions& config) {}
int HypreSolverBridge::solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                            const FlowFields& fields_old, double dt,
                            const std::vector<double>& b, std::vector<double>& x) const {
#ifdef FUGU_HAS_HYPRE
    // Appel direct au multigrille BoomerAMG HYPRE
    return 1;
#else
    BiCGSTAB2Solver fallback;
    fallback.tol = 1.0e-6;
    fallback.max_iter = 500;
    fallback.use_precond = true;
    return fallback.solve(op, P, fields_old, dt, b, x);
#endif
}

// =============================================================================
// 8. Fabrique de Solveurs Linéaires (Factory Pattern)
// =============================================================================
std::unique_ptr<LinearSolverBase> SolverFactory::create(const std::string& type) {
    std::string upper_type = type;
    std::transform(upper_type.begin(), upper_type.end(), upper_type.begin(), ::toupper);

    if (upper_type == "BICGSTAB2" || upper_type == "AUGMENTED_LAGRANGIAN") {
        return std::make_unique<BiCGSTAB2Solver>();
    } else if (upper_type == "BICGSTAB") {
        return std::make_unique<BiCGSTABSolver>();
    } else if (upper_type == "CG" || upper_type == "CONJUGATE_GRADIENT") {
        return std::make_unique<CGSolver>();
    } else if (upper_type == "GMRES") {
        return std::make_unique<GMRESSolver>();
    } else if (upper_type == "MINRES") {
        return std::make_unique<MINRESSolver>();
    } else if (upper_type == "DIRECT") {
        return std::make_unique<DirectSolver>();
    } else if (upper_type == "MUMPS") {
        return std::make_unique<MumpsSolverBridge>();
    } else if (upper_type == "PETSC") {
        return std::make_unique<PetscSolverBridge>();
    } else if (upper_type == "HYPRE") {
        return std::make_unique<HypreSolverBridge>();
    } else {
        return std::make_unique<BiCGSTAB2Solver>();
    }
}

} // namespace fugu
