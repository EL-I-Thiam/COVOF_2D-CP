#include "covof/Preconditioner.hpp"
#include "covof/NavierStokesSolver.hpp"
#include "covof/MpiComm.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace covof {
namespace {
    template <typename IdxFn, typename SyncFn>
    void solve_csr_block_ssor(const CSRMatrix& A,
                              const Mesh2D& mesh,
                              int ni, int nj,
                              IdxFn halo_idx, SyncFn sync_fn,
                              const std::vector<double>& bnd_W,
                              const std::vector<double>& bnd_E,
                              const std::vector<double>& bnd_S,
                              const std::vector<double>& bnd_N,
                              const std::vector<double>& rhs, int rhs_offset,
                              std::vector<double>& x_flat,
                              std::vector<double>& x_halo,
                              int n_sweeps, double omega,
                              bool sync_all_interior = false) {
        const int n_rows = ni * nj;
        if (A.nrows != n_rows || n_rows == 0) return;

        std::fill(x_flat.begin(), x_flat.begin() + n_rows, 0.0);
        std::fill(x_halo.begin(), x_halo.end(), 0.0);

        const bool need_sync = (mesh.num_procs > 1) ||
                               !mesh.is_boundary_left || !mesh.is_boundary_right ||
                               !mesh.is_boundary_bottom || !mesh.is_boundary_top;
        const double w = (omega > 0.1 && omega < 1.95) ? omega : 1.0;

        auto relax_row = [&](int i, int j) {
            int row = (j - 1) * ni + (i - 1);
            double d = A.diag[row];
            if (std::abs(d) < 1.0e-30) d = 1.0;

            double rhs_eff = rhs[rhs_offset + row] - A.offdiag_dot(row, x_flat);
            if (i == 1   && bnd_W[row] != 0.0) rhs_eff += bnd_W[row] * x_halo[halo_idx(0, j)];
            if (i == ni  && bnd_E[row] != 0.0) rhs_eff += bnd_E[row] * x_halo[halo_idx(ni + 1, j)];
            if (j == 1   && bnd_S[row] != 0.0) rhs_eff += bnd_S[row] * x_halo[halo_idx(i, 0)];
            if (j == nj  && bnd_N[row] != 0.0) rhs_eff += bnd_N[row] * x_halo[halo_idx(i, nj + 1)];

            double val = (1.0 - w) * x_flat[row] + w * (rhs_eff / d);
            x_flat[row] = val;
            if (sync_all_interior || i == 1 || i == ni || j == 1 || j == nj) {
                x_halo[halo_idx(i, j)] = val;
            }
        };

        for (int sweep = 0; sweep < n_sweeps; ++sweep) {
            if (need_sync && sweep > 0) sync_fn(mesh, x_halo);
            for (int j = 1; j <= nj; ++j) {
                for (int i = 1; i <= ni; ++i) {
                    relax_row(i, j);
                }
            }
            if (need_sync) sync_fn(mesh, x_halo);
            for (int j = nj; j >= 1; --j) {
                for (int i = ni; i >= 1; --i) {
                    relax_row(i, j);
                }
            }
        }
    }
} // namespace
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
    if (!ns_solver) return;

    const Mesh2D& mesh = ns_solver->mesh;
    const int nu = ns_solver->n_owned_u;
    const int nv = ns_solver->n_owned_v;
    const int np = ns_solver->n_owned_p;
    const int n  = ns_solver->n_owned_total;
    const int nx = mesh.nx_local;
    const int ny = mesh.ny_local;
    const int u_nx = nx + (mesh.is_boundary_right ? 1 : 0);
    const int v_ny = ny + (mesh.is_boundary_top ? 1 : 0);

    r.assign(n, 0.0);
    double w = (omega > 0.1 && omega < 1.95) ? omega : 1.0;

    if (ns_solver->Fuu.nrows == nu && ns_solver->Fvv.nrows == nv) {
        std::vector<double> r_u(nu, 0.0), r_v(nv, 0.0);
        std::vector<double> ru_halo(static_cast<size_t>((mesh.nx_ghost + 1) * mesh.ny_ghost), 0.0);
        std::vector<double> rv_halo(static_cast<size_t>(mesh.nx_ghost * (mesh.ny_ghost + 1)), 0.0);

        solve_csr_block_ssor(
            ns_solver->Fuu, mesh, u_nx, ny,
            [&](int i, int j) { return mesh.u_idx(i, j); }, MpiComm::sync_u,
            ns_solver->u_bnd_W, ns_solver->u_bnd_E, ns_solver->u_bnd_S, ns_solver->u_bnd_N,
            z, 0, r_u, ru_halo, 2, w);

        solve_csr_block_ssor(
            ns_solver->Fvv, mesh, nx, v_ny,
            [&](int i, int j) { return mesh.v_idx(i, j); }, MpiComm::sync_v,
            ns_solver->v_bnd_W, ns_solver->v_bnd_E, ns_solver->v_bnd_S, ns_solver->v_bnd_N,
            z, nu, r_v, rv_halo, 2, w);

        std::copy(r_u.begin(), r_u.end(), r.begin());
        std::copy(r_v.begin(), r_v.end(), r.begin() + nu);

        if (ns_solver->is_monolithic() && ns_solver->SchurLaplace.nrows == np) {
            std::vector<double> r_p(np, 0.0);
            std::vector<double> rp_halo(static_cast<size_t>(mesh.nx_ghost * mesh.ny_ghost), 0.0);
            solve_csr_block_ssor(
                ns_solver->SchurLaplace, mesh, nx, ny,
                [&](int i, int j) { return mesh.cell_idx(i, j); }, MpiComm::sync_cell_scalar,
                ns_solver->schur_bnd_W, ns_solver->schur_bnd_E, ns_solver->schur_bnd_S, ns_solver->schur_bnd_N,
                z, nu + nv, r_p, rp_halo, 2, w);
            std::copy(r_p.begin(), r_p.end(), r.begin() + nu + nv);
        }
        return;
    }

    // Repli diagonal si les matrices CSR ne sont pas assemblees
    for (int i = 0; i < nu; ++i) {
        double d = (ns_solver->diag_u.size() > (size_t)i && std::abs(ns_solver->diag_u[i]) > 1.0e-15)
                   ? ns_solver->diag_u[i] : 1.0;
        r[i] = z[i] / d;
    }
    for (int i = 0; i < nv; ++i) {
        double d = (ns_solver->diag_v.size() > (size_t)i && std::abs(ns_solver->diag_v[i]) > 1.0e-15)
                   ? ns_solver->diag_v[i] : 1.0;
        r[nu + i] = z[nu + i] / d;
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

    const Mesh2D& mesh = ns_solver->mesh;
    nx = mesh.nx_local;
    ny = mesh.ny_local;
    u_nx = nx + (mesh.is_boundary_right ? 1 : 0);
    v_ny = ny + (mesh.is_boundary_top ? 1 : 0);
    dx = mesh.dx;
    dy = mesh.dy;
    dvol = dx * dy;
    r_al = ns_solver->r_al;

    const int nu = ns_solver->n_owned_u;
    const int nv = ns_solver->n_owned_v;
    const int np = ns_solver->n_owned_p;

    if (r_p.size() != static_cast<size_t>(np)) r_p.assign(np, 0.0);
    if (rhs_u.size() != static_cast<size_t>(nu)) rhs_u.assign(nu, 0.0);
    if (rhs_v.size() != static_cast<size_t>(nv)) rhs_v.assign(nv, 0.0);
    if (r_u.size() != static_cast<size_t>(nu)) r_u.assign(nu, 0.0);
    if (r_v.size() != static_cast<size_t>(nv)) r_v.assign(nv, 0.0);

    const size_t p_halo_sz = static_cast<size_t>(mesh.nx_ghost * mesh.ny_ghost);
    const size_t u_halo_sz = static_cast<size_t>((mesh.nx_ghost + 1) * mesh.ny_ghost);
    const size_t v_halo_sz = static_cast<size_t>(mesh.nx_ghost * (mesh.ny_ghost + 1));
    if (rp_halo.size() != p_halo_sz) rp_halo.assign(p_halo_sz, 0.0);
    if (ru_halo.size() != u_halo_sz) ru_halo.assign(u_halo_sz, 0.0);
    if (rv_halo.size() != v_halo_sz) rv_halo.assign(v_halo_sz, 0.0);
}

void BlockTriangularPreconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    if (!ns_solver) return;

    const Mesh2D& mesh = ns_solver->mesh;
    const int nu = ns_solver->n_owned_u;
    const int nv = ns_solver->n_owned_v;
    const int np = ns_solver->n_owned_p;
    const int n  = ns_solver->n_owned_total;

    r.assign(n, 0.0);

    if (ns_solver->is_monolithic()) {
        const int p_offset = nu + nv;

        // 1. Résolution parallèle MPI du complément de Schur S * r_p = z_p via SchurLaplace (CSR)
        solve_csr_block_ssor(
            ns_solver->SchurLaplace, mesh, nx, ny,
            [&](int i, int j) { return mesh.cell_idx(i, j); }, MpiComm::sync_cell_scalar,
            ns_solver->schur_bnd_W, ns_solver->schur_bnd_E,
            ns_solver->schur_bnd_S, ns_solver->schur_bnd_N,
            z, p_offset, r_p, rp_halo, n_schur_sweeps, ssor_omega, true);

        // Élimination GLOBALE du mode constant de pression + synchronisation des halos pour G * r_p
        if (!ns_solver->compressible) {
            double local_sum = 0.0;
            for (int c = 0; c < np; ++c) local_sum += r_p[c];
            double global_sum_rp = MpiComm::global_sum(mesh, local_sum);
            double mean_rp = global_sum_rp / static_cast<double>(mesh.nx_global * mesh.ny_global);
            for (int j = 1; j <= ny; ++j) {
                for (int i = 1; i <= nx; ++i) {
                    int c = (j - 1) * nx + (i - 1);
                    r_p[c] -= mean_rp;
                    rp_halo[mesh.cell_idx(i, j)] = r_p[c];
                }
            }
        }
        if (mesh.num_procs > 1 || !mesh.is_boundary_left || !mesh.is_boundary_right ||
            !mesh.is_boundary_bottom || !mesh.is_boundary_top) {
            MpiComm::sync_cell_scalar(mesh, rp_halo);
        }

        // 2. Second membre pour vitesse : rhs_u = z_u - G_u * r_p, rhs_v = z_v - G_v * r_p
        rhs_u.assign(nu, 0.0);
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= u_nx; ++i) {
                int dof = (j - 1) * u_nx + (i - 1);
                bool is_wall = (i == 1 && mesh.is_boundary_left) ||
                               (i == u_nx && mesh.is_boundary_right) ||
                               (ns_solver->is_u_face_in_obstacle(i, j) && ns_solver->get_u_obstacle_perm(i, j) <= 1.0e-12);
                if (is_wall) {
                    rhs_u[dof] = z[dof];
                } else {
                    double grad_p = (rp_halo[mesh.cell_idx(i, j)] - rp_halo[mesh.cell_idx(i - 1, j)]) * dy;
                    rhs_u[dof] = z[dof] - grad_p;
                }
            }
        }

        rhs_v.assign(nv, 0.0);
        for (int j = 1; j <= v_ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int dof = (j - 1) * nx + (i - 1);
                bool is_wall = (j == 1 && mesh.is_boundary_bottom &&
                                (ns_solver->bc.bottom.type == BCType::DIRICHLET || ns_solver->bc.bottom.type == BCType::FREE_SLIP)) ||
                               (j == v_ny && mesh.is_boundary_top &&
                                (ns_solver->bc.top.type == BCType::DIRICHLET || ns_solver->bc.top.type == BCType::FREE_SLIP)) ||
                               (ns_solver->is_v_face_in_obstacle(i, j) && ns_solver->get_v_obstacle_perm(i, j) <= 1.0e-12);
                if (is_wall) {
                    rhs_v[dof] = z[nu + dof];
                } else {
                    double grad_p = (rp_halo[mesh.cell_idx(i, j)] - rp_halo[mesh.cell_idx(i, j - 1)]) * dx;
                    rhs_v[dof] = z[nu + dof] - grad_p;
                }
            }
        }

        // 3. Résolution parallèle MPI de Fu * r_u = rhs_u et Fv * r_v = rhs_v via Fuu / Fvv (CSR)
        solve_csr_block_ssor(
            ns_solver->Fuu, mesh, u_nx, ny,
            [&](int i, int j) { return mesh.u_idx(i, j); }, MpiComm::sync_u,
            ns_solver->u_bnd_W, ns_solver->u_bnd_E,
            ns_solver->u_bnd_S, ns_solver->u_bnd_N,
            rhs_u, 0, r_u, ru_halo, n_sweeps, 1.0);

        solve_csr_block_ssor(
            ns_solver->Fvv, mesh, nx, v_ny,
            [&](int i, int j) { return mesh.v_idx(i, j); }, MpiComm::sync_v,
            ns_solver->v_bnd_W, ns_solver->v_bnd_E,
            ns_solver->v_bnd_S, ns_solver->v_bnd_N,
            rhs_v, 0, r_v, rv_halo, n_sweeps, 1.0);

        // 4. Assemblage dans r
        for (int i = 0; i < nu; ++i) r[i] = r_u[i];
        for (int i = 0; i < nv; ++i) r[nu + i] = r_v[i];
        for (int i = 0; i < np; ++i) r[p_offset + i] = r_p[i];

    } else {
        // Mode Lagrangien augmenté : préconditionnement parallèle sur Fuu et Fvv (CSR)
        if (ns_solver->Fuu.nrows == nu && ns_solver->Fvv.nrows == nv) {
            solve_csr_block_ssor(
                ns_solver->Fuu, mesh, u_nx, ny,
                [&](int i, int j) { return mesh.u_idx(i, j); }, MpiComm::sync_u,
                ns_solver->u_bnd_W, ns_solver->u_bnd_E,
                ns_solver->u_bnd_S, ns_solver->u_bnd_N,
                z, 0, r_u, ru_halo, 2, 1.0);

            solve_csr_block_ssor(
                ns_solver->Fvv, mesh, nx, v_ny,
                [&](int i, int j) { return mesh.v_idx(i, j); }, MpiComm::sync_v,
                ns_solver->v_bnd_W, ns_solver->v_bnd_E,
                ns_solver->v_bnd_S, ns_solver->v_bnd_N,
                z, nu, r_v, rv_halo, 2, 1.0);

            for (int i = 0; i < nu; ++i) r[i] = r_u[i];
            for (int i = 0; i < nv; ++i) r[nu + i] = r_v[i];
        } else {
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
    } else if (upper_type == "BLOCK_TRIANGULAR" || upper_type == "SCHUR" ||
               upper_type == "BT_SBP" || upper_type == "BT-SBP" ||
               upper_type == "BLOCK" || upper_type == "BLOCK_PRECONDITIONER" || upper_type == "BLOCK_PRECOND") {
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
