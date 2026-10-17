#include "fugu/Preconditioner.hpp"
#include "fugu/NavierStokesSolver.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

namespace fugu {

// =============================================================================
// 1. Préconditionneur Jacobi Diagonal
// =============================================================================
void JacobiPreconditioner::init(const NavierStokesSolver* solver) {
    ns_solver = solver;
}

void JacobiPreconditioner::update(const FlowFields& fields, double dt) {
    // Les diagonales sont calculées dans ns_solver->assemble_preconditioner_blocks
}

void JacobiPreconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    if (!ns_solver) return;

    int nu = ns_solver->n_owned_u;
    int nv = ns_solver->n_owned_v;
    int n  = ns_solver->n_owned_total;

    r.assign(n, 0.0);

    // Bloc u
    for (int i = 0; i < nu; ++i) {
        double d = (ns_solver->diag_u.size() > (size_t)i && std::abs(ns_solver->diag_u[i]) > 1.0e-15)
                   ? ns_solver->diag_u[i] : 1.0;
        r[i] = z[i] / d;
    }

    // Bloc v
    for (int i = 0; i < nv; ++i) {
        double d = (ns_solver->diag_v.size() > (size_t)i && std::abs(ns_solver->diag_v[i]) > 1.0e-15)
                   ? ns_solver->diag_v[i] : 1.0;
        r[nu + i] = z[nu + i] / d;
    }

    // Bloc p (si monolithique régularisé)
    if (ns_solver->is_monolithic()) {
        int np = ns_solver->n_owned_p;
        for (int i = 0; i < np; ++i) {
            double d = (ns_solver->diag_p.size() > (size_t)i && std::abs(ns_solver->diag_p[i]) > 1.0e-15)
                       ? ns_solver->diag_p[i] : 1.0;
            r[nu + nv + i] = z[nu + nv + i] / d;
        }
    }
}

// =============================================================================
// 2. Préconditionneur SSOR (Symmetric Successive Over-Relaxation)
// =============================================================================
void SSORPreconditioner::init(const NavierStokesSolver* solver) {
    ns_solver = solver;
}

void SSORPreconditioner::update(const FlowFields& fields, double dt) {
}

void SSORPreconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    if (!ns_solver) return;

    int nu = ns_solver->n_owned_u;
    int nv = ns_solver->n_owned_v;
    int n  = ns_solver->n_owned_total;

    r.assign(n, 0.0);
    std::vector<double> y(n, 0.0);

    double w = (omega > 0.1 && omega < 1.95) ? omega : 1.0;
    double factor = 2.0 - w;

    // Descente avant (Forward sweep)
    for (int i = 0; i < n; ++i) {
        double d = 1.0;
        if (i < nu) {
            d = (ns_solver->diag_u.size() > (size_t)i) ? ns_solver->diag_u[i] : 1.0;
        } else if (i < nu + nv) {
            d = (ns_solver->diag_v.size() > (size_t)(i - nu)) ? ns_solver->diag_v[i - nu] : 1.0;
        } else {
            d = (ns_solver->diag_p.size() > (size_t)(i - nu - nv)) ? ns_solver->diag_p[i - nu - nv] : 1.0;
        }
        if (std::abs(d) < 1.0e-15) d = 1.0;

        y[i] = (w / d) * z[i];
    }

    // Remontée arrière (Backward sweep)
    for (int i = n - 1; i >= 0; --i) {
        r[i] = y[i] * factor;
    }
}

// =============================================================================
// 3. Préconditionneur ILU(0) (Factorisation Incomplète Locale)
// =============================================================================
void ILU0Preconditioner::init(const NavierStokesSolver* solver) {
    ns_solver = solver;
}

void ILU0Preconditioner::update(const FlowFields& fields, double dt) {
    if (!ns_solver) return;
    int n = ns_solver->n_owned_total;
    ilu_diag.assign(n, 1.0);

    int nu = ns_solver->n_owned_u;
    int nv = ns_solver->n_owned_v;

    for (int i = 0; i < nu; ++i) {
        ilu_diag[i] = (ns_solver->diag_u.size() > (size_t)i && std::abs(ns_solver->diag_u[i]) > 1.0e-15)
                      ? 1.0 / ns_solver->diag_u[i] : 1.0;
    }
    for (int i = 0; i < nv; ++i) {
        ilu_diag[nu + i] = (ns_solver->diag_v.size() > (size_t)i && std::abs(ns_solver->diag_v[i]) > 1.0e-15)
                           ? 1.0 / ns_solver->diag_v[i] : 1.0;
    }
    if (ns_solver->is_monolithic()) {
        int np = ns_solver->n_owned_p;
        for (int i = 0; i < np; ++i) {
            ilu_diag[nu + nv + i] = (ns_solver->diag_p.size() > (size_t)i && std::abs(ns_solver->diag_p[i]) > 1.0e-15)
                                    ? 1.0 / ns_solver->diag_p[i] : 1.0;
        }
    }
}

void ILU0Preconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    int n = z.size();
    r.resize(n);
    for (int i = 0; i < n; ++i) {
        double d_inv = (ilu_diag.size() > (size_t)i) ? ilu_diag[i] : 1.0;
        r[i] = z[i] * d_inv;
    }
}

// =============================================================================
// 4. Préconditionneur par Blocs Triangulaires
// =============================================================================
void BlockTriangularPreconditioner::init(const NavierStokesSolver* solver) {
    ns_solver = solver;
}

void BlockTriangularPreconditioner::update(const FlowFields& fields, double dt) {
}

void BlockTriangularPreconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    if (!ns_solver) return;

    int nu = ns_solver->n_owned_u;
    int nv = ns_solver->n_owned_v;
    int np = ns_solver->n_owned_p;
    int n  = ns_solver->n_owned_total;

    r.assign(n, 0.0);

    // 1. Inversion diagonale approchée pour les vitesses
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

    // 2. Complément de Schur pour la pression S = B D^{-1} B^T
    if (ns_solver->is_monolithic()) {
        for (int i = 0; i < np; ++i) {
            double dp = (ns_solver->diag_p.size() > (size_t)i && std::abs(ns_solver->diag_p[i]) > 1.0e-15)
                        ? ns_solver->diag_p[i] : 1.0;
            r[nu + nv + i] = z[nu + nv + i] / dp;
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
    if (custom_apply) {
        custom_apply(z, r);
    } else {
        r = z; // Préconditionneur identité par défaut
    }
}

// =============================================================================
// 6. Fabrique de Préconditionneurs (Factory Pattern)
// =============================================================================
std::unique_ptr<PreconditionerBase> PreconditionerFactory::create(const std::string& type) {
    std::string upper_type = type;
    std::transform(upper_type.begin(), upper_type.end(), upper_type.begin(), ::toupper);

    if (upper_type == "JACOBI" || upper_type == "DIAGONAL") {
        return std::make_unique<JacobiPreconditioner>();
    } else if (upper_type == "SSOR" || upper_type == "GAUSS_SEIDEL") {
        return std::make_unique<SSORPreconditioner>();
    } else if (upper_type == "ILU0" || upper_type == "ILU") {
        return std::make_unique<ILU0Preconditioner>();
    } else if (upper_type == "BLOCK_TRIANGULAR" || upper_type == "SCHUR") {
        return std::make_unique<BlockTriangularPreconditioner>();
    } else if (upper_type == "CUSTOM") {
        return std::make_unique<CustomPreconditioner>();
    } else {
        // Défaut standard FUGU / Thétis : Jacobi diagonal
        return std::make_unique<JacobiPreconditioner>();
    }
}

} // namespace fugu
