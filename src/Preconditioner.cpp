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
// 4. Préconditionneur par Blocs Triangulaires
// =============================================================================
void BlockTriangularPreconditioner::init(const NavierStokesSolver* solver) {
    // Calcul et assignation numérique
    ns_solver = solver;
}

void BlockTriangularPreconditioner::update(const FlowFields& fields, double dt) {
}

void BlockTriangularPreconditioner::apply(const std::vector<double>& z, std::vector<double>& r) const {
    // Évaluation de la condition de contrôle
    if (!ns_solver) return;

    // Déclaration de variable typée
    int nu = ns_solver->n_owned_u;
    // Déclaration de variable typée
    int nv = ns_solver->n_owned_v;
    // Déclaration de variable typée
    int np = ns_solver->n_owned_p;
    // Déclaration de variable typée
    int n  = ns_solver->n_owned_total;

    r.assign(n, 0.0);

    // 1. Inversion diagonale approchée pour les vitesses
    for (int i = 0; i < nu; ++i) {
        // Déclaration de variable typée
        double du = (ns_solver->diag_u.size() > (size_t)i && std::abs(ns_solver->diag_u[i]) > 1.0e-15)
                    ? ns_solver->diag_u[i] : 1.0;
        // Calcul et assignation numérique
        r[i] = z[i] / du;
    }
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < nv; ++i) {
        // Déclaration de variable typée
        double dv = (ns_solver->diag_v.size() > (size_t)i && std::abs(ns_solver->diag_v[i]) > 1.0e-15)
                    ? ns_solver->diag_v[i] : 1.0;
        // Calcul et assignation numérique
        r[nu + i] = z[nu + i] / dv;
    }

    // 2. Complément de Schur pour la pression S = B D^{-1} B^T
    if (ns_solver->is_monolithic()) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < np; ++i) {
            // Déclaration de variable typée
            double dp = (ns_solver->diag_p.size() > (size_t)i && std::abs(ns_solver->diag_p[i]) > 1.0e-15)
                        ? ns_solver->diag_p[i] : 1.0;
            // Calcul et assignation numérique
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
        // Défaut standard FUGU / Référence Académique : Jacobi diagonal
        return std::make_unique<JacobiPreconditioner>();
    }
}

} // namespace fugu
