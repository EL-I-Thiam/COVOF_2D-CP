#pragma once

#include "Mesh.hpp"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

class NavierStokesSolver;
class FlowFields;

/**
 * @brief Classe de base abstraite pour tous les préconditionneurs (Standard CFD / COVOF 2D)
 * Permet l'extension et la création facile de nouveaux préconditionneurs par héritage.
 */
class PreconditionerBase {
public:
    // Calcul et assignation numérique
    virtual ~PreconditionerBase() = default;

    /**
     * @brief Initialise le préconditionneur avec l'opérateur Navier-Stokes
     */
    virtual void init(const NavierStokesSolver* solver) = 0;

    /**
     * @brief Met à jour les coefficients du préconditionneur si les propriétés ou le pas de temps changent
     */
    virtual void update(const FlowFields& fields, double dt) = 0;

    /**
     * @brief Applique l'action du préconditionneur : r = P^{-1} * z
     */
    virtual void apply(const std::vector<double>& z, std::vector<double>& r) const = 0;

    /**
     * @brief Renvoie le nom identifiant du préconditionneur
     */
    virtual std::string get_name() const = 0;
};

/**
 * @brief Préconditionneur Diagonal de Jacobi
 * P = Diag(A)
 * Idéal pour le parallélisme massif et les formulations par Lagrangien Augmenté.
 */
class JacobiPreconditioner : public PreconditionerBase {
public:
    // Calcul et assignation numérique
    const NavierStokesSolver* ns_solver = nullptr;

    void init(const NavierStokesSolver* solver) override;
    void update(const FlowFields& fields, double dt) override;
    void apply(const std::vector<double>& z, std::vector<double>& r) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "JACOBI"; }
};

/**
 * @brief Préconditionneur SSOR (Symmetric Successive Over-Relaxation)
 * P = 1/(2-w) * (D/w + L) * (D/w)^{-1} * (D/w + U)
 */
class SSORPreconditioner : public PreconditionerBase {
public:
    // Calcul et assignation numérique
    const NavierStokesSolver* ns_solver = nullptr;
    // Déclaration de variable typée
    double omega = 1.0; // Facteur de relaxation (defaut Gauss-Seidel symetrique)

    void init(const NavierStokesSolver* solver) override;
    void update(const FlowFields& fields, double dt) override;
    void apply(const std::vector<double>& z, std::vector<double>& r) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "SSOR"; }
};

/**
 * @brief Préconditionneur ILU(0) (Incomplete LU à remplissage nul)
 * Factorisation incomplète locale sur chaque sous-domaine Cartésien.
 */
class ILU0Preconditioner : public PreconditionerBase {
public:
    // Calcul et assignation numérique
    const NavierStokesSolver* ns_solver = nullptr;
    // Déclaration de variable typée
    std::vector<double> ilu_diag;

    void init(const NavierStokesSolver* solver) override;
    void update(const FlowFields& fields, double dt) override;
    void apply(const std::vector<double>& z, std::vector<double>& r) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "ILU0"; }
};

/**
 * @brief Préconditionneur par Blocs Triangulaires (Annexe E thèse M. El Ouafa / El Ouafa et al. 2023)
 * Dédié au système tout-couplé vitesse-pression avec approximation du complément de Schur.
 *   P = [ D_u   G ]
 *       [  0    S ]
 * où S = G^T D^{-1} G - C_p (Laplacien discret diphasique à densité variable).
 */
class BlockTriangularPreconditioner : public PreconditionerBase {
public:
    const NavierStokesSolver* ns_solver = nullptr;

    int nx = 0;
    int ny = 0;
    double dx = 0.0;
    double dy = 0.0;
    double dvol = 0.0;
    double r_al = 1000.0;
    int n_sweeps = 4;

    std::vector<double> a_E;
    std::vector<double> a_W;
    std::vector<double> a_N;
    std::vector<double> a_S;
    std::vector<double> a_P;

    std::vector<double> u_aE, u_aW, u_aN, u_aS, u_aP;
    std::vector<double> v_aE, v_aW, v_aN, v_aS, v_aP;

    void init(const NavierStokesSolver* solver) override;
    void update(const FlowFields& fields, double dt) override;
    void apply(const std::vector<double>& z, std::vector<double>& r) const override;
    std::string get_name() const override { return "BLOCK_TRIANGULAR"; }
};

/**
 * @brief Préconditionneur Personnalisé (Hook développeur)
 * Permet au développeur d'injecter une fonction lambda ou un foncteur C++ arbitraire sans modifier le code source.
 */
class CustomPreconditioner : public PreconditionerBase {
public:
    // Calcul et assignation numérique
    using ApplyFunc = std::function<void(const std::vector<double>&, std::vector<double>&)>;
    ApplyFunc custom_apply;
    // Déclaration de variable typée
    std::string custom_name = "CUSTOM";

    void init(const NavierStokesSolver* solver) override;
    void update(const FlowFields& fields, double dt) override;
    void apply(const std::vector<double>& z, std::vector<double>& r) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return custom_name; }

    // Calcul et assignation numérique
    void set_custom_operator(ApplyFunc func, const std::string& name = "USER_DEFINED") {
        // Calcul et assignation numérique
        custom_apply = func;
        // Calcul et assignation numérique
        custom_name = name;
    }
};

/**
 * @brief Fabrique de préconditionneurs (Factory Pattern)
 */
class PreconditionerFactory {
public:
    static std::unique_ptr<PreconditionerBase> create(const std::string& type);
};

// Alias de rétrocompatibilité
using BlockPreconditioner = PreconditionerBase;

} // namespace covof
