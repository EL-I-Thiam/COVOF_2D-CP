#pragma once

#include "Types.hpp"
#include "Preconditioner.hpp"
#include <vector>
#include <string>
#include <memory>

namespace fugu {

class NavierStokesSolver;
class FlowFields;

/**
 * @brief Interface de base abstraite pour tous les solveurs linéaires (standards CFD / FUGU 2D)
 */
class LinearSolverBase {
public:
    // Calcul et assignation numérique
    virtual ~LinearSolverBase() = default;

    /**
     * @brief Configure le solveur linéaire
     */
    virtual void init(const SolverOptions& config) = 0;

    /**
     * @brief Résout le système linéaire préconditionné A * x = b
     * @return Nombre d'itérations effectuées pour converger
     */
    virtual int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
                      const FlowFields& fields_old, double dt,
                      // Calcul et assignation numérique
                      const std::vector<double>& b, std::vector<double>& x) const = 0;

    /**
     * @brief Renvoie le nom identifiant du solveur
     */
    virtual std::string get_name() const = 0;
};

/**
 * @brief Solveur BiCGSTAB(2) - Sleijpen & van der Vorst (1993)
 * Minimisation polynomiale résiduelle d'ordre 2 pour valeurs propres complexes.
 */
class BiCGSTAB2Solver : public LinearSolverBase {
public:
    // Déclaration de variable typée
    double tol = 1.0e-6;
    // Déclaration de variable typée
    int max_iter = 500;
    // Déclaration de variable typée
    bool use_precond = true;

    void init(const SolverOptions& config) override;
    // Déclaration de variable typée
    int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
              const FlowFields& fields_old, double dt,
              const std::vector<double>& b, std::vector<double>& x) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "BICGSTAB2"; }
};

/**
 * @brief Solveur BiCGSTAB standard - van der Vorst (1992)
 * Méthode de gradient biconjugué stabilisé d'ordre 1.
 */
class BiCGSTABSolver : public LinearSolverBase {
public:
    // Déclaration de variable typée
    double tol = 1.0e-6;
    // Déclaration de variable typée
    int max_iter = 500;
    // Déclaration de variable typée
    bool use_precond = true;

    void init(const SolverOptions& config) override;
    // Déclaration de variable typée
    int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
              const FlowFields& fields_old, double dt,
              const std::vector<double>& b, std::vector<double>& x) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "BICGSTAB"; }
};

/**
 * @brief Solveur Gradient Conjugué Préconditionné (PCG) - Hestenes & Stiefel (1952)
 * Optimal pour systèmes symétriques définis positifs (Lagrangien Augmenté, Laplacien, Énergie).
 */
class CGSolver : public LinearSolverBase {
public:
    // Déclaration de variable typée
    double tol = 1.0e-6;
    // Déclaration de variable typée
    int max_iter = 500;
    // Déclaration de variable typée
    bool use_precond = true;

    void init(const SolverOptions& config) override;
    // Déclaration de variable typée
    int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
              const FlowFields& fields_old, double dt,
              const std::vector<double>& b, std::vector<double>& x) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "CG"; }
};

/**
 * @brief Solveur GMRES(m) - Sa'ad & Schultz (1986)
 * Résidu minimal généralisé avec redémarrages (Krylov orthogonalisé par Arnoldi et rotations de Givens).
 */
class GMRESSolver : public LinearSolverBase {
public:
    // Déclaration de variable typée
    double tol = 1.0e-6;
    // Déclaration de variable typée
    int max_iter = 500;
    // Déclaration de variable typée
    int restart = 30; // Dimension du sous-espace de Krylov
    // Déclaration de variable typée
    bool use_precond = true;

    void init(const SolverOptions& config) override;
    // Déclaration de variable typée
    int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
              const FlowFields& fields_old, double dt,
              const std::vector<double>& b, std::vector<double>& x) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "GMRES"; }
};

/**
 * @brief Solveur MINRES - Paige & Saunders (1975)
 * Résidu minimal pour systèmes symétriques définis ou indéfinis.
 */
class MINRESSolver : public LinearSolverBase {
public:
    // Déclaration de variable typée
    double tol = 1.0e-6;
    // Déclaration de variable typée
    int max_iter = 500;
    // Déclaration de variable typée
    bool use_precond = true;

    void init(const SolverOptions& config) override;
    // Déclaration de variable typée
    int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
              const FlowFields& fields_old, double dt,
              const std::vector<double>& b, std::vector<double>& x) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "MINRES"; }
};

/**
 * @brief Solveur Direct Local / Bande LU
 * Résolution exacte par décomposition directe (proxy et repli autonome pour MUMPS).
 */
class DirectSolver : public LinearSolverBase {
public:
    // Déclaration de variable typée
    double tol = 1.0e-12;
    // Déclaration de variable typée
    int max_iter = 1;

    void init(const SolverOptions& config) override;
    // Déclaration de variable typée
    int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
              const FlowFields& fields_old, double dt,
              const std::vector<double>& b, std::vector<double>& x) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "DIRECT"; }
};

/**
 * @brief Interface / Passerelle pour Solveur Direct MUMPS (MUltifrontal Massively Parallel Solver)
 */
class MumpsSolverBridge : public LinearSolverBase {
public:
    void init(const SolverOptions& config) override;
    // Déclaration de variable typée
    int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
              const FlowFields& fields_old, double dt,
              const std::vector<double>& b, std::vector<double>& x) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "MUMPS"; }
};

/**
 * @brief Interface / Passerelle pour Bibliothèque PETSc (Portable, Extensible Toolkit for Scientific Computation)
 */
class PetscSolverBridge : public LinearSolverBase {
public:
    // Déclaration de variable typée
    std::string ksp_type = "bcgs";
    // Déclaration de variable typée
    std::string pc_type  = "bjacobi";

    void init(const SolverOptions& config) override;
    // Déclaration de variable typée
    int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
              const FlowFields& fields_old, double dt,
              const std::vector<double>& b, std::vector<double>& x) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "PETSC"; }
};

/**
 * @brief Interface / Passerelle pour Bibliothèque HYPRE (High Performance Preconditioners / BoomerAMG)
 */
class HypreSolverBridge : public LinearSolverBase {
public:
    void init(const SolverOptions& config) override;
    // Déclaration de variable typée
    int solve(const NavierStokesSolver& op, const PreconditionerBase& P,
              const FlowFields& fields_old, double dt,
              const std::vector<double>& b, std::vector<double>& x) const override;
    // Déclaration de variable typée
    std::string get_name() const override { return "HYPRE"; }
};

/**
 * @brief Fabrique de Solveurs Linéaires (Factory Pattern Référence Académique / FUGU 2D)
 */
class SolverFactory {
public:
    static std::unique_ptr<LinearSolverBase> create(const std::string& type);
};

// Alias de rétrocompatibilité
using LinearSolver = LinearSolverBase;

} // namespace fugu
