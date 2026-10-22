#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "Config.hpp"
#include "CSRMatrix.hpp"
#include <vector>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

/**
 * @brief Opérateur couplé monolithique pour Navier-Stokes diphasique
 * Implémente l'action du système point-selle A * x et la construction du second membre b.
 */
class NavierStokesSolver {
public:
    Mesh2D mesh;
    DomainBC bc;
    // Déclaration de variable typée
    bool enable_convection = true;
    // Déclaration de variable typée
    bool enable_energy = false;
    // Déclaration de variable typée
    double gx = 0.0;
    // Déclaration de variable typée
    double gy = 0.0;
    // Déclaration de variable typée
    double beta = 3.33e-3;
    // Déclaration de variable typée
    double T_ref = 0.0;
    // Déclaration de variable typée
    double Cp = 1006.0;
    // Déclaration de variable typée
    double k_thermal = 0.026;
    // Déclaration de variable typée
    double Ra = 1.0e5;
    // Déclaration de variable typée
    double Pr = 0.71;
    // Déclaration de variable typée
    std::string solver_type = "AUGMENTED_LAGRANGIAN";
    // Déclaration de variable typée
    double r_al = 500.0;
    // Déclaration de variable typée
    int uzawa_iter = 1;
    // Déclaration de variable typée
    std::vector<Obstacle> obstacles;

    inline bool is_monolithic() const {
        // Retour de fonction
        return (solver_type == "MONOLITHIC");
    }

    inline bool is_augmented_lagrangian() const {
        // Retour de fonction
        return (solver_type != "MONOLITHIC");
    }

    inline bool is_point_in_obstacle(double x, double y) const {
        // Boucle d'itération algorithmique
        for (const auto& obs : obstacles) {
            // Évaluation de la condition de contrôle
            if (obs.type == "CYLINDER") {
                // Déclaration de variable typée
                double dx = x - obs.xc;
                // Déclaration de variable typée
                double dy = y - obs.yc;
                // Évaluation de la condition de contrôle
                if (dx * dx + dy * dy <= obs.radius * obs.radius) return true;
            } else {
                // Évaluation de la condition de contrôle
                if (x >= obs.x0 - 1e-9 && x <= obs.x1 + 1e-9 &&
                    // Calcul et assignation numérique
                    y >= obs.y0 - 1e-9 && y <= obs.y1 + 1e-9) {
                    // Retour de fonction
                    return true;
                }
            }
        }
        // Retour de fonction
        return false;
    }

    inline bool is_u_face_in_obstacle(int i, int j) const {
        // Déclaration de variable typée
        double x = mesh.xu(i);
        // Déclaration de variable typée
        double y = mesh.yc(j);
        // Retour de fonction
        return is_point_in_obstacle(x, y);
    }

    inline bool is_v_face_in_obstacle(int i, int j) const {
        // Déclaration de variable typée
        double x = mesh.xc(i);
        // Déclaration de variable typée
        double y = mesh.yv(j);
        // Retour de fonction
        return is_point_in_obstacle(x, y);
    }

    // Dimensions locales des inconnues propres (sans duplication)
    int n_owned_u = 0;
    // Déclaration de variable typée
    int n_owned_v = 0;
    // Déclaration de variable typée
    int n_owned_p = 0;
    // Déclaration de variable typée
    int n_owned_total = 0;

    // Matrice bloc locale pour le préconditionnement
    CSRMatrix Fuu;
    CSRMatrix Fvv;
    CSRMatrix SchurLaplace;
    // Déclaration de variable typée
    std::vector<double> diag_u;
    // Déclaration de variable typée
    std::vector<double> diag_v;
    // Déclaration de variable typée
    std::vector<double> diag_p;

    // Calcul et assignation numérique
    NavierStokesSolver() = default;

    /**
     * @brief Initialise les indexations et structures locales
     */
    void init(const Mesh2D& m, const Config& cfg);

    /**
     * @brief Applique l'opérateur couplé global : y = A(x) avec communications MPI
     */
    void apply_system_matrix(const FlowFields& fields_old, double dt,
                             const std::vector<double>& x, std::vector<double>& y) const;

    /**
     * @brief Construit le second membre b = rhs du système monolithique
     */
    void assemble_rhs(const FlowFields& fields, double dt, std::vector<double>& rhs) const;

    /**
     * @brief Construit les matrices locales des blocs Fuu, Fvv et du complément de Schur pour le préconditionneur
     */
    void assemble_preconditioner_blocks(const FlowFields& fields, double dt);

    /**
     * @brief Convertit les champs décalés (u, v, p) en vecteur 1D distribué
     */
    void pack_vector(const FlowFields& fields, std::vector<double>& vec) const;

    /**
     * @brief Déballe le vecteur 1D distribué dans les champs décalés (u, v, p)
     */
    void unpack_vector(const std::vector<double>& vec, FlowFields& fields, bool homogeneous = false) const;

    /**
     * @brief Met à jour le champ de pression via le Lagrangien augmenté : p^{n+1} = p^n - r * div(u^{n+1})
     */
    void update_pressure_al(FlowFields& fields) const;

    /**
     * @brief Résout l'équation de transport-diffusion de la température T^{n+1}
     */
    void solve_energy(FlowFields& fields, double dt) const;

    /**
     * @brief Calcule le nombre de Nusselt moyen sur la paroi chaude (x = 0)
     */
    double compute_mean_nusselt_hot_wall(const FlowFields& fields) const;

    // Convertisseurs d'indices locaux
    inline int local_u_dof(int i, int j) const {
        // Retour de fonction
        return (j - 1) * (mesh.nx_local + (mesh.is_boundary_right ? 1 : 0)) + (i - 1);
    }
    inline int local_v_dof(int i, int j) const {
        // Retour de fonction
        return (j - 1) * mesh.nx_local + (i - 1);
    }
    inline int local_p_dof(int i, int j) const {
        // Retour de fonction
        return (j - 1) * mesh.nx_local + (i - 1);
    }
};

} // namespace covof
