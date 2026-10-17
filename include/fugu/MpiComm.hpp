#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "Types.hpp"
#include <vector>

namespace fugu {

/**
 * @brief Gestionnaire des communications MPI et des échanges de mailles fantômes (halos)
 */
class MpiComm {
public:
    /**
     * @brief Échange les halos d'un champ scalaire centré (p, C, rho, etc.)
     */
    static void sync_cell_scalar(const Mesh2D& mesh, std::vector<double>& field);

    /**
     * @brief Échange les halos de la vitesse horizontale u (faces verticales)
     */
    static void sync_u(const Mesh2D& mesh, std::vector<double>& u);

    /**
     * @brief Échange les halos de la vitesse verticale v (faces horizontales)
     */
    static void sync_v(const Mesh2D& mesh, std::vector<double>& v);

    /**
     * @brief Applique les conditions aux limites physiques sur les mailles fantômes extérieures
     * @param homogeneous Si true, applique des CL homogènes (pour l'opérateur linéaire A*x)
     */
    static void apply_boundary_conditions(const Mesh2D& mesh, const DomainBC& bc, FlowFields& fields, bool homogeneous = false);

    /**
     * @brief Applique les conditions aux limites thermiques sur les mailles fantômes de la température T
     */
    static void apply_thermal_boundary_conditions(const Mesh2D& mesh, const DomainBC& bc, FlowFields& fields);

    /**
     * @brief Calcule le produit scalaire global de deux vecteurs distribués
     */
    static double global_dot_product(const Mesh2D& mesh, const std::vector<double>& a, const std::vector<double>& b, int size);

    /**
     * @brief Calcule la norme L2 globale d'un vecteur distribué
     */
    static double global_norm2(const Mesh2D& mesh, const std::vector<double>& v, int size);

    /**
     * @brief Réduit la valeur maximale scalaire sur tous les processus MPI (ex: CFL)
     */
    static double global_max(const Mesh2D& mesh, double local_val);

    /**
     * @brief Réduit la somme globale d'un scalaire sur tous les processus MPI (ex: intégrales)
     */
    static double global_sum(const Mesh2D& mesh, double local_val);
};

} // namespace fugu
