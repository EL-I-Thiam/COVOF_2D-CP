#pragma once

#include "Mesh.hpp"
#include <vector>

namespace fugu {

/**
 * @brief Champs scalaires et vectoriels sur la grille décalée MAC
 * Tous les champs contiennent une couche de mailles fantômes pour le parallélisme MPI.
 */
class FlowFields {
public:
    // Champs cinématiques et thermodynamiques principaux
    std::vector<double> u;  ///< Vitesse horizontale sur les faces verticales
    // Déclaration de variable typée
    std::vector<double> v;  ///< Vitesse verticale sur les faces horizontales
    // Déclaration de variable typée
    std::vector<double> p;  ///< Pression scalaire au centre des cellules
    // Déclaration de variable typée
    std::vector<double> C;  ///< Fonction indicatrice de phase au centre des cellules (C in [0, 1])
    // Déclaration de variable typée
    std::vector<double> T;  ///< Température scalaire au centre des cellules [K ou adimensionnelle]

    // Propriétés physiques locales effectives (Lois de mélange)
    std::vector<double> rho;     ///< Masse volumique au centre des mailles
    // Déclaration de variable typée
    std::vector<double> rho_u;   ///< Masse volumique interpolée sur les faces u
    // Déclaration de variable typée
    std::vector<double> rho_v;   ///< Masse volumique interpolée sur les faces v
    // Déclaration de variable typée
    std::vector<double> mu;      ///< Viscosité dynamique au centre des mailles
    // Déclaration de variable typée
    std::vector<double> mu_node; ///< Viscosité dynamique harmonique aux nœuds (coins de maille)

    // Géométrie de l'interface et tension superficielle CSF
    std::vector<double> kappa;     ///< Courbure de l'interface au centre des cellules
    // Déclaration de variable typée
    std::vector<double> norm_x;    ///< Composante X du vecteur normal
    // Déclaration de variable typée
    std::vector<double> norm_y;    ///< Composante Y du vecteur normal
    // Déclaration de variable typée
    std::vector<double> F_sigma_x; ///< Force capillaire CSF selon X sur les faces u
    // Déclaration de variable typée
    std::vector<double> F_sigma_y; ///< Force capillaire CSF selon Y sur les faces v

    // Diagnostics
    std::vector<double> div_u;     ///< Divergence de la vitesse au centre des mailles

    // Calcul et assignation numérique
    FlowFields() = default;

    /**
     * @brief Alloue la mémoire selon les dimensions locales du maillage avec mailles fantômes
     */
    void allocate(const Mesh2D& mesh) {
        // Déclaration de variable typée
        int n_cells = mesh.nx_ghost * mesh.ny_ghost;
        // Déclaration de variable typée
        int n_u = (mesh.nx_ghost + 1) * mesh.ny_ghost;
        // Déclaration de variable typée
        int n_v = mesh.nx_ghost * (mesh.ny_ghost + 1);
        // Déclaration de variable typée
        int n_nodes = (mesh.nx_ghost + 1) * (mesh.ny_ghost + 1);

        u.assign(n_u, 0.0);
        v.assign(n_v, 0.0);
        p.assign(n_cells, 0.0);
        C.assign(n_cells, 0.0);
        T.assign(n_cells, 0.0);

        rho.assign(n_cells, 1.0);
        rho_u.assign(n_u, 1.0);
        rho_v.assign(n_v, 1.0);
        mu.assign(n_cells, 1.0);
        mu_node.assign(n_nodes, 1.0);

        kappa.assign(n_cells, 0.0);
        norm_x.assign(n_cells, 0.0);
        norm_y.assign(n_cells, 0.0);
        F_sigma_x.assign(n_u, 0.0);
        F_sigma_y.assign(n_v, 0.0);

        div_u.assign(n_cells, 0.0);
    }
};

} // namespace fugu
