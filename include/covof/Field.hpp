#pragma once

#include "Mesh.hpp"
#include <vector>

namespace covof {
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

    // Diagnostics et pénalité locale AAL (Vincent et al. 2004, Éq. 14)
    std::vector<double> div_u;     ///< Divergence de la vitesse au centre des mailles
    std::vector<double> r_aal;     ///< Paramètre de pénalité adaptatif local r(t, M) au centre des mailles [Pa.s]

    // Champs pour formulation compressible / All-Mach (Fluids 2024)
    std::vector<double> chi_T;     ///< Compressibilité isotherme locale au centre des cellules [1/Pa]
    std::vector<double> p_old;     ///< Pression au pas précédent p^n pour terme temporel / convectif
    std::vector<double> T_old;     ///< Température au pas précédent
    std::vector<double> u_old;     ///< Vitesse axiale au pas précédent pour extrapolation
    std::vector<double> v_old;     ///< Vitesse transversale au pas précédent pour extrapolation
    std::vector<double> u_nm1;     ///< Vitesse axiale au pas n-1 (Gear / BDF2 / Adams-Bashforth)
    std::vector<double> v_nm1;     ///< Vitesse transversale au pas n-1
    std::vector<double> rho_old;   ///< Masse volumique au pas précédent
    std::vector<double> rho_u_old; ///< Masse volumique sur les faces u au pas précédent
    std::vector<double> rho_v_old; ///< Masse volumique sur les faces v au pas précédent
    std::vector<double> rho_u_nm1; ///< Masse volumique faces u au pas n-1
    std::vector<double> rho_v_nm1; ///< Masse volumique faces v au pas n-1
    std::vector<double> C_s;       ///< Fraction volumique lissée SVOF (Éq. 4-5 Fluids 2021)
    bool has_nm1 = false;          ///< Vrai dès le pas 2 (autorise Gear / AB2)
    std::vector<double> energy;    ///< Énergie totale massique e = u + 0.5*||v||^2 [J/kg]
    std::vector<double> rho_e;     ///< Énergie totale volumique rho*e [J/m^3]
    std::vector<double> rhou_star; ///< Quantité de mouvement intermédiaire (rho*u)* après SSP-RK3
    std::vector<double> rhov_star; ///< Quantité de mouvement intermédiaire (rho*v)* après SSP-RK3
    std::vector<double> rhoe_star; ///< Énergie intermédiaire (rho*e)* après SSP-RK3
    std::vector<double> C_tilde;   ///< Fraction volumique lissée par diffusion pour courbure CSF

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
        T.assign(n_cells, 300.0);

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
        r_aal.assign(n_cells, 500.0);

        chi_T.assign(n_cells, 0.0);
        p_old.assign(n_cells, 0.0);
        T_old.assign(n_cells, 300.0);
        u_old.assign(n_u, 0.0);
        v_old.assign(n_v, 0.0);
        u_nm1.assign(n_u, 0.0);
        v_nm1.assign(n_v, 0.0);
        rho_old.assign(n_cells, 1.0);
        rho_u_old.assign(n_u, 1.0);
        rho_v_old.assign(n_v, 1.0);
        rho_u_nm1.assign(n_u, 1.0);
        rho_v_nm1.assign(n_v, 1.0);
        C_s.assign(n_cells, 0.0);
        has_nm1 = false;
        energy.assign(n_cells, 0.0);
        rho_e.assign(n_cells, 0.0);
        rhou_star.assign(n_u, 0.0);
        rhov_star.assign(n_v, 0.0);
        rhoe_star.assign(n_cells, 0.0);
        C_tilde.assign(n_cells, 0.0);
    }
};

} // namespace covof
