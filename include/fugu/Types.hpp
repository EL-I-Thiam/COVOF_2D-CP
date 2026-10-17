#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cmath>

namespace fugu {

/**
 * @brief Types de conditions aux limites physiques (Annexe B thèse M. El Ouafa)
 */
enum class BCType {
    DIRICHLET,  ///< Vitesse imposée (adhérence u=0, paroi mobile u=U_wall, ou profil d'entrée)
    NEUMANN,    ///< Dérivée normale nulle (sortie libre dU/dn = 0)
    FREE_SLIP,  ///< Glissement libre (composante normale nulle, contrainte tangentielle nulle)
    PERIODIC    ///< Conditions aux limites périodiques
};

/**
 * @brief Spécification des conditions aux limites sur les 4 frontières du domaine 2D
 */
struct BoundaryCondition {
    BCType type = BCType::DIRICHLET;
    double u_val = 0.0;  ///< Valeur de u imposée si Dirichlet
    double v_val = 0.0;  ///< Valeur de v imposée si Dirichlet
    std::string profile = "UNIFORM"; ///< "UNIFORM", "PARABOLIC", "POISEUILLE_STEP"
    double u_mean = 0.0; ///< Vitesse moyenne pour profils paraboliques
    double y_step = 0.0; ///< Hauteur de la marche pour POISEUILLE_STEP
    BCType T_type = BCType::NEUMANN; ///< Type de CL pour l'énergie (DIRICHLET ou NEUMANN)
    double T_val = 0.0;             ///< Température imposée si DIRICHLET
};

/**
 * @brief Obstacle immergé avec pénalisation de Brinkman (Thétis standard)
 */
struct Obstacle {
    std::string type = "BOX"; ///< Forme géométrique ("BOX", "CYLINDER")
    double x0 = 0.0;          ///< Borne inférieure X [m]
    double y0 = 0.0;          ///< Borne inférieure Y [m]
    double x1 = 0.0;          ///< Borne supérieure X [m]
    double y1 = 0.0;          ///< Borne supérieure Y [m]
    double xc = 0.0;          ///< Centre cylindre X [m]
    double yc = 0.0;          ///< Centre cylindre Y [m]
    double radius = 0.0;      ///< Rayon cylindre [m]
    double permeability = 1.0e-14; ///< Perméabilité de Darcy-Brinkman K
};

struct DomainBC {
    BoundaryCondition left;    ///< Frontière x = 0
    BoundaryCondition right;   ///< Frontière x = Lx
    BoundaryCondition bottom;  ///< Frontière y = 0
    BoundaryCondition top;     ///< Frontière y = Ly
};

/**
 * @brief Propriétés physiques d'une phase fluide
 */
struct FluidProperties {
    double rho = 1.0;         ///< Masse volumique [kg/m^3]
    double mu  = 1.0;         ///< Viscosité dynamique [Pa.s]
    double Cp = 1006.0;       ///< Capacité calorifique à pression constante [J/(kg.K)]
    double k_thermal = 0.026; ///< Conductivité thermique [W/(m.K)]
    double beta = 3.33e-3;    ///< Coefficient de dilatation thermique volumique [1/K]
};

/**
 * @brief Types de configurations initiales de la fraction volumique C
 */
enum class InitialConditionType {
    SINGLE_PHASE,    ///< Écoulement monophasique (C = 1 partout)
    DROPLET,         ///< Goutte (Fluide 1 à l'intérieur, Fluide 2 à l'extérieur)
    BUBBLE,          ///< Bulle (Fluide 2 à l'intérieur, Fluide 1 à l'extérieur)
    DAM_BREAK,       ///< Colonne rectangulaire d'eau (rupture de barrage)
    SHEAR_LAYER,     ///< Nappe liquide cisaillée avec perturbation
    RAYLEIGH_TAYLOR, ///< Interface perturbée avec fluide lourd au-dessus
    VORTEX_SHEAR     ///< Tâche circulaire déformée par vortex avec cisaillement
};

/**
 * @brief Paramètres géométriques de la condition initiale diphasique
 */
struct InitialConditionConfig {
    InitialConditionType type = InitialConditionType::SINGLE_PHASE;
    double center_x = 0.5;   ///< Centre de la goutte en X [m]
    double center_y = 0.5;   ///< Centre de la goutte en Y [m]
    double radius   = 0.25;  ///< Rayon de la goutte [m]
    double box_x0   = 0.0;   ///< Coin inf-gauche du rectangle (Dam break)
    double box_y0   = 0.0;
    double box_x1   = 1.0;   ///< Coin sup-droit du rectangle (Dam break)
    double box_y1   = 1.0;
    double pert_amp = 0.01;  ///< Amplitude de perturbation (Rayleigh-Taylor)
    double pert_k   = 1.0;   ///< Nombre d'onde de perturbation
};

/**
 * @brief Options de paramétrage du solveur linéaire de Krylov BiCGSTAB(2)
 */
struct SolverOptions {
    std::string type = "AUGMENTED_LAGRANGIAN"; ///< AUGMENTED_LAGRANGIAN ou MONOLITHIC
    double tol = 1.0e-7;                        ///< Résidu relatif d'arrêt ||r_k|| / ||r_0||
    int max_iter = 1000;                        ///< Nombre maximum d'itérations
    double r_al = 500.0;                        ///< Paramètre de pénalité de divergence r [Pa.s ou kg/(m.s)]
    int uzawa_iter = 1;                         ///< Nombre d'itérations d'Uzawa-Lagrangien augmenté
    bool use_precond = true;                    ///< Activer le préconditionneur
    std::string precond_type = "JACOBI";        ///< JACOBI, BLOCK_TRIANGULAR, ou NONE
};

/**
 * @brief Options de gestion des sorties et de la visualisation ParaView
 */
struct OutputConfig {
    std::string output_dir = "output"; ///< Répertoire d'écriture
    std::string case_name  = "fugu";   ///< Préfixe des fichiers VTK
    int vtk_interval = 10;             ///< Pas d'export VTK (tous les N pas de temps)
    int log_interval = 1;              ///< Pas d'affichage console et CSV
    bool write_csv_history = true;     ///< Écrire l'historique temporel (centre de masse, vitesse...)
};

} // namespace fugu
