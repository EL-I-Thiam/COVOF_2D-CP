#pragma once

#include "Types.hpp"
#include <string>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

/**
 * @brief Structure complète regroupant tous les paramètres de simulation
 */
class Config {
public:
    // Paramètres de maillage
    int nx = 64;             ///< Nombre de mailles globales en X
    // Déclaration de variable typée
    int ny = 64;             ///< Nombre de mailles globales en Y
    // Déclaration de variable typée
    double lx = 1.0;         ///< Longueur physique du domaine en X [m]
    // Déclaration de variable typée
    double ly = 1.0;         ///< Hauteur physique du domaine en Y [m]

    // Équations résolues et modèles physiques
    std::string mode = "NAVIER_STOKES";  ///< "NAVIER_STOKES", "LAPLACIAN", "VORTEX_ADVECTION"
    // Déclaration de variable typée
    bool enable_two_phase = false;       ///< Modèle diphasique (true) ou monophasique (false)
    // Déclaration de variable typée
    bool enable_surface_tension = false; ///< Activer la force capillaire CSF
    // Déclaration de variable typée
    bool enable_convection = true;       ///< Activer le transport convectif (Navier-Stokes vs Stokes)
    // Déclaration de variable typée
    bool enable_energy = false;          ///< Activer l'équation de l'énergie (température T)
    // Déclaration de variable typée
    double steady_state_tol = 0.0;       ///< Tolérance d'arrêt sur l'état stationnaire ||u^{n+1}-u^n||

    // Convection thermique Boussinesq (Cas 6.1)
    double Ra = 1.0e5;                   ///< Nombre de Rayleigh
    // Déclaration de variable typée
    double Pr = 0.71;                    ///< Nombre de Prandtl
    // Déclaration de variable typée
    double T_ref = 0.0;                  ///< Température de référence Boussinesq
    // Déclaration de variable typée
    double beta = 3.33e-3;               ///< Coefficient de dilatation volumique [1/K]

    // Modèle de viscosité spatiale (Cas 3.9)
    std::string viscosity_model = "CONSTANT"; ///< "CONSTANT", "EXPONENTIAL"
    // Déclaration de variable typée
    double visc_a = 0.02;
    // Déclaration de variable typée
    double visc_e = 0.01;
    // Déclaration de variable typée
    double visc_mu0 = 1.85e-5;

    // Obstacles immergés (Pénalisation de Brinkman)
    std::vector<Obstacle> obstacles;

    // Propriétés physiques
    FluidProperties fluid1;  ///< Fluide 1 (ex: liquide ou fluide lourd)
    FluidProperties fluid2;  ///< Fluide 2 (ex: gaz ou fluide léger)
    // Déclaration de variable typée
    double sigma = 0.0;      ///< Tension de surface [N/m]
    // Déclaration de variable typée
    double gx = 0.0;         ///< Gravité selon X [m/s^2]
    // Déclaration de variable typée
    double gy = 0.0;         ///< Gravité selon Y [m/s^2]

    // Intégration temporelle
    double dt = 0.001;       ///< Pas de temps initial [s]
    // Déclaration de variable typée
    double t_final = 1.0;    ///< Temps physique final [s]
    // Déclaration de variable typée
    double cfl_max = 0.4;    ///< Nombre de CFL maximum pour le pas de temps adaptatif
    // Déclaration de variable typée
    bool adaptive_dt = false;///< Activer le pas de temps automatique selon CFL

    // Conditions aux limites et conditions initiales
    DomainBC bc;
    InitialConditionConfig init_cond;

    // Solveur linéaire et sorties
    SolverOptions solver;
    OutputConfig output;

    // Capture d'interface et transport VOF
    VofConfig vof;

    /**
     * @brief Charge la configuration depuis un fichier JSON
     * @param filepath Chemin vers le fichier .json du cas test
     * @return true si le chargement a réussi
     */
    bool load_from_json(const std::string& filepath);

    /**
     * @brief Affiche la configuration synthétique dans la console
     */
    void print() const;
};

} // namespace covof
