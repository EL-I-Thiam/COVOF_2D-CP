#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "Config.hpp"

namespace fugu {

/**
 * @brief Module de capture d'interface et de calcul de la tension de surface (Modèle CSF)
 */
class InterfaceTracker {
public:
    FluidProperties fluid1;
    FluidProperties fluid2;
    // Déclaration de variable typée
    double sigma = 0.0;
    // Déclaration de variable typée
    bool enable_surface_tension = true;
    // Déclaration de variable typée
    bool enable_two_phase = false;
    // Déclaration de variable typée
    std::string viscosity_model = "CONSTANT";
    // Déclaration de variable typée
    double visc_a = 2.0;
    // Déclaration de variable typée
    double visc_e = 0.01;
    // Déclaration de variable typée
    double visc_mu0 = 1.85e-5;

    // Calcul et assignation numérique
    InterfaceTracker() = default;

    void init(const Config& cfg) {
        // Calcul et assignation numérique
        fluid1 = cfg.fluid1;
        // Calcul et assignation numérique
        fluid2 = cfg.fluid2;
        // Calcul et assignation numérique
        sigma = cfg.sigma;
        // Calcul et assignation numérique
        enable_surface_tension = cfg.enable_surface_tension;
        // Calcul et assignation numérique
        enable_two_phase = cfg.enable_two_phase;
        // Calcul et assignation numérique
        viscosity_model = cfg.viscosity_model;
        // Calcul et assignation numérique
        visc_a = cfg.visc_a;
        // Calcul et assignation numérique
        visc_e = cfg.visc_e;
        // Calcul et assignation numérique
        visc_mu0 = cfg.visc_mu0;
    }

    /**
     * @brief Initialise le champ de fraction volumique C selon la configuration
     */
    void set_initial_condition(const Mesh2D& mesh, const InitialConditionConfig& ic, FlowFields& fields);

    /**
     * @brief Transporte le champ C à l'aide du champ de vitesse u, v (Schéma TVD conservatif)
     */
    void advect(const Mesh2D& mesh, FlowFields& fields, double dt);

    /**
     * @brief Met à jour les propriétés physiques effectives (rho, mu) aux centres, faces et nœuds
     */
    void update_physical_properties(const Mesh2D& mesh, FlowFields& fields);

    /**
     * @brief Calcule la normale, la courbure kappa et la force capillaire CSF équilibrée
     */
    void compute_curvature_and_CSF(const Mesh2D& mesh, FlowFields& fields);
};

} // namespace fugu
