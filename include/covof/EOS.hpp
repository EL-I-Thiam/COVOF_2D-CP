#pragma once

#include "Types.hpp"
#include <string>
#include <cmath>
#include <algorithm>

namespace covof {

/**
 * @brief Modèle d'Équation d'État (EOS) pour les écoulements compressibles / All-Mach
 * Référence : Fluids 2024, 9, 90 (El Ouafa et al.) et Le Métayer & Saurel (2016)
 */
class EOS {
public:
    enum class Type {
        INCOMPRESSIBLE,
        LINEAR,
        IDEAL_GAS,
        NASG
    };

    static Type parse_type(const std::string& name) {
        if (name == "LINEAR" || name == "ACOUSTIC") return Type::LINEAR;
        if (name == "IDEAL_GAS" || name == "PERFECT_GAS") return Type::IDEAL_GAS;
        if (name == "NASG" || name == "STIFFENED_GAS") return Type::NASG;
        return Type::INCOMPRESSIBLE;
    }

    /**
     * @brief Calcule le coefficient de compressibilité isotherme chi_T = (1/rho)*(drho/dp)_T [1/Pa]
     */
    static double compute_chi_T(Type type, const FluidProperties& props, double p, double T, double rho) {
        switch (type) {
            case Type::INCOMPRESSIBLE:
                return 0.0;
            case Type::LINEAR:
                return props.chi_T;
            case Type::IDEAL_GAS: {
                double p_pos = std::max(1.0, p);
                return 1.0 / p_pos;
            }
            case Type::NASG: {
                double p_total = std::max(1.0, p + props.p_inf);
                double rho_pos = std::max(1.0e-3, rho);
                return (1.0 - rho_pos * props.b_nasg) / (rho_pos * p_total);
            }
            default:
                return 0.0;
        }
    }

    /**
     * @brief Calcule la masse volumique rho(p, T) [kg/m^3]
     */
    static double compute_density(Type type, const FluidProperties& props, double p, double T, double p_ref = 1.01325e5) {
        switch (type) {
            case Type::INCOMPRESSIBLE:
                return props.rho;
            case Type::LINEAR:
                return props.rho * (1.0 + props.chi_T * (p - p_ref));
            case Type::IDEAL_GAS: {
                double T_pos = std::max(1.0, T);
                double r = (props.r_gas > 0.0) ? props.r_gas : 287.05;
                return p / (r * T_pos);
            }
            case Type::NASG: {
                return props.rho;
            }
            default:
                return props.rho;
        }
    }

    /**
     * @brief Calcule la température à partir de l'énergie totale e et de la vitesse v [K]
     * Éq. 11 de l'article : T = (gamma - 1)/r * (e - ||v||^2 / 2)
     */
    static double compute_temperature(Type type, const FluidProperties& props, double e, double v_sq) {
        if (type == Type::INCOMPRESSIBLE || type == Type::LINEAR) {
            return 300.0; // Isotherme par défaut
        }
        double u = std::max(1.0, e - 0.5 * v_sq);
        double gamma = (props.gamma > 1.0) ? props.gamma : 1.4;
        double r = (props.r_gas > 0.0) ? props.r_gas : 287.05;
        return (gamma - 1.0) / r * u;
    }

    /**
     * @brief Vitesse du son c_s [m/s]
     */
    static double compute_sound_speed(Type type, const FluidProperties& props, double p, double rho, double T) {
        switch (type) {
            case Type::INCOMPRESSIBLE:
                return 1.0e5; // Vitesse acoustique quasi-infinie
            case Type::LINEAR: {
                if (props.chi_T > 0.0 && rho > 0.0) return 1.0 / std::sqrt(rho * props.chi_T);
                return 1500.0;
            }
            case Type::IDEAL_GAS: {
                double gamma = (props.gamma > 1.0) ? props.gamma : 1.4;
                double r = (props.r_gas > 0.0) ? props.r_gas : 287.05;
                return std::sqrt(gamma * r * std::max(1.0, T));
            }
            default:
                return 1500.0;
        }
    }
};

} // namespace covof
