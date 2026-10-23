#pragma once

#include "Mesh.hpp"
#include <string>
#include <vector>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

/**
 * @brief Énumération des méthodes de reconstruction géométrique d'interface
 */
enum class ReconstructionMethod {
    SLIC,   ///< Simple Line Interface Calculation (Noh & Woodward 1976)
    PLIC,   ///< Piecewise Linear Interface Calculation (Youngs 1982 / Scardovelli & Zaleski 1999)
    LVIRA,  ///< Least Squares VOF Interface Reconstruction Algorithm (Pilliod & Puckett 2004)
    ELVIRA  ///< Efficient LVIRA avec 6 candidats discrets (Pilliod & Puckett 2004)
};

/**
 * @brief Énumération des algorithmes d'advection VOF
 */
enum class AdvectionMethod {
    VOF_TVD,      ///< Advection algébrique Volumes Finis TVD haute résolution
    CIAM,         ///< Calcul d'Interface Affine par Morceaux (Strang splitting directionnel)
    WEYMOUTH_YUE  ///< Advection géométrique unsplit multidimensionnelle (Weymouth & Yue 2010)
};

/**
 * @brief Énumération des limiteurs de pente TVD / NVD pour l'advection d'interface
 */
enum class TvdLimiter {
    SUPERBEE,  ///< Limiteur Superbee de Roe (hautement compressif pour interfaces)
    VAN_LEER,  ///< Limiteur harmonique doux de Van Leer
    MUSCL,     ///< Limiteur linéaire par morceaux MUSCL
    HRIC       ///< High Resolution Interface Capturing (Muzaferija & Peric 1999)
};

/**
 * @brief Structure représentant un segment de droite d'interface affine dans une cellule
 * Équation locale : n_x * x + n_y * y + alpha = 0, orientée vers le fluide 2 (gaz)
 */
struct InterfaceSegment {
    double nx = 0.0;    ///< Composante normale unitaire selon X
    double ny = 0.0;    ///< Composante normale unitaire selon Y
    double alpha = 0.0; ///< Paramètre affine de position de l'interface
};

/**
 * @brief Classe statique fournissant les algorithmes géométriques de reconstruction et de découpe
 */
class InterfaceReconstruction {
public:
    /**
     * @brief Convertit une chaîne de caractères en méthode de reconstruction
     */
    static ReconstructionMethod parse_reconstruction_method(const std::string& str);

    /**
     * @brief Convertit une chaîne de caractères en méthode d'advection
     */
    static AdvectionMethod parse_advection_method(const std::string& str);

    /**
     * @brief Convertit une chaîne de caractères en limiteur TVD
     */
    static TvdLimiter parse_tvd_limiter(const std::string& str);

    /**
     * @brief Calcule l'aire exacte de l'intersection entre un rectangle [xmin, xmax] x [ymin, ymax]
     *        et le demi-plan fluide nx * x + ny * y + alpha <= 0 (Sutherland-Hodgman + Gauss Shoelace)
     */
    static double cut_box_area(double xmin, double xmax, double ymin, double ymax,
                               double nx, double ny, double alpha);

    /**
     * @brief Calcule le paramètre affine alpha conservant exactement le volume cible C_target * dx * dy
     */
    static double compute_alpha(double xmin, double xmax, double ymin, double ymax,
                                double nx, double ny, double C_target);

    /**
     * @brief Reconstruction SLIC (Simple Line Interface Calculation) sur l'ensemble du maillage
     */
    static void reconstruct_slic(const Mesh2D& mesh, const std::vector<double>& C,
                                 std::vector<InterfaceSegment>& segments);

    /**
     * @brief Reconstruction PLIC (Youngs + Scardovelli & Zaleski) sur l'ensemble du maillage
     */
    static void reconstruct_plic(const Mesh2D& mesh, const std::vector<double>& C,
                                 std::vector<InterfaceSegment>& segments);

    /**
     * @brief Reconstruction LVIRA (Least Squares VOF Interface Reconstruction Algorithm)
     */
    static void reconstruct_lvira(const Mesh2D& mesh, const std::vector<double>& C,
                                  std::vector<InterfaceSegment>& segments);

    /**
     * @brief Reconstruction ELVIRA (Efficient LVIRA avec 6 candidats de fonctions de hauteur)
     */
    static void reconstruct_elvira(const Mesh2D& mesh, const std::vector<double>& C,
                                   std::vector<InterfaceSegment>& segments);

    /**
     * @brief Point d'entrée générique de reconstruction selon la méthode configurée
     */
    static void reconstruct(ReconstructionMethod method, const Mesh2D& mesh,
                            const std::vector<double>& C,
                            std::vector<InterfaceSegment>& segments);

    /**
     * @brief Évalue le limiteur de flux TVD / NVD sélectionné
     */
    static double evaluate_limiter(TvdLimiter limiter, double r, double c_donor = 0.5, double c_upwind = 0.0, double c_acceptor = 1.0);
};

} // namespace covof
