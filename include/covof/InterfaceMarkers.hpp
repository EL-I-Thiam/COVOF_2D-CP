#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "Types.hpp"
#include "InterfaceReconstruction.hpp"
#include <vector>
#include <string>

namespace covof {

/**
 * @brief Particule lagrangienne d'interface non connectée (Marker-VOF / Point-Set)
 * Aucune connectivité ni topologie de maillage de surface : insensible aux ruptures et coalescences.
 */
struct Marker {
    double x = 0.0;      ///< Abscisse lagrangienne dans le domaine global [m]
    double y = 0.0;      ///< Ordonnée lagrangienne dans le domaine global [m]
    double nx = 0.0;     ///< Composante X de la normale unitaire (+grad C / |grad C|)
    double ny = 0.0;     ///< Composante Y de la normale unitaire (+grad C / |grad C|)
    double kappa = 0.0;  ///< Courbure locale estimée au marqueur [1/m]
    int cell_i = 1;      ///< Indice local X de la cellule hôte (1..nx_local)
    int cell_j = 1;      ///< Indice local Y de la cellule hôte (1..ny_local)
    bool active = true;  ///< Indicateur d'activité du marqueur
};

/**
 * @brief Gestionnaire du nuage de marqueurs lagrangiens non connectés avec
 *        Interpolation Conservatrice de la Vitesse (CVI) sur grille MAC,
 *        advection RK2, migration MPI 2D et régression parabolique locale de courbure.
 */
class InterfaceMarkers {
public:
    std::vector<Marker> cloud;

    InterfaceMarkers() = default;

    /**
     * @brief Ensemencement initial des marqueurs sur les segments PLIC/ELVIRA des cellules mixtes
     */
    void init_from_vof(const Mesh2D& mesh, const VofConfig& vof_cfg, const InitialConditionConfig& ic,
                       const DomainBC& bc, FlowFields& fields);

    /**
     * @brief Interpolation Conservatrice de la Vitesse (CVI) H(div)-conforme à divergence nulle exacte
     *        sur la grille décalée MAC : div(u_CVI)(x, y) == (div_h u)_{i,j} en tout point (x, y)
     */
    void interpolate_velocity_cvi(const Mesh2D& mesh, const FlowFields& fields,
                                  double xp, double yp, double& up, double& vp) const;

    /**
     * @brief Interpolation bilinéaire MAC standard (sans correction solénoïdale CVI, pour comparaison)
     */
    void interpolate_velocity_bilinear(const Mesh2D& mesh, const FlowFields& fields,
                                       double xp, double yp, double& up, double& vp) const;

    /**
     * @brief Advection lagrangienne d'ordre 2 Runge-Kutta explicite (RK2 Heun / point-milieu) avec CVI
     */
    void advect_rk2(const Mesh2D& mesh, const VofConfig& vof_cfg, const FlowFields& fields, double dt);

    /**
     * @brief Migration parallèle MPI 2 passes (X puis Y) et repliement périodique des marqueurs
     */
    void migrate_mpi(const Mesh2D& mesh, const VofConfig& vof_cfg, const DomainBC& bc);

    /**
     * @brief Élagage (pruning par sous-octants) et ré-ensemencement local (reseeding) guidé par C
     */
    void prune_and_reseed(const Mesh2D& mesh, const VofConfig& vof_cfg, const DomainBC& bc, FlowFields& fields);

    /**
     * @brief Calcul de la normale et de la courbure par régression parabolique pondérée dans le repère
     *        local tangent-normal (tau, eta), avec échange MPI de marqueurs fantômes dans une bande 2.5*h
     */
    void compute_curvature_from_markers(const Mesh2D& mesh, const VofConfig& vof_cfg,
                                        const DomainBC& bc, FlowFields& fields);

    /**
     * @brief Export du nuage de marqueurs au format VTK Legacy ASCII POLYDATA pour ParaView
     */
    void write_vtk(const Mesh2D& mesh, const std::string& output_dir,
                   const std::string& case_name, int step) const;
};

} // namespace covof
