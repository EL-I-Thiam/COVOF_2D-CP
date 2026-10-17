#pragma once

#include "Config.hpp"
#include "Mesh.hpp"
#include "Field.hpp"
#include "InterfaceTracker.hpp"
#include "NavierStokesSolver.hpp"
#include "Preconditioner.hpp"
#include "LinearSolver.hpp"
#include "VTKWriter.hpp"

namespace fugu {

/**
 * @brief Orchestrateur principal de la simulation CFD diphasique tout-couplée 2D
 */
class Simulation {
public:
    Config config;
    Mesh2D mesh;
    FlowFields fields;
    InterfaceTracker tracker;
    NavierStokesSolver ns_solver;
    std::unique_ptr<PreconditionerBase> precond;
    std::unique_ptr<LinearSolverBase> linear_solver;
    VTKWriter vtk_writer;

    Simulation() = default;

    /**
     * @brief Configure et initialise l'ensemble des modules physiques et numériques
     */
    bool init(MPI_Comm comm, const std::string& config_file);

    /**
     * @brief Exécute la boucle temporelle jusqu'à t_final
     */
    void run();

    /**
     * @brief Résout le problème de vérification du Laplacien analytique (Cas 3.1 Thétis)
     */
    void run_laplacian();

    /**
     * @brief Résout le problème d'advection d'une tâche dans un vortex avec cisaillement (Cas 3.19 Thétis)
     */
    void run_vortex_advection();

    /**
     * @brief Calcule le pas de temps adaptatif selon les critères CFL, viscosité et tension superficielle
     */
    double compute_time_step() const;
};

} // namespace fugu
