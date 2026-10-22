// ==============================================================================
// Projet        : COVOF 2D (Convection & Volume-Of-Fluid 2D CFD Solver)
// Fichier       : main.cpp
// Description   : Point d'entrée principal de l'exécutable COVOF 2D C++17 / OpenMPI.
//                 Initialise l'environnement parallèle MPI, analyse la ligne
//                 de commande, charge la configuration JSON, orchestre l'exécution
//                 de la simulation numérique et gère la libération des ressources.
// Auteur        : Elhadji Ibrahima Thiam
// Standard      : C++17 / MPI 3+
// ==============================================================================

#include "covof/Simulation.hpp" // Orchestrateur principal de simulation (grille, solveur, temps)
#include <mpi.h>               // Bibliothèque Message Passing Interface pour le calcul parallèle
#include <iostream>            // Flux d'entrée/sortie standard pour l'affichage console
#include <string>              // Gestion des chaînes de caractères pour les chemins de fichiers

// Point d'entrée standard du programme en ligne de commande
int main(int argc, char** argv) {
    // 1. Initialisation de l'environnement d'exécution parallèle MPI
    // Distribue les arguments de la ligne de commande à tous les processus participants
    MPI_Init(&argc, &argv);

    // Récupère l'identifiant unique (rang) du processus courant dans le communicateur global
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Récupère le nombre total de processus alloués pour l'exécution parallèle
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Fichier de configuration par défaut si aucun argument n'est fourni par l'utilisateur
    std::string config_file = "cases/cavity.json";

    // 2. Traitement des arguments de la ligne de commande
    if (argc > 1) {
        std::string arg = argv[1];
        // Détection de la demande d'aide (--help ou -h)
        if (arg == "-h" || arg == "--help") {
            // Seul le processus maître (rang 0) affiche le message d'aide pour éviter la duplication
            if (rank == 0) {
                std::cout << "================================================================================\n"
                          << "   COVOF 2D — Solveur CFD Navier-Stokes Incompressible Parallèle (C++17/MPI)   \n"
                          << "================================================================================\n"
                          << "Usage :\n"
                          << "  mpirun -np <N> ./bin/covof2d [fichier_configuration.json]\n\n"
                          << "Exemples d'exécution :\n"
                          << "  mpirun -np 4 ./bin/covof2d cases/cavity_re1000.json\n"
                          << "  mpirun -np 4 ./bin/covof2d cases/rising_bubble.json\n"
                          << "  mpirun -np 2 ./bin/covof2d cases/static_droplet.json\n"
                          << "  mpirun -np 4 ./bin/covof2d cases/dam_break.json\n"
                          << "================================================================================\n";
            }
            // Clôture propre de l'environnement MPI avant de quitter
            MPI_Finalize();
            return 0;
        }
        // Utilisation du chemin spécifié en premier argument
        config_file = arg;
    }

    // Affichage de la bannière d'accueil par le processus racine
    if (rank == 0) {
        std::cout << "================================================================================\n"
                  << "   COVOF 2D — Solveur CFD Navier-Stokes Incompressible Parallèle (C++17/MPI)   \n"
                  << "================================================================================\n"
                  << "  Nombre de processus MPI  : " << size << "\n"
                  << "  Fichier de configuration : " << config_file << "\n"
                  << "================================================================================\n";
    }

    // 3. Bloc d'exécution sécurisé sous surveillance des exceptions C++
    try {
        // Instanciation de l'objet Simulation encapsulant maillage, champs, solveurs et exports
        covof::Simulation sim;

        // Initialisation de la simulation : lecture du JSON, partitionnement spatial, allocation
        if (!sim.init(MPI_COMM_WORLD, config_file)) {
            // En cas d'erreur de parsing ou de cohérence des données, signalement sur le rang 0
            if (rank == 0) {
                std::cerr << "[ERREUR CRITIQUE] Échec de l'initialisation de la simulation COVOF 2D.\n"
                          << "                  Vérifiez la validité du fichier JSON : " << config_file << "\n";
            }
            // Interruption d'urgence coordonnée de tous les processus du communicateur MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        // Exécution de la boucle temporelle principale de calcul Navier-Stokes / VOF
        sim.run();

    } catch (const std::exception& e) {
        // Capture de toute exception standard (dépassement mémoire, assertion, I/O)
        if (rank == 0) {
            std::cerr << "[EXCEPTION INTERCEPTÉE] " << e.what() << "\n";
        }
        // Interruption immédiate de l'ensemble de la grappe de calcul
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // 4. Clôture normale et libération de l'environnement parallèle OpenMPI
    MPI_Finalize();

    // Code de retour standard indiquant le succès de l'exécution
    return 0;
}
