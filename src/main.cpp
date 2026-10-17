#include "fugu/Simulation.hpp"
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string config_file = "cases/cavity.json";
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            if (rank == 0) {
                std::cout << "Usage: mpirun -np <N> ./fugu2d [fichier_configuration.json]\n"
                          << "Exemples :\n"
                          << "  mpirun -np 4 ./bin/fugu2d cases/cavity.json\n"
                          << "  mpirun -np 4 ./bin/fugu2d cases/rising_bubble.json\n"
                          << "  mpirun -np 2 ./bin/fugu2d cases/static_droplet.json\n"
                          << "  mpirun -np 4 ./bin/fugu2d cases/dam_break.json\n";
            }
            MPI_Finalize();
            return 0;
        }
        config_file = arg;
    }

    try {
        fugu::Simulation sim;
        if (!sim.init(MPI_COMM_WORLD, config_file)) {
            if (rank == 0) {
                std::cerr << "Erreur critique : Echec de l'initialisation de la simulation FUGU 2D.\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        sim.run();

    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Exception non interceptee : " << e.what() << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}
