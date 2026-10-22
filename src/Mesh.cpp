#include "fugu/Mesh.hpp"
#include <iostream>
#include <cassert>

namespace fugu {

void Mesh2D::init(MPI_Comm parent_comm, int nx_g, int ny_g, double length_x, double length_y,
                  // Déclaration de variable typée
                  bool periodic_x, bool periodic_y) {
    // Calcul et assignation numérique
    nx_global = nx_g;
    // Calcul et assignation numérique
    ny_global = ny_g;
    // Calcul et assignation numérique
    lx = length_x;
    // Calcul et assignation numérique
    ly = length_y;
    // Calcul de la résolution géométrique de cellule
    dx = lx / nx_global;
    // Calcul de la résolution géométrique de cellule
    dy = ly / ny_global;

    // Déclaration de variable typée
    int parent_size, parent_rank;
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Comm_size(parent_comm, &parent_size);
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Comm_rank(parent_comm, &parent_rank);

    // 1. Détermination de la topologie cartésienne 2D optimale
    dims[0] = 0;
    // Calcul et assignation numérique
    dims[1] = 0;
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Dims_create(parent_size, 2, dims);

    // Déclaration de variable typée
    int periods[2] = {periodic_x ? 1 : 0, periodic_y ? 1 : 0};
    // Déclaration de variable typée
    int reorder = 1;
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Cart_create(parent_comm, 2, dims, periods, reorder, &comm);
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Comm_rank(comm, &rank);
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Comm_size(comm, &num_procs);
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Cart_coords(comm, rank, 2, coords);

    // 2. Découverte des voisins MPI
    MPI_Cart_shift(comm, 0, 1, &rank_left, &rank_right);
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Cart_shift(comm, 1, 1, &rank_bottom, &rank_top);

    // Calcul et assignation numérique
    is_boundary_left   = (!periodic_x && (coords[0] == 0));
    // Calcul et assignation numérique
    is_boundary_right  = (!periodic_x && (coords[0] == dims[0] - 1));
    // Calcul et assignation numérique
    is_boundary_bottom = (!periodic_y && (coords[1] == 0));
    // Calcul et assignation numérique
    is_boundary_top    = (!periodic_y && (coords[1] == dims[1] - 1));

    // 3. Répartition équitable des mailles locales
    auto partition_dim = [](int n_global, int n_procs, int coord, int& n_local, int& offset) {
        // Déclaration de variable typée
        int base = n_global / n_procs;
        // Déclaration de variable typée
        int rem  = n_global % n_procs;
        // Évaluation de la condition de contrôle
        if (coord < rem) {
            // Calcul et assignation numérique
            n_local = base + 1;
            // Calcul et assignation numérique
            offset  = coord * (base + 1);
        } else {
            // Calcul et assignation numérique
            n_local = base;
            // Calcul et assignation numérique
            offset  = rem * (base + 1) + (coord - rem) * base;
        }
    };

    partition_dim(nx_global, dims[0], coords[0], nx_local, i_offset);
    partition_dim(ny_global, dims[1], coords[1], ny_local, j_offset);

    // Calcul et assignation numérique
    nx_ghost = nx_local + 2;
    // Calcul et assignation numérique
    ny_ghost = ny_local + 2;
}

} // namespace fugu
