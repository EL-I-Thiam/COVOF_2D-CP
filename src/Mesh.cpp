#include "fugu/Mesh.hpp"
#include <iostream>
#include <cassert>

namespace fugu {

void Mesh2D::init(MPI_Comm parent_comm, int nx_g, int ny_g, double length_x, double length_y,
                  bool periodic_x, bool periodic_y) {
    nx_global = nx_g;
    ny_global = ny_g;
    lx = length_x;
    ly = length_y;
    dx = lx / nx_global;
    dy = ly / ny_global;

    int parent_size, parent_rank;
    MPI_Comm_size(parent_comm, &parent_size);
    MPI_Comm_rank(parent_comm, &parent_rank);

    // 1. Détermination de la topologie cartésienne 2D optimale
    dims[0] = 0;
    dims[1] = 0;
    MPI_Dims_create(parent_size, 2, dims);

    int periods[2] = {periodic_x ? 1 : 0, periodic_y ? 1 : 0};
    int reorder = 1;
    MPI_Cart_create(parent_comm, 2, dims, periods, reorder, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);
    MPI_Cart_coords(comm, rank, 2, coords);

    // 2. Découverte des voisins MPI
    MPI_Cart_shift(comm, 0, 1, &rank_left, &rank_right);
    MPI_Cart_shift(comm, 1, 1, &rank_bottom, &rank_top);

    is_boundary_left   = (!periodic_x && (coords[0] == 0));
    is_boundary_right  = (!periodic_x && (coords[0] == dims[0] - 1));
    is_boundary_bottom = (!periodic_y && (coords[1] == 0));
    is_boundary_top    = (!periodic_y && (coords[1] == dims[1] - 1));

    // 3. Répartition équitable des mailles locales
    auto partition_dim = [](int n_global, int n_procs, int coord, int& n_local, int& offset) {
        int base = n_global / n_procs;
        int rem  = n_global % n_procs;
        if (coord < rem) {
            n_local = base + 1;
            offset  = coord * (base + 1);
        } else {
            n_local = base;
            offset  = rem * (base + 1) + (coord - rem) * base;
        }
    };

    partition_dim(nx_global, dims[0], coords[0], nx_local, i_offset);
    partition_dim(ny_global, dims[1], coords[1], ny_local, j_offset);

    nx_ghost = nx_local + 2;
    ny_ghost = ny_local + 2;
}

} // namespace fugu
