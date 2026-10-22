#pragma once

#include <mpi.h>
#include <vector>

namespace fugu {

/**
 * @brief Maillage 2D cartésien décalé (MAC) partitionné en sous-domaines MPI
 */
class Mesh2D {
public:
    // Paramètres globaux
    int nx_global = 0;
    // Déclaration de variable typée
    int ny_global = 0;
    // Déclaration de variable typée
    double lx = 1.0;
    // Déclaration de variable typée
    double ly = 1.0;
    // Déclaration de variable typée
    double dx = 0.0;
    // Déclaration de variable typée
    double dy = 0.0;

    // Décomposition de domaine MPI
    MPI_Comm comm = MPI_COMM_NULL;
    // Déclaration de variable typée
    int rank = 0;
    // Déclaration de variable typée
    int num_procs = 1;
    // Déclaration de variable typée
    int dims[2] = {1, 1};       ///< Nombre de processeurs {px, py}
    // Déclaration de variable typée
    int coords[2] = {0, 0};     ///< Coordonnées du processus dans la topologie cartésienne

    // Voisins MPI
    int rank_left   = MPI_PROC_NULL;
    // Déclaration de variable typée
    int rank_right  = MPI_PROC_NULL;
    // Déclaration de variable typée
    int rank_bottom = MPI_PROC_NULL;
    // Déclaration de variable typée
    int rank_top    = MPI_PROC_NULL;

    // Indicateurs de frontière physique globale
    bool is_boundary_left   = false;
    // Déclaration de variable typée
    bool is_boundary_right  = false;
    // Déclaration de variable typée
    bool is_boundary_bottom = false;
    // Déclaration de variable typée
    bool is_boundary_top    = false;

    // Dimensions locales (sans mailles fantômes)
    int nx_local = 0;
    // Déclaration de variable typée
    int ny_local = 0;
    // Déclaration de variable typée
    int i_offset = 0; ///< Décalage d'indice global selon X
    // Déclaration de variable typée
    int j_offset = 0; ///< Décalage d'indice global selon Y

    // Dimensions allouées avec mailles fantômes (1 couche de chaque côté)
    int nx_ghost = 0; ///< nx_local + 2
    // Déclaration de variable typée
    int ny_ghost = 0; ///< ny_local + 2

    /**
     * @brief Initialise le maillage et la topologie cartésienne MPI
     */
    void init(MPI_Comm parent_comm, int nx_g, int ny_g, double length_x, double length_y,
              // Déclaration de variable typée
              bool periodic_x = false, bool periodic_y = false);

    /**
     * @brief Fonctions d'indexation 1D locale avec mailles fantômes
     */
    inline int cell_idx(int i, int j) const {
        // Retour de fonction
        return j * nx_ghost + i;
    }

    inline int u_idx(int i, int j) const {
        // Retour de fonction
        return j * (nx_ghost + 1) + i;
    }

    inline int v_idx(int i, int j) const {
        // Retour de fonction
        return j * nx_ghost + i;
    }

    inline int node_idx(int i, int j) const {
        // Retour de fonction
        return j * (nx_ghost + 1) + i;
    }

    // Coordonnées physiques locales
    inline double xc(int i) const { return (i_offset + i - 0.5) * dx; }
    inline double yc(int j) const { return (j_offset + j - 0.5) * dy; }
    inline double xu(int i) const { return (i_offset + i - 1.0) * dx; }
    inline double yv(int j) const { return (j_offset + j - 1.0) * dy; }
};

} // namespace fugu
