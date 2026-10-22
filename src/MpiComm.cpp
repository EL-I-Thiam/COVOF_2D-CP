#include "covof/MpiComm.hpp"
#include <algorithm>
#include <cmath>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

void MpiComm::sync_cell_scalar(const Mesh2D& mesh, std::vector<double>& field) {
    // Évaluation de la condition de contrôle
    if (mesh.num_procs == 1) return;

    // Déclaration de variable typée
    int nx = mesh.nx_local;
    // Déclaration de variable typée
    int ny = mesh.ny_local;

    // Déclaration de variable typée
    std::vector<double> send_left(ny), recv_left(ny);
    // Déclaration de variable typée
    std::vector<double> send_right(ny), recv_right(ny);
    // Déclaration de variable typée
    std::vector<double> send_bot(nx), recv_bot(nx);
    // Déclaration de variable typée
    std::vector<double> send_top(nx), recv_top(nx);

    // Préparation des buffers X
    for (int j = 1; j <= ny; ++j) {
        // Calcul et assignation numérique
        send_left[j - 1]  = field[mesh.cell_idx(1, j)];
        // Calcul et assignation numérique
        send_right[j - 1] = field[mesh.cell_idx(nx, j)];
    }

    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Request reqs[8];
    // Déclaration de variable typée
    int n_reqs = 0;

    // Échanges Est-Ouest
    if (mesh.rank_left != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_left.data(), ny, MPI_DOUBLE, mesh.rank_left, 10, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_left.data(), ny, MPI_DOUBLE, mesh.rank_left, 11, mesh.comm, &reqs[n_reqs++]);
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_right != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_right.data(), ny, MPI_DOUBLE, mesh.rank_right, 11, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_right.data(), ny, MPI_DOUBLE, mesh.rank_right, 10, mesh.comm, &reqs[n_reqs++]);
    }

    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    // Calcul et assignation numérique
    n_reqs = 0;

    // Évaluation de la condition de contrôle
    if (mesh.rank_left != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) field[mesh.cell_idx(0, j)] = recv_left[j - 1];
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_right != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) field[mesh.cell_idx(nx + 1, j)] = recv_right[j - 1];
    }

    // Préparation des buffers Y
    for (int i = 1; i <= nx; ++i) {
        // Calcul et assignation numérique
        send_bot[i - 1] = field[mesh.cell_idx(i, 1)];
        // Calcul et assignation numérique
        send_top[i - 1] = field[mesh.cell_idx(i, ny)];
    }

    // Échanges Nord-Sud
    if (mesh.rank_bottom != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_bot.data(), nx, MPI_DOUBLE, mesh.rank_bottom, 20, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_bot.data(), nx, MPI_DOUBLE, mesh.rank_bottom, 21, mesh.comm, &reqs[n_reqs++]);
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_top != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_top.data(), nx, MPI_DOUBLE, mesh.rank_top, 21, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_top.data(), nx, MPI_DOUBLE, mesh.rank_top, 20, mesh.comm, &reqs[n_reqs++]);
    }

    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    // Évaluation de la condition de contrôle
    if (mesh.rank_bottom != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) field[mesh.cell_idx(i, 0)] = recv_bot[i - 1];
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_top != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) field[mesh.cell_idx(i, ny + 1)] = recv_top[i - 1];
    }
}

void MpiComm::sync_u(const Mesh2D& mesh, std::vector<double>& u) {
    // Évaluation de la condition de contrôle
    if (mesh.num_procs == 1) return;

    // Déclaration de variable typée
    int nx = mesh.nx_local;
    // Déclaration de variable typée
    int ny = mesh.ny_local;

    // u est défini sur les faces verticales i de 1 à nx+1 pour l'intérieur
    std::vector<double> send_left(ny), recv_left(ny);
    // Déclaration de variable typée
    std::vector<double> send_right(ny), recv_right(ny);
    // Déclaration de variable typée
    std::vector<double> send_bot(nx + 2), recv_bot(nx + 2);
    // Déclaration de variable typée
    std::vector<double> send_top(nx + 2), recv_top(nx + 2);

    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= ny; ++j) {
        // Calcul et assignation numérique
        send_left[j - 1]  = u[mesh.u_idx(1, j)];
        // Calcul et assignation numérique
        send_right[j - 1] = u[mesh.u_idx(nx, j)];
    }

    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Request reqs[8];
    // Déclaration de variable typée
    int n_reqs = 0;

    // Évaluation de la condition de contrôle
    if (mesh.rank_left != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_left.data(), ny, MPI_DOUBLE, mesh.rank_left, 30, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_left.data(), ny, MPI_DOUBLE, mesh.rank_left, 31, mesh.comm, &reqs[n_reqs++]);
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_right != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_right.data(), ny, MPI_DOUBLE, mesh.rank_right, 31, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_right.data(), ny, MPI_DOUBLE, mesh.rank_right, 30, mesh.comm, &reqs[n_reqs++]);
    }

    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    // Calcul et assignation numérique
    n_reqs = 0;

    // Évaluation de la condition de contrôle
    if (mesh.rank_left != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) u[mesh.u_idx(0, j)] = recv_left[j - 1];
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_right != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) u[mesh.u_idx(nx + 1, j)] = recv_right[j - 1];
    }

    // Échange selon Y
    for (int i = 0; i <= nx + 1; ++i) {
        // Calcul et assignation numérique
        send_bot[i] = u[mesh.u_idx(i, 1)];
        // Calcul et assignation numérique
        send_top[i] = u[mesh.u_idx(i, ny)];
    }

    // Évaluation de la condition de contrôle
    if (mesh.rank_bottom != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_bot.data(), nx + 2, MPI_DOUBLE, mesh.rank_bottom, 40, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_bot.data(), nx + 2, MPI_DOUBLE, mesh.rank_bottom, 41, mesh.comm, &reqs[n_reqs++]);
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_top != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_top.data(), nx + 2, MPI_DOUBLE, mesh.rank_top, 41, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_top.data(), nx + 2, MPI_DOUBLE, mesh.rank_top, 40, mesh.comm, &reqs[n_reqs++]);
    }

    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    // Évaluation de la condition de contrôle
    if (mesh.rank_bottom != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i <= nx + 1; ++i) u[mesh.u_idx(i, 0)] = recv_bot[i];
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_top != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i <= nx + 1; ++i) u[mesh.u_idx(i, ny + 1)] = recv_top[i];
    }
}

void MpiComm::sync_v(const Mesh2D& mesh, std::vector<double>& v) {
    // Évaluation de la condition de contrôle
    if (mesh.num_procs == 1) return;

    // Déclaration de variable typée
    int nx = mesh.nx_local;
    // Déclaration de variable typée
    int ny = mesh.ny_local;

    // v est défini sur les faces horizontales j de 1 à ny+1 pour l'intérieur
    std::vector<double> send_left(ny + 2), recv_left(ny + 2);
    // Déclaration de variable typée
    std::vector<double> send_right(ny + 2), recv_right(ny + 2);
    // Déclaration de variable typée
    std::vector<double> send_bot(nx), recv_bot(nx);
    // Déclaration de variable typée
    std::vector<double> send_top(nx), recv_top(nx);

    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 0; j <= ny + 1; ++j) {
        // Calcul et assignation numérique
        send_left[j]  = v[mesh.v_idx(1, j)];
        // Calcul et assignation numérique
        send_right[j] = v[mesh.v_idx(nx, j)];
    }

    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Request reqs[8];
    // Déclaration de variable typée
    int n_reqs = 0;

    // Évaluation de la condition de contrôle
    if (mesh.rank_left != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_left.data(), ny + 2, MPI_DOUBLE, mesh.rank_left, 50, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_left.data(), ny + 2, MPI_DOUBLE, mesh.rank_left, 51, mesh.comm, &reqs[n_reqs++]);
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_right != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_right.data(), ny + 2, MPI_DOUBLE, mesh.rank_right, 51, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_right.data(), ny + 2, MPI_DOUBLE, mesh.rank_right, 50, mesh.comm, &reqs[n_reqs++]);
    }

    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    // Calcul et assignation numérique
    n_reqs = 0;

    // Évaluation de la condition de contrôle
    if (mesh.rank_left != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 0; j <= ny + 1; ++j) v[mesh.v_idx(0, j)] = recv_left[j];
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_right != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 0; j <= ny + 1; ++j) v[mesh.v_idx(nx + 1, j)] = recv_right[j];
    }

    // Échange selon Y
    for (int i = 1; i <= nx; ++i) {
        // Calcul et assignation numérique
        send_bot[i - 1] = v[mesh.v_idx(i, 1)];
        // Calcul et assignation numérique
        send_top[i - 1] = v[mesh.v_idx(i, ny)];
    }

    // Évaluation de la condition de contrôle
    if (mesh.rank_bottom != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_bot.data(), nx, MPI_DOUBLE, mesh.rank_bottom, 60, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_bot.data(), nx, MPI_DOUBLE, mesh.rank_bottom, 61, mesh.comm, &reqs[n_reqs++]);
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_top != MPI_PROC_NULL) {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Irecv(recv_top.data(), nx, MPI_DOUBLE, mesh.rank_top, 61, mesh.comm, &reqs[n_reqs++]);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Isend(send_top.data(), nx, MPI_DOUBLE, mesh.rank_top, 60, mesh.comm, &reqs[n_reqs++]);
    }

    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    // Évaluation de la condition de contrôle
    if (mesh.rank_bottom != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) v[mesh.v_idx(i, 0)] = recv_bot[i - 1];
    }
    // Évaluation de la condition de contrôle
    if (mesh.rank_top != MPI_PROC_NULL) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) v[mesh.v_idx(i, ny + 1)] = recv_top[i - 1];
    }
}

void MpiComm::apply_boundary_conditions(const Mesh2D& mesh, const DomainBC& bc, FlowFields& f, bool homogeneous) {
    // Déclaration de variable typée
    int nx = mesh.nx_local;
    // Déclaration de variable typée
    int ny = mesh.ny_local;

    // Déclaration de variable typée
    double u_left_val   = homogeneous ? 0.0 : bc.left.u_val;
    // Déclaration de variable typée
    double v_left_val   = homogeneous ? 0.0 : bc.left.v_val;
    // Déclaration de variable typée
    double u_right_val  = homogeneous ? 0.0 : bc.right.u_val;
    // Déclaration de variable typée
    double v_right_val  = homogeneous ? 0.0 : bc.right.v_val;
    // Déclaration de variable typée
    double u_bottom_val = homogeneous ? 0.0 : bc.bottom.u_val;
    // Déclaration de variable typée
    double v_bottom_val = homogeneous ? 0.0 : bc.bottom.v_val;
    // Déclaration de variable typée
    double u_top_val    = homogeneous ? 0.0 : bc.top.u_val;
    // Déclaration de variable typée
    double v_top_val    = homogeneous ? 0.0 : bc.top.v_val;

    // 0. Traitement de la périodicité si non gérée par MPI (ex: 1 seul processeur dans la direction)
    if (bc.left.type == BCType::PERIODIC && mesh.dims[0] == 1) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) {
            // Calcul de la résolution géométrique de cellule
            f.u[mesh.u_idx(0, j)]      = f.u[mesh.u_idx(nx, j)];
            // Calcul de la résolution géométrique de cellule
            f.u[mesh.u_idx(nx + 1, j)] = f.u[mesh.u_idx(1, j)];
            // Calcul de la résolution géométrique de cellule
            f.v[mesh.v_idx(0, j)]      = f.v[mesh.v_idx(nx, j)];
            // Calcul de la résolution géométrique de cellule
            f.v[mesh.v_idx(nx + 1, j)] = f.v[mesh.v_idx(1, j)];
            // Calcul de la résolution géométrique de cellule
            f.p[mesh.cell_idx(0, j)]      = f.p[mesh.cell_idx(nx, j)];
            // Calcul de la résolution géométrique de cellule
            f.p[mesh.cell_idx(nx + 1, j)] = f.p[mesh.cell_idx(1, j)];
            // Calcul de la résolution géométrique de cellule
            f.C[mesh.cell_idx(0, j)]      = f.C[mesh.cell_idx(nx, j)];
            // Calcul de la résolution géométrique de cellule
            f.C[mesh.cell_idx(nx + 1, j)] = f.C[mesh.cell_idx(1, j)];
        }
    }

    // Évaluation de la condition de contrôle
    if (bc.bottom.type == BCType::PERIODIC && mesh.dims[1] == 1) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Calcul de la résolution géométrique de cellule
            f.v[mesh.v_idx(i, 0)]      = f.v[mesh.v_idx(i, ny)];
            // Calcul de la résolution géométrique de cellule
            f.v[mesh.v_idx(i, ny + 1)] = f.v[mesh.v_idx(i, 1)];
            // Calcul de la résolution géométrique de cellule
            f.u[mesh.u_idx(i, 0)]      = f.u[mesh.u_idx(i, ny)];
            // Calcul de la résolution géométrique de cellule
            f.u[mesh.u_idx(i, ny + 1)] = f.u[mesh.u_idx(i, 1)];
            // Calcul de la résolution géométrique de cellule
            f.p[mesh.cell_idx(i, 0)]      = f.p[mesh.cell_idx(i, ny)];
            // Calcul de la résolution géométrique de cellule
            f.p[mesh.cell_idx(i, ny + 1)] = f.p[mesh.cell_idx(i, 1)];
            // Calcul de la résolution géométrique de cellule
            f.C[mesh.cell_idx(i, 0)]      = f.C[mesh.cell_idx(i, ny)];
            // Calcul de la résolution géométrique de cellule
            f.C[mesh.cell_idx(i, ny + 1)] = f.C[mesh.cell_idx(i, 1)];
        }
    }

    // 1. Bord gauche physique (x = 0)
    if (mesh.is_boundary_left) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) {
            // Évaluation de la condition de contrôle
            if (bc.left.type == BCType::DIRICHLET) {
                // Déclaration de variable typée
                double u_inlet = u_left_val;
                // Évaluation de la condition de contrôle
                if (!homogeneous) {
                    // Évaluation de la condition de contrôle
                    if (bc.left.profile == "POISEUILLE_STEP") {
                        // Déclaration de variable typée
                        double y = mesh.yc(j);
                        // Évaluation de la condition de contrôle
                        if (y < bc.left.y_step) {
                            // Calcul et assignation numérique
                            u_inlet = 0.0;
                        } else {
                            // Déclaration de variable typée
                            double h = mesh.ly - bc.left.y_step;
                            // Calcul et assignation numérique
                            u_inlet = 6.0 * bc.left.u_mean * (y - bc.left.y_step) * (mesh.ly - y) / (h * h);
                        }
                    // Calcul et assignation numérique
                    } else if (bc.left.profile == "PARABOLIC") {
                        // Déclaration de variable typée
                        double y = mesh.yc(j);
                        // Déclaration de variable typée
                        double h = mesh.ly;
                        // Calcul et assignation numérique
                        u_inlet = 6.0 * bc.left.u_mean * y * (h - y) / (h * h);
                    }
                    // Calcul de la résolution géométrique de cellule
                    f.u[mesh.u_idx(1, j)] = u_inlet;
                }
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(0, j)] = u_inlet;
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(0, j)] = 2.0 * v_left_val - f.v[mesh.v_idx(1, j)];
            // Calcul et assignation numérique
            } else if (bc.left.type == BCType::FREE_SLIP) {
                // Évaluation de la condition de contrôle
                if (!homogeneous) f.u[mesh.u_idx(1, j)] = 0.0;
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(0, j)] = 0.0;
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(0, j)] = f.v[mesh.v_idx(1, j)];
            // Calcul et assignation numérique
            } else if (bc.left.type == BCType::NEUMANN) {
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(0, j)] = f.u[mesh.u_idx(1, j)];
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(0, j)] = f.v[mesh.v_idx(1, j)];
            }
            // Calcul de la résolution géométrique de cellule
            f.p[mesh.cell_idx(0, j)] = f.p[mesh.cell_idx(1, j)];
            // Calcul de la résolution géométrique de cellule
            f.C[mesh.cell_idx(0, j)] = f.C[mesh.cell_idx(1, j)];
        }
    }

    // 2. Bord droit physique (x = Lx)
    if (mesh.is_boundary_right) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) {
            // Évaluation de la condition de contrôle
            if (bc.right.type == BCType::DIRICHLET) {
                // Évaluation de la condition de contrôle
                if (!homogeneous) f.u[mesh.u_idx(nx + 1, j)] = u_right_val;
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(nx + 2, j)] = u_right_val;
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(nx + 1, j)] = 2.0 * v_right_val - f.v[mesh.v_idx(nx, j)];
            // Calcul et assignation numérique
            } else if (bc.right.type == BCType::FREE_SLIP) {
                // Évaluation de la condition de contrôle
                if (!homogeneous) f.u[mesh.u_idx(nx + 1, j)] = 0.0;
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(nx + 2, j)] = 0.0;
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(nx + 1, j)] = f.v[mesh.v_idx(nx, j)];
            // Calcul et assignation numérique
            } else if (bc.right.type == BCType::NEUMANN) {
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(nx + 2, j)] = f.u[mesh.u_idx(nx + 1, j)];
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(nx + 1, j)] = f.v[mesh.v_idx(nx, j)];
            }
            // Calcul de la résolution géométrique de cellule
            f.p[mesh.cell_idx(nx + 1, j)] = f.p[mesh.cell_idx(nx, j)];
            // Calcul de la résolution géométrique de cellule
            f.C[mesh.cell_idx(nx + 1, j)] = f.C[mesh.cell_idx(nx, j)];
        }
    }

    // 3. Bord inférieur physique (y = 0)
    if (mesh.is_boundary_bottom) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Évaluation de la condition de contrôle
            if (bc.bottom.type == BCType::DIRICHLET) {
                // Évaluation de la condition de contrôle
                if (!homogeneous) f.v[mesh.v_idx(i, 1)] = v_bottom_val;
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(i, 0)] = v_bottom_val;
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(i, 0)] = 2.0 * u_bottom_val - f.u[mesh.u_idx(i, 1)];
            // Calcul et assignation numérique
            } else if (bc.bottom.type == BCType::FREE_SLIP) {
                // Évaluation de la condition de contrôle
                if (!homogeneous) f.v[mesh.v_idx(i, 1)] = 0.0;
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(i, 0)] = 0.0;
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(i, 0)] = f.u[mesh.u_idx(i, 1)];
            // Calcul et assignation numérique
            } else if (bc.bottom.type == BCType::NEUMANN) {
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(i, 0)] = f.u[mesh.u_idx(i, 1)];
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(i, 0)] = f.v[mesh.v_idx(i, 1)];
            }
            // Calcul de la résolution géométrique de cellule
            f.p[mesh.cell_idx(i, 0)] = f.p[mesh.cell_idx(i, 1)];
            // Calcul de la résolution géométrique de cellule
            f.C[mesh.cell_idx(i, 0)] = f.C[mesh.cell_idx(i, 1)];
        }
    }

    // 4. Bord supérieur physique (y = Ly)
    if (mesh.is_boundary_top) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Évaluation de la condition de contrôle
            if (bc.top.type == BCType::DIRICHLET) {
                // Évaluation de la condition de contrôle
                if (!homogeneous) f.v[mesh.v_idx(i, ny + 1)] = v_top_val;
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(i, ny + 2)] = v_top_val;
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(i, ny + 1)] = 2.0 * u_top_val - f.u[mesh.u_idx(i, ny)];
            // Calcul et assignation numérique
            } else if (bc.top.type == BCType::FREE_SLIP) {
                // Évaluation de la condition de contrôle
                if (!homogeneous) f.v[mesh.v_idx(i, ny + 1)] = 0.0;
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(i, ny + 2)] = 0.0;
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(i, ny + 1)] = f.u[mesh.u_idx(i, ny)];
            // Calcul et assignation numérique
            } else if (bc.top.type == BCType::NEUMANN) {
                // Calcul de la résolution géométrique de cellule
                f.u[mesh.u_idx(i, ny + 1)] = f.u[mesh.u_idx(i, ny)];
                // Calcul de la résolution géométrique de cellule
                f.v[mesh.v_idx(i, ny + 2)] = f.v[mesh.v_idx(i, ny + 1)];
            }
            // Calcul de la résolution géométrique de cellule
            f.p[mesh.cell_idx(i, ny + 1)] = f.p[mesh.cell_idx(i, ny)];
            // Calcul de la résolution géométrique de cellule
            f.C[mesh.cell_idx(i, ny + 1)] = f.C[mesh.cell_idx(i, ny)];
        }
    }
}

void MpiComm::apply_thermal_boundary_conditions(const Mesh2D& mesh, const DomainBC& bc, FlowFields& f) {
    // Déclaration de variable typée
    int nx = mesh.nx_local;
    // Déclaration de variable typée
    int ny = mesh.ny_local;

    // Périodicité si 1 seul processeur dans la direction
    if (bc.left.T_type == BCType::PERIODIC && mesh.dims[0] == 1) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) {
            // Calcul de la résolution géométrique de cellule
            f.T[mesh.cell_idx(0, j)]      = f.T[mesh.cell_idx(nx, j)];
            // Calcul de la résolution géométrique de cellule
            f.T[mesh.cell_idx(nx + 1, j)] = f.T[mesh.cell_idx(1, j)];
        }
    }
    // Évaluation de la condition de contrôle
    if (bc.bottom.T_type == BCType::PERIODIC && mesh.dims[1] == 1) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Calcul de la résolution géométrique de cellule
            f.T[mesh.cell_idx(i, 0)]      = f.T[mesh.cell_idx(i, ny)];
            // Calcul de la résolution géométrique de cellule
            f.T[mesh.cell_idx(i, ny + 1)] = f.T[mesh.cell_idx(i, 1)];
        }
    }

    // Bord gauche physique (x = 0)
    if (mesh.is_boundary_left) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) {
            // Évaluation de la condition de contrôle
            if (bc.left.T_type == BCType::DIRICHLET) {
                // Calcul de la résolution géométrique de cellule
                f.T[mesh.cell_idx(0, j)] = 2.0 * bc.left.T_val - f.T[mesh.cell_idx(1, j)];
            } else { // NEUMANN / Adiabatique
                // Calcul de la résolution géométrique de cellule
                f.T[mesh.cell_idx(0, j)] = f.T[mesh.cell_idx(1, j)];
            }
        }
    }

    // Bord droit physique (x = Lx)
    if (mesh.is_boundary_right) {
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= ny; ++j) {
            // Évaluation de la condition de contrôle
            if (bc.right.T_type == BCType::DIRICHLET) {
                // Calcul de la résolution géométrique de cellule
                f.T[mesh.cell_idx(nx + 1, j)] = 2.0 * bc.right.T_val - f.T[mesh.cell_idx(nx, j)];
            } else { // NEUMANN
                // Calcul de la résolution géométrique de cellule
                f.T[mesh.cell_idx(nx + 1, j)] = f.T[mesh.cell_idx(nx, j)];
            }
        }
    }

    // Bord inférieur physique (y = 0)
    if (mesh.is_boundary_bottom) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Évaluation de la condition de contrôle
            if (bc.bottom.T_type == BCType::DIRICHLET) {
                // Calcul de la résolution géométrique de cellule
                f.T[mesh.cell_idx(i, 0)] = 2.0 * bc.bottom.T_val - f.T[mesh.cell_idx(i, 1)];
            } else { // NEUMANN / Adiabatique
                // Calcul de la résolution géométrique de cellule
                f.T[mesh.cell_idx(i, 0)] = f.T[mesh.cell_idx(i, 1)];
            }
        }
    }

    // Bord supérieur physique (y = Ly)
    if (mesh.is_boundary_top) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Évaluation de la condition de contrôle
            if (bc.top.T_type == BCType::DIRICHLET) {
                // Calcul de la résolution géométrique de cellule
                f.T[mesh.cell_idx(i, ny + 1)] = 2.0 * bc.top.T_val - f.T[mesh.cell_idx(i, ny)];
            } else { // NEUMANN / Adiabatique
                // Calcul de la résolution géométrique de cellule
                f.T[mesh.cell_idx(i, ny + 1)] = f.T[mesh.cell_idx(i, ny)];
            }
        }
    }
}

// Déclaration de variable typée
double MpiComm::global_dot_product(const Mesh2D& mesh, const std::vector<double>& a, const std::vector<double>& b, int size) {
    // Déclaration de variable typée
    double local_sum = 0.0;
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < size; ++i) {
        // Calcul et assignation numérique
        local_sum += a[i] * b[i];
    }
    // Déclaration de variable typée
    double global_sum = 0.0;
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mesh.comm);
    // Retour de fonction
    return global_sum;
}

// Déclaration de variable typée
double MpiComm::global_norm2(const Mesh2D& mesh, const std::vector<double>& v, int size) {
    // Retour de fonction
    return std::sqrt(global_dot_product(mesh, v, v, size));
}

// Déclaration de variable typée
double MpiComm::global_max(const Mesh2D& mesh, double local_val) {
    // Déclaration de variable typée
    double global_val = 0.0;
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MAX, mesh.comm);
    // Retour de fonction
    return global_val;
}

// Déclaration de variable typée
double MpiComm::global_sum(const Mesh2D& mesh, double local_val) {
    // Déclaration de variable typée
    double global_val = 0.0;
    // Appel de communication ou synchronisation parallèle OpenMPI
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, mesh.comm);
    // Retour de fonction
    return global_val;
}

} // namespace covof
