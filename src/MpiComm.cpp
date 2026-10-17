#include "fugu/MpiComm.hpp"
#include <algorithm>
#include <cmath>

namespace fugu {

void MpiComm::sync_cell_scalar(const Mesh2D& mesh, std::vector<double>& field) {
    if (mesh.num_procs == 1) return;

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;

    std::vector<double> send_left(ny), recv_left(ny);
    std::vector<double> send_right(ny), recv_right(ny);
    std::vector<double> send_bot(nx), recv_bot(nx);
    std::vector<double> send_top(nx), recv_top(nx);

    // Préparation des buffers X
    for (int j = 1; j <= ny; ++j) {
        send_left[j - 1]  = field[mesh.cell_idx(1, j)];
        send_right[j - 1] = field[mesh.cell_idx(nx, j)];
    }

    MPI_Request reqs[8];
    int n_reqs = 0;

    // Échanges Est-Ouest
    if (mesh.rank_left != MPI_PROC_NULL) {
        MPI_Irecv(recv_left.data(), ny, MPI_DOUBLE, mesh.rank_left, 10, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_left.data(), ny, MPI_DOUBLE, mesh.rank_left, 11, mesh.comm, &reqs[n_reqs++]);
    }
    if (mesh.rank_right != MPI_PROC_NULL) {
        MPI_Irecv(recv_right.data(), ny, MPI_DOUBLE, mesh.rank_right, 11, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_right.data(), ny, MPI_DOUBLE, mesh.rank_right, 10, mesh.comm, &reqs[n_reqs++]);
    }

    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    n_reqs = 0;

    if (mesh.rank_left != MPI_PROC_NULL) {
        for (int j = 1; j <= ny; ++j) field[mesh.cell_idx(0, j)] = recv_left[j - 1];
    }
    if (mesh.rank_right != MPI_PROC_NULL) {
        for (int j = 1; j <= ny; ++j) field[mesh.cell_idx(nx + 1, j)] = recv_right[j - 1];
    }

    // Préparation des buffers Y
    for (int i = 1; i <= nx; ++i) {
        send_bot[i - 1] = field[mesh.cell_idx(i, 1)];
        send_top[i - 1] = field[mesh.cell_idx(i, ny)];
    }

    // Échanges Nord-Sud
    if (mesh.rank_bottom != MPI_PROC_NULL) {
        MPI_Irecv(recv_bot.data(), nx, MPI_DOUBLE, mesh.rank_bottom, 20, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_bot.data(), nx, MPI_DOUBLE, mesh.rank_bottom, 21, mesh.comm, &reqs[n_reqs++]);
    }
    if (mesh.rank_top != MPI_PROC_NULL) {
        MPI_Irecv(recv_top.data(), nx, MPI_DOUBLE, mesh.rank_top, 21, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_top.data(), nx, MPI_DOUBLE, mesh.rank_top, 20, mesh.comm, &reqs[n_reqs++]);
    }

    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    if (mesh.rank_bottom != MPI_PROC_NULL) {
        for (int i = 1; i <= nx; ++i) field[mesh.cell_idx(i, 0)] = recv_bot[i - 1];
    }
    if (mesh.rank_top != MPI_PROC_NULL) {
        for (int i = 1; i <= nx; ++i) field[mesh.cell_idx(i, ny + 1)] = recv_top[i - 1];
    }
}

void MpiComm::sync_u(const Mesh2D& mesh, std::vector<double>& u) {
    if (mesh.num_procs == 1) return;

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;

    // u est défini sur les faces verticales i de 1 à nx+1 pour l'intérieur
    std::vector<double> send_left(ny), recv_left(ny);
    std::vector<double> send_right(ny), recv_right(ny);
    std::vector<double> send_bot(nx + 2), recv_bot(nx + 2);
    std::vector<double> send_top(nx + 2), recv_top(nx + 2);

    for (int j = 1; j <= ny; ++j) {
        send_left[j - 1]  = u[mesh.u_idx(1, j)];
        send_right[j - 1] = u[mesh.u_idx(nx, j)];
    }

    MPI_Request reqs[8];
    int n_reqs = 0;

    if (mesh.rank_left != MPI_PROC_NULL) {
        MPI_Irecv(recv_left.data(), ny, MPI_DOUBLE, mesh.rank_left, 30, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_left.data(), ny, MPI_DOUBLE, mesh.rank_left, 31, mesh.comm, &reqs[n_reqs++]);
    }
    if (mesh.rank_right != MPI_PROC_NULL) {
        MPI_Irecv(recv_right.data(), ny, MPI_DOUBLE, mesh.rank_right, 31, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_right.data(), ny, MPI_DOUBLE, mesh.rank_right, 30, mesh.comm, &reqs[n_reqs++]);
    }

    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    n_reqs = 0;

    if (mesh.rank_left != MPI_PROC_NULL) {
        for (int j = 1; j <= ny; ++j) u[mesh.u_idx(0, j)] = recv_left[j - 1];
    }
    if (mesh.rank_right != MPI_PROC_NULL) {
        for (int j = 1; j <= ny; ++j) u[mesh.u_idx(nx + 1, j)] = recv_right[j - 1];
    }

    // Échange selon Y
    for (int i = 0; i <= nx + 1; ++i) {
        send_bot[i] = u[mesh.u_idx(i, 1)];
        send_top[i] = u[mesh.u_idx(i, ny)];
    }

    if (mesh.rank_bottom != MPI_PROC_NULL) {
        MPI_Irecv(recv_bot.data(), nx + 2, MPI_DOUBLE, mesh.rank_bottom, 40, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_bot.data(), nx + 2, MPI_DOUBLE, mesh.rank_bottom, 41, mesh.comm, &reqs[n_reqs++]);
    }
    if (mesh.rank_top != MPI_PROC_NULL) {
        MPI_Irecv(recv_top.data(), nx + 2, MPI_DOUBLE, mesh.rank_top, 41, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_top.data(), nx + 2, MPI_DOUBLE, mesh.rank_top, 40, mesh.comm, &reqs[n_reqs++]);
    }

    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    if (mesh.rank_bottom != MPI_PROC_NULL) {
        for (int i = 0; i <= nx + 1; ++i) u[mesh.u_idx(i, 0)] = recv_bot[i];
    }
    if (mesh.rank_top != MPI_PROC_NULL) {
        for (int i = 0; i <= nx + 1; ++i) u[mesh.u_idx(i, ny + 1)] = recv_top[i];
    }
}

void MpiComm::sync_v(const Mesh2D& mesh, std::vector<double>& v) {
    if (mesh.num_procs == 1) return;

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;

    // v est défini sur les faces horizontales j de 1 à ny+1 pour l'intérieur
    std::vector<double> send_left(ny + 2), recv_left(ny + 2);
    std::vector<double> send_right(ny + 2), recv_right(ny + 2);
    std::vector<double> send_bot(nx), recv_bot(nx);
    std::vector<double> send_top(nx), recv_top(nx);

    for (int j = 0; j <= ny + 1; ++j) {
        send_left[j]  = v[mesh.v_idx(1, j)];
        send_right[j] = v[mesh.v_idx(nx, j)];
    }

    MPI_Request reqs[8];
    int n_reqs = 0;

    if (mesh.rank_left != MPI_PROC_NULL) {
        MPI_Irecv(recv_left.data(), ny + 2, MPI_DOUBLE, mesh.rank_left, 50, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_left.data(), ny + 2, MPI_DOUBLE, mesh.rank_left, 51, mesh.comm, &reqs[n_reqs++]);
    }
    if (mesh.rank_right != MPI_PROC_NULL) {
        MPI_Irecv(recv_right.data(), ny + 2, MPI_DOUBLE, mesh.rank_right, 51, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_right.data(), ny + 2, MPI_DOUBLE, mesh.rank_right, 50, mesh.comm, &reqs[n_reqs++]);
    }

    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);
    n_reqs = 0;

    if (mesh.rank_left != MPI_PROC_NULL) {
        for (int j = 0; j <= ny + 1; ++j) v[mesh.v_idx(0, j)] = recv_left[j];
    }
    if (mesh.rank_right != MPI_PROC_NULL) {
        for (int j = 0; j <= ny + 1; ++j) v[mesh.v_idx(nx + 1, j)] = recv_right[j];
    }

    // Échange selon Y
    for (int i = 1; i <= nx; ++i) {
        send_bot[i - 1] = v[mesh.v_idx(i, 1)];
        send_top[i - 1] = v[mesh.v_idx(i, ny)];
    }

    if (mesh.rank_bottom != MPI_PROC_NULL) {
        MPI_Irecv(recv_bot.data(), nx, MPI_DOUBLE, mesh.rank_bottom, 60, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_bot.data(), nx, MPI_DOUBLE, mesh.rank_bottom, 61, mesh.comm, &reqs[n_reqs++]);
    }
    if (mesh.rank_top != MPI_PROC_NULL) {
        MPI_Irecv(recv_top.data(), nx, MPI_DOUBLE, mesh.rank_top, 61, mesh.comm, &reqs[n_reqs++]);
        MPI_Isend(send_top.data(), nx, MPI_DOUBLE, mesh.rank_top, 60, mesh.comm, &reqs[n_reqs++]);
    }

    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    if (mesh.rank_bottom != MPI_PROC_NULL) {
        for (int i = 1; i <= nx; ++i) v[mesh.v_idx(i, 0)] = recv_bot[i - 1];
    }
    if (mesh.rank_top != MPI_PROC_NULL) {
        for (int i = 1; i <= nx; ++i) v[mesh.v_idx(i, ny + 1)] = recv_top[i - 1];
    }
}

void MpiComm::apply_boundary_conditions(const Mesh2D& mesh, const DomainBC& bc, FlowFields& f, bool homogeneous) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;

    double u_left_val   = homogeneous ? 0.0 : bc.left.u_val;
    double v_left_val   = homogeneous ? 0.0 : bc.left.v_val;
    double u_right_val  = homogeneous ? 0.0 : bc.right.u_val;
    double v_right_val  = homogeneous ? 0.0 : bc.right.v_val;
    double u_bottom_val = homogeneous ? 0.0 : bc.bottom.u_val;
    double v_bottom_val = homogeneous ? 0.0 : bc.bottom.v_val;
    double u_top_val    = homogeneous ? 0.0 : bc.top.u_val;
    double v_top_val    = homogeneous ? 0.0 : bc.top.v_val;

    // 0. Traitement de la périodicité si non gérée par MPI (ex: 1 seul processeur dans la direction)
    if (bc.left.type == BCType::PERIODIC && mesh.dims[0] == 1) {
        for (int j = 1; j <= ny; ++j) {
            f.u[mesh.u_idx(0, j)]      = f.u[mesh.u_idx(nx, j)];
            f.u[mesh.u_idx(nx + 1, j)] = f.u[mesh.u_idx(1, j)];
            f.v[mesh.v_idx(0, j)]      = f.v[mesh.v_idx(nx, j)];
            f.v[mesh.v_idx(nx + 1, j)] = f.v[mesh.v_idx(1, j)];
            f.p[mesh.cell_idx(0, j)]      = f.p[mesh.cell_idx(nx, j)];
            f.p[mesh.cell_idx(nx + 1, j)] = f.p[mesh.cell_idx(1, j)];
            f.C[mesh.cell_idx(0, j)]      = f.C[mesh.cell_idx(nx, j)];
            f.C[mesh.cell_idx(nx + 1, j)] = f.C[mesh.cell_idx(1, j)];
        }
    }

    if (bc.bottom.type == BCType::PERIODIC && mesh.dims[1] == 1) {
        for (int i = 1; i <= nx; ++i) {
            f.v[mesh.v_idx(i, 0)]      = f.v[mesh.v_idx(i, ny)];
            f.v[mesh.v_idx(i, ny + 1)] = f.v[mesh.v_idx(i, 1)];
            f.u[mesh.u_idx(i, 0)]      = f.u[mesh.u_idx(i, ny)];
            f.u[mesh.u_idx(i, ny + 1)] = f.u[mesh.u_idx(i, 1)];
            f.p[mesh.cell_idx(i, 0)]      = f.p[mesh.cell_idx(i, ny)];
            f.p[mesh.cell_idx(i, ny + 1)] = f.p[mesh.cell_idx(i, 1)];
            f.C[mesh.cell_idx(i, 0)]      = f.C[mesh.cell_idx(i, ny)];
            f.C[mesh.cell_idx(i, ny + 1)] = f.C[mesh.cell_idx(i, 1)];
        }
    }

    // 1. Bord gauche physique (x = 0)
    if (mesh.is_boundary_left) {
        for (int j = 1; j <= ny; ++j) {
            if (bc.left.type == BCType::DIRICHLET) {
                double u_inlet = u_left_val;
                if (!homogeneous) {
                    if (bc.left.profile == "POISEUILLE_STEP") {
                        double y = mesh.yc(j);
                        if (y < bc.left.y_step) {
                            u_inlet = 0.0;
                        } else {
                            double h = mesh.ly - bc.left.y_step;
                            u_inlet = 6.0 * bc.left.u_mean * (y - bc.left.y_step) * (mesh.ly - y) / (h * h);
                        }
                    } else if (bc.left.profile == "PARABOLIC") {
                        double y = mesh.yc(j);
                        double h = mesh.ly;
                        u_inlet = 6.0 * bc.left.u_mean * y * (h - y) / (h * h);
                    }
                    f.u[mesh.u_idx(1, j)] = u_inlet;
                }
                f.u[mesh.u_idx(0, j)] = u_inlet;
                f.v[mesh.v_idx(0, j)] = 2.0 * v_left_val - f.v[mesh.v_idx(1, j)];
            } else if (bc.left.type == BCType::FREE_SLIP) {
                if (!homogeneous) f.u[mesh.u_idx(1, j)] = 0.0;
                f.u[mesh.u_idx(0, j)] = 0.0;
                f.v[mesh.v_idx(0, j)] = f.v[mesh.v_idx(1, j)];
            } else if (bc.left.type == BCType::NEUMANN) {
                f.u[mesh.u_idx(0, j)] = f.u[mesh.u_idx(1, j)];
                f.v[mesh.v_idx(0, j)] = f.v[mesh.v_idx(1, j)];
            }
            f.p[mesh.cell_idx(0, j)] = f.p[mesh.cell_idx(1, j)];
            f.C[mesh.cell_idx(0, j)] = f.C[mesh.cell_idx(1, j)];
        }
    }

    // 2. Bord droit physique (x = Lx)
    if (mesh.is_boundary_right) {
        for (int j = 1; j <= ny; ++j) {
            if (bc.right.type == BCType::DIRICHLET) {
                if (!homogeneous) f.u[mesh.u_idx(nx + 1, j)] = u_right_val;
                f.u[mesh.u_idx(nx + 2, j)] = u_right_val;
                f.v[mesh.v_idx(nx + 1, j)] = 2.0 * v_right_val - f.v[mesh.v_idx(nx, j)];
            } else if (bc.right.type == BCType::FREE_SLIP) {
                if (!homogeneous) f.u[mesh.u_idx(nx + 1, j)] = 0.0;
                f.u[mesh.u_idx(nx + 2, j)] = 0.0;
                f.v[mesh.v_idx(nx + 1, j)] = f.v[mesh.v_idx(nx, j)];
            } else if (bc.right.type == BCType::NEUMANN) {
                f.u[mesh.u_idx(nx + 2, j)] = f.u[mesh.u_idx(nx + 1, j)];
                f.v[mesh.v_idx(nx + 1, j)] = f.v[mesh.v_idx(nx, j)];
            }
            f.p[mesh.cell_idx(nx + 1, j)] = f.p[mesh.cell_idx(nx, j)];
            f.C[mesh.cell_idx(nx + 1, j)] = f.C[mesh.cell_idx(nx, j)];
        }
    }

    // 3. Bord inférieur physique (y = 0)
    if (mesh.is_boundary_bottom) {
        for (int i = 1; i <= nx; ++i) {
            if (bc.bottom.type == BCType::DIRICHLET) {
                if (!homogeneous) f.v[mesh.v_idx(i, 1)] = v_bottom_val;
                f.v[mesh.v_idx(i, 0)] = v_bottom_val;
                f.u[mesh.u_idx(i, 0)] = 2.0 * u_bottom_val - f.u[mesh.u_idx(i, 1)];
            } else if (bc.bottom.type == BCType::FREE_SLIP) {
                if (!homogeneous) f.v[mesh.v_idx(i, 1)] = 0.0;
                f.v[mesh.v_idx(i, 0)] = 0.0;
                f.u[mesh.u_idx(i, 0)] = f.u[mesh.u_idx(i, 1)];
            } else if (bc.bottom.type == BCType::NEUMANN) {
                f.u[mesh.u_idx(i, 0)] = f.u[mesh.u_idx(i, 1)];
                f.v[mesh.v_idx(i, 0)] = f.v[mesh.v_idx(i, 1)];
            }
            f.p[mesh.cell_idx(i, 0)] = f.p[mesh.cell_idx(i, 1)];
            f.C[mesh.cell_idx(i, 0)] = f.C[mesh.cell_idx(i, 1)];
        }
    }

    // 4. Bord supérieur physique (y = Ly)
    if (mesh.is_boundary_top) {
        for (int i = 1; i <= nx; ++i) {
            if (bc.top.type == BCType::DIRICHLET) {
                if (!homogeneous) f.v[mesh.v_idx(i, ny + 1)] = v_top_val;
                f.v[mesh.v_idx(i, ny + 2)] = v_top_val;
                f.u[mesh.u_idx(i, ny + 1)] = 2.0 * u_top_val - f.u[mesh.u_idx(i, ny)];
            } else if (bc.top.type == BCType::FREE_SLIP) {
                if (!homogeneous) f.v[mesh.v_idx(i, ny + 1)] = 0.0;
                f.v[mesh.v_idx(i, ny + 2)] = 0.0;
                f.u[mesh.u_idx(i, ny + 1)] = f.u[mesh.u_idx(i, ny)];
            } else if (bc.top.type == BCType::NEUMANN) {
                f.u[mesh.u_idx(i, ny + 1)] = f.u[mesh.u_idx(i, ny)];
                f.v[mesh.v_idx(i, ny + 2)] = f.v[mesh.v_idx(i, ny + 1)];
            }
            f.p[mesh.cell_idx(i, ny + 1)] = f.p[mesh.cell_idx(i, ny)];
            f.C[mesh.cell_idx(i, ny + 1)] = f.C[mesh.cell_idx(i, ny)];
        }
    }
}

void MpiComm::apply_thermal_boundary_conditions(const Mesh2D& mesh, const DomainBC& bc, FlowFields& f) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;

    // Périodicité si 1 seul processeur dans la direction
    if (bc.left.T_type == BCType::PERIODIC && mesh.dims[0] == 1) {
        for (int j = 1; j <= ny; ++j) {
            f.T[mesh.cell_idx(0, j)]      = f.T[mesh.cell_idx(nx, j)];
            f.T[mesh.cell_idx(nx + 1, j)] = f.T[mesh.cell_idx(1, j)];
        }
    }
    if (bc.bottom.T_type == BCType::PERIODIC && mesh.dims[1] == 1) {
        for (int i = 1; i <= nx; ++i) {
            f.T[mesh.cell_idx(i, 0)]      = f.T[mesh.cell_idx(i, ny)];
            f.T[mesh.cell_idx(i, ny + 1)] = f.T[mesh.cell_idx(i, 1)];
        }
    }

    // Bord gauche physique (x = 0)
    if (mesh.is_boundary_left) {
        for (int j = 1; j <= ny; ++j) {
            if (bc.left.T_type == BCType::DIRICHLET) {
                f.T[mesh.cell_idx(0, j)] = 2.0 * bc.left.T_val - f.T[mesh.cell_idx(1, j)];
            } else { // NEUMANN / Adiabatique
                f.T[mesh.cell_idx(0, j)] = f.T[mesh.cell_idx(1, j)];
            }
        }
    }

    // Bord droit physique (x = Lx)
    if (mesh.is_boundary_right) {
        for (int j = 1; j <= ny; ++j) {
            if (bc.right.T_type == BCType::DIRICHLET) {
                f.T[mesh.cell_idx(nx + 1, j)] = 2.0 * bc.right.T_val - f.T[mesh.cell_idx(nx, j)];
            } else { // NEUMANN
                f.T[mesh.cell_idx(nx + 1, j)] = f.T[mesh.cell_idx(nx, j)];
            }
        }
    }

    // Bord inférieur physique (y = 0)
    if (mesh.is_boundary_bottom) {
        for (int i = 1; i <= nx; ++i) {
            if (bc.bottom.T_type == BCType::DIRICHLET) {
                f.T[mesh.cell_idx(i, 0)] = 2.0 * bc.bottom.T_val - f.T[mesh.cell_idx(i, 1)];
            } else { // NEUMANN / Adiabatique
                f.T[mesh.cell_idx(i, 0)] = f.T[mesh.cell_idx(i, 1)];
            }
        }
    }

    // Bord supérieur physique (y = Ly)
    if (mesh.is_boundary_top) {
        for (int i = 1; i <= nx; ++i) {
            if (bc.top.T_type == BCType::DIRICHLET) {
                f.T[mesh.cell_idx(i, ny + 1)] = 2.0 * bc.top.T_val - f.T[mesh.cell_idx(i, ny)];
            } else { // NEUMANN / Adiabatique
                f.T[mesh.cell_idx(i, ny + 1)] = f.T[mesh.cell_idx(i, ny)];
            }
        }
    }
}

double MpiComm::global_dot_product(const Mesh2D& mesh, const std::vector<double>& a, const std::vector<double>& b, int size) {
    double local_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        local_sum += a[i] * b[i];
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mesh.comm);
    return global_sum;
}

double MpiComm::global_norm2(const Mesh2D& mesh, const std::vector<double>& v, int size) {
    return std::sqrt(global_dot_product(mesh, v, v, size));
}

double MpiComm::global_max(const Mesh2D& mesh, double local_val) {
    double global_val = 0.0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MAX, mesh.comm);
    return global_val;
}

double MpiComm::global_sum(const Mesh2D& mesh, double local_val) {
    double global_val = 0.0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, mesh.comm);
    return global_val;
}

} // namespace fugu
