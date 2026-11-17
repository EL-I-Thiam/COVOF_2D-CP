#include "covof/InterfaceMarkers.hpp"
#include "covof/MpiComm.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace covof {

// =============================================================================
// 1. INITIALISATION DES MARQUEURS SUR LES SEGMENTS PLIC DES CELLULES MIXTES
// =============================================================================

void InterfaceMarkers::init_from_vof(const Mesh2D& mesh, const VofConfig& vof_cfg,
                                     const InitialConditionConfig& ic,
                                     const DomainBC& bc, FlowFields& fields) {
    cloud.clear();
    if (!vof_cfg.use_markers) return;

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    int n_per_cell = std::max(2, vof_cfg.markers_per_cell);
    cloud.reserve(static_cast<size_t>(4 * std::max(nx, ny) * n_per_cell));

    ReconstructionMethod recon_method = InterfaceReconstruction::parse_reconstruction_method(vof_cfg.reconstruction_method);
    if (vof_cfg.reconstruction_method == "NONE" || vof_cfg.reconstruction_method == "SLIC") {
        recon_method = ReconstructionMethod::ELVIRA;
    }

    std::vector<InterfaceSegment> segments((nx + 2) * (ny + 2));
    InterfaceReconstruction::reconstruct(recon_method, mesh, fields.C, segments);

    bool is_circular_ic = (ic.type == InitialConditionType::DROPLET ||
                           ic.type == InitialConditionType::BUBBLE ||
                           ic.type == InitialConditionType::CYLINDER ||
                           ic.type == InitialConditionType::VORTEX_SHEAR) &&
                          (ic.radius > 0.0);

    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int idx = mesh.cell_idx(i, j);
            double c = fields.C[idx];
            if (c <= 1.0e-2 || c >= 1.0 - 1.0e-2) continue;

            double xmin = mesh.xv(i);
            double xmax = mesh.xv(i + 1);
            double ymin = mesh.yv(j);
            double ymax = mesh.yv(j + 1);

            const auto& seg = segments[idx];
            double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
            if (InterfaceReconstruction::clip_segment_in_cell(xmin, xmax, ymin, ymax,
                                                              seg.nx, seg.ny, seg.alpha,
                                                              x1, y1, x2, y2)) {
                for (int m = 0; m < n_per_cell; ++m) {
                    double s = (static_cast<double>(m) + 0.5) / static_cast<double>(n_per_cell);
                    double xp = x1 + s * (x2 - x1);
                    double yp = y1 + s * (y2 - y1);

                    Marker mk;
                    if (is_circular_ic) {
                        double rx = xp - ic.center_x;
                        double ry = yp - ic.center_y;
                        double r_dist = std::hypot(rx, ry);
                        if (r_dist > 1.0e-12 && std::abs(r_dist - ic.radius) <= 0.75 * std::hypot(mesh.dx, mesh.dy)) {
                            xp = ic.center_x + ic.radius * (rx / r_dist);
                            yp = ic.center_y + ic.radius * (ry / r_dist);
                            if (xp < xmin - 1.0e-12 || xp > xmax + 1.0e-12 ||
                                yp < ymin - 1.0e-12 || yp > ymax + 1.0e-12) {
                                continue;
                            }
                            double sgn_c1 = (ic.type == InitialConditionType::DROPLET ||
                                             ic.type == InitialConditionType::VORTEX_SHEAR) ? -1.0 : 1.0;
                            mk.nx = sgn_c1 * (rx / r_dist);
                            mk.ny = sgn_c1 * (ry / r_dist);
                            mk.kappa = -sgn_c1 / ic.radius;
                        } else {
                            mk.nx = -seg.nx;
                            mk.ny = -seg.ny;
                            mk.kappa = 0.0;
                        }
                    } else {
                        // Convention : seg.nx pointe le long de -grad C ; marker.nx pointe le long de +grad C
                        mk.nx = -seg.nx;
                        mk.ny = -seg.ny;
                        mk.kappa = 0.0;
                    }

                    mk.x = xp;
                    mk.y = yp;
                    mk.cell_i = i;
                    mk.cell_j = j;
                    mk.active = true;
                    cloud.push_back(mk);
                }
            }
        }
    }

    compute_curvature_from_markers(mesh, vof_cfg, bc, fields);
}

// =============================================================================
// 2. INTERPOLATION CONSERVATRICE DE LA VITESSE (CVI) SUR GRILLE MAC
//    Variante B exacte (Wang et al., 2015 ; Pusok et al., 2017) :
//    - Interpole u sur sa boîte décalée [x_i^f, x_{i+1}^f] x [y_{j_u}^c, y_{j_u+1}^c]
//      avec j_u in {j-1, j} ;
//    - Interpole v sur sa boîte décalée [x_{i_v}^c, x_{i_v+1}^c] x [y_j^f, y_{j+1}^f]
//      avec i_v in {i-1, i} ;
//    - Garantit analytiquement dans les 4 sous-quadrants de (i, j) :
//      du_CVI/dx + dv_CVI/dy = (u(i+1,j)-u(i,j))/dx + (v(i,j+1)-v(i,j))/dy = D_{i,j}
//      en restant strictement dans le halo unique (0:nx+1, 0:ny+1).
// =============================================================================

void InterfaceMarkers::interpolate_velocity_cvi(const Mesh2D& mesh, const FlowFields& fields,
                                                double xp, double yp, double& up, double& vp) const {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    double x_loc_origin = mesh.xu(1);
    double y_loc_origin = mesh.yv(1);

    int i = static_cast<int>(std::floor((xp - x_loc_origin) / dx)) + 1;
    int j = static_cast<int>(std::floor((yp - y_loc_origin) / dy)) + 1;

    i = std::clamp(i, 1, nx);
    j = std::clamp(j, 1, ny);

    // Indices des boîtes décalées MAC (j_u in {j-1, j} subset [0, ny], i_v in {i-1, i} subset [0, nx])
    int j_u = (yp < mesh.yc(j)) ? (j - 1) : j;
    int i_v = (xp < mesh.xc(i)) ? (i - 1) : i;

    // Coordonnées réduites locales dans chacune des boîtes décalées
    double xi_u  = std::clamp((xp - mesh.xu(i))   / dx, 0.0, 1.0);
    double eta_u = std::clamp((yp - mesh.yc(j_u)) / dy, 0.0, 1.0);
    double xi_v  = std::clamp((xp - mesh.xc(i_v)) / dx, 0.0, 1.0);
    double eta_v = std::clamp((yp - mesh.yv(j))   / dy, 0.0, 1.0);

    // Sommets MAC de la boîte décalée u : [x_i^f, x_{i+1}^f] x [y_{j_u}^c, y_{j_u+1}^c]
    double u_sw = fields.u[mesh.u_idx(i,     j_u    )];
    double u_se = fields.u[mesh.u_idx(i + 1, j_u    )];
    double u_nw = fields.u[mesh.u_idx(i,     j_u + 1)];
    double u_ne = fields.u[mesh.u_idx(i + 1, j_u + 1)];

    // Sommets MAC de la boîte décalée v : [x_{i_v}^c, x_{i_v+1}^c] x [y_j^f, y_{j+1}^f]
    double v_sw = fields.v[mesh.v_idx(i_v,     j    )];
    double v_se = fields.v[mesh.v_idx(i_v + 1, j    )];
    double v_nw = fields.v[mesh.v_idx(i_v,     j + 1)];
    double v_ne = fields.v[mesh.v_idx(i_v + 1, j + 1)];

    double du_s = u_se - u_sw;
    double du_n = u_ne - u_nw;
    double dv_w = v_nw - v_sw;
    double dv_e = v_ne - v_se;

    // Partie bilinéaire sur chaque boîte décalée MAC
    double u_bilin = (1.0 - eta_u) * ((1.0 - xi_u) * u_sw + xi_u * u_se)
                   +        eta_u  * ((1.0 - xi_u) * u_nw + xi_u * u_ne);
    double v_bilin = (1.0 - xi_v)  * ((1.0 - eta_v) * v_sw + eta_v * v_nw)
                   +        xi_v   * ((1.0 - eta_v) * v_se + eta_v * v_ne);

    // Correction quadratique solénoïdale CVI H(div)-conforme (Variante B exacte) :
    // garantit d(u_CVI)/dx + d(v_CVI)/dy == (u(i+1,j) - u(i,j))/dx + (v(i,j+1) - v(i,j))/dy == D_{i,j}
    double corr_u = 0.5 * (dx / dy) * xi_u  * (1.0 - xi_u)  * (dv_e - dv_w);
    double corr_v = 0.5 * (dy / dx) * eta_v * (1.0 - eta_v) * (du_n - du_s);

    up = u_bilin + corr_u;
    vp = v_bilin + corr_v;
}

void InterfaceMarkers::interpolate_velocity_bilinear(const Mesh2D& mesh, const FlowFields& fields,
                                                     double xp, double yp, double& up, double& vp) const {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    int i = static_cast<int>(std::floor((xp - mesh.xu(1)) / dx)) + 1;
    int j = static_cast<int>(std::floor((yp - mesh.yv(1)) / dy)) + 1;
    i = std::clamp(i, 1, nx);
    j = std::clamp(j, 1, ny);

    int j_u = (yp < mesh.yc(j)) ? (j - 1) : j;
    int i_v = (xp < mesh.xc(i)) ? (i - 1) : i;

    double xi_u  = std::clamp((xp - mesh.xu(i))   / dx, 0.0, 1.0);
    double eta_u = std::clamp((yp - mesh.yc(j_u)) / dy, 0.0, 1.0);
    double xi_v  = std::clamp((xp - mesh.xc(i_v)) / dx, 0.0, 1.0);
    double eta_v = std::clamp((yp - mesh.yv(j))   / dy, 0.0, 1.0);

    double u_sw = fields.u[mesh.u_idx(i,     j_u    )];
    double u_se = fields.u[mesh.u_idx(i + 1, j_u    )];
    double u_nw = fields.u[mesh.u_idx(i,     j_u + 1)];
    double u_ne = fields.u[mesh.u_idx(i + 1, j_u + 1)];

    double v_sw = fields.v[mesh.v_idx(i_v,     j    )];
    double v_se = fields.v[mesh.v_idx(i_v + 1, j    )];
    double v_nw = fields.v[mesh.v_idx(i_v,     j + 1)];
    double v_ne = fields.v[mesh.v_idx(i_v + 1, j + 1)];

    up = (1.0 - eta_u) * ((1.0 - xi_u) * u_sw + xi_u * u_se)
       +        eta_u  * ((1.0 - xi_u) * u_nw + xi_u * u_ne);
    vp = (1.0 - xi_v)  * ((1.0 - eta_v) * v_sw + eta_v * v_nw)
       +        xi_v   * ((1.0 - eta_v) * v_se + eta_v * v_ne);
}

// =============================================================================
// 3. ADVECTION LAGRANGIENNE RK2 EXPLICITE (HEUN D'ORDRE 2)
// =============================================================================

void InterfaceMarkers::advect_rk2(const Mesh2D& mesh, const VofConfig& vof_cfg,
                                  const FlowFields& fields, double dt) {
    if (!vof_cfg.use_markers || cloud.empty()) return;

    bool use_bilinear = (vof_cfg.marker_advection_scheme == "BILINEAR_RK2" ||
                         vof_cfg.marker_advection_scheme == "BILINEAR");

    for (auto& mk : cloud) {
        if (!mk.active) continue;
        double x0 = mk.x;
        double y0 = mk.y;
        double u1 = 0.0, v1 = 0.0;
        double u2 = 0.0, v2 = 0.0;

        if (use_bilinear) {
            interpolate_velocity_bilinear(mesh, fields, x0, y0, u1, v1);
            double x_star = x0 + dt * u1;
            double y_star = y0 + dt * v1;
            interpolate_velocity_bilinear(mesh, fields, x_star, y_star, u2, v2);
        } else {
            interpolate_velocity_cvi(mesh, fields, x0, y0, u1, v1);
            double x_star = x0 + dt * u1;
            double y_star = y0 + dt * v1;
            interpolate_velocity_cvi(mesh, fields, x_star, y_star, u2, v2);
        }

        mk.x = x0 + 0.5 * dt * (u1 + u2);
        mk.y = y0 + 0.5 * dt * (v1 + v2);
    }
}

// =============================================================================
// 4. MIGRATION PARALLÈLE MPI 2 PASSES (X PUIS Y) ET CONDITIONS PÉRIODIQUES
// =============================================================================

void InterfaceMarkers::migrate_mpi(const Mesh2D& mesh, const VofConfig& vof_cfg, const DomainBC& bc) {
    if (!vof_cfg.use_markers) return;

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    double x_min_loc = mesh.xu(1);
    double x_max_loc = mesh.xu(nx + 1);
    double y_min_loc = mesh.yv(1);
    double y_max_loc = mesh.yv(ny + 1);

    bool per_x = (bc.left.type == BCType::PERIODIC || bc.right.type == BCType::PERIODIC);
    bool per_y = (bc.bottom.type == BCType::PERIODIC || bc.top.type == BCType::PERIODIC);
    const double eps_b = 1.0e-12;

    // Repliement périodique global ou réflexion sur parois physiques
    for (auto& mk : cloud) {
        if (!mk.active) continue;
        if (per_x) {
            if (mk.x < 0.0)      mk.x += mesh.lx;
            if (mk.x >= mesh.lx) mk.x -= mesh.lx;
        } else {
            if (mesh.is_boundary_left  && mk.x < 0.0)      mk.x = eps_b * dx;
            if (mesh.is_boundary_right && mk.x >= mesh.lx) mk.x = mesh.lx - eps_b * dx;
        }
        if (per_y) {
            if (mk.y < 0.0)      mk.y += mesh.ly;
            if (mk.y >= mesh.ly) mk.y -= mesh.ly;
        } else {
            if (mesh.is_boundary_bottom && mk.y < 0.0)      mk.y = eps_b * dy;
            if (mesh.is_boundary_top    && mk.y >= mesh.ly) mk.y = mesh.ly - eps_b * dy;
        }
    }

    if (mesh.num_procs > 1) {
        // --- PASSE 1 : ÉCHANGE SELON X (GAUCHE / DROITE) ---
        std::vector<double> send_L, send_R;
        std::vector<Marker> stay_x;
        stay_x.reserve(cloud.size());

        for (auto& mk : cloud) {
            if (!mk.active) continue;
            if (mk.x < x_min_loc && mesh.rank_left != MPI_PROC_NULL) {
                send_L.insert(send_L.end(), {mk.x, mk.y, mk.nx, mk.ny, mk.kappa});
            } else if (mk.x >= x_max_loc && mesh.rank_right != MPI_PROC_NULL) {
                send_R.insert(send_R.end(), {mk.x, mk.y, mk.nx, mk.ny, mk.kappa});
            } else {
                stay_x.push_back(mk);
            }
        }

        int n_send_L = static_cast<int>(send_L.size() / 5);
        int n_send_R = static_cast<int>(send_R.size() / 5);
        int n_recv_L = 0, n_recv_R = 0;

        MPI_Sendrecv(&n_send_L, 1, MPI_INT, mesh.rank_left,  401,
                     &n_recv_R, 1, MPI_INT, mesh.rank_right, 401,
                     mesh.comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&n_send_R, 1, MPI_INT, mesh.rank_right, 402,
                     &n_recv_L, 1, MPI_INT, mesh.rank_left,  402,
                     mesh.comm, MPI_STATUS_IGNORE);

        std::vector<double> recv_L(static_cast<size_t>(std::max(1, 5 * n_recv_L)), 0.0);
        std::vector<double> recv_R(static_cast<size_t>(std::max(1, 5 * n_recv_R)), 0.0);
        if (send_L.empty()) send_L.push_back(0.0);
        if (send_R.empty()) send_R.push_back(0.0);

        MPI_Sendrecv(send_L.data(), 5 * n_send_L, MPI_DOUBLE, mesh.rank_left,  403,
                     recv_R.data(), 5 * n_recv_R, MPI_DOUBLE, mesh.rank_right, 403,
                     mesh.comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_R.data(), 5 * n_send_R, MPI_DOUBLE, mesh.rank_right, 404,
                     recv_L.data(), 5 * n_recv_L, MPI_DOUBLE, mesh.rank_left,  404,
                     mesh.comm, MPI_STATUS_IGNORE);

        for (int k = 0; k < n_recv_L; ++k) {
            Marker mk;
            mk.x = recv_L[5 * k];
            mk.y = recv_L[5 * k + 1];
            mk.nx = recv_L[5 * k + 2];
            mk.ny = recv_L[5 * k + 3];
            mk.kappa = recv_L[5 * k + 4];
            mk.active = true;
            stay_x.push_back(mk);
        }
        for (int k = 0; k < n_recv_R; ++k) {
            Marker mk;
            mk.x = recv_R[5 * k];
            mk.y = recv_R[5 * k + 1];
            mk.nx = recv_R[5 * k + 2];
            mk.ny = recv_R[5 * k + 3];
            mk.kappa = recv_R[5 * k + 4];
            mk.active = true;
            stay_x.push_back(mk);
        }

        // --- PASSE 2 : ÉCHANGE SELON Y (BAS / HAUT) ---
        std::vector<double> send_B, send_T;
        cloud.clear();
        cloud.reserve(stay_x.size());

        for (auto& mk : stay_x) {
            if (!mk.active) continue;
            if (mk.y < y_min_loc && mesh.rank_bottom != MPI_PROC_NULL) {
                send_B.insert(send_B.end(), {mk.x, mk.y, mk.nx, mk.ny, mk.kappa});
            } else if (mk.y >= y_max_loc && mesh.rank_top != MPI_PROC_NULL) {
                send_T.insert(send_T.end(), {mk.x, mk.y, mk.nx, mk.ny, mk.kappa});
            } else {
                cloud.push_back(mk);
            }
        }

        int n_send_B = static_cast<int>(send_B.size() / 5);
        int n_send_T = static_cast<int>(send_T.size() / 5);
        int n_recv_B = 0, n_recv_T = 0;

        MPI_Sendrecv(&n_send_B, 1, MPI_INT, mesh.rank_bottom, 405,
                     &n_recv_T, 1, MPI_INT, mesh.rank_top,    405,
                     mesh.comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&n_send_T, 1, MPI_INT, mesh.rank_top,    406,
                     &n_recv_B, 1, MPI_INT, mesh.rank_bottom, 406,
                     mesh.comm, MPI_STATUS_IGNORE);

        std::vector<double> recv_B(static_cast<size_t>(std::max(1, 5 * n_recv_B)), 0.0);
        std::vector<double> recv_T(static_cast<size_t>(std::max(1, 5 * n_recv_T)), 0.0);
        if (send_B.empty()) send_B.push_back(0.0);
        if (send_T.empty()) send_T.push_back(0.0);

        MPI_Sendrecv(send_B.data(), 5 * n_send_B, MPI_DOUBLE, mesh.rank_bottom, 407,
                     recv_T.data(), 5 * n_recv_T, MPI_DOUBLE, mesh.rank_top,    407,
                     mesh.comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_T.data(), 5 * n_send_T, MPI_DOUBLE, mesh.rank_top,    408,
                     recv_B.data(), 5 * n_recv_B, MPI_DOUBLE, mesh.rank_bottom, 408,
                     mesh.comm, MPI_STATUS_IGNORE);

        for (int k = 0; k < n_recv_B; ++k) {
            Marker mk;
            mk.x = recv_B[5 * k];
            mk.y = recv_B[5 * k + 1];
            mk.nx = recv_B[5 * k + 2];
            mk.ny = recv_B[5 * k + 3];
            mk.kappa = recv_B[5 * k + 4];
            mk.active = true;
            cloud.push_back(mk);
        }
        for (int k = 0; k < n_recv_T; ++k) {
            Marker mk;
            mk.x = recv_T[5 * k];
            mk.y = recv_T[5 * k + 1];
            mk.nx = recv_T[5 * k + 2];
            mk.ny = recv_T[5 * k + 3];
            mk.kappa = recv_T[5 * k + 4];
            mk.active = true;
            cloud.push_back(mk);
        }
    }

    // Mise à jour des indices de cellule locale (cell_i, cell_j)
    for (auto& mk : cloud) {
        if (!mk.active) continue;
        int i = static_cast<int>(std::floor((mk.x - x_min_loc) / dx)) + 1;
        int j = static_cast<int>(std::floor((mk.y - y_min_loc) / dy)) + 1;
        mk.cell_i = std::clamp(i, 1, nx);
        mk.cell_j = std::clamp(j, 1, ny);
    }
}

// =============================================================================
// 5. ÉLAGAGE (PRUNING) ET RÉ-ENSEMENCEMENT (RESEEDING) GUIDÉ PAR VOF C
// =============================================================================

void InterfaceMarkers::prune_and_reseed(const Mesh2D& mesh, const VofConfig& vof_cfg,
                                        const DomainBC& /*bc*/, FlowFields& fields) {
    if (!vof_cfg.use_markers) return;

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;
    int n_per_cell = std::max(2, vof_cfg.markers_per_cell);
    int max_per_cell = std::max(n_per_cell + 2, vof_cfg.marker_max_per_cell);

    // 1. Suppression des marqueurs orphelins situés loin d'une cellule mixte (voisinage 3x3 pur)
    std::vector<int> cell_counts((nx + 2) * (ny + 2), 0);
    for (auto& mk : cloud) {
        if (!mk.active) continue;
        int ci = mk.cell_i;
        int cj = mk.cell_j;
        if (ci < 1 || ci > nx || cj < 1 || cj > ny) {
            mk.active = false;
            continue;
        }
        double c_cell = fields.C[mesh.cell_idx(ci, cj)];
        if (c_cell <= 0.01 || c_cell >= 1.0 - 0.01) {
            bool near_mixed = false;
            for (int dj = -1; dj <= 1 && !near_mixed; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    double c_nb = fields.C[mesh.cell_idx(ci + di, cj + dj)];
                    if (c_nb > 0.01 && c_nb < 1.0 - 0.01) {
                        near_mixed = true;
                        break;
                    }
                }
            }
            if (!near_mixed) {
                mk.active = false;
                continue;
            }
        }
        cell_counts[mesh.cell_idx(ci, cj)]++;
    }

    // 2. Fusion par sous-quadrants 2x2 dans les cellules surpeuplées (> max_per_cell) avec projection PLIC
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            if (cell_counts[mesh.cell_idx(i, j)] <= max_per_cell) continue;

            double xc_mid = mesh.xc(i);
            double yc_mid = mesh.yc(j);
            double q_x[4] = {0.0, 0.0, 0.0, 0.0};
            double q_y[4] = {0.0, 0.0, 0.0, 0.0};
            double q_nx[4] = {0.0, 0.0, 0.0, 0.0};
            double q_ny[4] = {0.0, 0.0, 0.0, 0.0};
            double q_kap[4] = {0.0, 0.0, 0.0, 0.0};
            int q_cnt[4] = {0, 0, 0, 0};

            for (auto& mk : cloud) {
                if (!mk.active) continue;
                if (mk.cell_i == i && mk.cell_j == j) {
                    int q = ((mk.x >= xc_mid) ? 1 : 0) + ((mk.y >= yc_mid) ? 2 : 0);
                    q_x[q] += mk.x;
                    q_y[q] += mk.y;
                    q_nx[q] += mk.nx;
                    q_ny[q] += mk.ny;
                    q_kap[q] += mk.kappa;
                    q_cnt[q]++;
                    mk.active = false;
                }
            }

            cell_counts[mesh.cell_idx(i, j)] = 0;
            double c_cell = fields.C[mesh.cell_idx(i, j)];

            for (int q = 0; q < 4; ++q) {
                if (q_cnt[q] > 0) {
                    Marker mk;
                    double mx = q_x[q] / q_cnt[q];
                    double my = q_y[q] / q_cnt[q];
                    double nrm = std::hypot(q_nx[q], q_ny[q]);
                    if (nrm > 1.0e-12) {
                        mk.nx = q_nx[q] / nrm;
                        mk.ny = q_ny[q] / nrm;
                    } else {
                        mk.nx = 0.0;
                        mk.ny = 1.0;
                    }
                    mk.kappa = q_kap[q] / q_cnt[q];

                    // Projection géométrique orthogonale sur le segment PLIC pour éviter la dérive concave
                    if (c_cell > 1.0e-4 && c_cell < 1.0 - 1.0e-4) {
                        double xmin = mesh.xv(i), xmax = mesh.xv(i + 1);
                        double ymin = mesh.yv(j), ymax = mesh.yv(j + 1);
                        double snx = -mk.nx;
                        double sny = -mk.ny;
                        double alpha = InterfaceReconstruction::compute_alpha(xmin, xmax, ymin, ymax, snx, sny, c_cell);
                        double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
                        if (InterfaceReconstruction::clip_segment_in_cell(xmin, xmax, ymin, ymax, snx, sny, alpha, x1, y1, x2, y2)) {
                            double seg_dx = x2 - x1;
                            double seg_dy = y2 - y1;
                            double seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy;
                            if (seg_len2 > 1.0e-14) {
                                double t_proj = ((mx - x1) * seg_dx + (my - y1) * seg_dy) / seg_len2;
                                t_proj = std::clamp(t_proj, 0.05, 0.95);
                                mx = x1 + t_proj * seg_dx;
                                my = y1 + t_proj * seg_dy;
                            }
                        }
                    }

                    mk.x = mx;
                    mk.y = my;
                    mk.cell_i = i;
                    mk.cell_j = j;
                    mk.active = true;
                    cloud.push_back(mk);
                    cell_counts[mesh.cell_idx(i, j)]++;
                }
            }
        }
    }

    // 3. Ré-ensemencement local (reseeding) dans les cellules mixtes franches (0.05 < C < 0.95)
    //    dépourvues de marqueurs
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int idx = mesh.cell_idx(i, j);
            double c = fields.C[idx];
            if (c > 0.05 && c < 0.95 && cell_counts[idx] == 0) {
                double c_mm = fields.C[mesh.cell_idx(i - 1, j - 1)];
                double c_0m = fields.C[mesh.cell_idx(i,     j - 1)];
                double c_pm = fields.C[mesh.cell_idx(i + 1, j - 1)];
                double c_m0 = fields.C[mesh.cell_idx(i - 1, j    )];
                double c_p0 = fields.C[mesh.cell_idx(i + 1, j    )];
                double c_mp = fields.C[mesh.cell_idx(i - 1, j + 1)];
                double c_0p = fields.C[mesh.cell_idx(i,     j + 1)];
                double c_pp = fields.C[mesh.cell_idx(i + 1, j + 1)];

                double mx = -((c_pp + 2.0 * c_p0 + c_pm) - (c_mp + 2.0 * c_m0 + c_mm)) / (8.0 * dx);
                double my = -((c_pp + 2.0 * c_0p + c_mp) - (c_pm + 2.0 * c_0m + c_mm)) / (8.0 * dy);
                double nrm = std::hypot(mx, my);
                double snx = (nrm > 1.0e-12) ? (mx / nrm) : 1.0;
                double sny = (nrm > 1.0e-12) ? (my / nrm) : 0.0;

                double xmin = mesh.xv(i);
                double xmax = mesh.xv(i + 1);
                double ymin = mesh.yv(j);
                double ymax = mesh.yv(j + 1);
                double alpha = InterfaceReconstruction::compute_alpha(xmin, xmax, ymin, ymax, snx, sny, c);

                double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
                if (InterfaceReconstruction::clip_segment_in_cell(xmin, xmax, ymin, ymax,
                                                                  snx, sny, alpha,
                                                                  x1, y1, x2, y2)) {
                    for (int m = 0; m < n_per_cell; ++m) {
                        double s = (static_cast<double>(m) + 0.5) / static_cast<double>(n_per_cell);
                        Marker mk;
                        mk.x = x1 + s * (x2 - x1);
                        mk.y = y1 + s * (y2 - y1);
                        mk.nx = -snx;
                        mk.ny = -sny;
                        mk.kappa = 0.0;
                        mk.cell_i = i;
                        mk.cell_j = j;
                        mk.active = true;
                        cloud.push_back(mk);
                    }
                }
            }
        }
    }

    // 4. Compactage in-place du tableau de marqueurs
    cloud.erase(std::remove_if(cloud.begin(), cloud.end(),
                               [](const Marker& mk) { return !mk.active; }),
                cloud.end());
}

// =============================================================================
// 6. RÉGRESSION PARABOLIQUE LOCALE DE LA NORMALE ET DE LA COURBURE
// =============================================================================

void InterfaceMarkers::compute_curvature_from_markers(const Mesh2D& mesh, const VofConfig& vof_cfg,
                                                      const DomainBC& bc, FlowFields& fields) {
    std::fill(fields.marker_nx.begin(), fields.marker_nx.end(), 0.0);
    std::fill(fields.marker_ny.begin(), fields.marker_ny.end(), 0.0);
    std::fill(fields.marker_kappa.begin(), fields.marker_kappa.end(), 0.0);
    std::fill(fields.has_marker_normal.begin(), fields.has_marker_normal.end(), 0);
    std::fill(fields.has_marker_kappa.begin(), fields.has_marker_kappa.end(), 0);

    if (!vof_cfg.use_markers) return;

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;
    double h = std::max(dx, dy);
    double r_fit = 2.5 * h;

    double x_min_loc = mesh.xu(1);
    double x_max_loc = mesh.xu(nx + 1);
    double y_min_loc = mesh.yv(1);
    double y_max_loc = mesh.yv(ny + 1);

    bool per_x = (bc.left.type == BCType::PERIODIC || bc.right.type == BCType::PERIODIC);
    bool per_y = (bc.bottom.type == BCType::PERIODIC || bc.top.type == BCType::PERIODIC);

    // Construction de la liste étendue ext_cloud (marqueurs locaux + fantômes MPI/périodiques dans la bande r_fit)
    std::vector<Marker> ext_cloud = cloud;

    if (mesh.num_procs > 1) {
        // --- PASSE 1 : HALO X (GAUCHE / DROITE) ---
        std::vector<double> send_L, send_R;
        for (const auto& mk : ext_cloud) {
            if (!mk.active) continue;
            if (mk.x - x_min_loc <= r_fit && mesh.rank_left != MPI_PROC_NULL) {
                double x_s = mk.x;
                if (per_x && mesh.coords[0] == 0) x_s += mesh.lx;
                send_L.insert(send_L.end(), {x_s, mk.y, mk.nx, mk.ny, mk.kappa});
            }
            if (x_max_loc - mk.x <= r_fit && mesh.rank_right != MPI_PROC_NULL) {
                double x_s = mk.x;
                if (per_x && mesh.coords[0] == mesh.dims[0] - 1) x_s -= mesh.lx;
                send_R.insert(send_R.end(), {x_s, mk.y, mk.nx, mk.ny, mk.kappa});
            }
        }

        int n_send_L = static_cast<int>(send_L.size() / 5);
        int n_send_R = static_cast<int>(send_R.size() / 5);
        int n_recv_L = 0, n_recv_R = 0;

        MPI_Sendrecv(&n_send_L, 1, MPI_INT, mesh.rank_left,  501,
                     &n_recv_R, 1, MPI_INT, mesh.rank_right, 501,
                     mesh.comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&n_send_R, 1, MPI_INT, mesh.rank_right, 502,
                     &n_recv_L, 1, MPI_INT, mesh.rank_left,  502,
                     mesh.comm, MPI_STATUS_IGNORE);

        std::vector<double> recv_L(static_cast<size_t>(std::max(1, 5 * n_recv_L)), 0.0);
        std::vector<double> recv_R(static_cast<size_t>(std::max(1, 5 * n_recv_R)), 0.0);
        if (send_L.empty()) send_L.push_back(0.0);
        if (send_R.empty()) send_R.push_back(0.0);

        MPI_Sendrecv(send_L.data(), 5 * n_send_L, MPI_DOUBLE, mesh.rank_left,  503,
                     recv_R.data(), 5 * n_recv_R, MPI_DOUBLE, mesh.rank_right, 503,
                     mesh.comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_R.data(), 5 * n_send_R, MPI_DOUBLE, mesh.rank_right, 504,
                     recv_L.data(), 5 * n_recv_L, MPI_DOUBLE, mesh.rank_left,  504,
                     mesh.comm, MPI_STATUS_IGNORE);

        for (int k = 0; k < n_recv_L; ++k) {
            Marker mk;
            mk.x = recv_L[5 * k];
            mk.y = recv_L[5 * k + 1];
            mk.nx = recv_L[5 * k + 2];
            mk.ny = recv_L[5 * k + 3];
            mk.kappa = recv_L[5 * k + 4];
            mk.active = true;
            ext_cloud.push_back(mk);
        }
        for (int k = 0; k < n_recv_R; ++k) {
            Marker mk;
            mk.x = recv_R[5 * k];
            mk.y = recv_R[5 * k + 1];
            mk.nx = recv_R[5 * k + 2];
            mk.ny = recv_R[5 * k + 3];
            mk.kappa = recv_R[5 * k + 4];
            mk.active = true;
            ext_cloud.push_back(mk);
        }

        // --- PASSE 2 : HALO Y (BAS / HAUT, INCLUANT LES COINS REÇUS EN X) ---
        std::vector<double> send_B, send_T;
        size_t n_after_x = ext_cloud.size();
        for (size_t idx_m = 0; idx_m < n_after_x; ++idx_m) {
            const auto& mk = ext_cloud[idx_m];
            if (!mk.active) continue;
            if (mk.y - y_min_loc <= r_fit && mesh.rank_bottom != MPI_PROC_NULL) {
                double y_s = mk.y;
                if (per_y && mesh.coords[1] == 0) y_s += mesh.ly;
                send_B.insert(send_B.end(), {mk.x, y_s, mk.nx, mk.ny, mk.kappa});
            }
            if (y_max_loc - mk.y <= r_fit && mesh.rank_top != MPI_PROC_NULL) {
                double y_s = mk.y;
                if (per_y && mesh.coords[1] == mesh.dims[1] - 1) y_s -= mesh.ly;
                send_T.insert(send_T.end(), {mk.x, y_s, mk.nx, mk.ny, mk.kappa});
            }
        }

        int n_send_B = static_cast<int>(send_B.size() / 5);
        int n_send_T = static_cast<int>(send_T.size() / 5);
        int n_recv_B = 0, n_recv_T = 0;

        MPI_Sendrecv(&n_send_B, 1, MPI_INT, mesh.rank_bottom, 505,
                     &n_recv_T, 1, MPI_INT, mesh.rank_top,    505,
                     mesh.comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&n_send_T, 1, MPI_INT, mesh.rank_top,    506,
                     &n_recv_B, 1, MPI_INT, mesh.rank_bottom, 506,
                     mesh.comm, MPI_STATUS_IGNORE);

        std::vector<double> recv_B(static_cast<size_t>(std::max(1, 5 * n_recv_B)), 0.0);
        std::vector<double> recv_T(static_cast<size_t>(std::max(1, 5 * n_recv_T)), 0.0);
        if (send_B.empty()) send_B.push_back(0.0);
        if (send_T.empty()) send_T.push_back(0.0);

        MPI_Sendrecv(send_B.data(), 5 * n_send_B, MPI_DOUBLE, mesh.rank_bottom, 507,
                     recv_T.data(), 5 * n_recv_T, MPI_DOUBLE, mesh.rank_top,    507,
                     mesh.comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(send_T.data(), 5 * n_send_T, MPI_DOUBLE, mesh.rank_top,    508,
                     recv_B.data(), 5 * n_recv_B, MPI_DOUBLE, mesh.rank_bottom, 508,
                     mesh.comm, MPI_STATUS_IGNORE);

        for (int k = 0; k < n_recv_B; ++k) {
            Marker mk;
            mk.x = recv_B[5 * k];
            mk.y = recv_B[5 * k + 1];
            mk.nx = recv_B[5 * k + 2];
            mk.ny = recv_B[5 * k + 3];
            mk.kappa = recv_B[5 * k + 4];
            mk.active = true;
            ext_cloud.push_back(mk);
        }
        for (int k = 0; k < n_recv_T; ++k) {
            Marker mk;
            mk.x = recv_T[5 * k];
            mk.y = recv_T[5 * k + 1];
            mk.nx = recv_T[5 * k + 2];
            mk.ny = recv_T[5 * k + 3];
            mk.kappa = recv_T[5 * k + 4];
            mk.active = true;
            ext_cloud.push_back(mk);
        }
    }

    // Copies périodiques locales si dims[d] == 1
    if (per_x && mesh.dims[0] == 1) {
        size_t n_cur = ext_cloud.size();
        for (size_t idx_m = 0; idx_m < n_cur; ++idx_m) {
            const auto& mk = ext_cloud[idx_m];
            if (mk.x <= r_fit) {
                Marker copy_mk = mk;
                copy_mk.x += mesh.lx;
                ext_cloud.push_back(copy_mk);
            }
            if (mesh.lx - mk.x <= r_fit) {
                Marker copy_mk = mk;
                copy_mk.x -= mesh.lx;
                ext_cloud.push_back(copy_mk);
            }
        }
    }
    if (per_y && mesh.dims[1] == 1) {
        size_t n_cur = ext_cloud.size();
        for (size_t idx_m = 0; idx_m < n_cur; ++idx_m) {
            const auto& mk = ext_cloud[idx_m];
            if (mk.y <= r_fit) {
                Marker copy_mk = mk;
                copy_mk.y += mesh.ly;
                ext_cloud.push_back(copy_mk);
            }
            if (mesh.ly - mk.y <= r_fit) {
                Marker copy_mk = mk;
                copy_mk.y -= mesh.ly;
                ext_cloud.push_back(copy_mk);
            }
        }
    }

    // Indexation spatiale sur la grille étendue [-1 .. nx+2] x [-1 .. ny+2] (taille (nx+4) x (ny+4))
    int stride_x = nx + 4;
    int stride_y = ny + 4;
    auto ext_idx = [&](int ii, int jj) {
        return (jj + 1) * stride_x + (ii + 1);
    };

    std::vector<int> head(static_cast<size_t>(stride_x * stride_y), -1);
    std::vector<int> next_mk(ext_cloud.size(), -1);

    for (size_t m = 0; m < ext_cloud.size(); ++m) {
        auto& mk = ext_cloud[m];
        if (!mk.active) continue;
        int ci = static_cast<int>(std::floor((mk.x - x_min_loc) / dx)) + 1;
        int cj = static_cast<int>(std::floor((mk.y - y_min_loc) / dy)) + 1;
        mk.cell_i = ci;
        mk.cell_j = cj;
        if (ci >= -1 && ci <= nx + 2 && cj >= -1 && cj <= ny + 2) {
            int b = ext_idx(ci, cj);
            next_mk[m] = head[b];
            head[b] = static_cast<int>(m);
        }
    }

    // Boucle sur les cellules locales (1..nx, 1..ny) au voisinage de l'interface
    double kap_max = 2.0 / std::min(dx, dy);
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int idx = mesh.cell_idx(i, j);
            double c = fields.C[idx];

            // Normale initiale de référence orientée le long de +grad C (vers le fluide 1)
            double c_mm = fields.C[mesh.cell_idx(i - 1, j - 1)];
            double c_0m = fields.C[mesh.cell_idx(i,     j - 1)];
            double c_pm = fields.C[mesh.cell_idx(i + 1, j - 1)];
            double c_m0 = fields.C[mesh.cell_idx(i - 1, j    )];
            double c_p0 = fields.C[mesh.cell_idx(i + 1, j    )];
            double c_mp = fields.C[mesh.cell_idx(i - 1, j + 1)];
            double c_0p = fields.C[mesh.cell_idx(i,     j + 1)];
            double c_pp = fields.C[mesh.cell_idx(i + 1, j + 1)];

            double gx = ((c_pp + 2.0 * c_p0 + c_pm) - (c_mp + 2.0 * c_m0 + c_mm)) / (8.0 * dx);
            double gy = ((c_pp + 2.0 * c_0p + c_mp) - (c_pm + 2.0 * c_0m + c_mm)) / (8.0 * dy);
            double grad_mag = std::hypot(gx, gy);

            if (grad_mag <= 1.0e-4 / h && head[ext_idx(i, j)] == -1) continue;

            double n_ref_x = 0.0, n_ref_y = 0.0;
            if (grad_mag > 1.0e-12) {
                n_ref_x = gx / grad_mag;
                n_ref_y = gy / grad_mag;
            } else {
                int p0 = head[ext_idx(i, j)];
                if (p0 == -1) continue;
                n_ref_x = ext_cloud[p0].nx;
                n_ref_y = ext_cloud[p0].ny;
            }

            double xc = mesh.xc(i);
            double yc = mesh.yc(j);
            double sum_nx = 0.0, sum_ny = 0.0;
            double x_bar = 0.0, y_bar = 0.0, sum_w = 0.0;
            double x_near = xc, y_near = yc;
            double d_min = 1.0e30;
            int n_pts = 0;

            for (int dj = -2; dj <= 2; ++dj) {
                for (int di = -2; di <= 2; ++di) {
                    int p = head[ext_idx(i + di, j + dj)];
                    while (p != -1) {
                        if (ext_cloud[p].nx * n_ref_x + ext_cloud[p].ny * n_ref_y >= 0.5) {
                            double d_cur = std::hypot(ext_cloud[p].x - xc, ext_cloud[p].y - yc);
                            if (d_cur <= r_fit) {
                                n_pts++;
                                sum_nx += ext_cloud[p].nx;
                                sum_ny += ext_cloud[p].ny;
                                if (di == 0 && dj == 0) {
                                    x_bar += ext_cloud[p].x;
                                    y_bar += ext_cloud[p].y;
                                    sum_w += 1.0;
                                }
                                if (d_cur < d_min) {
                                    d_min = d_cur;
                                    x_near = ext_cloud[p].x;
                                    y_near = ext_cloud[p].y;
                                }
                            }
                        }
                        p = next_mk[p];
                    }
                }
            }

            if (n_pts < 3) continue;
            if (sum_w > 0.0) {
                x_bar /= sum_w;
                y_bar /= sum_w;
            } else {
                x_bar = x_near;
                y_bar = y_near;
            }

            double norm_avg = std::hypot(sum_nx, sum_ny);
            double e_eta_x = (norm_avg > 1.0e-12) ? (sum_nx / norm_avg) : n_ref_x;
            double e_eta_y = (norm_avg > 1.0e-12) ? (sum_ny / norm_avg) : n_ref_y;
            double e_tau_x = -e_eta_y;
            double e_tau_y =  e_eta_x;

            // Assemblage du système normal 3x3 en coordonnées adimensionnées (tau/h, eta/h) :
            // eta_h(tau_h) = a_hat * tau_h^2 + b_hat * tau_h + c_hat
            double S0 = 0.0, S1 = 0.0, S2 = 0.0, S3 = 0.0, S4 = 0.0;
            double T0 = 0.0, T1 = 0.0, T2 = 0.0;

            for (int dj = -2; dj <= 2; ++dj) {
                for (int di = -2; di <= 2; ++di) {
                    int p = head[ext_idx(i + di, j + dj)];
                    while (p != -1) {
                        if (ext_cloud[p].nx * e_eta_x + ext_cloud[p].ny * e_eta_y >= 0.5) {
                            double dx_p = ext_cloud[p].x - x_bar;
                            double dy_p = ext_cloud[p].y - y_bar;
                            double tau_h = (dx_p * e_tau_x + dy_p * e_tau_y) / h;
                            double eta_h = (dx_p * e_eta_x + dy_p * e_eta_y) / h;
                            double d2_h  = tau_h * tau_h + eta_h * eta_h;
                            if (d2_h <= 2.5 * 2.5) {
                                double w_p = std::exp(-d2_h / (1.5 * 1.5));
                                double t2 = tau_h * tau_h;
                                S0 += w_p;
                                S1 += w_p * tau_h;
                                S2 += w_p * t2;
                                S3 += w_p * t2 * tau_h;
                                S4 += w_p * t2 * t2;
                                T0 += w_p * eta_h;
                                T1 += w_p * eta_h * tau_h;
                                T2 += w_p * eta_h * t2;
                            }
                        }
                        p = next_mk[p];
                    }
                }
            }

            // Régularisation de Tikhonov légère sur la courbure pour conditionnement inconditionnel
            S4 += 1.0e-9;
            S2 += 1.0e-12;
            S0 += 1.0e-12;

            double M[3][3] = {
                {S4, S3, S2},
                {S3, S2, S1},
                {S2, S1, S0}
            };
            double rhs[3] = {T2, T1, T0};

            double det_M = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                         - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                         + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

            if (std::abs(det_M) <= 1.0e-14) continue;

            double a_hat = ( rhs[0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                           - M[0][1] * (rhs[1] * M[2][2] - M[1][2] * rhs[2])
                           + M[0][2] * (rhs[1] * M[2][1] - M[1][1] * rhs[2]) ) / det_M;

            double b_hat = ( M[0][0] * (rhs[1] * M[2][2] - M[1][2] * rhs[2])
                           - rhs[0] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                           + M[0][2] * (M[1][0] * rhs[2] - rhs[1] * M[2][0]) ) / det_M;

            double kap_val = (2.0 * (a_hat / h)) / std::pow(1.0 + b_hat * b_hat, 1.5);
            kap_val = std::clamp(kap_val, -kap_max, kap_max);

            double m_nx = e_eta_x - b_hat * e_tau_x;
            double m_ny = e_eta_y - b_hat * e_tau_y;
            double n_len = std::hypot(m_nx, m_ny);
            if (n_len > 1.0e-12) {
                m_nx /= n_len;
                m_ny /= n_len;
            } else {
                m_nx = e_eta_x;
                m_ny = e_eta_y;
            }

            fields.marker_nx[idx] = m_nx;
            fields.marker_ny[idx] = m_ny;
            fields.marker_kappa[idx] = kap_val;
            if (c > 0.01 && c < 1.0 - 0.01 && n_pts >= 3) {
                fields.has_marker_normal[idx] = 1;
            }
            fields.has_marker_kappa[idx] = 1;

            // Mise à jour de la normale et de la courbure des marqueurs locaux de la cellule (i, j)
            int p = head[ext_idx(i, j)];
            while (p != -1) {
                if (p < static_cast<int>(cloud.size()) &&
                    (ext_cloud[p].nx * e_eta_x + ext_cloud[p].ny * e_eta_y >= 0.5)) {
                    double dx_p = ext_cloud[p].x - x_bar;
                    double dy_p = ext_cloud[p].y - y_bar;
                    double tau_h = (dx_p * e_tau_x + dy_p * e_tau_y) / h;
                    double slope_p = 2.0 * a_hat * tau_h + b_hat;
                    double p_nx = e_eta_x - slope_p * e_tau_x;
                    double p_ny = e_eta_y - slope_p * e_tau_y;
                    double p_len = std::hypot(p_nx, p_ny);
                    if (p_len > 1.0e-12) {
                        cloud[p].nx = p_nx / p_len;
                        cloud[p].ny = p_ny / p_len;
                    }
                    cloud[p].kappa = (2.0 * (a_hat / h)) / std::pow(1.0 + slope_p * slope_p, 1.5);
                }
                p = next_mk[p];
            }
        }
    }

    // Synchronisation MPI des champs géométriques des marqueurs sur les mailles fantômes
    MpiComm::sync_cell_scalar(mesh, fields.marker_nx);
    MpiComm::sync_cell_scalar(mesh, fields.marker_ny);
    MpiComm::sync_cell_scalar(mesh, fields.marker_kappa);

    std::vector<double> flag_kap(fields.has_marker_kappa.size(), 0.0);
    std::vector<double> flag_nrm(fields.has_marker_normal.size(), 0.0);
    for (size_t k = 0; k < flag_kap.size(); ++k) {
        flag_kap[k] = fields.has_marker_kappa[k] ? 1.0 : 0.0;
        flag_nrm[k] = fields.has_marker_normal[k] ? 1.0 : 0.0;
    }
    MpiComm::sync_cell_scalar(mesh, flag_kap);
    MpiComm::sync_cell_scalar(mesh, flag_nrm);
    for (size_t k = 0; k < flag_kap.size(); ++k) {
        fields.has_marker_kappa[k]  = (flag_kap[k] > 0.5) ? 1 : 0;
        fields.has_marker_normal[k] = (flag_nrm[k] > 0.5) ? 1 : 0;
    }
}

// =============================================================================
// 7. EXPORT VTK POLYDATA DU NUAGE DE MARQUEURS POUR PARAVIEW
// =============================================================================

void InterfaceMarkers::write_vtk(const Mesh2D& mesh, const std::string& output_dir,
                                 const std::string& case_name, int step) const {
    int n_loc = static_cast<int>(cloud.size());
    std::vector<double> loc_buf(static_cast<size_t>(std::max(1, 5 * n_loc)), 0.0);
    for (int m = 0; m < n_loc; ++m) {
        loc_buf[5 * m]     = cloud[m].x;
        loc_buf[5 * m + 1] = cloud[m].y;
        loc_buf[5 * m + 2] = cloud[m].nx;
        loc_buf[5 * m + 3] = cloud[m].ny;
        loc_buf[5 * m + 4] = cloud[m].kappa;
    }

    std::vector<int> counts(static_cast<size_t>(mesh.num_procs), 0);
    MPI_Gather(&n_loc, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, mesh.comm);

    std::vector<int> counts5(static_cast<size_t>(mesh.num_procs), 0);
    std::vector<int> displs5(static_cast<size_t>(mesh.num_procs), 0);
    int n_tot = 0;
    if (mesh.rank == 0) {
        for (int r = 0; r < mesh.num_procs; ++r) {
            counts5[r] = 5 * counts[r];
            displs5[r] = 5 * n_tot;
            n_tot += counts[r];
        }
    }

    std::vector<double> glob_buf(static_cast<size_t>(std::max(1, 5 * n_tot)), 0.0);
    MPI_Gatherv(loc_buf.data(), 5 * n_loc, MPI_DOUBLE,
                glob_buf.data(), counts5.data(), displs5.data(), MPI_DOUBLE,
                0, mesh.comm);

    if (mesh.rank == 0) {
        std::filesystem::create_directories(output_dir);
        std::ostringstream fname;
        fname << output_dir << "/" << case_name << "_markers_"
              << std::setw(6) << std::setfill('0') << step << ".vtk";

        std::ofstream ofs(fname.str());
        if (!ofs.is_open()) return;

        ofs << "# vtk DataFile Version 3.0\n"
            << "COVOF 2D Unconnected Lagrangian Interface Markers\n"
            << "ASCII\n"
            << "DATASET POLYDATA\n"
            << "POINTS " << n_tot << " double\n"
            << std::scientific << std::setprecision(8);

        for (int m = 0; m < n_tot; ++m) {
            ofs << glob_buf[5 * m] << " " << glob_buf[5 * m + 1] << " 0.0\n";
        }

        ofs << "VERTICES " << n_tot << " " << (2 * n_tot) << "\n";
        for (int m = 0; m < n_tot; ++m) {
            ofs << "1 " << m << "\n";
        }

        ofs << "POINT_DATA " << n_tot << "\n"
            << "VECTORS MarkerNormal double\n";
        for (int m = 0; m < n_tot; ++m) {
            ofs << glob_buf[5 * m + 2] << " " << glob_buf[5 * m + 3] << " 0.0\n";
        }

        ofs << "SCALARS MarkerCurvature double 1\n"
            << "LOOKUP_TABLE default\n";
        for (int m = 0; m < n_tot; ++m) {
            ofs << glob_buf[5 * m + 4] << "\n";
        }
        ofs.close();
    }
}

} // namespace covof
