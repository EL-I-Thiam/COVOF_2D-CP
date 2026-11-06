#include "covof/InterfaceTracker.hpp"
#include "covof/MpiComm.hpp"
#include "covof/EOS.hpp"
#include <cmath>
#include <algorithm>

namespace covof {
void InterfaceTracker::set_initial_condition(const Mesh2D& mesh, const InitialConditionConfig& ic, FlowFields& fields) {
    // Déclaration de variable typée
    double eps = 1.5 * std::max(mesh.dx, mesh.dy);

    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 0; j <= mesh.ny_local + 1; ++j) {
        // Déclaration de variable typée
        double y = mesh.yc(j);
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i <= mesh.nx_local + 1; ++i) {
            // Déclaration de variable typée
            double x = mesh.xc(i);
            // Déclaration de variable typée
            int idx = mesh.cell_idx(i, j);

            // Évaluation de la condition de contrôle
            if (ic.type == InitialConditionType::SINGLE_PHASE) {
                // Calcul de la résolution géométrique de cellule
                fields.C[idx] = 1.0;
            // Calcul et assignation numérique
            } else if (ic.type == InitialConditionType::DROPLET) {
                // Déclaration de variable typée
                double dist = std::sqrt((x - ic.center_x) * (x - ic.center_x) +
                                        (y - ic.center_y) * (y - ic.center_y)) - ic.radius;
                // Évaluation de la condition de contrôle
                if (dist < -eps) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 1.0;
                } else if (dist > eps) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.0;
                } else {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.5 - dist / (2.0 * eps) - (1.0 / (2.0 * M_PI)) * std::sin(M_PI * dist / eps);
                }
            // Calcul et assignation numérique
            } else if (ic.type == InitialConditionType::BUBBLE || ic.type == InitialConditionType::CYLINDER) {
                // Déclaration de variable typée
                double dist = std::sqrt((x - ic.center_x) * (x - ic.center_x) +
                                        (y - ic.center_y) * (y - ic.center_y)) - ic.radius;
                if (ic.type == InitialConditionType::CYLINDER || vof_cfg.reconstruction_method == "PLIC") {
                    // Intégration géométrique nette par sous-échantillonnage 16x16 pour mailles coupées (Cas 3.1 Fluids 2021)
                    double x_min = mesh.xv(i);
                    double x_max = mesh.xv(i + 1);
                    double y_min = mesh.yv(j);
                    double y_max = mesh.yv(j + 1);

                    double d00 = std::hypot(x_min - ic.center_x, y_min - ic.center_y);
                    double d10 = std::hypot(x_max - ic.center_x, y_min - ic.center_y);
                    double d01 = std::hypot(x_min - ic.center_x, y_max - ic.center_y);
                    double d11 = std::hypot(x_max - ic.center_x, y_max - ic.center_y);

                    if (d00 <= ic.radius && d10 <= ic.radius && d01 <= ic.radius && d11 <= ic.radius) {
                        fields.C[idx] = 0.0; // Intérieur cylindre / bulle : fluide 2
                    } else if (d00 >= ic.radius && d10 >= ic.radius && d01 >= ic.radius && d11 >= ic.radius) {
                        double d_mid = std::hypot(x - ic.center_x, y - ic.center_y);
                        double diag = 0.5 * std::hypot(mesh.dx, mesh.dy);
                        if (d_mid > ic.radius + diag) {
                            fields.C[idx] = 1.0; // Extérieur : fluide 1
                        } else {
                            int count_out = 0;
                            const int N_SUB = 16;
                            for (int sj = 0; sj < N_SUB; ++sj) {
                                double ys = y_min + (sj + 0.5) * (mesh.dy / N_SUB);
                                for (int si = 0; si < N_SUB; ++si) {
                                    double xs = x_min + (si + 0.5) * (mesh.dx / N_SUB);
                                    if (std::hypot(xs - ic.center_x, ys - ic.center_y) > ic.radius) {
                                        count_out++;
                                    }
                                }
                            }
                            fields.C[idx] = static_cast<double>(count_out) / (N_SUB * N_SUB);
                        }
                    } else {
                        int count_out = 0;
                        const int N_SUB = 16;
                        for (int sj = 0; sj < N_SUB; ++sj) {
                            double ys = y_min + (sj + 0.5) * (mesh.dy / N_SUB);
                            for (int si = 0; si < N_SUB; ++si) {
                                double xs = x_min + (si + 0.5) * (mesh.dx / N_SUB);
                                if (std::hypot(xs - ic.center_x, ys - ic.center_y) > ic.radius) {
                                    count_out++;
                                }
                            }
                        }
                        fields.C[idx] = static_cast<double>(count_out) / (N_SUB * N_SUB);
                    }
                } else {
                    // Évaluation de la condition de contrôle diffuse standard
                    if (dist < -eps) {
                        fields.C[idx] = 0.0; // Intérieur bulle : fluide 2
                    } else if (dist > eps) {
                        fields.C[idx] = 1.0; // Extérieur : fluide 1 (eau)
                    } else {
                        fields.C[idx] = 0.5 + dist / (2.0 * eps) + (1.0 / (2.0 * M_PI)) * std::sin(M_PI * dist / eps);
                    }
                }
            // Calcul et assignation numérique
            } else if (ic.type == InitialConditionType::DAM_BREAK) {
                // Évaluation de la condition de contrôle
                if (x >= ic.box_x0 && x <= ic.box_x1 && y >= ic.box_y0 && y <= ic.box_y1) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 1.0;
                } else {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.0;
                }
            // Calcul et assignation numérique
            } else if (ic.type == InitialConditionType::SHEAR_LAYER) {
                // Déclaration de variable typée
                double int_y = ic.center_y + ic.pert_amp * std::sin(2.0 * M_PI * ic.pert_k * x / mesh.lx);
                // Calcul de la résolution géométrique de cellule
                fields.C[idx] = (y <= int_y) ? 1.0 : 0.0;
            // Calcul et assignation numérique pour nappe liquide (Cas 3.2 Fluids 2021)
            } else if (ic.type == InitialConditionType::LIQUID_SHEET) {
                // Déclaration de la position centrale et demi-épaisseur de nappe
                double y_c = (ic.center_y > 0.0) ? ic.center_y : 0.5 * mesh.ly;
                // Calcul de la demi-épaisseur de la nappe liquide
                double half_thick = 0.5 * ic.thickness;
                // Calcul de la distance géométrique par rapport à la surface de la nappe
                double dist = std::abs(y - y_c) - half_thick;
                // Évaluation de l'indicatrice : C = 0 dans le liquide (fluide 2), C = 1 dans le gaz (fluide 1)
                if (dist < -eps) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.0;
                // Branche alternative conditionnelle
                } else if (dist > eps) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 1.0;
                // Branche par défaut pour transition continue
                } else {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.5 + dist / (2.0 * eps) + (1.0 / (2.0 * M_PI)) * std::sin(M_PI * dist / eps);
                }
            // Calcul et assignation numérique pour jet liquide visqueux (Cas 3.3 Fluids 2021)
            } else if (ic.type == InitialConditionType::VISCOUS_JET) {
                // Déclaration du centre et de la demi-largeur de l'injecteur
                double x_c = (ic.center_x > 0.0) ? ic.center_x : 0.5 * mesh.lx;
                // Calcul de la demi-largeur de l'injecteur
                double half_w = (ic.jet_width > 0.0) ? 0.5 * ic.jet_width : 0.04;
                // Déclaration de la hauteur initiale de l'amorce de jet sous la paroi supérieure
                double stub_height = 2.0 * mesh.dy;
                // Évaluation de l'indicatrice : C = 0 dans l'amorce liquide, C = 1 dans la cavité de gaz
                if (y >= mesh.ly - stub_height && std::abs(x - x_c) <= half_w) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.0;
                } else {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 1.0;
                }
            // Calcul et assignation numérique
            } else if (ic.type == InitialConditionType::VORTEX_SHEAR) {
                // Déclaration de variable typée
                double dist = std::sqrt((x - ic.center_x) * (x - ic.center_x) +
                                        (y - ic.center_y) * (y - ic.center_y)) - ic.radius;
                // Évaluation de la condition de contrôle
                if (dist < -eps) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 1.0;
                } else if (dist > eps) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.0;
                } else {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.5 - dist / (2.0 * eps) - (1.0 / (2.0 * M_PI)) * std::sin(M_PI * dist / eps);
                }
            // Calcul et assignation numérique
            } else if (ic.type == InitialConditionType::RAYLEIGH_TAYLOR) {
                // Déclaration de variable typée
                double int_y = ic.center_y + ic.pert_amp * std::cos(2.0 * M_PI * ic.pert_k * x / mesh.lx);
                // Calcul de la résolution géométrique de cellule
                fields.C[idx] = (y >= int_y) ? 1.0 : 0.0;
            }
        }
    }

    MpiComm::sync_cell_scalar(mesh, fields.C);
    if (enable_svof_smoothing) {
        smooth_svof_helmholtz(mesh, fields);
    }
    update_physical_properties(mesh, fields);
    compute_curvature_and_CSF(mesh, fields);
}

void InterfaceTracker::smooth_svof_helmholtz(const Mesh2D& mesh, FlowFields& fields) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    if (fields.C_s.size() != fields.C.size()) {
        fields.C_s.assign(fields.C.size(), 0.0);
    }

    double k = (svof_filter_width > 0.0) ? svof_filter_width : 1.0;
    double D = k * dx * dy;
    double wx = D / (dx * dx);
    double wy = D / (dy * dy);

    // Initialisation avec C
    fields.C_s = fields.C;

    // Résolution itérative Gauss-Seidel symétrique du système strictement diagonal dominant
    // (I - D Delta) C_s = C avec flux nul dC_s/dn = 0 sur les frontières (Éq. 4-5 Fluids 2021)
    for (int iter = 0; iter < 25; ++iter) {
        for (int j = 1; j <= ny; ++j) {
            if (mesh.is_boundary_left)  fields.C_s[mesh.cell_idx(0, j)]      = fields.C_s[mesh.cell_idx(1, j)];
            if (mesh.is_boundary_right) fields.C_s[mesh.cell_idx(nx + 1, j)] = fields.C_s[mesh.cell_idx(nx, j)];
        }
        for (int i = 1; i <= nx; ++i) {
            if (mesh.is_boundary_bottom) fields.C_s[mesh.cell_idx(i, 0)]      = fields.C_s[mesh.cell_idx(i, 1)];
            if (mesh.is_boundary_top)    fields.C_s[mesh.cell_idx(i, ny + 1)] = fields.C_s[mesh.cell_idx(i, ny)];
        }
        MpiComm::sync_cell_scalar(mesh, fields.C_s);

        // Balayage avant
        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int idx = mesh.cell_idx(i, j);
                double a_E = (i == nx && mesh.is_boundary_right) ? 0.0 : wx;
                double a_W = (i == 1  && mesh.is_boundary_left)  ? 0.0 : wx;
                double a_N = (j == ny && mesh.is_boundary_top)   ? 0.0 : wy;
                double a_S = (j == 1  && mesh.is_boundary_bottom) ? 0.0 : wy;
                double a_P = 1.0 + a_E + a_W + a_N + a_S;

                double sum_nb = a_E * fields.C_s[mesh.cell_idx(i + 1, j)] +
                                a_W * fields.C_s[mesh.cell_idx(i - 1, j)] +
                                a_N * fields.C_s[mesh.cell_idx(i, j + 1)] +
                                a_S * fields.C_s[mesh.cell_idx(i, j - 1)];

                fields.C_s[idx] = (fields.C[idx] + sum_nb) / a_P;
            }
        }

        // Balayage arrière pour accélération spectrale symétrique
        for (int j = ny; j >= 1; --j) {
            for (int i = nx; i >= 1; --i) {
                int idx = mesh.cell_idx(i, j);
                double a_E = (i == nx && mesh.is_boundary_right) ? 0.0 : wx;
                double a_W = (i == 1  && mesh.is_boundary_left)  ? 0.0 : wx;
                double a_N = (j == ny && mesh.is_boundary_top)   ? 0.0 : wy;
                double a_S = (j == 1  && mesh.is_boundary_bottom) ? 0.0 : wy;
                double a_P = 1.0 + a_E + a_W + a_N + a_S;

                double sum_nb = a_E * fields.C_s[mesh.cell_idx(i + 1, j)] +
                                a_W * fields.C_s[mesh.cell_idx(i - 1, j)] +
                                a_N * fields.C_s[mesh.cell_idx(i, j + 1)] +
                                a_S * fields.C_s[mesh.cell_idx(i, j - 1)];

                fields.C_s[idx] = (fields.C[idx] + sum_nb) / a_P;
            }
        }
    }

    // Bornage physique strict dans [0, 1]
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int idx = mesh.cell_idx(i, j);
            fields.C_s[idx] = std::clamp(fields.C_s[idx], 0.0, 1.0);
        }
    }

    // Synchronisation finale
    for (int j = 1; j <= ny; ++j) {
        if (mesh.is_boundary_left)  fields.C_s[mesh.cell_idx(0, j)]      = fields.C_s[mesh.cell_idx(1, j)];
        if (mesh.is_boundary_right) fields.C_s[mesh.cell_idx(nx + 1, j)] = fields.C_s[mesh.cell_idx(nx, j)];
    }
    for (int i = 1; i <= nx; ++i) {
        if (mesh.is_boundary_bottom) fields.C_s[mesh.cell_idx(i, 0)]      = fields.C_s[mesh.cell_idx(i, 1)];
        if (mesh.is_boundary_top)    fields.C_s[mesh.cell_idx(i, ny + 1)] = fields.C_s[mesh.cell_idx(i, ny)];
    }
    fields.C_s[mesh.cell_idx(0, 0)]           = fields.C_s[mesh.cell_idx(1, 1)];
    fields.C_s[mesh.cell_idx(nx + 1, 0)]      = fields.C_s[mesh.cell_idx(nx, 1)];
    fields.C_s[mesh.cell_idx(0, ny + 1)]      = fields.C_s[mesh.cell_idx(1, ny)];
    fields.C_s[mesh.cell_idx(nx + 1, ny + 1)] = fields.C_s[mesh.cell_idx(nx, ny)];
    MpiComm::sync_cell_scalar(mesh, fields.C_s);
}

void InterfaceTracker::update_physical_properties(const Mesh2D& mesh, FlowFields& fields) {
    // Déclaration de variable typée
    int nx = mesh.nx_local;
    // Déclaration de variable typée
    int ny = mesh.ny_local;

    // Pour un écoulement monophasique, propriétés uniformes exactes
    if (!enable_two_phase) {
        std::fill(fields.rho.begin(), fields.rho.end(), fluid1.rho);
        std::fill(fields.rho_u.begin(), fields.rho_u.end(), fluid1.rho);
        std::fill(fields.rho_v.begin(), fields.rho_v.end(), fluid1.rho);
        // Évaluation de la condition de contrôle
        if (viscosity_model == "EXPONENTIAL") {
            // Loi de viscosité exponentielle : mu(z) = mu0 * exp(a * (z/e)^2) avec z = y - e
            for (int j = 0; j <= ny + 1; ++j) {
                double y = mesh.yc(j);
                double z = y - visc_e;
                double z_ratio = z / visc_e;
                double mu_val = visc_mu0 * std::exp(visc_a * z_ratio * z_ratio);
                for (int i = 0; i <= nx + 1; ++i) {
                    fields.mu[mesh.cell_idx(i, j)] = mu_val;
                }
            }
            for (int j = 0; j <= ny + 1; ++j) {
                double y = mesh.yv(j);
                double z = y - visc_e;
                double z_ratio = z / visc_e;
                double mu_node_val = visc_mu0 * std::exp(visc_a * z_ratio * z_ratio);
                for (int i = 0; i <= nx + 1; ++i) {
                    fields.mu_node[mesh.node_idx(i, j)] = mu_node_val;
                }
            }
        } else if (viscosity_model == "ASYMMETRIC_EXPONENTIAL") {
            for (int j = 0; j <= ny + 1; ++j) {
                double y = mesh.yc(j);
                double z = y - visc_e;
                double mu_val = visc_mu0 * std::exp(-visc_a * z / visc_e);
                for (int i = 0; i <= nx + 1; ++i) {
                    fields.mu[mesh.cell_idx(i, j)] = mu_val;
                }
            }
            for (int j = 0; j <= ny + 1; ++j) {
                double y = mesh.yv(j);
                double z = y - visc_e;
                double mu_node_val = visc_mu0 * std::exp(-visc_a * z / visc_e);
                for (int i = 0; i <= nx + 1; ++i) {
                    fields.mu_node[mesh.node_idx(i, j)] = mu_node_val;
                }
            }
        } else {
            std::fill(fields.mu.begin(), fields.mu.end(), fluid1.mu);
            std::fill(fields.mu_node.begin(), fields.mu_node.end(), fluid1.mu);
        }
        // Retour de fonction
        return;
    }

    // Choix du champ couleur : lissé si SVOF activé, sinon standard C
    const std::vector<double>& c_src = (enable_svof_smoothing && !fields.C_s.empty()) ? fields.C_s : fields.C;

    // Remplissage des mailles fantômes pour C (bords physiques et coins)
    for (int j = 1; j <= ny; ++j) {
        fields.C[mesh.cell_idx(0, j)]      = fields.C[mesh.cell_idx(1, j)];
        fields.C[mesh.cell_idx(nx + 1, j)] = fields.C[mesh.cell_idx(nx, j)];
    }
    for (int i = 1; i <= nx; ++i) {
        fields.C[mesh.cell_idx(i, 0)]      = fields.C[mesh.cell_idx(i, 1)];
        fields.C[mesh.cell_idx(i, ny + 1)] = fields.C[mesh.cell_idx(i, ny)];
    }
    fields.C[mesh.cell_idx(0, 0)]           = fields.C[mesh.cell_idx(1, 1)];
    fields.C[mesh.cell_idx(nx + 1, 0)]      = fields.C[mesh.cell_idx(nx, 1)];
    fields.C[mesh.cell_idx(0, ny + 1)]      = fields.C[mesh.cell_idx(1, ny)];
    fields.C[mesh.cell_idx(nx + 1, ny + 1)]  = fields.C[mesh.cell_idx(nx, ny)];

    // 1. Masse volumique et viscosité au centre des cellules
    for (int j = 0; j <= ny + 1; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i <= nx + 1; ++i) {
            // Déclaration de variable typée
            int idx = mesh.cell_idx(i, j);
            // Déclaration de variable typée
            double c = std::clamp(c_src[idx], 0.0, 1.0);
            if (!compressible) {
                fields.rho[idx] = c * fluid1.rho + (1.0 - c) * fluid2.rho;
            } else {
                double p_val = (fields.p.size() > (size_t)idx) ? fields.p[idx] : 1.01325e5;
                double T_val = (fields.T.size() > (size_t)idx) ? fields.T[idx] : 300.0;
                double rho1 = EOS::compute_density(EOS::parse_type(fluid1.eos), fluid1, p_val, T_val);
                double rho2 = EOS::compute_density(EOS::parse_type(fluid2.eos), fluid2, p_val, T_val);
                fields.rho[idx] = c * rho1 + (1.0 - c) * rho2;
            }
            fields.mu[idx]  = c * fluid1.mu  + (1.0 - c) * fluid2.mu;

            if (compressible) {
                double p_val = (fields.p.size() > (size_t)idx) ? fields.p[idx] : 1.01325e5;
                double T_val = (fields.T.size() > (size_t)idx) ? fields.T[idx] : 300.0;
                double chi1 = EOS::compute_chi_T(EOS::parse_type(fluid1.eos), fluid1, p_val, T_val, fluid1.rho);
                double chi2 = EOS::compute_chi_T(EOS::parse_type(fluid2.eos), fluid2, p_val, T_val, fields.rho[idx]);
                if (fields.chi_T.size() > (size_t)idx) {
                    fields.chi_T[idx] = c * chi1 + (1.0 - c) * chi2;
                }
            }
        }
    }

    // Remplissage des mailles fantômes pour rho et mu (avec prise en compte des CL périodiques)
    for (int j = 1; j <= ny; ++j) {
        if (!mesh.is_boundary_left) {
            fields.rho[mesh.cell_idx(0, j)]      = fields.rho[mesh.cell_idx(nx, j)];
            fields.rho[mesh.cell_idx(nx + 1, j)] = fields.rho[mesh.cell_idx(1, j)];
            fields.mu[mesh.cell_idx(0, j)]       = fields.mu[mesh.cell_idx(nx, j)];
            fields.mu[mesh.cell_idx(nx + 1, j)]  = fields.mu[mesh.cell_idx(1, j)];
        } else {
            fields.rho[mesh.cell_idx(0, j)]      = fields.rho[mesh.cell_idx(1, j)];
            fields.rho[mesh.cell_idx(nx + 1, j)] = fields.rho[mesh.cell_idx(nx, j)];
            fields.mu[mesh.cell_idx(0, j)]       = fields.mu[mesh.cell_idx(1, j)];
            fields.mu[mesh.cell_idx(nx + 1, j)]  = fields.mu[mesh.cell_idx(nx, j)];
        }
    }
    for (int i = 1; i <= nx; ++i) {
        if (!mesh.is_boundary_bottom) {
            fields.rho[mesh.cell_idx(i, 0)]      = fields.rho[mesh.cell_idx(i, ny)];
            fields.rho[mesh.cell_idx(i, ny + 1)] = fields.rho[mesh.cell_idx(i, 1)];
            fields.mu[mesh.cell_idx(i, 0)]       = fields.mu[mesh.cell_idx(i, ny)];
            fields.mu[mesh.cell_idx(i, ny + 1)]  = fields.mu[mesh.cell_idx(i, 1)];
        } else {
            fields.rho[mesh.cell_idx(i, 0)]      = fields.rho[mesh.cell_idx(i, 1)];
            fields.rho[mesh.cell_idx(i, ny + 1)] = fields.rho[mesh.cell_idx(i, ny)];
            fields.mu[mesh.cell_idx(i, 0)]       = fields.mu[mesh.cell_idx(i, 1)];
            fields.mu[mesh.cell_idx(i, ny + 1)]  = fields.mu[mesh.cell_idx(i, ny)];
        }
    }
    fields.mu[mesh.cell_idx(0, 0)]           = fields.mu[mesh.cell_idx(1, 1)];
    fields.mu[mesh.cell_idx(nx + 1, 0)]      = fields.mu[mesh.cell_idx(nx, 1)];
    fields.mu[mesh.cell_idx(0, ny + 1)]      = fields.mu[mesh.cell_idx(1, ny)];
    fields.mu[mesh.cell_idx(nx + 1, ny + 1)]  = fields.mu[mesh.cell_idx(nx, ny)];
    fields.rho[mesh.cell_idx(0, 0)]          = fields.rho[mesh.cell_idx(1, 1)];
    fields.rho[mesh.cell_idx(nx + 1, 0)]     = fields.rho[mesh.cell_idx(nx, 1)];
    fields.rho[mesh.cell_idx(0, ny + 1)]     = fields.rho[mesh.cell_idx(1, ny)];
    fields.rho[mesh.cell_idx(nx + 1, ny + 1)] = fields.rho[mesh.cell_idx(nx, ny)];

    // 2. Masse volumique sur les faces u
    for (int j = 1; j <= ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx + 1; ++i) {
            // Déclaration de variable typée
            int idx_u = mesh.u_idx(i, j);
            // Calcul de la résolution géométrique de cellule
            fields.rho_u[idx_u] = 0.5 * (fields.rho[mesh.cell_idx(i - 1, j)] + fields.rho[mesh.cell_idx(i, j)]);
        }
    }

    // 3. Masse volumique sur les faces v
    for (int j = 1; j <= ny + 1; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Déclaration de variable typée
            int idx_v = mesh.v_idx(i, j);
            // Calcul de la résolution géométrique de cellule
            fields.rho_v[idx_v] = 0.5 * (fields.rho[mesh.cell_idx(i, j - 1)] + fields.rho[mesh.cell_idx(i, j)]);
        }
    }

    // 4. Viscosité harmonique aux nœuds (coins)
    for (int j = 1; j <= ny + 1; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx + 1; ++i) {
            // Déclaration de variable typée
            int n_idx = mesh.node_idx(i, j);
            if (enable_svof_smoothing) {
                // Éq. 8-9 Fluids 2021 : interpolation nodale de C_s et viscosité harmonique de cisaillement
                double c_node = 0.25 * (c_src[mesh.cell_idx(i - 1, j - 1)] +
                                        c_src[mesh.cell_idx(i, j - 1)] +
                                        c_src[mesh.cell_idx(i - 1, j)] +
                                        c_src[mesh.cell_idx(i, j)]);
                c_node = std::clamp(c_node, 0.0, 1.0);
                double denom = fluid2.mu * c_node + fluid1.mu * (1.0 - c_node);
                fields.mu_node[n_idx] = (fluid1.mu * fluid2.mu) / std::max(1.0e-30, denom);
            } else {
                double m_SW = fields.mu[mesh.cell_idx(i - 1, j - 1)];
                double m_SE = fields.mu[mesh.cell_idx(i, j - 1)];
                double m_NW = fields.mu[mesh.cell_idx(i - 1, j)];
                double m_NE = fields.mu[mesh.cell_idx(i, j)];

                fields.mu_node[n_idx] = 4.0 / (1.0 / m_SW + 1.0 / m_SE + 1.0 / m_NW + 1.0 / m_NE);
            }
        }
    }
}

void InterfaceTracker::compute_curvature_and_CSF(const Mesh2D& mesh, FlowFields& fields) {
    // Évaluation de la condition de contrôle
    if (!enable_surface_tension || sigma == 0.0) {
        std::fill(fields.F_sigma_x.begin(), fields.F_sigma_x.end(), 0.0);
        std::fill(fields.F_sigma_y.begin(), fields.F_sigma_y.end(), 0.0);
        // Retour de fonction
        return;
    }

    // Déclaration de variable typée
    int nx = mesh.nx_local;
    // Déclaration de variable typée
    int ny = mesh.ny_local;
    // Déclaration de variable typée
    double dx = mesh.dx;
    // Déclaration de variable typée
    double dy = mesh.dy;

    // 1. Calcul du gradient et du vecteur normal unitaire aux centres
    const std::vector<double>& c_field =
        (enable_diffusion_smoothing && !fields.C_tilde.empty()) ? fields.C_tilde :
        ((enable_svof_smoothing && !fields.C_s.empty()) ? fields.C_s : fields.C);
    for (int j = 1; j <= ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Déclaration de variable typée
            int idx = mesh.cell_idx(i, j);
            // Déclaration de variable typée
            double dC_dx = (c_field[mesh.cell_idx(i + 1, j)] - c_field[mesh.cell_idx(i - 1, j)]) / (2.0 * dx);
            // Déclaration de variable typée
            double dC_dy = (c_field[mesh.cell_idx(i, j + 1)] - c_field[mesh.cell_idx(i, j - 1)]) / (2.0 * dy);
            // Déclaration de variable typée
            double grad_mag = std::sqrt(dC_dx * dC_dx + dC_dy * dC_dy);
            // Évaluation de la condition de contrôle
            if (grad_mag > 1.0e-3 / dx) {
                // Calcul de la résolution géométrique de cellule
                fields.norm_x[idx] = dC_dx / grad_mag;
                // Calcul de la résolution géométrique de cellule
                fields.norm_y[idx] = dC_dy / grad_mag;
            } else {
                // Calcul de la résolution géométrique de cellule
                fields.norm_x[idx] = 0.0;
                // Calcul de la résolution géométrique de cellule
                fields.norm_y[idx] = 0.0;
            }
        }
    }
    MpiComm::sync_cell_scalar(mesh, fields.norm_x);
    MpiComm::sync_cell_scalar(mesh, fields.norm_y);

    // 2. Calcul de la courbure kappa = -div(n)
    for (int j = 1; j <= ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Déclaration de variable typée
            int idx = mesh.cell_idx(i, j);
            // Évaluation de la condition de contrôle
            if (fields.norm_x[idx] == 0.0 && fields.norm_y[idx] == 0.0) {
                // Calcul de la résolution géométrique de cellule
                fields.kappa[idx] = 0.0;
            } else {
                // Déclaration de variable typée
                double dnx_dx = (fields.norm_x[mesh.cell_idx(i + 1, j)] - fields.norm_x[mesh.cell_idx(i - 1, j)]) / (2.0 * dx);
                // Déclaration de variable typée
                double dny_dy = (fields.norm_y[mesh.cell_idx(i, j + 1)] - fields.norm_y[mesh.cell_idx(i, j - 1)]) / (2.0 * dy);
                // Calcul de la résolution géométrique de cellule
                fields.kappa[idx] = -(dnx_dx + dny_dy);
            }
        }
    }
    MpiComm::sync_cell_scalar(mesh, fields.kappa);

    // 3. Force capillaire CSF équilibrée (balanced-force) sur les faces décalées
    // Selon X sur faces u
    for (int j = 1; j <= ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx + 1; ++i) {
            // Déclaration de variable typée
            int idx_u = mesh.u_idx(i, j);
            // Déclaration de variable typée
            double kap_avg = 0.5 * (fields.kappa[mesh.cell_idx(i - 1, j)] + fields.kappa[mesh.cell_idx(i, j)]);
            // Déclaration de variable typée
            double grad_C = (fields.C[mesh.cell_idx(i, j)] - fields.C[mesh.cell_idx(i - 1, j)]) / dx;
            // Calcul de la résolution géométrique de cellule
            fields.F_sigma_x[idx_u] = sigma * kap_avg * grad_C;
        }
    }

    // Selon Y sur faces v
    for (int j = 1; j <= ny + 1; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Déclaration de variable typée
            int idx_v = mesh.v_idx(i, j);
            // Déclaration de variable typée
            double kap_avg = 0.5 * (fields.kappa[mesh.cell_idx(i, j - 1)] + fields.kappa[mesh.cell_idx(i, j)]);
            // Déclaration de variable typée
            double grad_C = (fields.C[mesh.cell_idx(i, j)] - fields.C[mesh.cell_idx(i, j - 1)]) / dy;
            // Calcul de la résolution géométrique de cellule
            fields.F_sigma_y[idx_v] = sigma * kap_avg * grad_C;
        }
    }
}

void InterfaceTracker::advect(const Mesh2D& mesh, FlowFields& fields, double dt) {
    // Sélection de la méthode d'advection configurée par l'utilisateur
    AdvectionMethod adv_method = InterfaceReconstruction::parse_advection_method(vof_cfg.advection_method);

    // Initialisation du vecteur de segments d'interface si nécessaire
    int total_cells = (mesh.nx_local + 2) * (mesh.ny_local + 2);
    if (static_cast<int>(interface_segments.size()) != total_cells) {
        interface_segments.resize(total_cells);
    }

    // Répartition vers l'algorithme d'advection sélectionné
    switch (adv_method) {
        case AdvectionMethod::CIAM:
            advect_ciam(mesh, fields, dt);
            break;
        case AdvectionMethod::WEYMOUTH_YUE:
            advect_weymouth_yue(mesh, fields, dt);
            break;
        case AdvectionMethod::VOF_TVD:
        default:
            advect_tvd(mesh, fields, dt);
            break;
    }

    // Incrémentation du compteur de pas pour Strang splitting alterné
    step_count++;

    // Synchronisation MPI des fractions volumiques sur les mailles fantômes
    MpiComm::sync_cell_scalar(mesh, fields.C);

    // Mise à jour des grandeurs physiques dérivées (masse volumique, viscosité, tension de surface)
    if (enable_svof_smoothing) {
        smooth_svof_helmholtz(mesh, fields);
    }
    update_physical_properties(mesh, fields);
    if (enable_diffusion_smoothing) {
        smooth_diffusion(mesh, fields);
    }
    compute_curvature_and_CSF(mesh, fields);
}

void InterfaceTracker::advect_tvd(const Mesh2D& mesh, FlowFields& fields, double dt) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    std::vector<double> C_new = fields.C;
    TvdLimiter limiter = InterfaceReconstruction::parse_tvd_limiter(vof_cfg.tvd_limiter);

    // Calcul des flux de fraction volumique sur les faces X
    std::vector<double> flux_x((nx + 2) * (ny + 2), 0.0);
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx + 1; ++i) {
            double u_face = fields.u[mesh.u_idx(i, j)];
            double c_L = fields.C[mesh.cell_idx(i - 1, j)];
            double c_R = fields.C[mesh.cell_idx(i, j)];

            double cfl_x = std::min(1.0, std::abs(u_face) * dt / dx);
            if (u_face >= 0.0) {
                int i_up = std::max(0, i - 2);
                double c_U = fields.C[mesh.cell_idx(i_up, j)];
                double dC_up = c_L - c_U;
                double dC_loc = c_R - c_L;
                double r = (std::abs(dC_loc) > 1.0e-12) ? dC_up / dC_loc : 0.0;
                double phi = InterfaceReconstruction::evaluate_limiter(limiter, r, c_L, c_U, c_R);
                flux_x[mesh.cell_idx(i, j)] = u_face * (c_L + 0.5 * phi * (1.0 - cfl_x) * dC_loc);
            } else {
                int i_down = std::min(nx + 1, i + 1);
                double c_U = fields.C[mesh.cell_idx(i_down, j)];
                double dC_up = c_U - c_R;
                double dC_loc = c_R - c_L;
                double r = (std::abs(dC_loc) > 1.0e-12) ? dC_up / dC_loc : 0.0;
                double phi = InterfaceReconstruction::evaluate_limiter(limiter, r, c_R, c_U, c_L);
                flux_x[mesh.cell_idx(i, j)] = u_face * (c_R - 0.5 * phi * (1.0 - cfl_x) * dC_loc);
            }
        }
    }

    // Calcul des flux sur les faces Y
    std::vector<double> flux_y((nx + 2) * (ny + 2), 0.0);
    for (int j = 1; j <= ny + 1; ++j) {
        for (int i = 1; i <= nx; ++i) {
            double v_face = fields.v[mesh.v_idx(i, j)];
            double c_B = fields.C[mesh.cell_idx(i, j - 1)];
            double c_T = fields.C[mesh.cell_idx(i, j)];
            double cfl_y = std::min(1.0, std::abs(v_face) * dt / dy);

            if (v_face >= 0.0) {
                int j_up = std::max(0, j - 2);
                double c_U = fields.C[mesh.cell_idx(i, j_up)];
                double dC_up = c_B - c_U;
                double dC_loc = c_T - c_B;
                double r = (std::abs(dC_loc) > 1.0e-12) ? dC_up / dC_loc : 0.0;
                double phi = InterfaceReconstruction::evaluate_limiter(limiter, r, c_B, c_U, c_T);
                flux_y[mesh.cell_idx(i, j)] = v_face * (c_B + 0.5 * phi * (1.0 - cfl_y) * dC_loc);
            } else {
                int j_down = std::min(ny + 1, j + 1);
                double c_U = fields.C[mesh.cell_idx(i, j_down)];
                double dC_up = c_U - c_T;
                double dC_loc = c_T - c_B;
                double r = (std::abs(dC_loc) > 1.0e-12) ? dC_up / dC_loc : 0.0;
                double phi = InterfaceReconstruction::evaluate_limiter(limiter, r, c_T, c_U, c_B);
                flux_y[mesh.cell_idx(i, j)] = v_face * (c_T - 0.5 * phi * (1.0 - cfl_y) * dC_loc);
            }
        }
    }

    // Mise à jour conservative directe
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int idx = mesh.cell_idx(i, j);
            double fx_in  = flux_x[mesh.cell_idx(i, j)];
            double fx_out = flux_x[mesh.cell_idx(i + 1, j)];
            double fy_in  = flux_y[mesh.cell_idx(i, j)];
            double fy_out = flux_y[mesh.cell_idx(i, j + 1)];

            C_new[idx] = fields.C[idx] - (dt / dx) * (fx_out - fx_in) - (dt / dy) * (fy_out - fy_in);
            C_new[idx] = std::clamp(C_new[idx], 0.0, 1.0);
        }
    }

    fields.C = C_new;
}

void InterfaceTracker::advect_ciam(const Mesh2D& mesh, FlowFields& fields, double dt) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    ReconstructionMethod recon_method = InterfaceReconstruction::parse_reconstruction_method(vof_cfg.reconstruction_method);
    bool x_first = (step_count % 2 == 0);

    // Fonction lambda pour un balayage directionnel X
    auto sweep_x = [&](std::vector<double>& C_in, std::vector<double>& C_out) {
        InterfaceReconstruction::reconstruct(recon_method, mesh, C_in, interface_segments);
        std::vector<double> flux_vol_x((nx + 2) * (ny + 2), 0.0);

        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx + 1; ++i) {
                double u_face = fields.u[mesh.u_idx(i, j)];
                double delta_x = std::abs(u_face) * dt;
                if (delta_x <= 1.0e-14) continue;

                if (u_face > 0.0) {
                    // Donneur cellule i-1
                    int donor_idx = mesh.cell_idx(i - 1, j);
                    double x_face_pos = mesh.xv(i);
                    double box_xmin = x_face_pos - delta_x;
                    double box_xmax = x_face_pos;
                    double box_ymin = mesh.yv(j);
                    double box_ymax = mesh.yv(j + 1);

                    const auto& seg = interface_segments[donor_idx];
                    flux_vol_x[mesh.cell_idx(i, j)] = InterfaceReconstruction::cut_box_area(box_xmin, box_xmax, box_ymin, box_ymax,
                                                                                           seg.nx, seg.ny, seg.alpha);
                } else {
                    // Donneur cellule i
                    int donor_idx = mesh.cell_idx(i, j);
                    double x_face_pos = mesh.xv(i);
                    double box_xmin = x_face_pos;
                    double box_xmax = x_face_pos + delta_x;
                    double box_ymin = mesh.yv(j);
                    double box_ymax = mesh.yv(j + 1);

                    const auto& seg = interface_segments[donor_idx];
                    flux_vol_x[mesh.cell_idx(i, j)] = -InterfaceReconstruction::cut_box_area(box_xmin, box_xmax, box_ymin, box_ymax,
                                                                                            seg.nx, seg.ny, seg.alpha);
                }
            }
        }

        // Raccordement périodique strict des flux géométriques VOF
        if (!mesh.is_boundary_left && mesh.num_procs == 1) {
            for (int j = 1; j <= ny; ++j) {
                flux_vol_x[mesh.cell_idx(1, j)] = flux_vol_x[mesh.cell_idx(nx + 1, j)];
            }
        }

        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int idx = mesh.cell_idx(i, j);
                double f_in  = flux_vol_x[mesh.cell_idx(i, j)];
                double f_out = flux_vol_x[mesh.cell_idx(i + 1, j)];
                double u_in  = fields.u[mesh.u_idx(i, j)];
                double u_out = fields.u[mesh.u_idx(i + 1, j)];

                // Mise à jour conservative 1D avec compensation de divergence
                double denom = dx - dt * (u_out - u_in);
                if (denom > 1.0e-12) {
                    C_out[idx] = (C_in[idx] * dx - (f_out - f_in) / dy) / denom;
                } else {
                    C_out[idx] = C_in[idx] - (f_out - f_in) / (dx * dy);
                }
                C_out[idx] = std::clamp(C_out[idx], 0.0, 1.0);
            }
        }
        MpiComm::sync_cell_scalar(mesh, C_out);
    };

    // Fonction lambda pour un balayage directionnel Y
    auto sweep_y = [&](std::vector<double>& C_in, std::vector<double>& C_out) {
        InterfaceReconstruction::reconstruct(recon_method, mesh, C_in, interface_segments);
        std::vector<double> flux_vol_y((nx + 2) * (ny + 2), 0.0);

        for (int j = 1; j <= ny + 1; ++j) {
            for (int i = 1; i <= nx; ++i) {
                double v_face = fields.v[mesh.v_idx(i, j)];
                double delta_y = std::abs(v_face) * dt;
                if (delta_y <= 1.0e-14) continue;

                if (v_face > 0.0) {
                    // Donneur cellule j-1
                    int donor_idx = mesh.cell_idx(i, j - 1);
                    double y_face_pos = mesh.yv(j);
                    double box_xmin = mesh.xv(i);
                    double box_xmax = mesh.xv(i + 1);
                    double box_ymin = y_face_pos - delta_y;
                    double box_ymax = y_face_pos;

                    const auto& seg = interface_segments[donor_idx];
                    flux_vol_y[mesh.cell_idx(i, j)] = InterfaceReconstruction::cut_box_area(box_xmin, box_xmax, box_ymin, box_ymax,
                                                                                           seg.nx, seg.ny, seg.alpha);
                } else {
                    // Donneur cellule j
                    int donor_idx = mesh.cell_idx(i, j);
                    double y_face_pos = mesh.yv(j);
                    double box_xmin = mesh.xv(i);
                    double box_xmax = mesh.xv(i + 1);
                    double box_ymin = y_face_pos;
                    double box_ymax = y_face_pos + delta_y;

                    const auto& seg = interface_segments[donor_idx];
                    flux_vol_y[mesh.cell_idx(i, j)] = -InterfaceReconstruction::cut_box_area(box_xmin, box_xmax, box_ymin, box_ymax,
                                                                                            seg.nx, seg.ny, seg.alpha);
                }
            }
        }

        // Raccordement périodique strict des flux géométriques VOF en Y
        if (!mesh.is_boundary_bottom && mesh.num_procs == 1) {
            for (int i = 1; i <= nx; ++i) {
                flux_vol_y[mesh.cell_idx(i, 1)] = flux_vol_y[mesh.cell_idx(i, ny + 1)];
            }
        }

        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int idx = mesh.cell_idx(i, j);
                double f_in  = flux_vol_y[mesh.cell_idx(i, j)];
                double f_out = flux_vol_y[mesh.cell_idx(i, j + 1)];
                double v_in  = fields.v[mesh.v_idx(i, j)];
                double v_out = fields.v[mesh.v_idx(i, j + 1)];

                double denom = dy - dt * (v_out - v_in);
                if (denom > 1.0e-12) {
                    C_out[idx] = (C_in[idx] * dy - (f_out - f_in) / dx) / denom;
                } else {
                    C_out[idx] = C_in[idx] - (f_out - f_in) / (dx * dy);
                }
                C_out[idx] = std::clamp(C_out[idx], 0.0, 1.0);
            }
        }
        MpiComm::sync_cell_scalar(mesh, C_out);
    };

    std::vector<double> C_star = fields.C;
    std::vector<double> C_final = fields.C;

    if (x_first) {
        // Balayage X puis Y
        sweep_x(fields.C, C_star);
        sweep_y(C_star, C_final);
    } else {
        // Balayage Y puis X
        sweep_y(fields.C, C_star);
        sweep_x(C_star, C_final);
    }

    fields.C = C_final;
}

void InterfaceTracker::advect_weymouth_yue(const Mesh2D& mesh, FlowFields& fields, double dt) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;
    double cell_area = dx * dy;

    ReconstructionMethod recon_method = InterfaceReconstruction::parse_reconstruction_method(vof_cfg.reconstruction_method);
    bool x_first = (step_count % 2 == 0);

    // 1. Fonction indicatrice de dilatation binaire C_d(i,j) figee au temps n (Weymouth & Yue, JCP 2010, Eq. 14)
    std::vector<double> C_d((nx + 2) * (ny + 2), 0.0);
    for (int j = 0; j <= ny + 1; ++j) {
        for (int i = 0; i <= nx + 1; ++i) {
            int idx = mesh.cell_idx(i, j);
            C_d[idx] = (fields.C[idx] >= 0.5) ? 1.0 : 0.0;
        }
    }

    // Remplissage des conditions aux limites fantômes pour C
    auto fill_ghost_C = [&](std::vector<double>& c_arr) {
        MpiComm::sync_cell_scalar(mesh, c_arr);
        if (mesh.is_boundary_left) {
            for (int j = 1; j <= ny; ++j) c_arr[mesh.cell_idx(0, j)] = c_arr[mesh.cell_idx(1, j)];
        }
        if (mesh.is_boundary_right) {
            for (int j = 1; j <= ny; ++j) c_arr[mesh.cell_idx(nx + 1, j)] = c_arr[mesh.cell_idx(nx, j)];
        }
        if (mesh.is_boundary_bottom) {
            for (int i = 1; i <= nx; ++i) c_arr[mesh.cell_idx(i, 0)] = c_arr[mesh.cell_idx(i, 1)];
        }
        if (mesh.is_boundary_top) {
            if (bc.top.profile == "JET_INLET") {
                double xc = (bc.top.jet_center > 0.0) ? bc.top.jet_center : 0.5 * mesh.lx;
                double half_w = (bc.top.jet_width > 0.0) ? 0.5 * bc.top.jet_width : 0.04;
                for (int i = 1; i <= nx; ++i) {
                    double x_c = mesh.xc(i);
                    if (std::abs(x_c - xc) <= half_w) {
                        c_arr[mesh.cell_idx(i, ny + 1)] = 0.0;
                    } else {
                        c_arr[mesh.cell_idx(i, ny + 1)] = 1.0;
                    }
                }
            } else {
                for (int i = 1; i <= nx; ++i) c_arr[mesh.cell_idx(i, ny + 1)] = c_arr[mesh.cell_idx(i, ny)];
            }
        }
        c_arr[mesh.cell_idx(0, 0)]           = c_arr[mesh.cell_idx(1, 1)];
        c_arr[mesh.cell_idx(nx + 1, 0)]      = c_arr[mesh.cell_idx(nx, 1)];
        c_arr[mesh.cell_idx(0, ny + 1)]      = c_arr[mesh.cell_idx(1, ny)];
        c_arr[mesh.cell_idx(nx + 1, ny + 1)]  = c_arr[mesh.cell_idx(nx, ny)];
    };

    // 2. Balayage directionnel 1D selon X (Weymouth & Yue, JCP 2010, Eq. 15)
    auto sweep_x = [&](std::vector<double>& C_in, std::vector<double>& C_out, bool clamp_out) {
        fill_ghost_C(C_in);
        InterfaceReconstruction::reconstruct(recon_method, mesh, C_in, interface_segments);
        std::vector<double> f_vol_x((nx + 2) * (ny + 2), 0.0);

        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx + 1; ++i) {
                double u_face = fields.u[mesh.u_idx(i, j)];
                double delta_x = std::abs(u_face) * dt;
                if (delta_x <= 1.0e-14) continue;

                double x_pos = mesh.xv(i);
                if (u_face > 0.0) {
                    int donor_idx = mesh.cell_idx(i - 1, j);
                    const auto& seg = interface_segments[donor_idx];
                    f_vol_x[mesh.cell_idx(i, j)] = InterfaceReconstruction::cut_box_area(
                        x_pos - delta_x, x_pos, mesh.yv(j), mesh.yv(j + 1), seg.nx, seg.ny, seg.alpha);
                } else {
                    int donor_idx = mesh.cell_idx(i, j);
                    const auto& seg = interface_segments[donor_idx];
                    f_vol_x[mesh.cell_idx(i, j)] = -InterfaceReconstruction::cut_box_area(
                        x_pos, x_pos + delta_x, mesh.yv(j), mesh.yv(j + 1), seg.nx, seg.ny, seg.alpha);
                }
            }
        }

        // Raccordement périodique strict des flux géométriques VOF
        if (!mesh.is_boundary_left && mesh.num_procs == 1) {
            for (int j = 1; j <= ny; ++j) {
                f_vol_x[mesh.cell_idx(1, j)] = f_vol_x[mesh.cell_idx(nx + 1, j)];
            }
        }

        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int idx = mesh.cell_idx(i, j);
                double fx_in  = f_vol_x[mesh.cell_idx(i, j)];
                double fx_out = f_vol_x[mesh.cell_idx(i + 1, j)];
                double u_in   = fields.u[mesh.u_idx(i, j)];
                double u_out  = fields.u[mesh.u_idx(i + 1, j)];

                // Advection conservatrice Weymouth & Yue (JCP 2010, Eq. 15) avec compensation de dilatation 1D
                C_out[idx] = C_in[idx] - (fx_out - fx_in) / cell_area + C_d[idx] * (dt / dx) * (u_out - u_in);
                if (clamp_out) {
                    C_out[idx] = std::clamp(C_out[idx], 0.0, 1.0);
                }
            }
        }
        MpiComm::sync_cell_scalar(mesh, C_out);
    };

    // 3. Balayage directionnel 1D selon Y (Weymouth & Yue, JCP 2010, Eq. 16)
    auto sweep_y = [&](std::vector<double>& C_in, std::vector<double>& C_out, bool clamp_out) {
        fill_ghost_C(C_in);
        InterfaceReconstruction::reconstruct(recon_method, mesh, C_in, interface_segments);
        std::vector<double> f_vol_y((nx + 2) * (ny + 2), 0.0);

        for (int j = 1; j <= ny + 1; ++j) {
            for (int i = 1; i <= nx; ++i) {
                double v_face = fields.v[mesh.v_idx(i, j)];
                double delta_y = std::abs(v_face) * dt;
                if (delta_y <= 1.0e-14) continue;

                double y_pos = mesh.yv(j);
                if (v_face > 0.0) {
                    int donor_idx = mesh.cell_idx(i, j - 1);
                    const auto& seg = interface_segments[donor_idx];
                    f_vol_y[mesh.cell_idx(i, j)] = InterfaceReconstruction::cut_box_area(
                        mesh.xv(i), mesh.xv(i + 1), y_pos - delta_y, y_pos, seg.nx, seg.ny, seg.alpha);
                } else {
                    int donor_idx = mesh.cell_idx(i, j);
                    const auto& seg = interface_segments[donor_idx];
                    f_vol_y[mesh.cell_idx(i, j)] = -InterfaceReconstruction::cut_box_area(
                        mesh.xv(i), mesh.xv(i + 1), y_pos, y_pos + delta_y, seg.nx, seg.ny, seg.alpha);
                }
            }
        }

        // Raccordement périodique strict des flux géométriques VOF en Y
        if (!mesh.is_boundary_bottom && mesh.num_procs == 1) {
            for (int i = 1; i <= nx; ++i) {
                f_vol_y[mesh.cell_idx(i, 1)] = f_vol_y[mesh.cell_idx(i, ny + 1)];
            }
        }

        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int idx = mesh.cell_idx(i, j);
                double fy_in  = f_vol_y[mesh.cell_idx(i, j)];
                double fy_out = f_vol_y[mesh.cell_idx(i, j + 1)];
                double v_in   = fields.v[mesh.v_idx(i, j)];
                double v_out  = fields.v[mesh.v_idx(i, j + 1)];

                // Advection conservatrice Weymouth & Yue (JCP 2010, Eq. 16) avec compensation de dilatation 1D
                C_out[idx] = C_in[idx] - (fy_out - fy_in) / cell_area + C_d[idx] * (dt / dy) * (v_out - v_in);
                if (clamp_out) {
                    C_out[idx] = std::clamp(C_out[idx], 0.0, 1.0);
                }
            }
        }
        MpiComm::sync_cell_scalar(mesh, C_out);
    };

    std::vector<double> C_star = fields.C;
    std::vector<double> C_final = fields.C;

    if (x_first) {
        sweep_x(fields.C, C_star, false);
        sweep_y(C_star, C_final, true);
    } else {
        sweep_y(fields.C, C_star, false);
        sweep_x(C_star, C_final, true);
    }

    fields.C = C_final;
}

void InterfaceTracker::smooth_diffusion(const Mesh2D& mesh, FlowFields& fields) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    fields.C_tilde = fields.C;
    double dtau = 0.2 * std::min(dx * dx, dy * dy);
    std::vector<double> C_next = fields.C_tilde;

    for (int step = 0; step < diffusion_smoothing_steps; ++step) {
        MpiComm::sync_cell_scalar(mesh, fields.C_tilde);

        for (int j = 1; j <= ny; ++j) {
            for (int i = 1; i <= nx; ++i) {
                int idx = mesh.cell_idx(i, j);
                double lap_x = (fields.C_tilde[mesh.cell_idx(i + 1, j)] - 2.0 * fields.C_tilde[idx] + fields.C_tilde[mesh.cell_idx(i - 1, j)]) / (dx * dx);
                double lap_y = (fields.C_tilde[mesh.cell_idx(i, j + 1)] - 2.0 * fields.C_tilde[idx] + fields.C_tilde[mesh.cell_idx(i, j - 1)]) / (dy * dy);
                C_next[idx] = fields.C_tilde[idx] + dtau * (lap_x + lap_y);
            }
        }
        fields.C_tilde = C_next;
    }
    MpiComm::sync_cell_scalar(mesh, fields.C_tilde);
}

} // namespace covof
