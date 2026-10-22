#include "fugu/InterfaceTracker.hpp"
#include "fugu/MpiComm.hpp"
#include <cmath>
#include <algorithm>

namespace fugu {

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
            } else if (ic.type == InitialConditionType::BUBBLE) {
                // Déclaration de variable typée
                double dist = std::sqrt((x - ic.center_x) * (x - ic.center_x) +
                                        (y - ic.center_y) * (y - ic.center_y)) - ic.radius;
                // Évaluation de la condition de contrôle
                if (dist < -eps) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.0; // Intérieur bulle : fluide 2
                } else if (dist > eps) {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 1.0; // Extérieur : fluide 1 (eau)
                } else {
                    // Calcul de la résolution géométrique de cellule
                    fields.C[idx] = 0.5 + dist / (2.0 * eps) + (1.0 / (2.0 * M_PI)) * std::sin(M_PI * dist / eps);
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
    update_physical_properties(mesh, fields);
    compute_curvature_and_CSF(mesh, fields);
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
            // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
            for (int j = 0; j <= ny + 1; ++j) {
                // Déclaration de variable typée
                double y = mesh.yc(j);
                // Déclaration de variable typée
                double z = y - visc_e;
                // Déclaration de variable typée
                double mu_val = visc_mu0 * std::exp(visc_a * (z * z) / (visc_e * visc_e));
                // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
                for (int i = 0; i <= nx + 1; ++i) {
                    // Calcul de la résolution géométrique de cellule
                    fields.mu[mesh.cell_idx(i, j)] = mu_val;
                }
            }
            // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
            for (int j = 0; j <= ny + 1; ++j) {
                // Déclaration de variable typée
                double y = mesh.yv(j);
                // Déclaration de variable typée
                double z = y - visc_e;
                // Déclaration de variable typée
                double mu_node_val = visc_mu0 * std::exp(visc_a * (z * z) / (visc_e * visc_e));
                // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
                for (int i = 0; i <= nx + 1; ++i) {
                    // Calcul de la résolution géométrique de cellule
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

    // Remplissage des coins pour C
    fields.C[mesh.cell_idx(0, 0)]           = fields.C[mesh.cell_idx(1, 1)];
    // Calcul de la résolution géométrique de cellule
    fields.C[mesh.cell_idx(nx + 1, 0)]      = fields.C[mesh.cell_idx(nx, 1)];
    // Calcul de la résolution géométrique de cellule
    fields.C[mesh.cell_idx(0, ny + 1)]      = fields.C[mesh.cell_idx(1, ny)];
    // Calcul de la résolution géométrique de cellule
    fields.C[mesh.cell_idx(nx + 1, ny + 1)]  = fields.C[mesh.cell_idx(nx, ny)];

    // 1. Masse volumique et viscosité au centre des cellules
    for (int j = 0; j <= ny + 1; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i <= nx + 1; ++i) {
            // Déclaration de variable typée
            int idx = mesh.cell_idx(i, j);
            // Déclaration de variable typée
            double c = std::clamp(fields.C[idx], 0.0, 1.0);
            // Calcul de la résolution géométrique de cellule
            fields.rho[idx] = c * fluid1.rho + (1.0 - c) * fluid2.rho;
            // Calcul de la résolution géométrique de cellule
            fields.mu[idx]  = c * fluid1.mu  + (1.0 - c) * fluid2.mu;
        }
    }

    // Calcul de la résolution géométrique de cellule
    fields.mu[mesh.cell_idx(0, 0)]           = fields.mu[mesh.cell_idx(1, 1)];
    // Calcul de la résolution géométrique de cellule
    fields.mu[mesh.cell_idx(nx + 1, 0)]      = fields.mu[mesh.cell_idx(nx, 1)];
    // Calcul de la résolution géométrique de cellule
    fields.mu[mesh.cell_idx(0, ny + 1)]      = fields.mu[mesh.cell_idx(1, ny)];
    // Calcul de la résolution géométrique de cellule
    fields.mu[mesh.cell_idx(nx + 1, ny + 1)]  = fields.mu[mesh.cell_idx(nx, ny)];

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
            // Déclaration de variable typée
            double m_SW = fields.mu[mesh.cell_idx(i - 1, j - 1)];
            // Déclaration de variable typée
            double m_SE = fields.mu[mesh.cell_idx(i, j - 1)];
            // Déclaration de variable typée
            double m_NW = fields.mu[mesh.cell_idx(i - 1, j)];
            // Déclaration de variable typée
            double m_NE = fields.mu[mesh.cell_idx(i, j)];

            // Calcul de la résolution géométrique de cellule
            fields.mu_node[n_idx] = 4.0 / (1.0 / m_SW + 1.0 / m_SE + 1.0 / m_NW + 1.0 / m_NE);
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
    for (int j = 1; j <= ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Déclaration de variable typée
            int idx = mesh.cell_idx(i, j);
            // Déclaration de variable typée
            double dC_dx = (fields.C[mesh.cell_idx(i + 1, j)] - fields.C[mesh.cell_idx(i - 1, j)]) / (2.0 * dx);
            // Déclaration de variable typée
            double dC_dy = (fields.C[mesh.cell_idx(i, j + 1)] - fields.C[mesh.cell_idx(i, j - 1)]) / (2.0 * dy);
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
    // Déclaration de variable typée
    int nx = mesh.nx_local;
    // Déclaration de variable typée
    int ny = mesh.ny_local;
    // Déclaration de variable typée
    double dx = mesh.dx;
    // Déclaration de variable typée
    double dy = mesh.dy;

    // Déclaration de variable typée
    std::vector<double> C_new = fields.C;

    // Schéma d'advection TVD avec limiteur de flux Superbee
    auto superbee = [&](double r) {
        // Retour de fonction
        return std::max(0.0, std::max(std::min(2.0 * r, 1.0), std::min(r, 2.0)));
    };

    // Calcul des flux de fraction volumique sur les faces X
    std::vector<double> flux_x((nx + 2) * (ny + 2), 0.0);
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx + 1; ++i) {
            // Déclaration de variable typée
            double u_face = fields.u[mesh.u_idx(i, j)];
            // Déclaration de variable typée
            double c_L = fields.C[mesh.cell_idx(i - 1, j)];
            // Déclaration de variable typée
            double c_R = fields.C[mesh.cell_idx(i, j)];

            // Évaluation de la condition de contrôle
            if (u_face >= 0.0) {
                // Déclaration de variable typée
                int i_up = std::max(0, i - 2);
                // Déclaration de variable typée
                double dC_up = c_L - fields.C[mesh.cell_idx(i_up, j)];
                // Déclaration de variable typée
                double dC_loc = c_R - c_L;
                // Déclaration de variable typée
                double r = (std::abs(dC_loc) > 1.0e-12) ? dC_up / dC_loc : 0.0;
                // Déclaration de variable typée
                double phi = superbee(r);
                // Calcul de la résolution géométrique de cellule
                flux_x[mesh.cell_idx(i, j)] = u_face * (c_L + 0.5 * phi * (1.0 - u_face * dt / dx) * dC_loc);
            } else {
                // Déclaration de variable typée
                int i_down = std::min(nx + 1, i + 1);
                // Déclaration de variable typée
                double dC_up = fields.C[mesh.cell_idx(i_down, j)] - c_R;
                // Déclaration de variable typée
                double dC_loc = c_R - c_L;
                // Déclaration de variable typée
                double r = (std::abs(dC_loc) > 1.0e-12) ? dC_up / dC_loc : 0.0;
                // Déclaration de variable typée
                double phi = superbee(r);
                // Calcul de la résolution géométrique de cellule
                flux_x[mesh.cell_idx(i, j)] = u_face * (c_R - 0.5 * phi * (1.0 + u_face * dt / dx) * dC_loc);
            }
        }
    }

    // Calcul des flux sur les faces Y
    std::vector<double> flux_y((nx + 2) * (ny + 2), 0.0);
    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= ny + 1; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Déclaration de variable typée
            double v_face = fields.v[mesh.v_idx(i, j)];
            // Déclaration de variable typée
            double c_B = fields.C[mesh.cell_idx(i, j - 1)];
            // Déclaration de variable typée
            double c_T = fields.C[mesh.cell_idx(i, j)];

            // Évaluation de la condition de contrôle
            if (v_face >= 0.0) {
                // Déclaration de variable typée
                int j_up = std::max(0, j - 2);
                // Déclaration de variable typée
                double dC_up = c_B - fields.C[mesh.cell_idx(i, j_up)];
                // Déclaration de variable typée
                double dC_loc = c_T - c_B;
                // Déclaration de variable typée
                double r = (std::abs(dC_loc) > 1.0e-12) ? dC_up / dC_loc : 0.0;
                // Déclaration de variable typée
                double phi = superbee(r);
                // Calcul de la résolution géométrique de cellule
                flux_y[mesh.cell_idx(i, j)] = v_face * (c_B + 0.5 * phi * (1.0 - v_face * dt / dy) * dC_loc);
            } else {
                // Déclaration de variable typée
                int j_down = std::min(ny + 1, j + 1);
                // Déclaration de variable typée
                double dC_up = fields.C[mesh.cell_idx(i, j_down)] - c_T;
                // Déclaration de variable typée
                double dC_loc = c_T - c_B;
                // Déclaration de variable typée
                double r = (std::abs(dC_loc) > 1.0e-12) ? dC_up / dC_loc : 0.0;
                // Déclaration de variable typée
                double phi = superbee(r);
                // Calcul de la résolution géométrique de cellule
                flux_y[mesh.cell_idx(i, j)] = v_face * (c_T - 0.5 * phi * (1.0 + v_face * dt / dy) * dC_loc);
            }
        }
    }

    // Mise à jour conservative
    for (int j = 1; j <= ny; ++j) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= nx; ++i) {
            // Déclaration de variable typée
            int idx = mesh.cell_idx(i, j);
            // Déclaration de variable typée
            double fx_in  = flux_x[mesh.cell_idx(i, j)];
            // Déclaration de variable typée
            double fx_out = flux_x[mesh.cell_idx(i + 1, j)];
            // Déclaration de variable typée
            double fy_in  = flux_y[mesh.cell_idx(i, j)];
            // Déclaration de variable typée
            double fy_out = flux_y[mesh.cell_idx(i, j + 1)];

            // Calcul de la résolution géométrique de cellule
            C_new[idx] = fields.C[idx] - (dt / dx) * (fx_out - fx_in) - (dt / dy) * (fy_out - fy_in);
            // Calcul de la résolution géométrique de cellule
            C_new[idx] = std::clamp(C_new[idx], 0.0, 1.0);
        }
    }

    // Calcul et assignation numérique
    fields.C = C_new;
    MpiComm::sync_cell_scalar(mesh, fields.C);

    // Mise à jour des grandeurs physiques dérivées
    update_physical_properties(mesh, fields);
    compute_curvature_and_CSF(mesh, fields);
}

} // namespace fugu
