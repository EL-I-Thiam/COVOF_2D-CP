#include "covof/InterfaceReconstruction.hpp"
#include <cmath>
#include <algorithm>
#include <array>
#include <iostream>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

// =============================================================================
// PARSING DES MÉTHODES ET LIMITATEURS
// =============================================================================

ReconstructionMethod InterfaceReconstruction::parse_reconstruction_method(const std::string& str) {
    if (str == "SLIC")   return ReconstructionMethod::SLIC;
    if (str == "LVIRA")  return ReconstructionMethod::LVIRA;
    if (str == "ELVIRA") return ReconstructionMethod::ELVIRA;
    return ReconstructionMethod::PLIC; // Défaut PLIC
}

AdvectionMethod InterfaceReconstruction::parse_advection_method(const std::string& str) {
    if (str == "CIAM")         return AdvectionMethod::CIAM;
    if (str == "WEYMOUTH_YUE") return AdvectionMethod::WEYMOUTH_YUE;
    return AdvectionMethod::VOF_TVD; // Défaut VOF-TVD
}

TvdLimiter InterfaceReconstruction::parse_tvd_limiter(const std::string& str) {
    if (str == "VAN_LEER") return TvdLimiter::VAN_LEER;
    if (str == "MUSCL")    return TvdLimiter::MUSCL;
    if (str == "HRIC")     return TvdLimiter::HRIC;
    return TvdLimiter::SUPERBEE; // Défaut Superbee
}

// =============================================================================
// GÉOMÉTRIE : DÉCOUPE D'UN RECTANGLE PAR UN DEMI-PLAN (SUTHERLAND-HODGMAN)
// =============================================================================

double InterfaceReconstruction::cut_box_area(double xmin, double xmax, double ymin, double ymax,
                                            double nx, double ny, double alpha) {
    // Vérification des dimensions de la boîte de contrôle
    double dx = xmax - xmin;
    double dy = ymax - ymin;
    if (dx <= 1.0e-14 || dy <= 1.0e-14) return 0.0;

    // Définition des 4 sommets du rectangle dans l'ordre trigonométrique
    struct Point { double x, y; };
    std::array<Point, 4> box_pts = {{
        {xmin, ymin},
        {xmax, ymin},
        {xmax, ymax},
        {xmin, ymax}
    }};

    // Calcul de la distance signée d(x, y) = nx * x + ny * y + alpha
    auto dist = [&](const Point& p) {
        return nx * p.x + ny * p.y + alpha;
    };

    // Polygone résultant de la troncature
    std::vector<Point> clipped;
    clipped.reserve(8);

    // Algorithme de Sutherland-Hodgman contre le demi-plan fluide d(x, y) <= 0
    for (size_t i = 0; i < 4; ++i) {
        Point p1 = box_pts[i];
        Point p2 = box_pts[(i + 1) % 4];
        double d1 = dist(p1);
        double d2 = dist(p2);

        if (d1 <= 0.0) {
            if (d2 <= 0.0) {
                // Les deux points sont à l'intérieur du fluide
                clipped.push_back(p2);
            } else {
                // Sortie du fluide : calcul du point d'intersection
                double t = -d1 / (d2 - d1);
                clipped.push_back({p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y)});
            }
        } else {
            if (d2 <= 0.0) {
                // Entrée dans le fluide : calcul de l'intersection puis ajout de p2
                double t = -d1 / (d2 - d1);
                clipped.push_back({p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y)});
                clipped.push_back(p2);
            }
            // Sinon les deux points sont hors du fluide, rien à ajouter
        }
    }

    // Si moins de 3 sommets, l'aire découpée est nulle
    if (clipped.size() < 3) return 0.0;

    // Calcul de l'aire par la formule de Green / Gauss (Shoelace formula)
    double area = 0.0;
    size_t m = clipped.size();
    for (size_t i = 0; i < m; ++i) {
        size_t next = (i + 1) % m;
        area += clipped[i].x * clipped[next].y - clipped[next].x * clipped[i].y;
    }
    return 0.5 * std::abs(area);
}

// =============================================================================
// INVERSION NUMÉRIQUE EXACTE : FRACTION VOLUMIQUE -> PARAMÈTRE ALPHA
// =============================================================================

double InterfaceReconstruction::compute_alpha(double xmin, double xmax, double ymin, double ymax,
                                             double nx, double ny, double C_target) {
    double total_area = (xmax - xmin) * (ymax - ymin);
    if (C_target <= 1.0e-7) return 1.0e9;   // Cellule quasi-vide
    if (C_target >= 1.0 - 1.0e-7) return -1.0e9; // Cellule quasi-pleine

    double target_area = std::clamp(C_target, 0.0, 1.0) * total_area;

    // Bornes extrêmes pour alpha
    // Les 4 sommets donnent l'étendue maximale de nx*x + ny*y
    std::array<double, 4> vals = {{
        nx * xmin + ny * ymin,
        nx * xmax + ny * ymin,
        nx * xmax + ny * ymax,
        nx * xmin + ny * ymax
    }};
    double min_val = *std::min_element(vals.begin(), vals.end());
    double max_val = *std::max_element(vals.begin(), vals.end());

    // Pour alpha_low = -max_val, nx*x + ny*y + alpha <= 0 est vide -> aire = 0
    // Pour alpha_high = -min_val, nx*x + ny*y + alpha <= 0 contient tout -> aire = total_area
    double alpha_low = -max_val - 1.0e-12;
    double alpha_high = -min_val + 1.0e-12;

    // Méthode de dichotomie / bisection pour trouver alpha exact à 1.0e-12
    for (int iter = 0; iter < 45; ++iter) {
        double alpha_mid = 0.5 * (alpha_low + alpha_high);
        double a = cut_box_area(xmin, xmax, ymin, ymax, nx, ny, alpha_mid);
        if (a < target_area) {
            alpha_low = alpha_mid;
        } else {
            alpha_high = alpha_mid;
        }
    }
    return 0.5 * (alpha_low + alpha_high);
}

// =============================================================================
// RECONSTRUCTION SLIC (NOH & WOODWARD 1976)
// =============================================================================

void InterfaceReconstruction::reconstruct_slic(const Mesh2D& mesh, const std::vector<double>& C,
                                              std::vector<InterfaceSegment>& segments) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;

    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int idx = mesh.cell_idx(i, j);
            double c = C[idx];

            if (c <= 1.0e-6) {
                segments[idx] = {1.0, 0.0, 1.0e9};
                continue;
            }
            if (c >= 1.0 - 1.0e-6) {
                segments[idx] = {1.0, 0.0, -1.0e9};
                continue;
            }

            double xmin = mesh.xv(i);
            double xmax = mesh.xv(i + 1);
            double ymin = mesh.yv(j);
            double ymax = mesh.yv(j + 1);

            double grad_x = std::abs(C[mesh.cell_idx(i + 1, j)] - C[mesh.cell_idx(i - 1, j)]);
            double grad_y = std::abs(C[mesh.cell_idx(i, j + 1)] - C[mesh.cell_idx(i, j - 1)]);

            double snx = 0.0, sny = 0.0;
            if (grad_x >= grad_y) {
                // Interface verticale
                if (C[mesh.cell_idx(i - 1, j)] >= C[mesh.cell_idx(i + 1, j)]) {
                    snx = 1.0; sny = 0.0; // Fluide à gauche
                } else {
                    snx = -1.0; sny = 0.0; // Fluide à droite
                }
            } else {
                // Interface horizontale
                if (C[mesh.cell_idx(i, j - 1)] >= C[mesh.cell_idx(i, j + 1)]) {
                    snx = 0.0; sny = 1.0; // Fluide en bas
                } else {
                    snx = 0.0; sny = -1.0; // Fluide en haut
                }
            }

            double alpha = compute_alpha(xmin, xmax, ymin, ymax, snx, sny, c);
            segments[idx] = {snx, sny, alpha};
        }
    }
}

// =============================================================================
// RECONSTRUCTION PLIC (YOUNGS 1982)
// =============================================================================

void InterfaceReconstruction::reconstruct_plic(const Mesh2D& mesh, const std::vector<double>& C,
                                              std::vector<InterfaceSegment>& segments) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int idx = mesh.cell_idx(i, j);
            double c = C[idx];

            if (c <= 1.0e-6) {
                segments[idx] = {1.0, 0.0, 1.0e9};
                continue;
            }
            if (c >= 1.0 - 1.0e-6) {
                segments[idx] = {1.0, 0.0, -1.0e9};
                continue;
            }

            // Calcul du gradient de Youngs sur stencil 3x3 centré
            double c_mm = C[mesh.cell_idx(i - 1, j - 1)];
            double c_0m = C[mesh.cell_idx(i,     j - 1)];
            double c_pm = C[mesh.cell_idx(i + 1, j - 1)];
            double c_m0 = C[mesh.cell_idx(i - 1, j    )];
            double c_p0 = C[mesh.cell_idx(i + 1, j    )];
            double c_mp = C[mesh.cell_idx(i - 1, j + 1)];
            double c_0p = C[mesh.cell_idx(i,     j + 1)];
            double c_pp = C[mesh.cell_idx(i + 1, j + 1)];

            double mx = -((c_pp + 2.0 * c_p0 + c_pm) - (c_mp + 2.0 * c_m0 + c_mm)) / (8.0 * dx);
            double my = -((c_pp + 2.0 * c_0p + c_mp) - (c_pm + 2.0 * c_0m + c_mm)) / (8.0 * dy);

            double norm = std::hypot(mx, my);
            double pnx = (norm > 1.0e-12) ? mx / norm : 1.0;
            double pny = (norm > 1.0e-12) ? my / norm : 0.0;

            double xmin = mesh.xv(i);
            double xmax = mesh.xv(i + 1);
            double ymin = mesh.yv(j);
            double ymax = mesh.yv(j + 1);

            double alpha = compute_alpha(xmin, xmax, ymin, ymax, pnx, pny, c);
            segments[idx] = {pnx, pny, alpha};
        }
    }
}

// =============================================================================
// RECONSTRUCTION ELVIRA (PILLIOD & PUCKETT 2004)
// =============================================================================

void InterfaceReconstruction::reconstruct_elvira(const Mesh2D& mesh, const std::vector<double>& C,
                                                std::vector<InterfaceSegment>& segments) {
    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int idx = mesh.cell_idx(i, j);
            double c = C[idx];

            if (c <= 1.0e-6) {
                segments[idx] = {1.0, 0.0, 1.0e9};
                continue;
            }
            if (c >= 1.0 - 1.0e-6) {
                segments[idx] = {1.0, 0.0, -1.0e9};
                continue;
            }

            double xmin = mesh.xv(i);
            double xmax = mesh.xv(i + 1);
            double ymin = mesh.yv(j);
            double ymax = mesh.yv(j + 1);

            // 1. Hauteurs de colonnes H_k (sommes sur y)
            double H_m1 = (C[mesh.cell_idx(i - 1, j - 1)] + C[mesh.cell_idx(i - 1, j)] + C[mesh.cell_idx(i - 1, j + 1)]) * dy;
            double H_0  = (C[mesh.cell_idx(i,     j - 1)] + C[mesh.cell_idx(i,     j)] + C[mesh.cell_idx(i,     j + 1)]) * dy;
            double H_p1 = (C[mesh.cell_idx(i + 1, j - 1)] + C[mesh.cell_idx(i + 1, j)] + C[mesh.cell_idx(i + 1, j + 1)]) * dy;

            // 2. Largeurs de lignes W_l (sommes sur x)
            double W_m1 = (C[mesh.cell_idx(i - 1, j - 1)] + C[mesh.cell_idx(i, j - 1)] + C[mesh.cell_idx(i + 1, j - 1)]) * dx;
            double W_0  = (C[mesh.cell_idx(i - 1, j    )] + C[mesh.cell_idx(i, j    )] + C[mesh.cell_idx(i + 1, j    )]) * dx;
            double W_p1 = (C[mesh.cell_idx(i - 1, j + 1)] + C[mesh.cell_idx(i, j + 1)] + C[mesh.cell_idx(i + 1, j + 1)]) * dx;

            // Détermination du signe de la normale
            double sgn_y = (C[mesh.cell_idx(i, j + 1)] < C[mesh.cell_idx(i, j - 1)]) ? 1.0 : -1.0;
            double sgn_x = (C[mesh.cell_idx(i + 1, j)] < C[mesh.cell_idx(i - 1, j)]) ? 1.0 : -1.0;

            // Ensemble des candidats de pentes
            std::vector<std::pair<double, double>> candidates;
            candidates.reserve(7);

            // Candidats verticaux (Height functions par colonnes)
            candidates.push_back({-sgn_y * (H_p1 - H_m1) / (2.0 * dx), sgn_y * 1.0}); // Centré
            candidates.push_back({-sgn_y * (H_p1 - H_0)  / (dx),       sgn_y * 1.0}); // Décentré avant
            candidates.push_back({-sgn_y * (H_0  - H_m1) / (dx),       sgn_y * 1.0}); // Décentré arrière

            // Candidats horizontaux (Height functions par lignes)
            candidates.push_back({sgn_x * 1.0, -sgn_x * (W_p1 - W_m1) / (2.0 * dy)}); // Centré
            candidates.push_back({sgn_x * 1.0, -sgn_x * (W_p1 - W_0)  / (dy)});       // Décentré avant
            candidates.push_back({sgn_x * 1.0, -sgn_x * (W_0  - W_m1) / (dy)});       // Décentré arrière

            // Candidat Youngs standard
            double mx = -((C[mesh.cell_idx(i+1,j+1)] + 2.0*C[mesh.cell_idx(i+1,j)] + C[mesh.cell_idx(i+1,j-1)]) -
                          (C[mesh.cell_idx(i-1,j+1)] + 2.0*C[mesh.cell_idx(i-1,j)] + C[mesh.cell_idx(i-1,j-1)])) / (8.0 * dx);
            double my = -((C[mesh.cell_idx(i+1,j+1)] + 2.0*C[mesh.cell_idx(i,j+1)] + C[mesh.cell_idx(i-1,j+1)]) -
                          (C[mesh.cell_idx(i+1,j-1)] + 2.0*C[mesh.cell_idx(i,j-1)] + C[mesh.cell_idx(i-1,j-1)])) / (8.0 * dy);
            candidates.push_back({mx, my});

            // Évaluation de l'erreur quadratique pour chaque candidat sur le stencil 3x3
            double best_err = 1.0e30;
            double best_nx = 1.0, best_ny = 0.0, best_alpha = 0.0;

            for (const auto& cand : candidates) {
                double cnorm = std::hypot(cand.first, cand.second);
                if (cnorm < 1.0e-12) continue;
                double cnx = cand.first / cnorm;
                double cny = cand.second / cnorm;

                double calpha = compute_alpha(xmin, xmax, ymin, ymax, cnx, cny, c);

                // Somme des carrés des erreurs sur les 8 voisins
                double err = 0.0;
                for (int dj = -1; dj <= 1; ++dj) {
                    for (int di = -1; di <= 1; ++di) {
                        if (di == 0 && dj == 0) continue;
                        double v_xmin = mesh.xv(i + di);
                        double v_xmax = mesh.xv(i + di + 1);
                        double v_ymin = mesh.yv(j + dj);
                        double v_ymax = mesh.yv(j + dj + 1);

                        double v_target = C[mesh.cell_idx(i + di, j + dj)] * dx * dy;
                        double v_recon  = cut_box_area(v_xmin, v_xmax, v_ymin, v_ymax, cnx, cny, calpha);
                        double diff = v_recon - v_target;
                        err += diff * diff;
                    }
                }

                if (err < best_err) {
                    best_err = err;
                    best_nx = cnx;
                    best_ny = cny;
                    best_alpha = calpha;
                }
            }

            segments[idx] = {best_nx, best_ny, best_alpha};
        }
    }
}

// =============================================================================
// RECONSTRUCTION LVIRA (PILLIOD & PUCKETT 2004)
// =============================================================================

void InterfaceReconstruction::reconstruct_lvira(const Mesh2D& mesh, const std::vector<double>& C,
                                                std::vector<InterfaceSegment>& segments) {
    // 1. Initialisation par ELVIRA pour obtenir un excellent point de départ
    reconstruct_elvira(mesh, C, segments);

    int nx = mesh.nx_local;
    int ny = mesh.ny_local;
    double dx = mesh.dx;
    double dy = mesh.dy;

    // 2. Raffinement par optimisation continue locale de l'angle theta
    for (int j = 1; j <= ny; ++j) {
        for (int i = 1; i <= nx; ++i) {
            int idx = mesh.cell_idx(i, j);
            double c = C[idx];
            if (c <= 1.0e-6 || c >= 1.0 - 1.0e-6) continue;

            double xmin = mesh.xv(i);
            double xmax = mesh.xv(i + 1);
            double ymin = mesh.yv(j);
            double ymax = mesh.yv(j + 1);

            double theta0 = std::atan2(segments[idx].ny, segments[idx].nx);

            // Fonction coût : erreur quadratique pour un angle theta donné
            auto cost_func = [&](double theta) {
                double lnx = std::cos(theta);
                double lny = std::sin(theta);
                double lalpha = compute_alpha(xmin, xmax, ymin, ymax, lnx, lny, c);

                double err = 0.0;
                for (int dj = -1; dj <= 1; ++dj) {
                    for (int di = -1; di <= 1; ++di) {
                        if (di == 0 && dj == 0) continue;
                        double v_xmin = mesh.xv(i + di);
                        double v_xmax = mesh.xv(i + di + 1);
                        double v_ymin = mesh.yv(j + dj);
                        double v_ymax = mesh.yv(j + dj + 1);

                        double v_target = C[mesh.cell_idx(i + di, j + dj)] * dx * dy;
                        double v_recon  = cut_box_area(v_xmin, v_xmax, v_ymin, v_ymax, lnx, lny, lalpha);
                        double diff = v_recon - v_target;
                        err += diff * diff;
                    }
                }
                return err;
            };

            // Recherche par Section Dorée (Golden Section Search) dans [theta0 - pi/6, theta0 + pi/6]
            const double invphi = (std::sqrt(5.0) - 1.0) / 2.0; // 0.6180339887...
            const double invphi2 = (3.0 - std::sqrt(5.0)) / 2.0; // 0.3819660113...

            double a = theta0 - 0.5;
            double b = theta0 + 0.5;
            double h = b - a;

            double c_pt = a + invphi2 * h;
            double d_pt = a + invphi * h;
            double fc = cost_func(c_pt);
            double fd = cost_func(d_pt);

            for (int iter = 0; iter < 12; ++iter) {
                if (fc < fd) {
                    b = d_pt;
                    d_pt = c_pt;
                    fd = fc;
                    h = invphi * h;
                    c_pt = a + invphi2 * h;
                    fc = cost_func(c_pt);
                } else {
                    a = c_pt;
                    c_pt = d_pt;
                    fc = fd;
                    h = invphi * h;
                    d_pt = a + invphi * h;
                    fd = cost_func(d_pt);
                }
            }

            double best_theta = 0.5 * (a + b);
            double best_nx = std::cos(best_theta);
            double best_ny = std::sin(best_theta);
            double best_alpha = compute_alpha(xmin, xmax, ymin, ymax, best_nx, best_ny, c);

            segments[idx] = {best_nx, best_ny, best_alpha};
        }
    }
}

// =============================================================================
// DISPATCHING GÉNÉRIQUE
// =============================================================================

void InterfaceReconstruction::reconstruct(ReconstructionMethod method, const Mesh2D& mesh,
                                          const std::vector<double>& C,
                                          std::vector<InterfaceSegment>& segments) {
    switch (method) {
        case ReconstructionMethod::SLIC:
            reconstruct_slic(mesh, C, segments);
            break;
        case ReconstructionMethod::PLIC:
            reconstruct_plic(mesh, C, segments);
            break;
        case ReconstructionMethod::LVIRA:
            reconstruct_lvira(mesh, C, segments);
            break;
        case ReconstructionMethod::ELVIRA:
            reconstruct_elvira(mesh, C, segments);
            break;
    }
}

// =============================================================================
// ÉVALUATION DES LIMITEURS DE FLUX (SUPERBEE, VAN LEER, MUSCL, HRIC)
// =============================================================================

double InterfaceReconstruction::evaluate_limiter(TvdLimiter limiter, double r, double c_donor, double c_upwind, double c_acceptor) {
    switch (limiter) {
        case TvdLimiter::SUPERBEE:
            // Limiteur Superbee de Roe (1985)
            return std::max(0.0, std::max(std::min(2.0 * r, 1.0), std::min(r, 2.0)));

        case TvdLimiter::VAN_LEER:
            // Limiteur harmonique de Van Leer (1974)
            return (r > 0.0) ? (2.0 * r / (1.0 + r)) : 0.0;

        case TvdLimiter::MUSCL:
            // Limiteur linéaire par morceaux MUSCL (Van Leer 1979)
            return std::max(0.0, std::min({2.0 * r, 0.5 * (1.0 + r), 2.0}));

        case TvdLimiter::HRIC: {
            // High Resolution Interface Capturing (Muzaferija & Peric 1999) dans l'espace NVD
            double denom = c_acceptor - c_upwind;
            if (std::abs(denom) < 1.0e-12) return 0.0;
            double c_tilde_D = (c_donor - c_upwind) / denom;

            double c_tilde_f = c_tilde_D;
            if (c_tilde_D >= 0.0 && c_tilde_D <= 1.0) {
                if (c_tilde_D < 0.5) {
                    c_tilde_f = 2.0 * c_tilde_D;
                } else {
                    c_tilde_f = 1.0;
                }
            }

            // Conversion de la valeur normalisée à la face en limiteur équivalent phi(r)
            // C_f = C_D + 0.5 * phi * (1 - C_D) dans le formalisme NVD
            if (std::abs(1.0 - c_tilde_D) > 1.0e-12) {
                double phi = 2.0 * (c_tilde_f - c_tilde_D) / (1.0 - c_tilde_D);
                return std::clamp(phi, 0.0, 2.0);
            }
            return 1.0;
        }
    }
    return 0.0;
}

} // namespace covof
