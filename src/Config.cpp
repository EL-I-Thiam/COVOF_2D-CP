#include "covof/Config.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

// Fonctions utilitaires locales pour parser le JSON simplement
static std::string trim(const std::string& str) {
    // Déclaration de variable typée
    size_t first = str.find_first_not_of(" \t\n\r");
    // Évaluation de la condition de contrôle
    if (first == std::string::npos) return "";
    // Déclaration de variable typée
    size_t last = str.find_last_not_of(" \t\n\r");
    // Retour de fonction
    return str.substr(first, (last - first + 1));
}

static std::string extract_value(const std::string& content, const std::string& key) {
    // Déclaration de variable typée
    std::string search = "\"" + key + "\"";
    // Déclaration de variable typée
    size_t pos = content.find(search);
    // Évaluation de la condition de contrôle
    if (pos == std::string::npos) return "";
    // Déclaration de variable typée
    size_t colon = content.find(':', pos + search.length());
    // Évaluation de la condition de contrôle
    if (colon == std::string::npos) return "";
    
    // Déclaration de variable typée
    size_t start = content.find_first_not_of(" \t\n\r", colon + 1);
    // Évaluation de la condition de contrôle
    if (start == std::string::npos) return "";

    // Évaluation de la condition de contrôle
    if (content[start] == '{' || content[start] == '[') {
        // Calcul et assignation numérique
        char open_c = content[start];
        // Calcul et assignation numérique
        char close_c = (open_c == '{') ? '}' : ']';
        // Déclaration de variable typée
        int depth = 1;
        // Déclaration de variable typée
        size_t end = start + 1;
        // Boucle itérative conditionnelle
        while (end < content.length() && depth > 0) {
            // Évaluation de la condition de contrôle
            if (content[end] == open_c) depth++;
            // Branche alternative conditionnelle
            else if (content[end] == close_c) depth--;
            end++;
        }
        // Retour de fonction
        return content.substr(start, end - start);
    // Calcul et assignation numérique
    } else if (content[start] == '"') {
        // Déclaration de variable typée
        size_t end = content.find('"', start + 1);
        // Évaluation de la condition de contrôle
        if (end == std::string::npos) return "";
        // Retour de fonction
        return content.substr(start + 1, end - start - 1);
    } else {
        // Déclaration de variable typée
        size_t end = content.find_first_of(",}\n\r", start);
        // Évaluation de la condition de contrôle
        if (end == std::string::npos) end = content.length();
        // Retour de fonction
        return trim(content.substr(start, end - start));
    }
}

static double get_double(const std::string& block, const std::string& key, double default_val) {
    // Déclaration de variable typée
    std::string val = extract_value(block, key);
    // Évaluation de la condition de contrôle
    if (val.empty()) return default_val;
    try { return std::stod(val); } catch (...) { return default_val; }
}

static int get_int(const std::string& block, const std::string& key, int default_val) {
    // Déclaration de variable typée
    std::string val = extract_value(block, key);
    // Évaluation de la condition de contrôle
    if (val.empty()) return default_val;
    try { return std::stoi(val); } catch (...) { return default_val; }
}

static bool get_bool(const std::string& block, const std::string& key, bool default_val) {
    // Déclaration de variable typée
    std::string val = extract_value(block, key);
    // Évaluation de la condition de contrôle
    if (val == "true" || val == "1") return true;
    // Évaluation de la condition de contrôle
    if (val == "false" || val == "0") return false;
    // Retour de fonction
    return default_val;
}

static std::string get_string(const std::string& block, const std::string& key, const std::string& default_val) {
    // Déclaration de variable typée
    std::string val = extract_value(block, key);
    // Retour de fonction
    return val.empty() ? default_val : val;
}

static BCType parse_bc_type(const std::string& str) {
    // Évaluation de la condition de contrôle
    if (str == "DIRICHLET") return BCType::DIRICHLET;
    // Évaluation de la condition de contrôle
    if (str == "NEUMANN") return BCType::NEUMANN;
    // Évaluation de la condition de contrôle
    if (str == "FREE_SLIP") return BCType::FREE_SLIP;
    // Évaluation de la condition de contrôle
    if (str == "PERIODIC") return BCType::PERIODIC;
    // Retour de fonction
    return BCType::DIRICHLET;
}

static void parse_single_bc(const std::string& block, BoundaryCondition& bc) {
    // Calcul et assignation numérique
    bc.type = parse_bc_type(get_string(block, "type", "DIRICHLET"));
    // Calcul et assignation numérique
    bc.u_val = get_double(block, "u", 0.0);
    // Calcul et assignation numérique
    bc.v_val = get_double(block, "v", 0.0);
    // Calcul et assignation numérique
    bc.profile = get_string(block, "profile", "UNIFORM");
    // Calcul et assignation numérique
    bc.u_mean = get_double(block, "u_mean", 0.0);
    // Calcul et assignation numérique
    bc.y_step = get_double(block, "y_step", 0.0);
    // Calcul et assignation numérique
    bc.T_type = parse_bc_type(get_string(block, "T_type", "NEUMANN"));
    // Calcul et assignation numérique
    bc.T_val = get_double(block, "T", 0.0);
}

// Déclaration de variable typée
bool Config::load_from_json(const std::string& filepath) {
    std::ifstream file(filepath);
    // Évaluation de la condition de contrôle
    if (!file.is_open()) {
        std::cerr << "[Config] Erreur: Impossible d'ouvrir le fichier JSON : " << filepath << std::endl;
        // Retour de fonction
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    // Déclaration de variable typée
    std::string json = buffer.str();

    // 0. Mode global et arrêt stationnaire
    mode = get_string(json, "mode", mode);
    // Calcul de la résolution géométrique de cellule
    steady_state_tol = get_double(json, "steady_state_tol", steady_state_tol);

    // 1. Maillage
    std::string mesh_block = extract_value(json, "mesh");
    // Évaluation de la condition de contrôle
    if (!mesh_block.empty()) {
        // Calcul et assignation numérique
        nx = get_int(mesh_block, "nx", nx);
        // Calcul et assignation numérique
        ny = get_int(mesh_block, "ny", ny);
        // Calcul et assignation numérique
        lx = get_double(mesh_block, "lx", lx);
        // Calcul et assignation numérique
        ly = get_double(mesh_block, "ly", ly);
    }

    // 2. Modèle physique
    std::string model_block = extract_value(json, "physics");
    // Évaluation de la condition de contrôle
    if (!model_block.empty()) {
        // Calcul et assignation numérique
        enable_two_phase = get_bool(model_block, "enable_two_phase", enable_two_phase);
        // Calcul et assignation numérique
        enable_surface_tension = get_bool(model_block, "enable_surface_tension", enable_surface_tension);
        // Calcul et assignation numérique
        enable_convection = get_bool(model_block, "enable_convection", enable_convection);
        // Calcul et assignation numérique
        enable_energy = get_bool(model_block, "enable_energy", enable_energy);
        // Calcul et assignation numérique
        sigma = get_double(model_block, "sigma", sigma);
        // Calcul et assignation numérique
        gx = get_double(model_block, "gx", gx);
        // Calcul et assignation numérique
        gy = get_double(model_block, "gy", gy);

        // Calcul et assignation numérique
        Ra = get_double(model_block, "Ra", Ra);
        // Calcul et assignation numérique
        Pr = get_double(model_block, "Pr", Pr);
        // Calcul et assignation numérique
        T_ref = get_double(model_block, "T_ref", T_ref);
        // Calcul et assignation numérique
        beta = get_double(model_block, "beta", beta);

        // Calcul et assignation numérique
        viscosity_model = get_string(model_block, "viscosity_model", viscosity_model);
        // Calcul et assignation numérique
        visc_a = get_double(model_block, "visc_a", visc_a);
        // Calcul et assignation numérique
        visc_e = get_double(model_block, "visc_e", visc_e);
        // Calcul et assignation numérique
        visc_mu0 = get_double(model_block, "visc_mu0", visc_mu0);

        // Déclaration de variable typée
        std::string f1_block = extract_value(model_block, "fluid1");
        // Évaluation de la condition de contrôle
        if (!f1_block.empty()) {
            // Calcul et assignation numérique
            fluid1.rho = get_double(f1_block, "rho", fluid1.rho);
            // Calcul et assignation numérique
            fluid1.mu  = get_double(f1_block, "mu", fluid1.mu);
            // Calcul et assignation numérique
            fluid1.Cp  = get_double(f1_block, "Cp", fluid1.Cp);
            // Calcul et assignation numérique
            fluid1.k_thermal = get_double(f1_block, "k_thermal", fluid1.k_thermal);
            // Calcul et assignation numérique
            fluid1.beta = get_double(f1_block, "beta", fluid1.beta);
        }

        // Déclaration de variable typée
        std::string f2_block = extract_value(model_block, "fluid2");
        // Évaluation de la condition de contrôle
        if (!f2_block.empty()) {
            // Calcul et assignation numérique
            fluid2.rho = get_double(f2_block, "rho", fluid2.rho);
            // Calcul et assignation numérique
            fluid2.mu  = get_double(f2_block, "mu", fluid2.mu);
            // Calcul et assignation numérique
            fluid2.Cp  = get_double(f2_block, "Cp", fluid2.Cp);
            // Calcul et assignation numérique
            fluid2.k_thermal = get_double(f2_block, "k_thermal", fluid2.k_thermal);
            // Calcul et assignation numérique
            fluid2.beta = get_double(f2_block, "beta", fluid2.beta);
        }
    }

    // 3. Intégration temporelle
    std::string time_block = extract_value(json, "time");
    // Évaluation de la condition de contrôle
    if (!time_block.empty()) {
        // Calcul et assignation numérique
        dt = get_double(time_block, "dt", dt);
        // Calcul et assignation numérique
        t_final = get_double(time_block, "t_final", t_final);
        // Calcul et assignation numérique
        cfl_max = get_double(time_block, "cfl_max", cfl_max);
        // Calcul et assignation numérique
        adaptive_dt = get_bool(time_block, "adaptive_dt", adaptive_dt);
    }

    // 4. Conditions aux limites
    std::string bc_block = extract_value(json, "boundary_conditions");
    // Évaluation de la condition de contrôle
    if (!bc_block.empty()) {
        // Déclaration de variable typée
        std::string b_left = extract_value(bc_block, "left");
        // Évaluation de la condition de contrôle
        if (!b_left.empty()) parse_single_bc(b_left, bc.left);

        // Déclaration de variable typée
        std::string b_right = extract_value(bc_block, "right");
        // Évaluation de la condition de contrôle
        if (!b_right.empty()) parse_single_bc(b_right, bc.right);

        // Déclaration de variable typée
        std::string b_bot = extract_value(bc_block, "bottom");
        // Évaluation de la condition de contrôle
        if (!b_bot.empty()) parse_single_bc(b_bot, bc.bottom);

        // Déclaration de variable typée
        std::string b_top = extract_value(bc_block, "top");
        // Évaluation de la condition de contrôle
        if (!b_top.empty()) parse_single_bc(b_top, bc.top);
    }

    // 5. Condition initiale
    std::string ic_block = extract_value(json, "initial_condition");
    // Évaluation de la condition de contrôle
    if (!ic_block.empty()) {
        // Déclaration de variable typée
        std::string type_str = get_string(ic_block, "type", "SINGLE_PHASE");
        // Évaluation de la condition de contrôle
        if (type_str == "DROPLET") init_cond.type = InitialConditionType::DROPLET;
        // Branche alternative conditionnelle pour bulle ou cylindre immergé
        else if (type_str == "BUBBLE" || type_str == "CYLINDER") init_cond.type = InitialConditionType::BUBBLE;
        // Branche alternative conditionnelle
        else if (type_str == "DAM_BREAK") init_cond.type = InitialConditionType::DAM_BREAK;
        // Branche alternative conditionnelle
        else if (type_str == "SHEAR_LAYER") init_cond.type = InitialConditionType::SHEAR_LAYER;
        // Branche alternative conditionnelle
        else if (type_str == "VORTEX_SHEAR") init_cond.type = InitialConditionType::VORTEX_SHEAR;
        // Branche alternative conditionnelle
        else if (type_str == "RAYLEIGH_TAYLOR") init_cond.type = InitialConditionType::RAYLEIGH_TAYLOR;
        // Branche par défaut
        else init_cond.type = InitialConditionType::SINGLE_PHASE;

        // Calcul et assignation numérique
        init_cond.center_x = get_double(ic_block, "center_x", init_cond.center_x);
        // Calcul et assignation numérique
        init_cond.center_y = get_double(ic_block, "center_y", init_cond.center_y);
        // Calcul et assignation numérique
        init_cond.radius   = get_double(ic_block, "radius", init_cond.radius);
        // Calcul et assignation numérique
        init_cond.box_x0   = get_double(ic_block, "box_x0", init_cond.box_x0);
        // Calcul et assignation numérique
        init_cond.box_y0   = get_double(ic_block, "box_y0", init_cond.box_y0);
        // Calcul et assignation numérique
        init_cond.box_x1   = get_double(ic_block, "box_x1", init_cond.box_x1);
        // Calcul et assignation numérique
        init_cond.box_y1   = get_double(ic_block, "box_y1", init_cond.box_y1);
        // Calcul et assignation numérique
        init_cond.pert_amp = get_double(ic_block, "pert_amp", init_cond.pert_amp);
        // Calcul et assignation numérique
        init_cond.pert_k   = get_double(ic_block, "pert_k", init_cond.pert_k);
    }

    // 6. Solveur linéaire
    std::string sol_block = extract_value(json, "solver");
    // Évaluation de la condition de contrôle
    if (!sol_block.empty()) {
        // Calcul et assignation numérique
        solver.type = get_string(sol_block, "type", solver.type);
        // Calcul et assignation numérique
        solver.tol = get_double(sol_block, "tol", solver.tol);
        // Calcul et assignation numérique
        solver.max_iter = get_int(sol_block, "max_iter", solver.max_iter);
        // Calcul et assignation numérique
        solver.r_al = get_double(sol_block, "r_al", solver.r_al);
        // Calcul et assignation numérique
        solver.uzawa_iter = get_int(sol_block, "uzawa_iter", solver.uzawa_iter);
        // Calcul et assignation numérique
        solver.use_precond = get_bool(sol_block, "use_precond", solver.use_precond);
        // Calcul et assignation numérique
        solver.precond_type = get_string(sol_block, "precond_type", solver.precond_type);
    }

    // 7. Sorties et visualisation
    std::string out_block = extract_value(json, "output");
    // Évaluation de la condition de contrôle
    if (!out_block.empty()) {
        // Calcul et assignation numérique
        output.output_dir = get_string(out_block, "output_dir", output.output_dir);
        // Calcul et assignation numérique
        output.case_name  = get_string(out_block, "case_name", output.case_name);
        // Calcul et assignation numérique
        output.vtk_interval = get_int(out_block, "vtk_interval", output.vtk_interval);
        // Calcul et assignation numérique
        output.log_interval = get_int(out_block, "log_interval", output.log_interval);
        // Calcul et assignation numérique
        output.write_csv_history = get_bool(out_block, "write_csv_history", output.write_csv_history);
    }

    // 8. Capture d'interface et transport VOF
    std::string vof_block = extract_value(json, "vof");
    // Évaluation de la condition de contrôle
    if (!vof_block.empty()) {
        // Calcul et assignation numérique
        vof.reconstruction_method = get_string(vof_block, "reconstruction_method", vof.reconstruction_method);
        // Calcul et assignation numérique
        vof.advection_method      = get_string(vof_block, "advection_method", vof.advection_method);
        // Calcul et assignation numérique
        vof.tvd_limiter           = get_string(vof_block, "tvd_limiter", vof.tvd_limiter);
        // Calcul et assignation numérique
        vof.cfl_vof               = get_double(vof_block, "cfl_vof", vof.cfl_vof);
    }

    // 9. Obstacles immergés (Pénalisation de Brinkman)
    obstacles.clear();
    // Déclaration de variable typée
    std::string obs_array = extract_value(json, "obstacles");
    // Évaluation de la condition de contrôle
    if (!obs_array.empty()) {
        // Déclaration de variable typée
        size_t pos = 0;
        // Boucle itérative conditionnelle
        while ((pos = obs_array.find('{', pos)) != std::string::npos) {
            // Déclaration de variable typée
            size_t end_pos = obs_array.find('}', pos);
            // Évaluation de la condition de contrôle
            if (end_pos == std::string::npos) break;
            // Déclaration de variable typée
            std::string obs_block = obs_array.substr(pos, end_pos - pos + 1);
            Obstacle obs;
            // Calcul et assignation numérique
            obs.type = get_string(obs_block, "type", "BOX");
            // Calcul et assignation numérique
            obs.x0 = get_double(obs_block, "x0", 0.0);
            // Calcul et assignation numérique
            obs.y0 = get_double(obs_block, "y0", 0.0);
            // Calcul et assignation numérique
            obs.x1 = get_double(obs_block, "x1", 0.0);
            // Calcul et assignation numérique
            obs.y1 = get_double(obs_block, "y1", 0.0);
            // Calcul et assignation numérique
            obs.xc = get_double(obs_block, "xc", 0.0);
            // Calcul et assignation numérique
            obs.yc = get_double(obs_block, "yc", 0.0);
            // Calcul et assignation numérique
            obs.radius = get_double(obs_block, "radius", 0.0);
            // Calcul et assignation numérique
            obs.permeability = get_double(obs_block, "permeability", 1.0e-14);
            obstacles.push_back(obs);
            // Calcul et assignation numérique
            pos = end_pos + 1;
        }
    }

    // Retour de fonction
    return true;
}

void Config::print() const {
    // Calcul et assignation numérique
    std::cout << "========================================================\n"
              << "       COVOF 2D PARALLELE - CONFIGURATION DU CAS         \n"
              // Calcul et assignation numérique
              << "========================================================\n"
              << " Mode calcul : " << mode << "\n"
              // Calcul et assignation numérique
              << " Maillage : " << nx << " x " << ny << " cellules (Lx = " << lx << " m, Ly = " << ly << " m)\n"
              // Calcul de la résolution géométrique de cellule
              << " dx = " << lx / nx << " m, dy = " << ly / ny << " m\n"
              << " Physique : " << (enable_two_phase ? "Diphasique" : "Monophasique")
              << " | Convection : " << (enable_convection ? "Oui" : "Non")
              << " | Tension de surface : " << (enable_surface_tension ? "Oui" : "Non") << "\n";
    // Évaluation de la condition de contrôle
    if (!obstacles.empty()) {
        std::cout << " Obstacles immerges : " << obstacles.size() << " defini(s) [Brinkman]\n";
        // Boucle d'itération algorithmique
        for (size_t i = 0; i < obstacles.size(); ++i) {
            std::cout << "   - Obstacle " << i+1 << " : [" << obstacles[i].x0 << ", " << obstacles[i].x1
                      // Calcul et assignation numérique
                      << "] x [" << obstacles[i].y0 << ", " << obstacles[i].y1 << "], K=" << obstacles[i].permeability << "\n";
        }
    }
    // Calcul et assignation numérique
    std::cout << " Fluide 1 : rho=" << fluid1.rho << " kg/m3, mu=" << fluid1.mu << " Pa.s\n";
    // Évaluation de la condition de contrôle
    if (enable_two_phase) {
        // Calcul et assignation numérique
        std::cout << " Fluide 2 : rho=" << fluid2.rho << " kg/m3, mu=" << fluid2.mu << " Pa.s\n"
                  // Calcul et assignation numérique
                  << " Tension superficielle sigma = " << sigma << " N/m\n"
                  << " VOF : Reconstruction=" << vof.reconstruction_method 
                  << " | Advection=" << vof.advection_method 
                  << " | Limiteur=" << vof.tvd_limiter << "\n";
    }
    // Calcul et assignation numérique
    std::cout << " Gravite : g = (" << gx << ", " << gy << ") m/s^2\n"
              // Calcul et assignation numérique
              << " Temps : dt = " << dt << " s, t_final = " << t_final << " s\n"
              // Calcul et assignation numérique
              << " Solveur : " << solver.type << " | BiCGSTAB(2), tol=" << solver.tol
              // Calcul et assignation numérique
              << ", max_iter=" << solver.max_iter << ", r_al=" << solver.r_al
              // Calcul et assignation numérique
              << ", uzawa_iter=" << solver.uzawa_iter << "\n"
              // Calcul et assignation numérique
              << " Sorties : Dossier '" << output.output_dir << "', intervalle VTK=" << output.vtk_interval << "\n"
              // Calcul et assignation numérique
              << "========================================================\n";
}

} // namespace covof
