#include "fugu/Config.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace fugu {

// Fonctions utilitaires locales pour parser le JSON simplement
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

static std::string extract_value(const std::string& content, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = content.find(search);
    if (pos == std::string::npos) return "";
    size_t colon = content.find(':', pos + search.length());
    if (colon == std::string::npos) return "";
    
    size_t start = content.find_first_not_of(" \t\n\r", colon + 1);
    if (start == std::string::npos) return "";

    if (content[start] == '{' || content[start] == '[') {
        char open_c = content[start];
        char close_c = (open_c == '{') ? '}' : ']';
        int depth = 1;
        size_t end = start + 1;
        while (end < content.length() && depth > 0) {
            if (content[end] == open_c) depth++;
            else if (content[end] == close_c) depth--;
            end++;
        }
        return content.substr(start, end - start);
    } else if (content[start] == '"') {
        size_t end = content.find('"', start + 1);
        if (end == std::string::npos) return "";
        return content.substr(start + 1, end - start - 1);
    } else {
        size_t end = content.find_first_of(",}\n\r", start);
        if (end == std::string::npos) end = content.length();
        return trim(content.substr(start, end - start));
    }
}

static double get_double(const std::string& block, const std::string& key, double default_val) {
    std::string val = extract_value(block, key);
    if (val.empty()) return default_val;
    try { return std::stod(val); } catch (...) { return default_val; }
}

static int get_int(const std::string& block, const std::string& key, int default_val) {
    std::string val = extract_value(block, key);
    if (val.empty()) return default_val;
    try { return std::stoi(val); } catch (...) { return default_val; }
}

static bool get_bool(const std::string& block, const std::string& key, bool default_val) {
    std::string val = extract_value(block, key);
    if (val == "true" || val == "1") return true;
    if (val == "false" || val == "0") return false;
    return default_val;
}

static std::string get_string(const std::string& block, const std::string& key, const std::string& default_val) {
    std::string val = extract_value(block, key);
    return val.empty() ? default_val : val;
}

static BCType parse_bc_type(const std::string& str) {
    if (str == "DIRICHLET") return BCType::DIRICHLET;
    if (str == "NEUMANN") return BCType::NEUMANN;
    if (str == "FREE_SLIP") return BCType::FREE_SLIP;
    if (str == "PERIODIC") return BCType::PERIODIC;
    return BCType::DIRICHLET;
}

static void parse_single_bc(const std::string& block, BoundaryCondition& bc) {
    bc.type = parse_bc_type(get_string(block, "type", "DIRICHLET"));
    bc.u_val = get_double(block, "u", 0.0);
    bc.v_val = get_double(block, "v", 0.0);
    bc.profile = get_string(block, "profile", "UNIFORM");
    bc.u_mean = get_double(block, "u_mean", 0.0);
    bc.y_step = get_double(block, "y_step", 0.0);
    bc.T_type = parse_bc_type(get_string(block, "T_type", "NEUMANN"));
    bc.T_val = get_double(block, "T", 0.0);
}

bool Config::load_from_json(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[Config] Erreur: Impossible d'ouvrir le fichier JSON : " << filepath << std::endl;
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    // 0. Mode global et arrêt stationnaire
    mode = get_string(json, "mode", mode);
    steady_state_tol = get_double(json, "steady_state_tol", steady_state_tol);

    // 1. Maillage
    std::string mesh_block = extract_value(json, "mesh");
    if (!mesh_block.empty()) {
        nx = get_int(mesh_block, "nx", nx);
        ny = get_int(mesh_block, "ny", ny);
        lx = get_double(mesh_block, "lx", lx);
        ly = get_double(mesh_block, "ly", ly);
    }

    // 2. Modèle physique
    std::string model_block = extract_value(json, "physics");
    if (!model_block.empty()) {
        enable_two_phase = get_bool(model_block, "enable_two_phase", enable_two_phase);
        enable_surface_tension = get_bool(model_block, "enable_surface_tension", enable_surface_tension);
        enable_convection = get_bool(model_block, "enable_convection", enable_convection);
        enable_energy = get_bool(model_block, "enable_energy", enable_energy);
        sigma = get_double(model_block, "sigma", sigma);
        gx = get_double(model_block, "gx", gx);
        gy = get_double(model_block, "gy", gy);

        Ra = get_double(model_block, "Ra", Ra);
        Pr = get_double(model_block, "Pr", Pr);
        T_ref = get_double(model_block, "T_ref", T_ref);
        beta = get_double(model_block, "beta", beta);

        viscosity_model = get_string(model_block, "viscosity_model", viscosity_model);
        visc_a = get_double(model_block, "visc_a", visc_a);
        visc_e = get_double(model_block, "visc_e", visc_e);
        visc_mu0 = get_double(model_block, "visc_mu0", visc_mu0);

        std::string f1_block = extract_value(model_block, "fluid1");
        if (!f1_block.empty()) {
            fluid1.rho = get_double(f1_block, "rho", fluid1.rho);
            fluid1.mu  = get_double(f1_block, "mu", fluid1.mu);
            fluid1.Cp  = get_double(f1_block, "Cp", fluid1.Cp);
            fluid1.k_thermal = get_double(f1_block, "k_thermal", fluid1.k_thermal);
            fluid1.beta = get_double(f1_block, "beta", fluid1.beta);
        }

        std::string f2_block = extract_value(model_block, "fluid2");
        if (!f2_block.empty()) {
            fluid2.rho = get_double(f2_block, "rho", fluid2.rho);
            fluid2.mu  = get_double(f2_block, "mu", fluid2.mu);
            fluid2.Cp  = get_double(f2_block, "Cp", fluid2.Cp);
            fluid2.k_thermal = get_double(f2_block, "k_thermal", fluid2.k_thermal);
            fluid2.beta = get_double(f2_block, "beta", fluid2.beta);
        }
    }

    // 3. Intégration temporelle
    std::string time_block = extract_value(json, "time");
    if (!time_block.empty()) {
        dt = get_double(time_block, "dt", dt);
        t_final = get_double(time_block, "t_final", t_final);
        cfl_max = get_double(time_block, "cfl_max", cfl_max);
        adaptive_dt = get_bool(time_block, "adaptive_dt", adaptive_dt);
    }

    // 4. Conditions aux limites
    std::string bc_block = extract_value(json, "boundary_conditions");
    if (!bc_block.empty()) {
        std::string b_left = extract_value(bc_block, "left");
        if (!b_left.empty()) parse_single_bc(b_left, bc.left);

        std::string b_right = extract_value(bc_block, "right");
        if (!b_right.empty()) parse_single_bc(b_right, bc.right);

        std::string b_bot = extract_value(bc_block, "bottom");
        if (!b_bot.empty()) parse_single_bc(b_bot, bc.bottom);

        std::string b_top = extract_value(bc_block, "top");
        if (!b_top.empty()) parse_single_bc(b_top, bc.top);
    }

    // 5. Condition initiale
    std::string ic_block = extract_value(json, "initial_condition");
    if (!ic_block.empty()) {
        std::string type_str = get_string(ic_block, "type", "SINGLE_PHASE");
        if (type_str == "DROPLET") init_cond.type = InitialConditionType::DROPLET;
        else if (type_str == "BUBBLE") init_cond.type = InitialConditionType::BUBBLE;
        else if (type_str == "DAM_BREAK") init_cond.type = InitialConditionType::DAM_BREAK;
        else if (type_str == "SHEAR_LAYER") init_cond.type = InitialConditionType::SHEAR_LAYER;
        else if (type_str == "VORTEX_SHEAR") init_cond.type = InitialConditionType::VORTEX_SHEAR;
        else if (type_str == "RAYLEIGH_TAYLOR") init_cond.type = InitialConditionType::RAYLEIGH_TAYLOR;
        else init_cond.type = InitialConditionType::SINGLE_PHASE;

        init_cond.center_x = get_double(ic_block, "center_x", init_cond.center_x);
        init_cond.center_y = get_double(ic_block, "center_y", init_cond.center_y);
        init_cond.radius   = get_double(ic_block, "radius", init_cond.radius);
        init_cond.box_x0   = get_double(ic_block, "box_x0", init_cond.box_x0);
        init_cond.box_y0   = get_double(ic_block, "box_y0", init_cond.box_y0);
        init_cond.box_x1   = get_double(ic_block, "box_x1", init_cond.box_x1);
        init_cond.box_y1   = get_double(ic_block, "box_y1", init_cond.box_y1);
        init_cond.pert_amp = get_double(ic_block, "pert_amp", init_cond.pert_amp);
        init_cond.pert_k   = get_double(ic_block, "pert_k", init_cond.pert_k);
    }

    // 6. Solveur linéaire
    std::string sol_block = extract_value(json, "solver");
    if (!sol_block.empty()) {
        solver.type = get_string(sol_block, "type", solver.type);
        solver.tol = get_double(sol_block, "tol", solver.tol);
        solver.max_iter = get_int(sol_block, "max_iter", solver.max_iter);
        solver.r_al = get_double(sol_block, "r_al", solver.r_al);
        solver.uzawa_iter = get_int(sol_block, "uzawa_iter", solver.uzawa_iter);
        solver.use_precond = get_bool(sol_block, "use_precond", solver.use_precond);
        solver.precond_type = get_string(sol_block, "precond_type", solver.precond_type);
    }

    // 7. Sorties et visualisation
    std::string out_block = extract_value(json, "output");
    if (!out_block.empty()) {
        output.output_dir = get_string(out_block, "output_dir", output.output_dir);
        output.case_name  = get_string(out_block, "case_name", output.case_name);
        output.vtk_interval = get_int(out_block, "vtk_interval", output.vtk_interval);
        output.log_interval = get_int(out_block, "log_interval", output.log_interval);
        output.write_csv_history = get_bool(out_block, "write_csv_history", output.write_csv_history);
    }

    // 8. Obstacles immergés (Pénalisation de Brinkman)
    obstacles.clear();
    std::string obs_array = extract_value(json, "obstacles");
    if (!obs_array.empty()) {
        size_t pos = 0;
        while ((pos = obs_array.find('{', pos)) != std::string::npos) {
            size_t end_pos = obs_array.find('}', pos);
            if (end_pos == std::string::npos) break;
            std::string obs_block = obs_array.substr(pos, end_pos - pos + 1);
            Obstacle obs;
            obs.type = get_string(obs_block, "type", "BOX");
            obs.x0 = get_double(obs_block, "x0", 0.0);
            obs.y0 = get_double(obs_block, "y0", 0.0);
            obs.x1 = get_double(obs_block, "x1", 0.0);
            obs.y1 = get_double(obs_block, "y1", 0.0);
            obs.xc = get_double(obs_block, "xc", 0.0);
            obs.yc = get_double(obs_block, "yc", 0.0);
            obs.radius = get_double(obs_block, "radius", 0.0);
            obs.permeability = get_double(obs_block, "permeability", 1.0e-14);
            obstacles.push_back(obs);
            pos = end_pos + 1;
        }
    }

    return true;
}

void Config::print() const {
    std::cout << "========================================================\n"
              << "       FUGU 2D PARALLELE - CONFIGURATION DU CAS         \n"
              << "========================================================\n"
              << " Mode calcul : " << mode << "\n"
              << " Maillage : " << nx << " x " << ny << " cellules (Lx = " << lx << " m, Ly = " << ly << " m)\n"
              << " dx = " << lx / nx << " m, dy = " << ly / ny << " m\n"
              << " Physique : " << (enable_two_phase ? "Diphasique" : "Monophasique")
              << " | Convection : " << (enable_convection ? "Oui" : "Non")
              << " | Tension de surface : " << (enable_surface_tension ? "Oui" : "Non") << "\n";
    if (!obstacles.empty()) {
        std::cout << " Obstacles immerges : " << obstacles.size() << " defini(s) [Brinkman]\n";
        for (size_t i = 0; i < obstacles.size(); ++i) {
            std::cout << "   - Obstacle " << i+1 << " : [" << obstacles[i].x0 << ", " << obstacles[i].x1
                      << "] x [" << obstacles[i].y0 << ", " << obstacles[i].y1 << "], K=" << obstacles[i].permeability << "\n";
        }
    }
    std::cout << " Fluide 1 : rho=" << fluid1.rho << " kg/m3, mu=" << fluid1.mu << " Pa.s\n";
    if (enable_two_phase) {
        std::cout << " Fluide 2 : rho=" << fluid2.rho << " kg/m3, mu=" << fluid2.mu << " Pa.s\n"
                  << " Tension superficielle sigma = " << sigma << " N/m\n";
    }
    std::cout << " Gravite : g = (" << gx << ", " << gy << ") m/s^2\n"
              << " Temps : dt = " << dt << " s, t_final = " << t_final << " s\n"
              << " Solveur : " << solver.type << " | BiCGSTAB(2), tol=" << solver.tol
              << ", max_iter=" << solver.max_iter << ", r_al=" << solver.r_al
              << ", uzawa_iter=" << solver.uzawa_iter << "\n"
              << " Sorties : Dossier '" << output.output_dir << "', intervalle VTK=" << output.vtk_interval << "\n"
              << "========================================================\n";
}

} // namespace fugu
