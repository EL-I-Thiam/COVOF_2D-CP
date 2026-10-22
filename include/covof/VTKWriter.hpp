#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include <string>
#include <fstream>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

/**
 * @brief Gestionnaire d'exportation pour ParaView (XML VTK .vtr/.pvtr) et diagnostics CSV
 */
class VTKWriter {
public:
    // Déclaration de variable typée
    std::string output_dir = "output";
    // Déclaration de variable typée
    std::string case_name  = "covof";
    std::ofstream csv_file;

    // Calcul et assignation numérique
    VTKWriter() = default;

    /**
     * @brief Initialise le répertoire de sortie et le fichier CSV d'historique
     */
    void init(const Mesh2D& mesh, const std::string& out_dir, const std::string& name);

    /**
     * @brief Exporte le pas de temps courant au format VTK XML (.vtr par processus + .pvtr maître)
     */
    void write_step(const Mesh2D& mesh, const FlowFields& fields, int step, double time);

    /**
     * @brief Calcule et enregistre les grandeurs physiques intégrales dans le fichier CSV
     */
    void write_diagnostics(const Mesh2D& mesh, const FlowFields& fields, int step, double time, double dt);

    void finalize() {
        // Évaluation de la condition de contrôle
        if (csv_file.is_open()) csv_file.close();
    }
};

} // namespace covof
