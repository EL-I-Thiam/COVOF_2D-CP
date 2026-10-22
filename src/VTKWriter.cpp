#include "covof/VTKWriter.hpp"
#include "covof/MpiComm.hpp"
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <cmath>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

void VTKWriter::init(const Mesh2D& mesh, const std::string& out_dir, const std::string& name) {
    // Calcul et assignation numérique
    output_dir = out_dir;
    // Calcul et assignation numérique
    case_name = name;

    // Évaluation de la condition de contrôle
    if (mesh.rank == 0) {
        mkdir(output_dir.c_str(), 0777);

        // Déclaration de variable typée
        std::string csv_path = output_dir + "/" + case_name + "_history.csv";
        csv_file.open(csv_path);
        // Évaluation de la condition de contrôle
        if (csv_file.is_open()) {
            csv_file << "step,time,dt,center_of_mass_y,rise_velocity,bubble_volume,kinetic_energy,max_u,max_div\n";
        }
    }
}

void VTKWriter::write_step(const Mesh2D& mesh, const FlowFields& fields, int step, double time) {
    std::ostringstream ss_step;
    ss_step << std::setfill('0') << std::setw(6) << step;
    // Déclaration de variable typée
    std::string step_str = ss_step.str();

    // Déclaration de variable typée
    int x_start = mesh.i_offset;
    // Déclaration de variable typée
    int x_end   = mesh.i_offset + mesh.nx_local;
    // Déclaration de variable typée
    int y_start = mesh.j_offset;
    // Déclaration de variable typée
    int y_end   = mesh.j_offset + mesh.ny_local;

    // 1. Écriture du fichier .vtr propre à chaque processus MPI
    std::ostringstream ss_vtr;
    ss_vtr << output_dir << "/" << case_name << "_p" << mesh.rank << "_" << step_str << ".vtr";
    std::ofstream vtr(ss_vtr.str());

    // Évaluation de la condition de contrôle
    if (vtr.is_open()) {
        // Calcul et assignation numérique
        vtr << "<?xml version=\"1.0\"?>\n"
            // Calcul et assignation numérique
            << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
            // Calcul et assignation numérique
            << "  <RectilinearGrid WholeExtent=\"" << x_start << " " << x_end << " "
            << y_start << " " << y_end << " 0 0\">\n"
            // Calcul et assignation numérique
            << "    <Piece Extent=\"" << x_start << " " << x_end << " "
            << y_start << " " << y_end << " 0 0\">\n"
            // Calcul et assignation numérique
            << "      <CellData Scalars=\"PhaseIndicator,Pressure,Density,Viscosity,Divergence\" Vectors=\"Velocity\">\n";

        // PhaseIndicator C
        vtr << "        <DataArray type=\"Float64\" Name=\"PhaseIndicator\" format=\"ascii\">\n";
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                vtr << fields.C[mesh.cell_idx(i, j)] << " ";
            }
            vtr << "\n";
        }
        vtr << "        </DataArray>\n";

        // Pressure p
        vtr << "        <DataArray type=\"Float64\" Name=\"Pressure\" format=\"ascii\">\n";
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                vtr << fields.p[mesh.cell_idx(i, j)] << " ";
            }
            vtr << "\n";
        }
        vtr << "        </DataArray>\n";

        // Density rho
        vtr << "        <DataArray type=\"Float64\" Name=\"Density\" format=\"ascii\">\n";
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                vtr << fields.rho[mesh.cell_idx(i, j)] << " ";
            }
            vtr << "\n";
        }
        vtr << "        </DataArray>\n";

        // Viscosity mu
        vtr << "        <DataArray type=\"Float64\" Name=\"Viscosity\" format=\"ascii\">\n";
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                vtr << fields.mu[mesh.cell_idx(i, j)] << " ";
            }
            vtr << "\n";
        }
        vtr << "        </DataArray>\n";

        // Divergence div(u)
        vtr << "        <DataArray type=\"Float64\" Name=\"Divergence\" format=\"ascii\">\n";
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                double div = (fields.u[mesh.u_idx(i + 1, j)] - fields.u[mesh.u_idx(i, j)]) / mesh.dx +
                             (fields.v[mesh.v_idx(i, j + 1)] - fields.v[mesh.v_idx(i, j)]) / mesh.dy;
                vtr << div << " ";
            }
            vtr << "\n";
        }
        vtr << "        </DataArray>\n";

        // Temperature T
        vtr << "        <DataArray type=\"Float64\" Name=\"Temperature\" format=\"ascii\">\n";
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                vtr << fields.T[mesh.cell_idx(i, j)] << " ";
            }
            vtr << "\n";
        }
        vtr << "        </DataArray>\n";

        // Velocity vector au centre des cellules
        vtr << "        <DataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 1; j <= mesh.ny_local; ++j) {
            // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
            for (int i = 1; i <= mesh.nx_local; ++i) {
                // Déclaration de variable typée
                double uc = 0.5 * (fields.u[mesh.u_idx(i, j)] + fields.u[mesh.u_idx(i + 1, j)]);
                // Déclaration de variable typée
                double vc = 0.5 * (fields.v[mesh.v_idx(i, j)] + fields.v[mesh.v_idx(i, j + 1)]);
                vtr << uc << " " << vc << " 0.0 ";
            }
            vtr << "\n";
        }
        vtr << "        </DataArray>\n";

        vtr << "      </CellData>\n"
            << "      <Coordinates>\n"
            // Calcul et assignation numérique
            << "        <DataArray type=\"Float64\" Name=\"XCoordinates\" format=\"ascii\">\n";
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i <= mesh.nx_local; ++i) {
            vtr << (mesh.i_offset + i) * mesh.dx << " ";
        }
        vtr << "\n        </DataArray>\n"
            // Calcul et assignation numérique
            << "        <DataArray type=\"Float64\" Name=\"YCoordinates\" format=\"ascii\">\n";
        // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
        for (int j = 0; j <= mesh.ny_local; ++j) {
            vtr << (mesh.j_offset + j) * mesh.dy << " ";
        }
        vtr << "\n        </DataArray>\n"
            // Calcul et assignation numérique
            << "        <DataArray type=\"Float64\" Name=\"ZCoordinates\" format=\"ascii\">\n"
            << "          0.0\n"
            << "        </DataArray>\n"
            << "      </Coordinates>\n"
            << "    </Piece>\n"
            << "  </RectilinearGrid>\n"
            << "</VTKFile>\n";
    }

    // 2. Écriture du fichier maître .pvtr par le processus racine (rank 0)
    if (mesh.rank == 0) {
        // Déclaration de variable typée
        std::vector<int> all_x_start(mesh.num_procs), all_x_end(mesh.num_procs);
        // Déclaration de variable typée
        std::vector<int> all_y_start(mesh.num_procs), all_y_end(mesh.num_procs);

        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Gather(&x_start, 1, MPI_INT, all_x_start.data(), 1, MPI_INT, 0, mesh.comm);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Gather(&x_end,   1, MPI_INT, all_x_end.data(),   1, MPI_INT, 0, mesh.comm);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Gather(&y_start, 1, MPI_INT, all_y_start.data(), 1, MPI_INT, 0, mesh.comm);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Gather(&y_end,   1, MPI_INT, all_y_end.data(),   1, MPI_INT, 0, mesh.comm);

        std::ostringstream ss_pvtr;
        ss_pvtr << output_dir << "/" << case_name << "_" << step_str << ".pvtr";
        std::ofstream pvtr(ss_pvtr.str());

        // Évaluation de la condition de contrôle
        if (pvtr.is_open()) {
            // Calcul et assignation numérique
            pvtr << "<?xml version=\"1.0\"?>\n"
                 // Calcul et assignation numérique
                 << "<VTKFile type=\"PRectilinearGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
                 // Calcul et assignation numérique
                 << "  <PRectilinearGrid WholeExtent=\"0 " << mesh.nx_global << " 0 " << mesh.ny_global << " 0 0\" GhostLevel=\"0\">\n"
                 // Calcul et assignation numérique
                 << "    <PCellData Scalars=\"PhaseIndicator,Pressure,Density,Viscosity,Divergence\" Vectors=\"Velocity\">\n"
                 // Calcul et assignation numérique
                 << "      <PDataArray type=\"Float64\" Name=\"PhaseIndicator\"/>\n"
                 // Calcul et assignation numérique
                 << "      <PDataArray type=\"Float64\" Name=\"Pressure\"/>\n"
                 // Calcul et assignation numérique
                 << "      <PDataArray type=\"Float64\" Name=\"Density\"/>\n"
                 // Calcul et assignation numérique
                 << "      <PDataArray type=\"Float64\" Name=\"Viscosity\"/>\n"
                 // Calcul et assignation numérique
                 << "      <PDataArray type=\"Float64\" Name=\"Divergence\"/>\n"
                 // Calcul et assignation numérique
                 << "      <PDataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\"/>\n"
                 << "    </PCellData>\n"
                 << "    <PCoordinates>\n"
                 // Calcul et assignation numérique
                 << "      <PDataArray type=\"Float64\" Name=\"XCoordinates\"/>\n"
                 // Calcul et assignation numérique
                 << "      <PDataArray type=\"Float64\" Name=\"YCoordinates\"/>\n"
                 // Calcul et assignation numérique
                 << "      <PDataArray type=\"Float64\" Name=\"ZCoordinates\"/>\n"
                 << "    </PCoordinates>\n";

            // Boucle d'itération algorithmique
            for (int p = 0; p < mesh.num_procs; ++p) {
                std::ostringstream piece_vtr;
                piece_vtr << case_name << "_p" << p << "_" << step_str << ".vtr";
                // Calcul et assignation numérique
                pvtr << "    <Piece Extent=\"" << all_x_start[p] << " " << all_x_end[p] << " "
                     // Calcul et assignation numérique
                     << all_y_start[p] << " " << all_y_end[p] << " 0 0\" Source=\"" << piece_vtr.str() << "\"/>\n";
            }

            pvtr << "  </PRectilinearGrid>\n"
                 << "</VTKFile>\n";
        }
    } else {
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Gather(&x_start, 1, MPI_INT, nullptr, 0, MPI_INT, 0, mesh.comm);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Gather(&x_end,   1, MPI_INT, nullptr, 0, MPI_INT, 0, mesh.comm);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Gather(&y_start, 1, MPI_INT, nullptr, 0, MPI_INT, 0, mesh.comm);
        // Appel de communication ou synchronisation parallèle OpenMPI
        MPI_Gather(&y_end,   1, MPI_INT, nullptr, 0, MPI_INT, 0, mesh.comm);
    }
}

void VTKWriter::write_diagnostics(const Mesh2D& mesh, const FlowFields& fields, int step, double time, double dt) {
    // Déclaration de variable typée
    double local_y_integral = 0.0;
    // Déclaration de variable typée
    double local_v_integral = 0.0;
    // Déclaration de variable typée
    double local_vol = 0.0;
    // Déclaration de variable typée
    double local_ke = 0.0;
    // Déclaration de variable typée
    double local_max_u = 0.0;
    // Déclaration de variable typée
    double local_max_div = 0.0;

    // Déclaration de variable typée
    double dvol = mesh.dx * mesh.dy;

    // Boucle spatiale sur les cellules locales selon l'axe vertical Y (ordonnées)
    for (int j = 1; j <= mesh.ny_local; ++j) {
        // Déclaration de variable typée
        double y = mesh.yc(j);
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 1; i <= mesh.nx_local; ++i) {
            // Déclaration de variable typée
            double c = fields.C[mesh.cell_idx(i, j)];
            // Déclaration de variable typée
            double indicator_bubble = 1.0 - c; // 1 si phase fluide 2 (bulle/goutte)

            // Calcul et assignation numérique
            local_vol += indicator_bubble * dvol;
            // Calcul et assignation numérique
            local_y_integral += y * indicator_bubble * dvol;

            // Déclaration de variable typée
            double uc = 0.5 * (fields.u[mesh.u_idx(i, j)] + fields.u[mesh.u_idx(i + 1, j)]);
            // Déclaration de variable typée
            double vc = 0.5 * (fields.v[mesh.v_idx(i, j)] + fields.v[mesh.v_idx(i, j + 1)]);

            // Calcul et assignation numérique
            local_v_integral += vc * indicator_bubble * dvol;

            // Déclaration de variable typée
            double speed2 = uc * uc + vc * vc;
            // Calcul et assignation numérique
            local_ke += 0.5 * fields.rho[mesh.cell_idx(i, j)] * speed2 * dvol;

            // Déclaration de variable typée
            double speed = std::sqrt(speed2);
            // Évaluation de la condition de contrôle
            if (speed > local_max_u) local_max_u = speed;

            // Déclaration de variable typée
            double div = std::abs((fields.u[mesh.u_idx(i + 1, j)] - fields.u[mesh.u_idx(i, j)]) / mesh.dx +
                                  (fields.v[mesh.v_idx(i, j + 1)] - fields.v[mesh.v_idx(i, j)]) / mesh.dy);
            // Évaluation de la condition de contrôle
            if (div > local_max_div) local_max_div = div;
        }
    }

    // Déclaration de variable typée
    double global_vol = MpiComm::global_sum(mesh, local_vol);
    // Déclaration de variable typée
    double global_y_integral = MpiComm::global_sum(mesh, local_y_integral);
    // Déclaration de variable typée
    double global_v_integral = MpiComm::global_sum(mesh, local_v_integral);
    // Déclaration de variable typée
    double global_ke = MpiComm::global_sum(mesh, local_ke);
    // Déclaration de variable typée
    double global_max_u = MpiComm::global_max(mesh, local_max_u);
    // Déclaration de variable typée
    double global_max_div = MpiComm::global_max(mesh, local_max_div);

    // Déclaration de variable typée
    double yc = (global_vol > 1.0e-12) ? global_y_integral / global_vol : 0.0;
    // Déclaration de variable typée
    double vc = (global_vol > 1.0e-12) ? global_v_integral / global_vol : 0.0;

    // Évaluation de la condition de contrôle
    if (mesh.rank == 0 && csv_file.is_open()) {
        csv_file << step << "," << time << "," << dt << ","
                 << yc << "," << vc << "," << global_vol << ","
                 << global_ke << "," << global_max_u << "," << global_max_div << "\n";
        csv_file.flush();
    }
}

} // namespace covof
