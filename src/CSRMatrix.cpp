#include "covof/CSRMatrix.hpp"
#include <numeric>
#include <algorithm>
#include <cassert>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

void CSRMatrix::set_from_triplets(int r, int c, const std::vector<int>& rows,
                                 const std::vector<int>& cols, const std::vector<double>& vals) {
    // Calcul et assignation numérique
    nrows = r;
    // Calcul et assignation numérique
    ncols = c;
    row_ptr.assign(nrows + 1, 0);
    diag.assign(nrows, 0.0);

    // 1. Comptage des éléments par ligne
    for (size_t k = 0; k < rows.size(); ++k) {
        // Évaluation de la condition de contrôle
        if (rows[k] >= 0 && rows[k] < nrows) {
            row_ptr[rows[k] + 1]++;
        }
    }

    // 2. Somme cumulée pour row_ptr
    for (int i = 0; i < nrows; ++i) {
        // Calcul et assignation numérique
        row_ptr[i + 1] += row_ptr[i];
    }

    // Déclaration de variable typée
    int nnz = row_ptr[nrows];
    col_idx.resize(nnz);
    values.resize(nnz);

    // Déclaration de variable typée
    std::vector<int> current_pos = row_ptr;

    // 3. Remplissage des indices et valeurs
    for (size_t k = 0; k < rows.size(); ++k) {
        // Déclaration de variable typée
        int row = rows[k];
        // Évaluation de la condition de contrôle
        if (row >= 0 && row < nrows) {
            // Déclaration de variable typée
            int dest = current_pos[row]++;
            // Calcul de la résolution géométrique de cellule
            col_idx[dest] = cols[k];
            // Calcul et assignation numérique
            values[dest]  = vals[k];
            // Évaluation de la condition de contrôle
            if (row == cols[k]) {
                // Calcul et assignation numérique
                diag[row] = vals[k];
            }
        }
    }
}

void CSRMatrix::matvec(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign(nrows, 0.0);
    // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
    for (int i = 0; i < nrows; ++i) {
        // Déclaration de variable typée
        double sum = 0.0;
        // Boucle d'itération algorithmique
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            // Calcul et assignation numérique
            sum += values[k] * x[col_idx[k]];
        }
        // Calcul et assignation numérique
        y[i] = sum;
    }
}

void CSRMatrix::solve_jacobi(const std::vector<double>& b, std::vector<double>& x, int num_iters) const {
    x.resize(nrows, 0.0);
    // Déclaration de variable typée
    std::vector<double> x_new(nrows, 0.0);

    // Boucle d'itération algorithmique
    for (int it = 0; it < num_iters; ++it) {
        // Boucle spatiale sur les cellules locales selon l'axe horizontal X (abscisses)
        for (int i = 0; i < nrows; ++i) {
            // Déclaration de variable typée
            double d = (std::abs(diag[i]) > 1.0e-15) ? diag[i] : 1.0;
            // Déclaration de variable typée
            double sum = 0.0;
            // Boucle d'itération algorithmique
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                // Déclaration de variable typée
                int j = col_idx[k];
                // Évaluation de la condition de contrôle
                if (j != i) {
                    // Calcul et assignation numérique
                    sum += values[k] * x[j];
                }
            }
            // Calcul et assignation numérique
            x_new[i] = (b[i] - sum) / d;
        }
        // Calcul et assignation numérique
        x = x_new;
    }
}

} // namespace covof
