#include "fugu/CSRMatrix.hpp"
#include <numeric>
#include <algorithm>
#include <cassert>

namespace fugu {

void CSRMatrix::set_from_triplets(int r, int c, const std::vector<int>& rows,
                                 const std::vector<int>& cols, const std::vector<double>& vals) {
    nrows = r;
    ncols = c;
    row_ptr.assign(nrows + 1, 0);
    diag.assign(nrows, 0.0);

    // 1. Comptage des éléments par ligne
    for (size_t k = 0; k < rows.size(); ++k) {
        if (rows[k] >= 0 && rows[k] < nrows) {
            row_ptr[rows[k] + 1]++;
        }
    }

    // 2. Somme cumulée pour row_ptr
    for (int i = 0; i < nrows; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    int nnz = row_ptr[nrows];
    col_idx.resize(nnz);
    values.resize(nnz);

    std::vector<int> current_pos = row_ptr;

    // 3. Remplissage des indices et valeurs
    for (size_t k = 0; k < rows.size(); ++k) {
        int row = rows[k];
        if (row >= 0 && row < nrows) {
            int dest = current_pos[row]++;
            col_idx[dest] = cols[k];
            values[dest]  = vals[k];
            if (row == cols[k]) {
                diag[row] = vals[k];
            }
        }
    }
}

void CSRMatrix::matvec(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign(nrows, 0.0);
    for (int i = 0; i < nrows; ++i) {
        double sum = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            sum += values[k] * x[col_idx[k]];
        }
        y[i] = sum;
    }
}

void CSRMatrix::solve_jacobi(const std::vector<double>& b, std::vector<double>& x, int num_iters) const {
    x.resize(nrows, 0.0);
    std::vector<double> x_new(nrows, 0.0);

    for (int it = 0; it < num_iters; ++it) {
        for (int i = 0; i < nrows; ++i) {
            double d = (std::abs(diag[i]) > 1.0e-15) ? diag[i] : 1.0;
            double sum = 0.0;
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                int j = col_idx[k];
                if (j != i) {
                    sum += values[k] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / d;
        }
        x = x_new;
    }
}

} // namespace fugu
