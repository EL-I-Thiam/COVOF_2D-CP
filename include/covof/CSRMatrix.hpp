#pragma once

#include "Mesh.hpp"
#include <vector>

namespace covof {
namespace fugu = covof; // Alias rétrocompatibilité

/**
 * @brief Structure de matrice creuse locale au format Compressed Sparse Row (CSR)
 * Utilisée pour les blocs de préconditionnement (Fuu, Fvv, Laplacien de Schur).
 */
class CSRMatrix {
public:
    // Déclaration de variable typée
    int nrows = 0;
    // Déclaration de variable typée
    int ncols = 0;
    // Déclaration de variable typée
    std::vector<int> row_ptr;     ///< Pointeurs de début de ligne (taille nrows + 1)
    // Déclaration de variable typée
    std::vector<int> col_idx;     ///< Indices de colonnes
    // Déclaration de variable typée
    std::vector<double> values;   ///< Valeurs non nulles
    // Déclaration de variable typée
    std::vector<double> diag;     ///< Diagonale explicite pour le préconditionnement Jacobi / Gauss-Seidel

    // Calcul et assignation numérique
    CSRMatrix() = default;

    /**
     * @brief Initialise les dimensions
     */
    void init(int r, int c) {
        // Calcul et assignation numérique
        nrows = r;
        // Calcul et assignation numérique
        ncols = c;
        row_ptr.assign(r + 1, 0);
        col_idx.clear();
        values.clear();
        diag.assign(r, 0.0);
    }

    /**
     * @brief Construit le format CSR à partir d'une liste de triplets (row, col, value)
     */
    void set_from_triplets(int r, int c, const std::vector<int>& rows,
                           const std::vector<int>& cols, const std::vector<double>& vals);

    /**
     * @brief Produit matrice-vecteur local : y = A * x
     */
    void matvec(const std::vector<double>& x, std::vector<double>& y) const;

    /**
     * @brief Résolution approchée par itérations de Jacobi : x = D^{-1} b
     */
    void solve_jacobi(const std::vector<double>& b, std::vector<double>& x, int num_iters = 5) const;
};

} // namespace covof
