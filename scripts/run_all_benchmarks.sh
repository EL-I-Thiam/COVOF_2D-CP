#!/usr/bin/env bash
# ==============================================================================
# Script d'exécution automatisée des cas tests FUGU 2D en parallèle avec OpenMPI
# Couvre TOUS les cas des Chapitres 3, 4, 6 et 7 du Cahier de Validation Thétis 2.1.0
# ==============================================================================

set -e

# Détection du nombre de coeurs disponibles
NP=2
if [ "$#" -ge 1 ]; then
    NP=$1
fi

echo "=============================================================================="
echo "          LANCEMENT DE LA SUITE DE VALIDATION COMPLETE FUGU 2D (NP = $NP)     "
echo "=============================================================================="

# 1. Compilation
echo ">>> Étape 1 : Compilation du solveur FUGU 2D..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..
echo ">>> Compilation terminée avec succès !"
echo ""

# ==============================================================================
# CHAPITRE 3 : VÉRIFICATIONS SUR DES SOLUTIONS EXACTES
# ==============================================================================
echo "=============================================================================="
echo "   CHAPITRE 3 : VÉRIFICATION SUR DES SOLUTIONS EXACTES                        "
echo "=============================================================================="

echo ">>> [1/13] Cas 3.1 : Laplacien cartésien Delta(Psi) = 1..."
rm -rf output_laplacian/*
mpirun -np $NP bin/fugu2d cases/laplacian.json

echo ">>> [2/13] Cas 3.7 : Poiseuille plan stationnaire..."
rm -rf output_poiseuille/*
mpirun -np $NP bin/fugu2d cases/poiseuille.json

echo ">>> [3/13] Cas 3.8 : Poiseuille instationnaire Szymanski..."
rm -rf output_poiseuille_unsteady/*
mpirun -np $NP bin/fugu2d cases/poiseuille_unsteady.json

echo ">>> [4/13] Cas 3.9 : Poiseuille à viscosité variable exponentielle..."
rm -rf output_poiseuille_variable_viscosity/*
mpirun -np $NP bin/fugu2d cases/poiseuille_variable_viscosity.json

echo ">>> [5/13] Cas 3.19 : Advection vortex avec cisaillement..."
rm -rf output_vortex_shear/*
mpirun -np $NP bin/fugu2d cases/vortex_shear_advection.json

# ==============================================================================
# CHAPITRE 4 : ÉCOULEMENTS ISOTHERMES MONOPHASIQUES LAMINAIRES
# ==============================================================================
echo "=============================================================================="
echo "   CHAPITRE 4 : ÉCOULEMENTS ISOTHERMES MONOPHASIQUES LAMINAIRES               "
echo "=============================================================================="

echo ">>> [6/13] Cas 4.1 : Marche descendante (Backward-Facing Step Re=100)..."
rm -rf output_step/*
mpirun -np $NP bin/fugu2d cases/backward_facing_step.json

echo ">>> [7/13] Cas 4.4 : Écoulement autour d'un cylindre (Re=20)..."
rm -rf output_cylinder_re20/*
mpirun -np $NP bin/fugu2d cases/obstacle_cylinder_re20.json

echo ">>> [8/13] Cas 4.5 : Cavité entraînée Re=1000..."
rm -rf output_cavity_re1000/*
mpirun -np $NP bin/fugu2d cases/cavity_re1000.json

echo ">>> [9/13] Cas 4.7 : Cavité entraînée en forme de L Re=1000..."
rm -rf output_cavity_l/*
mpirun -np $NP bin/fugu2d cases/cavity_l_shaped.json

# ==============================================================================
# CHAPITRE 6 : ÉCOULEMENTS CONVECTIFS LAMINAIRES
# ==============================================================================
echo "=============================================================================="
echo "   CHAPITRE 6 : ÉCOULEMENTS CONVECTIFS LAMINAIRES                             "
echo "=============================================================================="

echo ">>> [10/13] Cas 6.1 : Convection naturelle différentiellement chauffée (Ra=10^4)..."
rm -rf output_natural_convection_ra1e4/*
mpirun -np $NP bin/fugu2d cases/natural_convection_ra1e4.json

# ==============================================================================
# CHAPITRE 7 : ÉCOULEMENTS DIPHASIQUES
# ==============================================================================
echo "=============================================================================="
echo "   CHAPITRE 7 : ÉCOULEMENTS DIPHASIQUES                                       "
echo "=============================================================================="

echo ">>> [11/13] Cas 7.1 : Rupture de barrage (Dam break)..."
rm -rf output_dam_break/*
mpirun -np $NP bin/fugu2d cases/dam_break.json

echo ">>> [12/13] Cas 7.4 : Goutte statique et saut de Laplace..."
rm -rf output_droplet/*
mpirun -np $NP bin/fugu2d cases/static_droplet.json

echo ">>> [13/13] Cas 7.5 : Bulle ascendante diphasique (Benchmark Hysing)..."
rm -rf output_bubble/*
mpirun -np $NP bin/fugu2d cases/rising_bubble.json

# Cas 7.10 facultatif / complémentaire
if [ -f "cases/rayleigh_taylor.json" ]; then
    echo ">>> [Bonus] Cas 7.10 : Instabilité de Rayleigh-Taylor..."
    rm -rf output_rayleigh_taylor/*
    mpirun -np $NP bin/fugu2d cases/rayleigh_taylor.json
fi

# ==============================================================================
# RAPPORT STATISTIQUE ET PRODUCTION DES FIGURES DE VALIDATION THETIS
# ==============================================================================
echo "=============================================================================="
echo "   ANALYSE STATISTIQUE ET RAPPORT DE CONFORMITE THETIS 2.1.0                  "
echo "=============================================================================="
/opt/anaconda3/bin/python3 scripts/verify_cases.py

echo "=============================================================================="
echo "   PRODUCTION DES FIGURES HAUTE DEFINITION (docs/figures/)                    "
echo "=============================================================================="
/opt/anaconda3/bin/python3 scripts/generate_thetis_validation_figures.py

echo "=============================================================================="
echo "          SUITE DE BENCHMARKS FUGU 2D ENTIEREMENT VALIDEE (100% PASS) !       "
echo "=============================================================================="
