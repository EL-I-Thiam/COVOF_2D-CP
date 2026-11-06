# COVOF 2D C++17 — High-Performance Multi-Phase CFD Solver

[![Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![MPI](https://img.shields.io/badge/MPI-OpenMPI%20%2F%20MPICH-green.svg)](https://www.open-mpi.org/)
[![License](https://img.shields.io/badge/License-Proprietary-red.svg)]()
[![Dual-Code](https://img.shields.io/badge/Dual--Code%20Parity-Fortran%202018-success.svg)]()

**COVOF 2D (C++17)** is a modern, massively parallel two-dimensional Computational Fluid Dynamics (CFD) platform dedicated to the direct numerical simulation (DNS) of incompressible and low-Mach multi-phase flows with high density and viscosity ratios. It features exact volume-of-fluid (VOF) interface tracking, coupled and fractional-step Navier-Stokes formulations, advanced block-preconditioned iterative Krylov solvers, and MPI 2D Cartesian domain decomposition.

COVOF 2D C++17 is developed in strict algorithmic parity with its twin implementation **COVOF 2D Fortran 2018**.

---

## Key Features and Capabilities

### 1. Navier-Stokes Formulation & Velocity-Pressure Coupling
- **Governing Equations**: Incompressible and low-Mach one-fluid Navier-Stokes with surface tension:
  $$\nabla \cdot \mathbf{u} = 0$$
  $$\rho(C)\left(\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}\right) = -\nabla p + \rho(C)\mathbf{g} + \nabla \cdot \left[\mu(C)\left(\nabla \mathbf{u} + \nabla \mathbf{u}^T\right)\right] + \mathbf{f}_\sigma$$
- **Coupling Algorithms**:
  - `MONOLITHIC` (`FC`): Fully-coupled saddle-point velocity-pressure system solved simultaneously.
  - `ADAPTATIVE_AUGMENTED_LAGRANGIAN` (`AAL`): Spatially localized divergence penalty $r(x,y,t) = K_{\mathrm{aal}}\,\mu_2\,|\nabla C|$ with Uzawa projection for robust handling of extreme viscosity contrasts ($\mu_2/\mu_1 > 10^5$).
  - `STANDARD_PROJECTION` (`SP`): Incremental fractional-step Chorin-Temam projection method with variable-density Poisson solve.
- **Time Integration**: Second-order backward differentiation (Gear / BDF2) and first-order Euler, with Adams-Bashforth convective linearization.
- **Convective Schemes**: Centered second-order, upwind, and TVD limiters (Superbee, Van Leer, Minmod).

### 2. High-Fidelity Interface Tracking (VOF)
- **Geometric VOF Schemes**:
  - `WEYMOUTH_YUE`: Unsplit Eulerian-Lagrangian flux advection ensuring strict mass conservation in solenoidal fields.
  - `CIAM`: Direction-split Classical Interface Advection Method.
- **Reconstruction Methods**:
  - `PLIC`: Piecewise Linear Interface Calculation with analytic normal calculation.
  - `ELVIRA`: Efficient Least-Squares Volume-of-Fluid Interface Reconstruction Algorithm.
  - `LVIRA`: Least-Squares Volume-of-Fluid Interface Reconstruction Algorithm.
  - `SLIC`: Simple Line Interface Calculation.
- **Algebraic TVD VOF**: High-resolution TVD limiters (Superbee, Van Leer, MUSCL, HRIC).
- **Smoothed VOF (sVOF)**: Normalized PDE-based indicator filtering for robust phase transitions and continuous fluid property evaluation across thin interfaces.
- **Surface Tension**: Continuum Surface Force (CSF) model with Height-Function (HF) curvature estimation.

### 3. High-Performance Linear Algebra & Preconditioning
- **Sparse Matrix Engine**: Compressed Sparse Row (CSR) storage with MPI-distributed matrix-vector operations.
- **Krylov Subspace Solvers**: BiCGSTAB, BiCGSTAB(2), GMRES, and Conjugate Gradient (CG).
- **Preconditioners**:
  - `BLOCK_TRIANGULAR`: Block Schur-complement preconditioner specifically designed for monolithic saddle-point systems.
  - `ILU0`: Incomplete LU factorization with zero fill-in.
  - `SSOR`: Symmetric Successive Over-Relaxation.
  - `JACOBI`: Diagonal scaling preconditioner.

### 4. Parallelization & I/O
- **Parallel Paradigm**: 2D Cartesian domain decomposition via OpenMPI / MPICH.
- **Halo Exchange**: Non-blocking asynchronous MPI point-to-point exchanges (`MPI_Isend` / `MPI_Irecv`).
- **Parallel Output**: Serial and parallel VTK XML Rectilinear Grid (`.vtr` / `.pvtr`) and CSV history files recording mass, kinetic energy, center of mass, and maximum divergence.

---

## Directory Structure

```text
COVOF_2D-C++/
├── CMakeLists.txt              # CMake build configuration
├── .gitignore                  # Git ignore rules
├── include/
│   └── covof/
│       ├── Config.hpp          # JSON configuration parser and simulation parameters
│       ├── CSRMatrix.hpp       # Compressed Sparse Row matrix format
│       ├── EOS.hpp             # Equation of state and multi-phase fluid properties
│       ├── Field.hpp           # 2D staggered and collocated scalar/vector fields
│       ├── InterfaceReconstruction.hpp # PLIC, ELVIRA, LVIRA, SLIC reconstruction
│       ├── InterfaceTracker.hpp# Weymouth-Yue, CIAM, and algebraic TVD VOF advection
│       ├── LinearSolver.hpp    # BiCGSTAB, BiCGSTAB2, GMRES, CG iterative solvers
│       ├── Mesh.hpp            # 2D non-uniform/uniform Cartesian staggered grid
│       ├── MpiComm.hpp         # MPI 2D Cartesian topology and halo exchange
│       ├── NavierStokesSolver.hpp # Monolithic, AAL, and Standard Projection solvers
│       ├── Preconditioner.hpp  # Block-triangular, ILU(0), SSOR, Jacobi preconditioners
│       ├── Simulation.hpp      # Main simulation time loop and physics orchestrator
│       ├── Types.hpp           # Core numerical types, enumerations, and constants
│       └── VTKWriter.hpp       # XML VTK (.vtr/.pvtr) visualizer export engine
└── src/
    ├── Config.cpp
    ├── CSRMatrix.cpp
    ├── InterfaceReconstruction.cpp
    ├── InterfaceTracker.cpp
    ├── LinearSolver.cpp
    ├── main.cpp                # Application entry point
    ├── Mesh.cpp
    ├── MpiComm.cpp
    ├── NavierStokesSolver.cpp
    ├── Preconditioner.cpp
    ├── Simulation.cpp
    └── VTKWriter.cpp
```

---

## Build and Installation

### Prerequisites
- **C++ Compiler**: Modern C++17 compliant compiler (`clang++` $\ge 12.0$ or `g++` $\ge 9.0$).
- **CMake**: Version $\ge 3.16$.
- **MPI**: OpenMPI $\ge 4.0$ or MPICH $\ge 3.3$.
- **BLAS / LAPACK**: Apple Accelerate Framework (macOS) or OpenBLAS / Intel MKL (Linux).

### Compilation Steps

```bash
# Clone the repository
git clone https://github.com/EL-I-Thiam/COVOF_2D-CP.git
cd COVOF_2D-CP

# Create build directory
mkdir -p build && cd build

# Configure and compile
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

The compiled binary is generated at `bin/covof2d_cpp` (with a convenience symlink `bin/covof2d`).

---

## Running a Simulation

### Serial Execution
```bash
./bin/covof2d_cpp path/to/case.json
```

### Parallel Execution (OpenMPI)
```bash
mpirun -np 4 ./bin/covof2d_cpp path/to/case.json
```

---

## Configuration File Structure (`case.json`)

A standard COVOF 2D configuration file specifies mesh resolution, fluid properties, time discretization, boundary conditions, linear solvers, and interface tracking:

```json
{
  "mode": "NAVIER_STOKES",
  "mesh": {
    "nx": 160,
    "ny": 160,
    "lx": 1.0,
    "ly": 1.0
  },
  "physics": {
    "enable_two_phase": true,
    "fluid1": { "rho": 1.1768, "mu": 0.00185 },
    "fluid2": { "rho": 1800.0, "mu": 500.0 },
    "gx": 0.0,
    "gy": -9.81,
    "sigma": 0.0,
    "enable_surface_tension": false,
    "convective_scheme": "CENTERED",
    "convective_linearization": "ADAMS_BASHFORTH",
    "enable_svof_smoothing": true,
    "svof_filter_width": 1.0
  },
  "time": {
    "dt": 0.0005,
    "t_final": 2.0,
    "time_scheme": "GEAR",
    "adaptive_dt": false,
    "cfl_max": 0.3
  },
  "initial_condition": {
    "type": "VISCOUS_JET",
    "xc": 0.5,
    "yc": 1.0,
    "r0": 0.04,
    "u0": 0.0,
    "v0": -0.6
  },
  "boundary_conditions": {
    "left":   { "type": "FREE_SLIP" },
    "right":  { "type": "FREE_SLIP" },
    "bottom": { "type": "FREE_SLIP" },
    "top": {
      "type": "DIRICHLET",
      "profile": "JET_INLET",
      "jet_center": 0.5,
      "jet_width": 0.08,
      "u": 0.0,
      "v": -0.6
    }
  },
  "solver": {
    "type": "MONOLITHIC",
    "r_al": 0.0,
    "linear_solver": "BICGSTAB2",
    "tol": 1e-06,
    "max_iter": 1000,
    "use_precond": true,
    "precond_type": "BLOCK_TRIANGULAR"
  },
  "vof": {
    "advection_method": "WEYMOUTH_YUE",
    "reconstruction_method": "PLIC",
    "tvd_limiter": "SUPERBEE",
    "cfl_vof": 0.3
  },
  "output": {
    "output_dir": "output/jet_buckling",
    "case_name": "jet_buckling",
    "vtk_interval": 100,
    "log_interval": 20,
    "write_csv_history": true
  }
}
```

---

## Validation Benchmark Suite

COVOF 2D C++17 has been rigorously verified against analytical, experimental, and benchmark DNS data across standard multi-phase and single-phase benchmarks:

1. **Viscous Jet Buckling Instability**: Periodic folding and air entrapment for low Reynolds numbers ($Re = 0.1728 < 0.52$, $H/D > 3\pi$) with exact flow-rate compensation (`JET_INLET`).
2. **Falling Dense Cylinder**: Free fall of a dense fluid cylinder ($\rho_2/\rho_1 = 1000$) through a lighter fluid, matching exact terminal velocities.
3. **Liquid Sheet Deformation**: Incompressible liquid sheet dynamics under strong density contrasts ($\rho_2/\rho_1 \in \{100, 1000\}$).
4. **Vortex Shear Advection**: High-order PLIC and TVD interface preservation under single-vortex shear stretching and reversal.
5. **Static Laplace Droplet**: Verification of Laplace-Young pressure jump $\Delta p = \sigma / R$ and spurious current damping below $10^{-6}\,\mathrm{m/s}$.
6. **Poiseuille Flow Suite**: Steady, transient (Szymanski series), and variable-viscosity channel flows with $L_2$ error norm $< 0.01\,\%$.
7. **Lid-Driven Cavity**: Square and L-shaped cavities up to $Re = 1000$.

---

## Dual-Code Parity Certification

Every numerical feature in COVOF 2D C++17 is validated to achieve strict parity with **COVOF 2D Fortran 2018**:
- Spatial velocity error: $\max |u^{\mathrm{CPP}} - u^{\mathrm{Fort}}| < 10^{-6}\,\mathrm{m/s}$
- Interface volume parity: $|V^{\mathrm{CPP}}(t) - V^{\mathrm{Fort}}(t)| / V_0 < 0.005\,\%$
- Divergence tolerance: $\|\nabla\cdot\mathbf{u}\|_\infty \le 10^{-7}\,\mathrm{s}^{-1}$ on identical grids

---

## Authors & Maintenance

- **Author**: Elhadji Ibrahima Thiam
- **Platform**: COVOF 2D Multi-Phase CFD Platform
