cmake_minimum_required(VERSION 3.16)
project(FUGU_2D_MPI CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization and warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-unused-parameter")

# Find OpenMPI / MPI
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# Include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/Config.cpp
    src/Mesh.cpp
    src/MpiComm.cpp
    src/CSRMatrix.cpp
    src/Preconditioner.cpp
    src/LinearSolver.cpp
    src/InterfaceTracker.cpp
    src/NavierStokesSolver.cpp
    src/VTKWriter.cpp
    src/Simulation.cpp
    src/main.cpp
)

# Executable target
add_executable(fugu2d ${SOURCES})
target_link_libraries(fugu2d PRIVATE MPI::MPI_CXX)

# Set runtime output directory
set_target_properties(fugu2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)
