cmake_minimum_required(VERSION 3.16)
project(COVOF_2D_CPP CXX)

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
    src/InterfaceReconstruction.cpp
    src/InterfaceTracker.cpp
    src/NavierStokesSolver.cpp
    src/VTKWriter.cpp
    src/Simulation.cpp
    src/main.cpp
)

# Executable target: covof2d
add_executable(covof2d ${SOURCES})
target_link_libraries(covof2d PRIVATE MPI::MPI_CXX)

set_target_properties(covof2d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

# Create compatibility symlink/target for fugu2d
add_custom_command(TARGET covof2d POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/covof2d
        ${CMAKE_CURRENT_SOURCE_DIR}/bin/fugu2d
    COMMENT "Creating compatibility symlink bin/fugu2d -> bin/covof2d"
)
