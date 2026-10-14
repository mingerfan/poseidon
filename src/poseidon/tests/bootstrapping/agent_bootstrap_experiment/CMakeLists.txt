cmake_minimum_required(VERSION 3.22)
project(PoseidonAgentBootstrapDiagnostic LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED POSEIDON_ROOT OR NOT DEFINED POSEIDON_CPU_BUILD)
 message(FATAL_ERROR "Provide the existing source and native CPU build paths")
endif()
add_library(poseidon_existing SHARED IMPORTED)
set_target_properties(poseidon_existing PROPERTIES
 IMPORTED_LOCATION "${POSEIDON_CPU_BUILD}/libposeidon_shared.so"
 INTERFACE_INCLUDE_DIRECTORIES "${POSEIDON_ROOT}/src;${POSEIDON_CPU_BUILD}")
add_executable(agent_bootstrap_experiment test_bootstrap_experiment.cpp)
target_link_libraries(agent_bootstrap_experiment PRIVATE poseidon_existing)
