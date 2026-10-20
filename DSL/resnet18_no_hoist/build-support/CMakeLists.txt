cmake_minimum_required(VERSION 3.26.4)
project(ResNet18FrameworkValidation LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(POSEIDON_SOURCE "" CACHE PATH "Pinned framework checkout with the included patches")
if(NOT EXISTS "${POSEIDON_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "Set POSEIDON_SOURCE to the framework checkout")
endif()
set(POSEIDON_USE_SPDLOG OFF CACHE BOOL "" FORCE)
add_subdirectory("${POSEIDON_SOURCE}" poseidon)
target_link_libraries(poseidon_shared PUBLIC spdlog::spdlog_header_only)
# Reuse the ordinary targets, definitions and runtime/crypto libraries.
foreach(target dacapo_mlp_vec_e2e poseidon_runtime_gpu_mlp_e2e poseidon_runtime_gpu_bootstrap_e2e)
    target_include_directories(${target} BEFORE PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/include" "${CMAKE_CURRENT_LIST_DIR}"
        "${POSEIDON_SOURCE}/src/poseidon/tests/runtime_api")
endforeach()
set_property(TARGET dacapo_mlp_vec_e2e PROPERTY SOURCES "${CMAKE_CURRENT_LIST_DIR}/dacapo_mlp_vec_e2e.cpp")
set_property(TARGET poseidon_runtime_gpu_mlp_e2e PROPERTY SOURCES "${CMAKE_CURRENT_LIST_DIR}/poseidon_gpu_mlp_e2e_test.cpp")
set_property(TARGET poseidon_runtime_gpu_bootstrap_e2e PROPERTY SOURCES "${CMAKE_CURRENT_LIST_DIR}/component-probe.cpp")
