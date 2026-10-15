cmake_minimum_required(VERSION 3.26.4)
project(PoseidonResNet20Dsl LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(POSEIDON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "Poseidon checkout")
set(POSEIDON_BUILD_RESNET20_S2C_FIRST OFF CACHE BOOL "" FORCE)
# The CPU core does not need an installed spdlog package for this diagnostic;
# the GPU/RMM target supplies its own bundled logging headers.
set(POSEIDON_USE_SPDLOG OFF CACHE BOOL "" FORCE)
add_subdirectory("${POSEIDON_ROOT}/src/poseidon/tests/bootstrapping" "${CMAKE_BINARY_DIR}/poseidon_bootstrap")
# ckks_encoder.h includes spdlog even when optional CPU logging is disabled.
# Reuse the bundled RMM dependency rather than requiring a system installation.
target_link_libraries(poseidon_shared PUBLIC spdlog::spdlog_header_only)
find_package(CUDAToolkit REQUIRED)
set(dsl_cupti CUDA::cupti)
if(NOT TARGET CUDA::cupti)
    # CUDA 12.2 installations can keep CUPTI only under extras/CUPTI.
    find_library(POSEIDON_DSL_CUPTI_LIBRARY NAMES cupti
        HINTS "${CUDAToolkit_BIN_DIR}/../extras/CUPTI/lib64" REQUIRED)
    find_path(POSEIDON_DSL_CUPTI_INCLUDE cupti_activity.h
        HINTS "${CUDAToolkit_BIN_DIR}/../extras/CUPTI/include" REQUIRED)
    add_library(poseidon_dsl_cupti UNKNOWN IMPORTED)
    set_target_properties(poseidon_dsl_cupti PROPERTIES
        IMPORTED_LOCATION "${POSEIDON_DSL_CUPTI_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${POSEIDON_DSL_CUPTI_INCLUDE}")
    set(dsl_cupti poseidon_dsl_cupti)
endif()
set(native "${CMAKE_CURRENT_LIST_DIR}/runtime/native_snapshot")
set(app "${native}/benchmark/resnet20_gpu")
add_executable(resnet20_dsl
    runtime/main_strict_native_dsl.cpp
    "${native}/bench/resnet20_s2c_first/plain_resnet20_reference.cpp"
    "${native}/bench/resnet20_s2c_first/gpu_activity_timing.cpp"
    "${app}/resnet20_weights.cpp")
target_link_libraries(resnet20_dsl PRIVATE poseidon::gpu ${dsl_cupti} ${CMAKE_DL_LIBS})
target_include_directories(resnet20_dsl PRIVATE
    "${app}" "${native}/Doc/analysis/resnet20_joint_chain_20260911")
target_compile_definitions(resnet20_dsl PRIVATE
    POSEIDON_GPU_RESNET20_SOURCE_DIR="${POSEIDON_ROOT}/bench/resnet20_s2c_first/support")
target_compile_options(resnet20_dsl PRIVATE -O1 -UNDEBUG -ffunction-sections -fdata-sections)
target_link_options(resnet20_dsl PRIVATE -Wl,--gc-sections)
set_target_properties(resnet20_dsl PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
