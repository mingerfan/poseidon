cmake_minimum_required(VERSION 3.26.4)

project(PoseidonGpuResNet50 LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(POSEIDON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH
    "Path to the Poseidon source tree")

set(POSEIDON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_DEPS OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)
set(CPM_LOCAL_PACKAGES_ONLY ON CACHE BOOL "" FORCE)
set(CPM_DOWNLOAD_ALL OFF CACHE BOOL "" FORCE)

add_subdirectory("${POSEIDON_ROOT}" "${CMAKE_BINARY_DIR}/poseidon_build")
find_package(CUDAToolkit REQUIRED)

set(POSEIDON_THIRD_PARTY_DIR "${POSEIDON_ROOT}/third_party" CACHE PATH "")
set(POSEIDON_BUNDLED_RMM_DIR "${POSEIDON_THIRD_PARTY_DIR}/rmm" CACHE PATH "")
if(NOT EXISTS "${POSEIDON_BUNDLED_RMM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Bundled RMM was not found at ${POSEIDON_BUNDLED_RMM_DIR}")
endif()

set(CPM_SOURCE_CACHE "${POSEIDON_THIRD_PARTY_DIR}/cpm" CACHE PATH "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(RMM_NVTX OFF CACHE BOOL "" FORCE)
set(USE_NVTX OFF CACHE BOOL "" FORCE)
set(RMM_LOGGING_LEVEL "OFF" CACHE STRING "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL_HO ON CACHE BOOL "" FORCE)
add_subdirectory(
    "${POSEIDON_BUNDLED_RMM_DIR}"
    "${CMAKE_BINARY_DIR}/third_party/rmm"
    EXCLUDE_FROM_ALL)

set(POSEIDON_GPU_SOURCES
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_ciphertext.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_key.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_parameter.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_plaintext.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_uploader.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_evaluator.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_double_hoist.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_elementwise_handler.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_keyswitch_handler.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_modswitch_handler.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_ntt_handler.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_tensor_core_gemm.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_elementwise_kernels.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_double_hoist_kernels.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_keyswitch_kernels.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_ntt_kernels.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_rescale_kernels.cu")

add_executable(poseidon_gpu_resnet50
    main.cpp
    gpu_ckks_runtime.cpp
    gpu_multiplexed_tensor.cpp
    gpu_resnet50_inference.cpp
    gpu_relu.cpp
    resnet50_config.cpp
    resnet50_topology.cpp
    resnet50_weights.cpp
    ${POSEIDON_GPU_SOURCES})

target_include_directories(poseidon_gpu_resnet50 PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${POSEIDON_ROOT}/src")
target_link_libraries(poseidon_gpu_resnet50 PRIVATE
    poseidon_shared
    CUDA::cudart
    rmm::rmm)
target_compile_options(poseidon_gpu_resnet50 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(poseidon_gpu_resnet50 PRIVATE
    POSEIDON_GPU_RESNET50_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}")
set_target_properties(poseidon_gpu_resnet50 PROPERTIES CUDA_ARCHITECTURES 70)
