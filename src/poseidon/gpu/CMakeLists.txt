find_package(CUDAToolkit REQUIRED)

set(POSEIDON_BUNDLED_RMM_DIR "${POSEIDON_INCLUDES_DIR}/third_party/rmm" CACHE PATH
    "Path to the bundled RMM source")
if(NOT EXISTS "${POSEIDON_BUNDLED_RMM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Bundled RMM was not found at ${POSEIDON_BUNDLED_RMM_DIR}")
endif()

if(NOT TARGET rmm::rmm)
    set(CPM_SOURCE_CACHE "${POSEIDON_INCLUDES_DIR}/third_party/cpm" CACHE PATH
        "Bundled CPM cache for offline RAPIDS/RMM configuration" FORCE)
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
        "${CMAKE_CURRENT_BINARY_DIR}/rmm"
        EXCLUDE_FROM_ALL)
endif()

add_library(poseidon_gpu STATIC
    gpu_ciphertext.cpp
    gpu_key.cpp
    gpu_parameter.cpp
    gpu_plaintext.cpp
    gpu_uploader.cpp
    gpu_evaluator.cpp
    gpu_elementwise_handler.cpp
    gpu_keyswitch_handler.cpp
    gpu_modswitch_handler.cpp
    gpu_ntt_handler.cpp
    gpu_tensor_core_gemm.cu
    kernels/gpu_elementwise_kernels.cu
    kernels/gpu_keyswitch_kernels.cu
    kernels/gpu_ntt_kernels.cu
    kernels/gpu_rescale_kernels.cu
)
add_library(poseidon::gpu ALIAS poseidon_gpu)

target_include_directories(poseidon_gpu
    PUBLIC
        ${POSEIDON_INCLUDES_DIR}/src
)
target_link_libraries(poseidon_gpu
    PUBLIC
        poseidon_shared
        CUDA::cudart
        rmm::rmm
)
target_compile_features(poseidon_gpu PUBLIC cxx_std_17)
target_compile_options(poseidon_gpu
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
