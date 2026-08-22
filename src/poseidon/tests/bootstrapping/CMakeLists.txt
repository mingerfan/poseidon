cmake_minimum_required(VERSION 3.26.4)

project(PoseidonGpuBootstrappingTests LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

option(POSEIDON_GPU_BOOTSTRAPPING_ENABLE_GPU_TESTS
    "Build GPU bootstrapping tests that require CUDA and RMM"
    ON)

set(POSEIDON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." CACHE PATH
    "Path to the Poseidon source tree")

set(POSEIDON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_DEPS OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL
    "Disable all FetchContent downloads for the offline GPU bootstrap test" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Disable FetchContent update checks for the offline GPU bootstrap test" FORCE)
set(CPM_LOCAL_PACKAGES_ONLY ON CACHE BOOL
    "Require CPM dependencies to be available locally" FORCE)
set(CPM_DOWNLOAD_ALL OFF CACHE BOOL "" FORCE)

add_subdirectory("${POSEIDON_ROOT}" "${CMAKE_BINARY_DIR}/poseidon_build")

if(POSEIDON_GPU_BOOTSTRAPPING_ENABLE_GPU_TESTS)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING
            "CUDA architectures for GPU bootstrapping tests" FORCE)
    endif()

    find_package(CUDAToolkit REQUIRED)

    set(POSEIDON_THIRD_PARTY_DIR "${POSEIDON_ROOT}/third_party" CACHE PATH
        "Path to bundled third-party dependencies")
    set(POSEIDON_BUNDLED_RMM_DIR "${POSEIDON_THIRD_PARTY_DIR}/rmm" CACHE PATH
        "Path to bundled RMM source")

    if(EXISTS "${POSEIDON_BUNDLED_RMM_DIR}/CMakeLists.txt")
        message(STATUS "RMM: using bundled source at ${POSEIDON_BUNDLED_RMM_DIR}")
        set(CPM_SOURCE_CACHE "${POSEIDON_THIRD_PARTY_DIR}/cpm" CACHE PATH
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
            "${CMAKE_BINARY_DIR}/third_party/rmm"
            EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR
            "Bundled RMM was not found at ${POSEIDON_BUNDLED_RMM_DIR}. "
            "For offline builds, copy RMM and its vendored dependencies into "
            "${POSEIDON_THIRD_PARTY_DIR}; otherwise install RMM and adjust this CMake file.")
    endif()

    set(POSEIDON_GPU_DATA_STRUCT_SOURCES
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_bootstrap_profile.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_ciphertext.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_key.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_memory.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_parameter.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_plaintext.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_uploader.cpp"
    )

    set(POSEIDON_GPU_OPERATOR_SOURCES
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
        "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_rescale_kernels.cu"
    )

    add_executable(test_gpu_bootstrap_modraise
        test_gpu_bootstrap_modraise.cpp
        ${POSEIDON_GPU_DATA_STRUCT_SOURCES}
        ${POSEIDON_GPU_OPERATOR_SOURCES}
    )

    target_link_libraries(test_gpu_bootstrap_modraise
        PRIVATE
            poseidon_shared
            CUDA::cudart
            rmm::rmm
    )
    target_compile_options(test_gpu_bootstrap_modraise
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    set_target_properties(test_gpu_bootstrap_modraise PROPERTIES
        CUDA_ARCHITECTURES 70)

    add_test(
        NAME test_gpu_bootstrap_modraise
        COMMAND test_gpu_bootstrap_modraise
    )
    set_tests_properties(test_gpu_bootstrap_modraise
        PROPERTIES SKIP_RETURN_CODE 77)
endif()
