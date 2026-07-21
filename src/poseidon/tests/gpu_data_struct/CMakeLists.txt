cmake_minimum_required(VERSION 3.26.4)

project(PoseidonGpuDataStructTests LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

option(POSEIDON_GPU_DATA_STRUCT_ENABLE_GPU_TESTS
    "Build GPU data-structure tests that require CUDA and RMM"
    ON)
option(POSEIDON_GPU_DATA_STRUCT_BUILD_CPU_TEST
    "Build the auxiliary CPU ciphertext-generation test"
    OFF)
option(POSEIDON_GPU_DATA_STRUCT_BUILD_GPU_STORAGE_TEST
    "Build the auxiliary GPU data-structure generation test"
    OFF)

set(POSEIDON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." CACHE PATH
    "Path to the Poseidon source tree")

set(POSEIDON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_DEPS OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_MSGSL OFF CACHE BOOL "" FORCE)

add_subdirectory("${POSEIDON_ROOT}" "${CMAKE_BINARY_DIR}/poseidon_build")

if(POSEIDON_GPU_DATA_STRUCT_BUILD_CPU_TEST)
    add_executable(test_cpu_ciphertext_generation
        test_cpu_ciphertext_generation.cpp
    )

    target_link_libraries(test_cpu_ciphertext_generation
        PRIVATE
            poseidon_shared
    )

    add_test(
        NAME test_cpu_ciphertext_generation
        COMMAND test_cpu_ciphertext_generation
    )
endif()

if(POSEIDON_GPU_DATA_STRUCT_ENABLE_GPU_TESTS)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING
            "CUDA architectures for GPU data-structure tests" FORCE)
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
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_ciphertext.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_key.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_parameter.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_plaintext.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_uploader.cpp"
    )

    set(POSEIDON_GPU_ELEMENTWISE_SOURCES
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_evaluator.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_elementwise_handler.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_keyswitch_handler.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_modswitch_handler.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_ntt_handler.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_tensor_core_gemm.cu"
        "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_elementwise_kernels.cu"
        "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_keyswitch_kernels.cu"
        "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_ntt_kernels.cu"
        "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_rescale_kernels.cu"
    )

    if(POSEIDON_GPU_DATA_STRUCT_BUILD_GPU_STORAGE_TEST)
        add_executable(test_gpu_data_struct_generation
            test_gpu_data_struct_generation.cpp
            ${POSEIDON_GPU_DATA_STRUCT_SOURCES}
        )

        target_link_libraries(test_gpu_data_struct_generation
            PRIVATE
                poseidon_shared
                CUDA::cudart
                rmm::rmm
        )

        add_test(
            NAME test_gpu_data_struct_generation
            COMMAND test_gpu_data_struct_generation
        )
        set_tests_properties(test_gpu_data_struct_generation
            PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    add_executable(demo_gpu_ciphertext_add_handler
        demo_gpu_ciphertext_add_handler.cpp
        ${POSEIDON_GPU_DATA_STRUCT_SOURCES}
        ${POSEIDON_GPU_ELEMENTWISE_SOURCES}
    )

    target_link_libraries(demo_gpu_ciphertext_add_handler
        PRIVATE
            poseidon_shared
            CUDA::cudart
            rmm::rmm
    )
    target_include_directories(demo_gpu_ciphertext_add_handler
        PRIVATE
            "${CMAKE_BINARY_DIR}/poseidon_build"
    )
    target_compile_options(demo_gpu_ciphertext_add_handler
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    set_target_properties(demo_gpu_ciphertext_add_handler PROPERTIES
        CUDA_ARCHITECTURES 75)
endif()
