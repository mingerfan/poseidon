cmake_minimum_required(VERSION 3.16)

project(PoseidonGpuDataStructTests LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

option(POSEIDON_GPU_DATA_STRUCT_ENABLE_GPU_TESTS
    "Build GPU data-structure tests that require CUDA and RMM"
    ON)

set(POSEIDON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." CACHE PATH
    "Path to the Poseidon source tree")

set(POSEIDON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_DEPS OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_MSGSL OFF CACHE BOOL "" FORCE)

add_subdirectory("${POSEIDON_ROOT}" "${CMAKE_BINARY_DIR}/poseidon_build")

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

if(POSEIDON_GPU_DATA_STRUCT_ENABLE_GPU_TESTS)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    find_package(CUDAToolkit REQUIRED)
    find_package(rmm CONFIG REQUIRED)

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
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_modswitch_handler.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_ntt_handler.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_elementwise_kernels.cu"
    )

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
endif()
