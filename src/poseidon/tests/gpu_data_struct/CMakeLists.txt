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
    find_package(CUDAToolkit REQUIRED)
    find_package(rmm CONFIG REQUIRED)

    set(POSEIDON_GPU_DATA_STRUCT_SOURCES
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_ciphertext.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_key.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_parameter.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_plaintext.cpp"
        "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_uploader.cpp"
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
endif()
