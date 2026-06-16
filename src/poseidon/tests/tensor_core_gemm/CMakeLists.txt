cmake_minimum_required(VERSION 3.18)

project(PoseidonTensorCoreGemmTests LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

enable_testing()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING
        "CUDA architectures for integer Tensor Core GEMM tests" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

set(POSEIDON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." CACHE PATH
    "Path to the Poseidon source tree")

add_library(poseidon_gpu_tensor_core_gemm STATIC
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_tensor_core_gemm.cu"
)

target_include_directories(poseidon_gpu_tensor_core_gemm
    PUBLIC
        "${POSEIDON_ROOT}/src"
)

target_compile_options(poseidon_gpu_tensor_core_gemm
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_link_libraries(poseidon_gpu_tensor_core_gemm
    PUBLIC
        CUDA::cudart
)

set_target_properties(poseidon_gpu_tensor_core_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

add_executable(tensor_core_gemm_test
    tensor_core_gemm_test.cpp
)

target_link_libraries(tensor_core_gemm_test
    PRIVATE
        poseidon_gpu_tensor_core_gemm
)

add_test(
    NAME tensor_core_gemm_test
    COMMAND tensor_core_gemm_test
)
set_tests_properties(tensor_core_gemm_test
    PROPERTIES SKIP_RETURN_CODE 77)

add_executable(tensor_core_gemm_bench
    tensor_core_gemm_bench.cpp
)

target_link_libraries(tensor_core_gemm_bench
    PRIVATE
        poseidon_gpu_tensor_core_gemm
)

add_test(
    NAME tensor_core_gemm_bench_smoke
    COMMAND tensor_core_gemm_bench --size 16 --mode u8 --warmup 0 --repeat 1 --verify-samples 4
)
set_tests_properties(tensor_core_gemm_bench_smoke
    PROPERTIES SKIP_RETURN_CODE 77)
