enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING
        "CUDA architectures for optional multi-GPU CUDA targets" FORCE)
endif()

add_executable(poseidon_mgpu_ckks_transfer_bench
    ckks_transfer_bench.cpp
    cuda_link_stub.cu
)

target_include_directories(poseidon_mgpu_ckks_transfer_bench
    PRIVATE
        ${POSEIDON_INCLUDES_DIR}/src
)

target_link_libraries(poseidon_mgpu_ckks_transfer_bench
    PRIVATE
        poseidon_mgpu_cuda_comm
        poseidon_mgpu_gpu_objects
        CUDA::cudart
        rmm::rmm
)
