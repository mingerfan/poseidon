find_package(CUDAToolkit REQUIRED)

if((POSEIDON_BUILD_MGPU_CUDA_COMM OR POSEIDON_BUILD_MGPU_NCCL_COMM)
    AND POSEIDON_BUILD_MGPU_GPU_OBJECTS)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING
            "CUDA architectures for optional multi-GPU CUDA targets" FORCE)
    endif()
endif()

if(POSEIDON_BUILD_MGPU_CUDA_COMM AND POSEIDON_BUILD_MGPU_GPU_OBJECTS)
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
endif()

if(POSEIDON_BUILD_MGPU_NCCL_COMM AND NOT POSEIDON_BUILD_MGPU_GPU_OBJECTS)
    message(FATAL_ERROR
        "poseidon_mgpu_nccl_transfer_bench requires "
        "POSEIDON_BUILD_MGPU_GPU_OBJECTS=ON because it constructs real GPU ciphertexts")
endif()

if(POSEIDON_BUILD_MGPU_NCCL_COMM AND POSEIDON_BUILD_MGPU_GPU_OBJECTS)
    add_executable(poseidon_mgpu_nccl_transfer_bench
        nccl_transfer_bench.cpp
    )

    target_include_directories(poseidon_mgpu_nccl_transfer_bench
        PRIVATE
            ${POSEIDON_INCLUDES_DIR}/src
    )

    target_link_libraries(poseidon_mgpu_nccl_transfer_bench
        PRIVATE
            poseidon_mgpu_nccl_comm
            poseidon_mgpu_gpu_objects
            CUDA::cudart
            rmm::rmm
    )
endif()
