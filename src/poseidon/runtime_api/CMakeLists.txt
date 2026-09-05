add_library(poseidon_runtime_cpu_api STATIC
    poseidon_cpu_api.cpp
)
add_library(poseidon::runtime_cpu_api ALIAS poseidon_runtime_cpu_api)

target_include_directories(poseidon_runtime_cpu_api
    PUBLIC
        ${POSEIDON_INCLUDES_DIR}/src
)
target_link_libraries(poseidon_runtime_cpu_api
    PUBLIC
        poseidon_shared
        ckks_runtime::core
)
target_compile_features(poseidon_runtime_cpu_api PUBLIC cxx_std_17)

if(POSEIDON_BUILD_CKKS_RUNTIME_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_link_libraries(poseidon_runtime_cpu_api PUBLIC MPI::MPI_CXX)
    target_compile_definitions(poseidon_runtime_cpu_api
        PUBLIC POSEIDON_RUNTIME_CPU_MPI=1)
endif()

add_library(poseidon_runtime_gpu_api STATIC
    poseidon_gpu_api.cpp
    communication/cuda_local_transfer.cpp
    communication/gpu_object_copy.cpp
)
add_library(poseidon::runtime_gpu_api ALIAS poseidon_runtime_gpu_api)

target_include_directories(poseidon_runtime_gpu_api
    PUBLIC
        ${POSEIDON_INCLUDES_DIR}/src
)
target_link_libraries(poseidon_runtime_gpu_api
    PUBLIC
        poseidon_shared
        poseidon::gpu
        ckks_runtime::core
)
target_compile_features(poseidon_runtime_gpu_api PUBLIC cxx_std_17)

if(POSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL)
    find_package(MPI REQUIRED COMPONENTS CXX)
    find_path(POSEIDON_NCCL_INCLUDE_DIR nccl.h
        HINTS ENV NCCL_ROOT ENV CUDA_HOME
        PATH_SUFFIXES include)
    find_library(POSEIDON_NCCL_LIBRARY NAMES nccl
        HINTS ENV NCCL_ROOT ENV CUDA_HOME
        PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
    if(NOT POSEIDON_NCCL_INCLUDE_DIR OR NOT POSEIDON_NCCL_LIBRARY)
        message(FATAL_ERROR
            "POSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL requires nccl.h and libnccl")
    endif()
    target_sources(poseidon_runtime_gpu_api PRIVATE
        communication/nccl_mpi_transport.cpp
    )
    target_include_directories(poseidon_runtime_gpu_api PUBLIC
        ${POSEIDON_NCCL_INCLUDE_DIR})
    target_link_libraries(poseidon_runtime_gpu_api PUBLIC
        MPI::MPI_CXX
        ${POSEIDON_NCCL_LIBRARY})
    target_compile_definitions(poseidon_runtime_gpu_api PUBLIC
        POSEIDON_RUNTIME_GPU_NCCL=1)
endif()

if(POSEIDON_BUILD_CKKS_RUNTIME_TESTS)
    add_executable(poseidon_runtime_cpu_api_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/poseidon_cpu_api_test.cpp
    )
    target_link_libraries(poseidon_runtime_cpu_api_tests
        PRIVATE
            poseidon_runtime_cpu_api
    )
    add_test(
        NAME poseidon_runtime_cpu_api_tests
        COMMAND poseidon_runtime_cpu_api_tests
    )
    add_executable(poseidon_runtime_cpu_mlp_e2e
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/poseidon_cpu_mlp_e2e_test.cpp
    )
    target_link_libraries(poseidon_runtime_cpu_mlp_e2e
        PRIVATE
            poseidon_runtime_cpu_api
            ckks_runtime::plan_json
    )
    if(POSEIDON_BUILD_CKKS_RUNTIME_MPI)
        add_test(
            NAME poseidon_runtime_cpu_api_mpi_2_tests
            COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:poseidon_runtime_cpu_api_tests> --mpi
                ${MPIEXEC_POSTFLAGS}
        )
        add_test(
            NAME poseidon_runtime_cpu_api_mpi_4_tests
            COMMAND
                ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:poseidon_runtime_cpu_api_tests> --mpi
                ${MPIEXEC_POSTFLAGS}
        )
    endif()
endif()

if(POSEIDON_BUILD_CKKS_RUNTIME_GPU_NCCL AND
   POSEIDON_BUILD_CKKS_RUNTIME_TESTS)
    add_executable(poseidon_nccl_mpi_smoke
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/nccl_mpi_smoke.cpp
    )
    target_link_libraries(poseidon_nccl_mpi_smoke PRIVATE
        poseidon_runtime_gpu_api)
    add_test(NAME poseidon_nccl_mpi_smoke
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:poseidon_nccl_mpi_smoke>
                --device-counts 1x1x1x1
                ${MPIEXEC_POSTFLAGS})

    add_executable(poseidon_gpu_mpi_transfer_test
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/poseidon_gpu_mpi_transfer_test.cpp
    )
    target_link_libraries(poseidon_gpu_mpi_transfer_test PRIVATE
        poseidon_runtime_gpu_api)

    add_executable(poseidon_gpu_mpi_plan_e2e
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/poseidon_gpu_mpi_plan_e2e.cpp
    )
    target_link_libraries(poseidon_gpu_mpi_plan_e2e PRIVATE
        poseidon_runtime_gpu_api
        ckks_runtime::plan_json)

    add_executable(poseidon_gpu_mpi_runtime_e2e
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/poseidon_gpu_mpi_runtime_e2e.cpp
    )
    target_link_libraries(poseidon_gpu_mpi_runtime_e2e PRIVATE
        poseidon_runtime_gpu_api
        ckks_runtime::plan_json)
endif()

if(POSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS)
    if(NOT POSEIDON_BUILD_CKKS_RUNTIME_TESTS)
        message(FATAL_ERROR
            "POSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS requires "
            "POSEIDON_BUILD_CKKS_RUNTIME_TESTS=ON")
    endif()
    add_executable(poseidon_runtime_gpu_api_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp
    )
    target_link_libraries(poseidon_runtime_gpu_api_tests
        PRIVATE
            poseidon_runtime_gpu_api
    )
    add_test(
        NAME poseidon_runtime_gpu_api_tests
        COMMAND poseidon_runtime_gpu_api_tests
    )

    add_executable(poseidon_runtime_gpu_plan_e2e
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/poseidon_gpu_plan_e2e.cpp
    )
    target_link_libraries(poseidon_runtime_gpu_plan_e2e
        PRIVATE
            poseidon_runtime_gpu_api
            ckks_runtime::plan_json
    )

    add_executable(poseidon_runtime_gpu_mlp_e2e
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/poseidon_gpu_mlp_e2e_test.cpp
    )
    target_link_libraries(poseidon_runtime_gpu_mlp_e2e
        PRIVATE
            poseidon_runtime_gpu_api
            ckks_runtime::plan_json
    )

    add_executable(poseidon_runtime_gpu_bootstrap_e2e
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/poseidon_gpu_bootstrap_e2e.cpp
    )
    target_link_libraries(poseidon_runtime_gpu_bootstrap_e2e
        PRIVATE
            poseidon_runtime_gpu_api
    )
    add_test(
        NAME poseidon_runtime_gpu_bootstrap_e2e
        COMMAND poseidon_runtime_gpu_bootstrap_e2e
    )
    set_tests_properties(poseidon_runtime_gpu_bootstrap_e2e
        PROPERTIES SKIP_RETURN_CODE 77)
endif()
