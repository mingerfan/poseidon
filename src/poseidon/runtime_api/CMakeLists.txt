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
endif()
