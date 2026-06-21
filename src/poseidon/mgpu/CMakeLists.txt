add_library(poseidon_mgpu STATIC
    comm/gpu_comm.cpp
    comm/gpu_object_copy.cpp
    comm/materialized_gpu_comm.cpp
    compiler/copy_insertion.cpp
    compiler/dacapo_adapter.cpp
    compiler/schedule_verifier.cpp
    compiler/static_placement.cpp
    compiler/static_schedule_pipeline.cpp
    ir/schedule.cpp
    ir/schedule_json.cpp
    runtime/comm_schedule_handler.cpp
    runtime/object_store.cpp
    runtime/schedule_interpreter.cpp
)

target_include_directories(poseidon_mgpu
    PUBLIC
        ${POSEIDON_INCLUDES_DIR}/src
)

target_link_libraries(poseidon_mgpu
    PUBLIC
        poseidon_shared
)

if(POSEIDON_BUILD_MGPU_TESTS)
    add_executable(poseidon_mgpu_ir_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_ir_test.cpp
    )

    add_executable(poseidon_mgpu_schedule_json_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_schedule_json_test.cpp
    )

    add_executable(poseidon_mgpu_comm_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_comm_test.cpp
    )

    add_executable(poseidon_mgpu_copy_insertion_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_copy_insertion_test.cpp
    )

    add_executable(poseidon_mgpu_object_copy_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_object_copy_test.cpp
    )

    add_executable(poseidon_mgpu_materialized_comm_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_materialized_comm_test.cpp
    )

    add_executable(poseidon_mgpu_dacapo_adapter_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_dacapo_adapter_test.cpp
    )

    add_executable(poseidon_mgpu_runtime_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_runtime_test.cpp
    )

    add_executable(poseidon_mgpu_static_placement_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_static_placement_test.cpp
    )

    add_executable(poseidon_mgpu_static_pipeline_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_static_pipeline_test.cpp
    )

    target_link_libraries(poseidon_mgpu_ir_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_schedule_json_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_comm_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_copy_insertion_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_object_copy_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_materialized_comm_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_dacapo_adapter_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_runtime_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_static_placement_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_static_pipeline_tests
        PRIVATE
            poseidon_mgpu
    )

    add_test(
        NAME poseidon_mgpu_ir_tests
        COMMAND poseidon_mgpu_ir_tests
    )

    add_test(
        NAME poseidon_mgpu_schedule_json_tests
        COMMAND poseidon_mgpu_schedule_json_tests
    )

    add_test(
        NAME poseidon_mgpu_comm_tests
        COMMAND poseidon_mgpu_comm_tests
    )

    add_test(
        NAME poseidon_mgpu_copy_insertion_tests
        COMMAND poseidon_mgpu_copy_insertion_tests
    )

    add_test(
        NAME poseidon_mgpu_object_copy_tests
        COMMAND poseidon_mgpu_object_copy_tests
    )

    add_test(
        NAME poseidon_mgpu_materialized_comm_tests
        COMMAND poseidon_mgpu_materialized_comm_tests
    )

    add_test(
        NAME poseidon_mgpu_dacapo_adapter_tests
        COMMAND poseidon_mgpu_dacapo_adapter_tests
    )

    add_test(
        NAME poseidon_mgpu_runtime_tests
        COMMAND poseidon_mgpu_runtime_tests
    )

    add_test(
        NAME poseidon_mgpu_static_placement_tests
        COMMAND poseidon_mgpu_static_placement_tests
    )

    add_test(
        NAME poseidon_mgpu_static_pipeline_tests
        COMMAND poseidon_mgpu_static_pipeline_tests
    )
endif()

option(POSEIDON_BUILD_MGPU_CUDA_COMM "Build optional CUDA peer communication backend" OFF)
message(STATUS "POSEIDON_BUILD_MGPU_CUDA_COMM: ${POSEIDON_BUILD_MGPU_CUDA_COMM}")

if(POSEIDON_BUILD_MGPU_CUDA_COMM)
    find_package(CUDAToolkit REQUIRED)

    add_library(poseidon_mgpu_cuda_comm STATIC
        comm/cuda_peer_comm.cpp
    )

    target_include_directories(poseidon_mgpu_cuda_comm
        PUBLIC
            ${POSEIDON_INCLUDES_DIR}/src
    )

    target_link_libraries(poseidon_mgpu_cuda_comm
        PUBLIC
            poseidon_mgpu
            CUDA::cudart
    )

    if(POSEIDON_BUILD_MGPU_TESTS)
        add_executable(poseidon_mgpu_cuda_comm_tests
            ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_cuda_comm_test.cpp
        )

        target_link_libraries(poseidon_mgpu_cuda_comm_tests
            PRIVATE
                poseidon_mgpu_cuda_comm
        )

        add_test(
            NAME poseidon_mgpu_cuda_comm_tests
            COMMAND poseidon_mgpu_cuda_comm_tests
        )
        set_tests_properties(poseidon_mgpu_cuda_comm_tests
            PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
