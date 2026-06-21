add_library(poseidon_mgpu STATIC
    comm/gpu_comm.cpp
    compiler/copy_insertion.cpp
    compiler/dacapo_adapter.cpp
    compiler/schedule_verifier.cpp
    ir/schedule.cpp
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

    add_executable(poseidon_mgpu_comm_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_comm_test.cpp
    )

    add_executable(poseidon_mgpu_copy_insertion_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_copy_insertion_test.cpp
    )

    add_executable(poseidon_mgpu_dacapo_adapter_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_dacapo_adapter_test.cpp
    )

    add_executable(poseidon_mgpu_runtime_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_runtime_test.cpp
    )

    target_link_libraries(poseidon_mgpu_ir_tests
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

    target_link_libraries(poseidon_mgpu_dacapo_adapter_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_runtime_tests
        PRIVATE
            poseidon_mgpu
    )

    add_test(
        NAME poseidon_mgpu_ir_tests
        COMMAND poseidon_mgpu_ir_tests
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
        NAME poseidon_mgpu_dacapo_adapter_tests
        COMMAND poseidon_mgpu_dacapo_adapter_tests
    )

    add_test(
        NAME poseidon_mgpu_runtime_tests
        COMMAND poseidon_mgpu_runtime_tests
    )
endif()
