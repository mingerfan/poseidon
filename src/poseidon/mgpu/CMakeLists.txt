add_library(poseidon_mgpu STATIC
    compiler/schedule_verifier.cpp
    ir/schedule.cpp
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

    target_link_libraries(poseidon_mgpu_ir_tests
        PRIVATE
            poseidon_mgpu
    )

    add_test(
        NAME poseidon_mgpu_ir_tests
        COMMAND poseidon_mgpu_ir_tests
    )
endif()
