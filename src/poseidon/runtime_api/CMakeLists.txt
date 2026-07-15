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
endif()
