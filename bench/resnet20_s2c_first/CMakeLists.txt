# The Runtime adapter and this application share one canonical GPU library.
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET poseidon::gpu)
    message(FATAL_ERROR "The S2C-first application requires poseidon::gpu")
endif()
if(NOT TARGET CUDA::cupti)
    message(FATAL_ERROR "The S2C-first activity collector requires CUDA CUPTI")
endif()

if(NOT TARGET ckks_runtime::plan_json)
    set(CKKS_RUNTIME_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(CKKS_RUNTIME_BUILD_EXPERIMENTS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${POSEIDON_INCLUDES_DIR}/third_party/ckks-runtime"
        "${CMAKE_CURRENT_BINARY_DIR}/ckks-runtime")
endif()

add_executable(poseidon_resnet20_s2c_first
    main.cpp plain_resnet20_reference.cpp gpu_activity_timing.cpp
    support/resnet20_weights.cpp)
target_link_libraries(poseidon_resnet20_s2c_first PRIVATE
    poseidon::gpu ckks_runtime::plan_json CUDA::cupti ${CMAKE_DL_LIBS})
target_include_directories(poseidon_resnet20_s2c_first PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/support"
    "${POSEIDON_INCLUDES_DIR}/third_party/nvtx/c/include"
    "${POSEIDON_INCLUDES_DIR}/Doc/analysis/resnet20_joint_chain_20260911")
target_compile_definitions(poseidon_resnet20_s2c_first PRIVATE
    POSEIDON_GPU_RESNET20_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/support")
# Preserve the previous application compiler optimization and dead-code policy.
target_compile_options(poseidon_resnet20_s2c_first PRIVATE
    -O1 -ffunction-sections -fdata-sections)
target_link_options(poseidon_resnet20_s2c_first PRIVATE -Wl,--gc-sections)
set_target_properties(poseidon_resnet20_s2c_first PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME resnet20_s2c_first_metadata
    COMMAND poseidon_resnet20_s2c_first --metadata-only
    "${POSEIDON_INCLUDES_DIR}/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt" 0)

add_executable(gpu_trace_plan_validation_test
    "${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/gpu_trace_plan_validation_test.cpp")
target_link_libraries(gpu_trace_plan_validation_test PRIVATE poseidon::gpu ckks_runtime::plan_json)
set_target_properties(gpu_trace_plan_validation_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME gpu_trace_plan_validation_test COMMAND gpu_trace_plan_validation_test)
set_tests_properties(gpu_trace_plan_validation_test PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_gpu_split_hoist
    "${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/gpu_split_hoist_test.cpp")
target_link_libraries(test_gpu_split_hoist PRIVATE poseidon::gpu)
set_target_properties(test_gpu_split_hoist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME gpu_split_hoist COMMAND test_gpu_split_hoist --small-only)
set_tests_properties(gpu_split_hoist PROPERTIES SKIP_RETURN_CODE 77)

foreach(_gpu_test IN ITEMS gpu_selected_prerotated_crop_test gpu_uploader_stream_order_test gpu_key_view_cache_test gpu_boot_optimizations_test)
    add_executable(${_gpu_test} "${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/${_gpu_test}.cpp")
    target_link_libraries(${_gpu_test} PRIVATE poseidon::gpu)
    set_target_properties(${_gpu_test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME ${_gpu_test} COMMAND ${_gpu_test})
    set_tests_properties(${_gpu_test} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

add_executable(test_resnet20_manual_boot
    "${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/runtime_api/manual_low_boot_test.cpp")
target_link_libraries(test_resnet20_manual_boot PRIVATE poseidon::gpu)
add_test(NAME resnet20_manual_boot_host COMMAND test_resnet20_manual_boot
    "${POSEIDON_INCLUDES_DIR}/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt" --host)

add_executable(test_resnet20_prepared_api_graph test_prepared_api_graph.cpp)
add_test(NAME resnet20_prepared_api_graph COMMAND test_resnet20_prepared_api_graph)
set_target_properties(test_resnet20_manual_boot test_resnet20_prepared_api_graph
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(test_homomorphic_dft_repack
    "${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/bootstrapping/test_homomorphic_dft_repack.cpp")
target_link_libraries(test_homomorphic_dft_repack PRIVATE poseidon_shared)
add_test(NAME test_homomorphic_dft_repack COMMAND test_homomorphic_dft_repack)

add_executable(test_resnet20_s2c_support test_support.cpp support/resnet20_weights.cpp)
target_include_directories(test_resnet20_s2c_support PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/support")
target_compile_definitions(test_resnet20_s2c_support PRIVATE
    POSEIDON_GPU_RESNET20_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/support")
add_test(NAME resnet20_s2c_support COMMAND test_resnet20_s2c_support)
