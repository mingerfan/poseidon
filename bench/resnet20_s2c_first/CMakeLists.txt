# The Runtime adapter and this application share one canonical GPU library.
find_package(CUDAToolkit REQUIRED)
if(NOT TARGET poseidon::gpu)
    message(FATAL_ERROR "The S2C-first application requires poseidon::gpu")
endif()
if(NOT TARGET CUDA::cupti)
    message(FATAL_ERROR "The S2C-first activity collector requires CUDA CUPTI")
endif()

add_executable(poseidon_resnet20_s2c_first
    main.cpp plain_resnet20_reference.cpp gpu_activity_timing.cpp
    support/resnet20_weights.cpp)
target_link_libraries(poseidon_resnet20_s2c_first PRIVATE
    poseidon::gpu CUDA::cupti ${CMAKE_DL_LIBS})
target_include_directories(poseidon_resnet20_s2c_first PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/support"
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

add_executable(test_homomorphic_dft_repack
    "${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/bootstrapping/test_homomorphic_dft_repack.cpp")
target_link_libraries(test_homomorphic_dft_repack PRIVATE poseidon_shared)
add_test(NAME test_homomorphic_dft_repack COMMAND test_homomorphic_dft_repack)

add_executable(test_resnet20_s2c_support test_support.cpp support/resnet20_weights.cpp)
target_include_directories(test_resnet20_s2c_support PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/support")
target_compile_definitions(test_resnet20_s2c_support PRIVATE
    POSEIDON_GPU_RESNET20_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/support")
add_test(NAME resnet20_s2c_support COMMAND test_resnet20_s2c_support)
