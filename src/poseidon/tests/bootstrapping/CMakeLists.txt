cmake_minimum_required(VERSION 3.26.4)

project(PoseidonGpuBootstrappingTests LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

option(POSEIDON_GPU_BOOTSTRAPPING_ENABLE_GPU_TESTS
    "Build GPU bootstrapping tests that require CUDA and RMM"
    ON)

set(POSEIDON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." CACHE PATH
    "Path to the Poseidon source tree")

set(POSEIDON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_DEPS OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL
    "Disable all FetchContent downloads for the offline GPU bootstrap test" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
    "Disable FetchContent update checks for the offline GPU bootstrap test" FORCE)
set(CPM_LOCAL_PACKAGES_ONLY ON CACHE BOOL
    "Require CPM dependencies to be available locally" FORCE)
set(CPM_DOWNLOAD_ALL OFF CACHE BOOL "" FORCE)

# Share the exact GPU library used by the Runtime adapter.
if(POSEIDON_GPU_BOOTSTRAPPING_ENABLE_GPU_TESTS)
    # CUDA must be enabled in the common parent of the library and test targets.
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(POSEIDON_BUILD_GPU ${POSEIDON_GPU_BOOTSTRAPPING_ENABLE_GPU_TESTS}
    CACHE BOOL "Build the standalone Poseidon GPU library" FORCE)
if(POSEIDON_BUILD_RESNET20_S2C_FIRST AND NOT POSEIDON_GPU_BOOTSTRAPPING_ENABLE_GPU_TESTS)
    message(FATAL_ERROR "S2C-first ResNet20 requires GPU support")
endif()
add_subdirectory("${POSEIDON_ROOT}" "${CMAKE_BINARY_DIR}/poseidon_build")

if(NOT TARGET test_homomorphic_dft_repack)
    add_executable(test_homomorphic_dft_repack test_homomorphic_dft_repack.cpp)
    target_link_libraries(test_homomorphic_dft_repack PRIVATE poseidon_shared)
    add_test(NAME test_homomorphic_dft_repack COMMAND test_homomorphic_dft_repack)
endif()

if(POSEIDON_GPU_BOOTSTRAPPING_ENABLE_GPU_TESTS)
    add_executable(test_gpu_bootstrap_modraise test_gpu_bootstrap_modraise.cpp)
    target_link_libraries(test_gpu_bootstrap_modraise PRIVATE poseidon::gpu)
    add_test(NAME test_gpu_bootstrap_modraise COMMAND test_gpu_bootstrap_modraise)
    set_tests_properties(test_gpu_bootstrap_modraise PROPERTIES SKIP_RETURN_CODE 77)
endif()

# CPU arithmetic check, independent of optional CUDA tests.
add_executable(test_portable_borrow test_portable_borrow.cpp)
target_include_directories(test_portable_borrow PRIVATE
    "${POSEIDON_ROOT}/src" "${CMAKE_BINARY_DIR}/poseidon_build")
enable_testing()
add_test(NAME test_portable_borrow COMMAND test_portable_borrow)

add_executable(test_bootstrap_parameter_budget test_bootstrap_parameter_budget.cpp)
target_include_directories(test_bootstrap_parameter_budget PRIVATE
    "${POSEIDON_ROOT}/src" "${CMAKE_BINARY_DIR}/poseidon_build")
target_link_libraries(test_bootstrap_parameter_budget PRIVATE poseidon_shared gmpxx gmp)
add_test(NAME test_bootstrap_parameter_budget COMMAND test_bootstrap_parameter_budget)
