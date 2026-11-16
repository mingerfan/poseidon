cmake_minimum_required(VERSION 3.26.4)
project(PoseidonGpuKeyload LANGUAGES C CXX)

# This is a standalone entry point, NOT an addition to the legacy bench target.
# Keep Runtime (and its JSON dependency), applications and other tests out.
foreach(_option IN ITEMS POSEIDON_BUILD_EXAMPLES POSEIDON_BUILD_BENCH
        POSEIDON_BUILD_CPU_TESTS POSEIDON_BUILD_CKKS_RUNTIME_API
        POSEIDON_BUILD_CKKS_RUNTIME_TESTS POSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS
        POSEIDON_BUILD_RESNET20_S2C_FIRST POSEIDON_BUILD_GPU_ENCODE_BENCH
        POSEIDON_USE_HARDWARE)
    set(${_option} OFF CACHE BOOL "Disabled by standalone GPU keyload bench" FORCE)
endforeach()
set(POSEIDON_BUILD_GPU ON CACHE BOOL "Build the standalone GPU library" FORCE)

# Defaults only: callers can still pass any of these as -DPOSEIDON_...=ON/OFF.
option(POSEIDON_BUILD_DEPS "Download missing Poseidon dependencies" OFF)
option(POSEIDON_USE_MSGSL "Use Microsoft GSL" OFF)
option(POSEIDON_USE_ZLIB "Enable zlib serialization" OFF)
option(POSEIDON_USE_ZSTD "Enable zstd serialization" OFF)
option(POSEIDON_USE_SPDLOG "Enable Poseidon spdlog logging" OFF)
option(POSEIDON_USE_OPENMP "Enable guarded Poseidon OpenMP paths" ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# A800 is SM80; set explicitly so the Poseidon child does not select its SM75 default.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures")
endif()
# CUDA must also be enabled in this parent scope: link features of poseidon::gpu
# propagate here, whereas enable_language() in the Poseidon child alone does not.
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

get_filename_component(_poseidon_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(_poseidon_binary "${CMAKE_CURRENT_BINARY_DIR}/poseidon")
add_subdirectory("${_poseidon_root}" "${_poseidon_binary}" EXCLUDE_FROM_ALL)
# Some CPU sources include spdlog unconditionally even with the root option OFF.
# Reuse RMM's already-configured header-only dependency (including external fmt)
# instead of accidentally including unrelated system headers without linking fmt.
if(NOT POSEIDON_USE_SPDLOG)
    if(NOT TARGET spdlog::spdlog_header_only)
        message(FATAL_ERROR "RMM did not expose spdlog::spdlog_header_only")
    endif()
    target_link_libraries(poseidon_shared PUBLIC spdlog::spdlog_header_only)
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)
# CUDAToolkit's imported targets were created in the child directory's scope.
find_package(CUDAToolkit REQUIRED)

execute_process(COMMAND git -C "${_poseidon_root}" rev-parse HEAD
    OUTPUT_VARIABLE KEYLOAD_SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT KEYLOAD_SOURCE_REVISION)
    set(KEYLOAD_SOURCE_REVISION "unknown")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" KEYLOAD_BENCH_SOURCE_SHA256)
foreach(_option IN ITEMS POSEIDON_USE_OPENMP POSEIDON_BUILD_DEPS
        POSEIDON_USE_MSGSL POSEIDON_USE_ZLIB POSEIDON_USE_ZSTD)
    set(KEYLOAD_${_option} "${${_option}}")
endforeach()
configure_file(build_info.h.in generated/keyload_build_info.h @ONLY)

add_executable(poseidon_gpu_keyload main.cpp)
target_compile_features(poseidon_gpu_keyload PRIVATE cxx_std_17)
target_compile_definitions(poseidon_gpu_keyload PRIVATE KEYLOAD_BUILD_CONFIGURATION="$<CONFIG>")
target_include_directories(poseidon_gpu_keyload PRIVATE
    "${_poseidon_binary}" "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(poseidon_gpu_keyload PRIVATE poseidon::gpu OpenMP::OpenMP_CXX)
set_target_properties(poseidon_gpu_keyload PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
# No automatic GPU/CPU keygen tests: even a configure/build must be safe on a login host.
