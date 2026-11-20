cmake_minimum_required(VERSION 3.22)
project(PtAgentR12NativeCpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

foreach(required POSEIDON_ROOT CKKS_RUNTIME_ROOT
                 POSEIDON_GENERATED_INCLUDE_DIR POSEIDON_NATIVE_LIBRARY)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "Set ${required} to the existing local path")
    endif()
    if(NOT IS_ABSOLUTE "${${required}}" OR NOT EXISTS "${${required}}")
        message(FATAL_ERROR "${required} must be an existing absolute path")
    endif()
    get_filename_component(${required} "${${required}}" REALPATH)
endforeach()

get_filename_component(expected_runtime
    "${POSEIDON_ROOT}/third_party/ckks-runtime" REALPATH)
if(NOT CKKS_RUNTIME_ROOT STREQUAL expected_runtime)
    message(FATAL_ERROR "CKKS_RUNTIME_ROOT must select this checkout's Runtime")
endif()
if(NOT EXISTS "${POSEIDON_GENERATED_INCLUDE_DIR}/poseidon/util/config.h")
    message(FATAL_ERROR "Generated Poseidon config.h is missing")
endif()

list(APPEND CMAKE_MODULE_PATH "${POSEIDON_ROOT}/cmake")
find_package(GMP REQUIRED MODULE)
find_package(Threads REQUIRED)

set(CKKS_RUNTIME_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CKKS_RUNTIME_BUILD_EXPERIMENTS OFF CACHE BOOL "" FORCE)
set(CKKS_RUNTIME_BUILD_MPI OFF CACHE BOOL "" FORCE)
add_subdirectory("${CKKS_RUNTIME_ROOT}" "${CMAKE_BINARY_DIR}/runtime")

add_library(r12_poseidon_existing SHARED IMPORTED)
set_target_properties(r12_poseidon_existing PROPERTIES
    IMPORTED_LOCATION "${POSEIDON_NATIVE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES
        "${POSEIDON_ROOT}/src;${POSEIDON_GENERATED_INCLUDE_DIR}")

add_library(r12_poseidon_cpu_api STATIC
    "${POSEIDON_ROOT}/src/poseidon/runtime_api/poseidon_cpu_api.cpp")
target_link_libraries(r12_poseidon_cpu_api PUBLIC
    r12_poseidon_existing ckks_runtime::core GMP::gmpxx Threads::Threads)

add_executable(mlp_client mlp_client.cpp)
target_link_libraries(mlp_client PRIVATE
    r12_poseidon_cpu_api ckks_runtime::plan_json ${CMAKE_DL_LIBS})
target_compile_definitions(mlp_client PRIVATE
    _GNU_SOURCE
    PT_R12_APPROVED_DSO="${POSEIDON_NATIVE_LIBRARY}")
get_filename_component(approved_library_dir "${POSEIDON_NATIVE_LIBRARY}" DIRECTORY)
set_target_properties(mlp_client PROPERTIES BUILD_RPATH "${approved_library_dir}")
