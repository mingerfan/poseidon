cmake_minimum_required(VERSION 3.26.4)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(BertMultiPlan LANGUAGES C CXX)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/BertDependencies.cmake")
bert_default_cuda_architectures()
enable_language(CUDA)

get_filename_component(_bert_default_bench_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(BERT_MULTI_PLAN_BENCH_ROOT "${_bert_default_bench_root}" CACHE PATH "Coherent bench helpers/adapters and prepared module source root")
bert_module_paths(COMPACT)
bert_setup_dependencies(PREFIX COMPACT GPU RUNTIME)

foreach(_bench_input IN ITEMS native_runner.cpp prepared_qk_adapter.h prepared_av_adapter.h
        compact_native/CMakeLists.txt compact_attention/CMakeLists.txt compact_av/CMakeLists.txt
        compact_attention/qk_reference.h compact_av/av_reference.h)
    if(NOT EXISTS "${BERT_MULTI_PLAN_BENCH_ROOT}/${_bench_input}")
        message(FATAL_ERROR "BERT_MULTI_PLAN_BENCH_ROOT is missing ${_bench_input}: ${BERT_MULTI_PLAN_BENCH_ROOT}")
    endif()
endforeach()
message(STATUS "Multi-plan coherent bench source root: ${BERT_MULTI_PLAN_BENCH_ROOT}")
set(COMPACT_BUILD_SETUP_CHECK OFF CACHE BOOL "Build optional standalone CPU Linear checks")
set(ATTENTION_BUILD_METADATA_CHECK OFF CACHE BOOL "Build optional standalone CPU QK check")
set(COMPACT_AV_BUILD_METADATA_CHECK OFF CACHE BOOL "Build optional standalone CPU AV checks")
set(COMPACT_BUILD_GPU ON CACHE BOOL "Build prepared Linear engine")
set(ATTENTION_BUILD_GPU ON CACHE BOOL "Build prepared QK engine")
set(COMPACT_AV_BUILD_GPU ON CACHE BOOL "Build prepared AV engine")
if(NOT COMPACT_BUILD_GPU OR NOT ATTENTION_BUILD_GPU OR NOT COMPACT_AV_BUILD_GPU)
    message(FATAL_ERROR "The multi-plan runner requires all three prepared GPU engines")
endif()
if(NOT TARGET bert::compact_linear)
    add_subdirectory("${BERT_MULTI_PLAN_BENCH_ROOT}/compact_native" "${CMAKE_CURRENT_BINARY_DIR}/compact-native")
endif()
if(NOT TARGET bert::prepared_qk)
    add_subdirectory("${BERT_MULTI_PLAN_BENCH_ROOT}/compact_attention" "${CMAKE_CURRENT_BINARY_DIR}/compact-attention")
endif()
if(NOT TARGET bert::prepared_av)
    add_subdirectory("${BERT_MULTI_PLAN_BENCH_ROOT}/compact_av" "${CMAKE_CURRENT_BINARY_DIR}/compact-av")
endif()

add_executable(bert_multi_plan_runner runner.cpp)
target_compile_definitions(bert_multi_plan_runner PRIVATE
    BERT_COMPACT_LINEAR=1 BERT_PREPARED_QK=1 BERT_PREPARED_AV=1
    "BERT_MULTI_PLAN_LEGACY_RUNNER=\"${BERT_MULTI_PLAN_BENCH_ROOT}/native_runner.cpp\"")
target_include_directories(bert_multi_plan_runner PRIVATE "${BERT_MULTI_PLAN_BENCH_ROOT}")
target_link_libraries(bert_multi_plan_runner PRIVATE bert::runtime_gpu_api bert::plan_json
    bert::compact_linear bert::prepared_qk bert::prepared_av CUDA::cufft)
set_target_properties(bert_multi_plan_runner PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# This ELF interposition aid is Linux-specific, not a prerequisite for the runner.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(bert_no_device_guard SHARED tests/no_device_guard.c)
    set_target_properties(bert_no_device_guard PROPERTIES
        C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
        C_VISIBILITY_PRESET hidden PREFIX "" OUTPUT_NAME "no_device_guard"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bert_no_device_guard PRIVATE -Wall -Wextra -Werror)
    endif()
endif()
