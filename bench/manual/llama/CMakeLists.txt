cmake_minimum_required(VERSION 3.26.4)
if(NOT TARGET poseidon::gpu)
    message(FATAL_ERROR "Add this folder from FHE_GPU_Agent after its poseidon::gpu target")
endif()
find_package(OpenMP REQUIRED)
add_executable(llama31_cerium_source_api source_benchmark.cpp support/checkpoint.cpp fast_linear_gpu.cu)
target_compile_features(llama31_cerium_source_api PRIVATE cxx_std_20)
target_include_directories(llama31_cerium_source_api PRIVATE support/include)
target_link_libraries(llama31_cerium_source_api PRIVATE poseidon::gpu OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})

# Separate manual, accuracy-gating entry. Legacy source_api remains unchanged.
add_executable(llama31_poseidon_manual manual_main.cpp support/checkpoint.cpp fast_linear_gpu.cu)
target_compile_features(llama31_poseidon_manual PRIVATE cxx_std_20)
target_include_directories(llama31_poseidon_manual PRIVATE support/include)
target_link_libraries(llama31_poseidon_manual PRIVATE poseidon::gpu OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})

# This target uses only std headers: no Context, keys, CUDA or FHE execution.
add_executable(llama31_manual_plan_tests tests/manual_plan_test.cpp)
target_compile_features(llama31_manual_plan_tests PRIVATE cxx_std_20)
target_include_directories(llama31_manual_plan_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Fixed public controls use this exact promoted implementation/backend, not
# historical binaries. Their diagnostic walls are not online performance.
foreach(_control IN ITEMS public_paired_projections public_paired_silu)
    add_executable(llama31_${_control} tests/${_control}.cpp)
    target_compile_features(llama31_${_control} PRIVATE cxx_std_20)
    target_include_directories(llama31_${_control} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} support/include)
    target_link_libraries(llama31_${_control} PRIVATE poseidon::gpu OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
endforeach()
add_executable(llama31_fast_probe fast_probe.cpp fast_linear_gpu.cu)
target_compile_features(llama31_fast_probe PRIVATE cxx_std_20)
target_include_directories(llama31_fast_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} support/include)
target_link_libraries(llama31_fast_probe PRIVATE poseidon::gpu OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})

add_executable(llama31_poseidon_dual dual_main.cpp support/checkpoint.cpp fast_linear_gpu.cu)
target_compile_features(llama31_poseidon_dual PRIVATE cxx_std_20)
target_include_directories(llama31_poseidon_dual PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} support/include)
target_link_libraries(llama31_poseidon_dual PRIVATE poseidon::gpu OpenMP::OpenMP_CXX ${CMAKE_DL_LIBS})
