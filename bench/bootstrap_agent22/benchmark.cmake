find_package(CUDAToolkit REQUIRED)
if(NOT TARGET CUDA::cupti)
    message(FATAL_ERROR "The Agent22 benchmark requires CUDA CUPTI")
endif()

set(agent22_activity_dir "${POSEIDON_AGENT22_BENCH_DIR}/../resnet20_s2c_first")
target_include_directories(poseidon_gpu PRIVATE "${POSEIDON_AGENT22_BENCH_DIR}")

add_executable(agent22_bootstrap
    "${POSEIDON_AGENT22_BENCH_DIR}/bootstrap.cpp"
    "${agent22_activity_dir}/gpu_activity_timing.cpp")
target_include_directories(agent22_bootstrap PRIVATE
    "${agent22_activity_dir}"
    "${POSEIDON_INCLUDES_DIR}/bench/resnet20_s2c_first")
target_link_libraries(agent22_bootstrap PRIVATE poseidon_gpu CUDA::cupti)
