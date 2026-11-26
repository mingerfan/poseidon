# Compile the pinned kernels under a private namespace. Other GPU operators keep
# their existing ABI and the benchmark and runtime use the same source archive.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(bench "${POSEIDON_INCLUDES_DIR}/bench/bootstrap_agent22")
set(snapshot "${bench}/vendor/upstream-be4bc9f5.tar.gz")
set(overlay "${bench}/vendor/local.patch")
set(inputs "${snapshot}" "${overlay}" "${CMAKE_CURRENT_SOURCE_DIR}/native.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/prepare.py")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${inputs})
set(generated "${CMAKE_CURRENT_BINARY_DIR}/include")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/prepare.py"
    "${snapshot}" "${overlay}" "${CMAKE_CURRENT_SOURCE_DIR}/native.patch" "${generated}"
    RESULT_VARIABLE status)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "Failed to prepare the private S2C-first GPU sources")
endif()
set(src "${generated}/poseidon/s2c_gpu")
file(GLOB kernels "${src}/*.cpp" "${src}/*.cu" "${src}/kernels/*.cu")
add_library(poseidon_s2c_boot STATIC ${kernels} native_boot.cpp)
target_include_directories(poseidon_s2c_boot PRIVATE "${generated}" "${bench}" "${bench}/../resnet20_s2c_first" "${POSEIDON_INCLUDES_DIR}/src")
target_compile_features(poseidon_s2c_boot PUBLIC cxx_std_17)
target_compile_options(poseidon_s2c_boot PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_link_libraries(poseidon_s2c_boot PRIVATE poseidon::gpu)
