find_package(CUDAToolkit REQUIRED)

set(POSEIDON_BUNDLED_RMM_DIR "${POSEIDON_INCLUDES_DIR}/third_party/rmm" CACHE PATH
    "Path to the bundled RMM source")
if(NOT EXISTS "${POSEIDON_BUNDLED_RMM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Bundled RMM was not found at ${POSEIDON_BUNDLED_RMM_DIR}")
endif()

foreach(_rmm_thirdparty_file IN ITEMS get_spdlog.cmake get_cccl.cmake get_nvtx.cmake)
    set(_rmm_thirdparty_path
        "${POSEIDON_BUNDLED_RMM_DIR}/cmake/thirdparty/${_rmm_thirdparty_file}")
    if(NOT EXISTS "${_rmm_thirdparty_path}")
        message(FATAL_ERROR
            "Bundled RMM is incomplete: ${_rmm_thirdparty_path} is missing. "
            "Recreate the offline bundle with scripts/package_offline_gpu_runtime.sh.")
    endif()
endforeach()
unset(_rmm_thirdparty_file)
unset(_rmm_thirdparty_path)

if(NOT TARGET rmm::rmm)
    set(CPM_SOURCE_CACHE "${POSEIDON_INCLUDES_DIR}/third_party/cpm" CACHE PATH
        "Bundled CPM cache for offline RAPIDS/RMM configuration" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(RMM_NVTX OFF CACHE BOOL "" FORCE)
    set(USE_NVTX OFF CACHE BOOL "" FORCE)
    set(RMM_LOGGING_LEVEL "OFF" CACHE STRING "" FORCE)
    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL_HO ON CACHE BOOL "" FORCE)
    add_subdirectory(
        "${POSEIDON_BUNDLED_RMM_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/rmm"
        EXCLUDE_FROM_ALL)
endif()

add_library(poseidon_gpu STATIC
    gpu_bootstrap_profile.cpp
    gpu_ciphertext.cpp
    gpu_double_hoist.cpp
    gpu_key.cpp
    gpu_memory.cpp
    gpu_parameter.cpp
    gpu_plaintext.cpp
    gpu_uploader.cpp
    gpu_evaluator.cpp
    gpu_elementwise_handler.cpp
    gpu_keyswitch_handler.cpp
    gpu_modswitch_handler.cpp
    gpu_ntt_handler.cpp
    gpu_tensor_core_gemm.cu
    kernels/gpu_double_hoist_kernels.cu
    kernels/gpu_elementwise_kernels.cu
    kernels/gpu_keyswitch_kernels.cu
    kernels/gpu_ntt_kernels.cu
    kernels/gpu_rescale_kernels.cu
)
add_library(poseidon::gpu ALIAS poseidon_gpu)

target_include_directories(poseidon_gpu
    PUBLIC
        ${POSEIDON_INCLUDES_DIR}/src
)
target_link_libraries(poseidon_gpu
    PUBLIC
        poseidon_shared
        CUDA::cudart
        rmm::rmm
)
target_compile_features(poseidon_gpu PUBLIC cxx_std_17)
target_compile_options(poseidon_gpu
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

option(POSEIDON_BUILD_GPU_ENCODE_BENCH "Build experimental CUDA/Tensor CKKS encoding benchmark" OFF)
if(POSEIDON_BUILD_GPU_ENCODE_BENCH)
    add_executable(poseidon_gpu_encode_bench
        ../tests/gpu_encode/gpu_encode_bench.cpp
        ../tests/gpu_encode/gpu_encode_experiment.cu)
    target_include_directories(poseidon_gpu_encode_bench PRIVATE
        ${POSEIDON_INCLUDES_DIR}/third_party/ckks-runtime/third_party/nlohmann)
    target_link_libraries(poseidon_gpu_encode_bench PRIVATE poseidon_gpu CUDA::cufft)
    target_compile_options(poseidon_gpu_encode_bench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set(POSEIDON_GPU_ENCODE_TILELANG_PYTHON "" CACHE FILEPATH
        "Optional Python with TileLang 0.1.14 to build experimental NTT kernels")
    set(POSEIDON_GPU_ENCODE_TILELANG_ARCH "89" CACHE STRING
        "CUDA architecture for experimental TileLang NTT kernels")
    if(POSEIDON_GPU_ENCODE_TILELANG_PYTHON)
        if(NOT POSEIDON_GPU_ENCODE_TILELANG_ARCH MATCHES "^[0-9]+$")
            message(FATAL_ERROR "TileLang architecture must be a single numeric SM version")
        endif()
        execute_process(COMMAND "${POSEIDON_GPU_ENCODE_TILELANG_PYTHON}" -c
            "import pathlib,tilelang; print(pathlib.Path(tilelang.__file__).parent)"
            RESULT_VARIABLE _tilelang_status OUTPUT_VARIABLE _tilelang_root
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT _tilelang_status EQUAL 0)
            message(FATAL_ERROR "Cannot import TileLang with POSEIDON_GPU_ENCODE_TILELANG_PYTHON")
        endif()
        set(_encode_dir "${CMAKE_CURRENT_SOURCE_DIR}/../tests/gpu_encode")
        set(_tilelang_generated "${CMAKE_CURRENT_BINARY_DIR}/tilelang-ntt-generated.cu")
        add_custom_command(OUTPUT "${_tilelang_generated}"
            COMMAND "${POSEIDON_GPU_ENCODE_TILELANG_PYTHON}"
                "${_encode_dir}/generate_tilelang_ntt.py"
                --out "${_tilelang_generated}" --arch "sm_${POSEIDON_GPU_ENCODE_TILELANG_ARCH}"
            DEPENDS "${_encode_dir}/generate_tilelang_ntt.py" "${_encode_dir}/tilelang_ntt.py"
            VERBATIM)
        # Compile separately: TileLang MMA uses the toolkit CUDA C++ headers;
        # Poseidon/RMM's vendored CCCL include paths must not reach this target.
        add_library(poseidon_encode_tilelang STATIC "${_tilelang_generated}")
        target_include_directories(poseidon_encode_tilelang PRIVATE
            "${_encode_dir}" "${_tilelang_root}/src" "${_tilelang_root}/3rdparty/cutlass/include")
        target_compile_features(poseidon_encode_tilelang PRIVATE cxx_std_17)
        target_compile_options(poseidon_encode_tilelang PRIVATE --default-stream=per-thread)
        set_target_properties(poseidon_encode_tilelang PROPERTIES
            CUDA_ARCHITECTURES "${POSEIDON_GPU_ENCODE_TILELANG_ARCH}")
        target_link_libraries(poseidon_gpu_encode_bench PRIVATE poseidon_encode_tilelang)
        target_compile_definitions(poseidon_gpu_encode_bench PRIVATE POSEIDON_ENCODE_TILELANG=1
            POSEIDON_ENCODE_TILELANG_SM=${POSEIDON_GPU_ENCODE_TILELANG_ARCH})
    endif()
    if(POSEIDON_BUILD_CKKS_RUNTIME_TESTS)
        add_test(NAME poseidon_gpu_encode_smoke
            COMMAND poseidon_gpu_encode_bench 8 2 40 0 1
                ${CMAKE_CURRENT_BINARY_DIR}/encode-smoke.json)
    endif()
endif()
