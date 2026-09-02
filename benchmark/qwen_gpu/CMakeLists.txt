cmake_minimum_required(VERSION 3.26.4)

project(PoseidonGpuQwen LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(POSEIDON_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE PATH
    "Path to the Poseidon source tree")
set(TRIDENT_ROOT "${POSEIDON_ROOT}/../Trident" CACHE PATH
    "Path to the Trident source tree")
set(QWEN_ROOT "${TRIDENT_ROOT}/qwen")
set(POSEIDON_GPU_SHARED_DIR "${CMAKE_CURRENT_LIST_DIR}/../resnet50_gpu")

if(NOT EXISTS "${QWEN_ROOT}/model/plain_qwen.cpp")
    message(FATAL_ERROR "Trident Qwen sources were not found at ${QWEN_ROOT}")
endif()

set(POSEIDON_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(POSEIDON_BUILD_DEPS OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZLIB OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_ZSTD OFF CACHE BOOL "" FORCE)
set(POSEIDON_USE_MSGSL OFF CACHE BOOL "" FORCE)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "" FORCE)
set(CPM_LOCAL_PACKAGES_ONLY ON CACHE BOOL "" FORCE)
set(CPM_DOWNLOAD_ALL OFF CACHE BOOL "" FORCE)

add_subdirectory("${POSEIDON_ROOT}" "${CMAKE_BINARY_DIR}/poseidon_build")
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

set(POSEIDON_THIRD_PARTY_DIR "${POSEIDON_ROOT}/third_party" CACHE PATH "")
set(POSEIDON_BUNDLED_RMM_DIR "${POSEIDON_THIRD_PARTY_DIR}/rmm" CACHE PATH "")
if(NOT EXISTS "${POSEIDON_BUNDLED_RMM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Bundled RMM was not found at ${POSEIDON_BUNDLED_RMM_DIR}")
endif()

set(CPM_SOURCE_CACHE "${POSEIDON_THIRD_PARTY_DIR}/cpm" CACHE PATH "" FORCE)
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
    "${CMAKE_BINARY_DIR}/third_party/rmm"
    EXCLUDE_FROM_ALL)

set(POSEIDON_GPU_SOURCES
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_ciphertext.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_key.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_parameter.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_plaintext.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_uploader.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_evaluator.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_double_hoist.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_elementwise_handler.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_keyswitch_handler.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_modswitch_handler.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_ntt_handler.cpp"
    "${POSEIDON_ROOT}/src/poseidon/gpu/gpu_tensor_core_gemm.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_elementwise_kernels.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_double_hoist_kernels.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_keyswitch_kernels.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_ntt_kernels.cu"
    "${POSEIDON_ROOT}/src/poseidon/gpu/kernels/gpu_rescale_kernels.cu")

set(QWEN_PLAIN_SOURCES
    "${QWEN_ROOT}/core/tensor.cpp"
    "${QWEN_ROOT}/io/safetensors.cpp"
    "${QWEN_ROOT}/model/qwen_config.cpp"
    "${QWEN_ROOT}/model/plain_attention.cpp"
    "${QWEN_ROOT}/model/plain_decoder.cpp"
    "${QWEN_ROOT}/model/plain_qwen.cpp"
    "${QWEN_ROOT}/ops/plain_ops.cpp")

set(QWEN_HE_REFERENCE_SOURCES
    "${QWEN_ROOT}/he/he_config.cpp"
    "${QWEN_ROOT}/he/he_runtime.cpp"
    "${QWEN_ROOT}/he/encrypted_tensor.cpp"
    "${QWEN_ROOT}/he/encrypted_ops.cpp"
    "${QWEN_ROOT}/he/approximation.cpp"
    "${QWEN_ROOT}/he/encrypted_attention.cpp"
    "${QWEN_ROOT}/he/comparison.cpp"
    "${QWEN_ROOT}/he/encrypted_decoder.cpp"
    "${QWEN_ROOT}/he/encrypted_model.cpp"
    "${QWEN_ROOT}/he/qwen25_05b_config.cpp"
    "${QWEN_ROOT}/he/validation_log.cpp"
    "${TRIDENT_ROOT}/resnet50/relu_approx.cpp")

add_executable(poseidon_gpu_qwen
    main.cpp
    qwen_gpu_tensor.cpp
    qwen_gpu_ops.cpp
    qwen_gpu_attention.cpp
    qwen_gpu_decoder.cpp
    "${POSEIDON_GPU_SHARED_DIR}/gpu_ckks_runtime.cpp"
    "${POSEIDON_GPU_SHARED_DIR}/resnet50_config.cpp"
    ${QWEN_PLAIN_SOURCES}
    ${QWEN_HE_REFERENCE_SOURCES}
    ${POSEIDON_GPU_SOURCES})

target_include_directories(poseidon_gpu_qwen PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${POSEIDON_GPU_SHARED_DIR}"
    "${QWEN_ROOT}"
    "${TRIDENT_ROOT}/resnet50"
    "${POSEIDON_ROOT}/src")
target_link_libraries(poseidon_gpu_qwen PRIVATE
    poseidon_shared CUDA::cudart rmm::rmm)
if(OpenMP_CXX_FOUND)
    target_link_libraries(poseidon_gpu_qwen PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(poseidon_gpu_qwen PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(poseidon_gpu_qwen PRIVATE
    POSEIDON_GPU_RESNET50_SOURCE_DIR="${POSEIDON_GPU_SHARED_DIR}"
    POSEIDON_GPU_QWEN_DEFAULT_MODEL_DIR="${QWEN_ROOT}/pretrained_parameters/Qwen2.5-0.5B")
set_target_properties(poseidon_gpu_qwen PROPERTIES CUDA_ARCHITECTURES 70)
