add_library(poseidon_mgpu STATIC
    comm/gpu_comm.cpp
    comm/gpu_object_copy.cpp
    comm/materialized_gpu_comm.cpp
    compiler/copy_insertion.cpp
    compiler/dacapo_adapter.cpp
    compiler/schedule_verifier.cpp
    compiler/static_placement.cpp
    compiler/static_schedule_pipeline.cpp
    ir/schedule.cpp
    ir/schedule_json.cpp
    runtime/comm_schedule_handler.cpp
    runtime/io_binding_handler.cpp
    runtime/object_store.cpp
    runtime/schedule_interpreter.cpp
)

target_include_directories(poseidon_mgpu
    PUBLIC
        ${POSEIDON_INCLUDES_DIR}/src
)

target_link_libraries(poseidon_mgpu
    PUBLIC
        poseidon_shared
)

if(POSEIDON_BUILD_MGPU_TESTS)
    add_executable(poseidon_mgpu_ir_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_ir_test.cpp
    )

    add_executable(poseidon_mgpu_schedule_json_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_schedule_json_test.cpp
    )

    add_executable(poseidon_mgpu_comm_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_comm_test.cpp
    )

    add_executable(poseidon_mgpu_copy_insertion_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_copy_insertion_test.cpp
    )

    add_executable(poseidon_mgpu_object_copy_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_object_copy_test.cpp
    )

    add_executable(poseidon_mgpu_materialized_comm_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_materialized_comm_test.cpp
    )

    add_executable(poseidon_mgpu_dacapo_adapter_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_dacapo_adapter_test.cpp
    )

    add_executable(poseidon_mgpu_runtime_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_runtime_test.cpp
    )

    add_executable(poseidon_mgpu_io_binding_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_io_binding_test.cpp
    )

    add_executable(poseidon_mgpu_static_placement_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_static_placement_test.cpp
    )

    add_executable(poseidon_mgpu_static_pipeline_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_static_pipeline_test.cpp
    )

    target_link_libraries(poseidon_mgpu_ir_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_schedule_json_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_comm_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_copy_insertion_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_object_copy_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_materialized_comm_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_dacapo_adapter_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_runtime_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_io_binding_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_static_placement_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_static_pipeline_tests
        PRIVATE
            poseidon_mgpu
    )

    add_test(
        NAME poseidon_mgpu_ir_tests
        COMMAND poseidon_mgpu_ir_tests
    )

    add_test(
        NAME poseidon_mgpu_schedule_json_tests
        COMMAND poseidon_mgpu_schedule_json_tests
    )

    add_test(
        NAME poseidon_mgpu_comm_tests
        COMMAND poseidon_mgpu_comm_tests
    )

    add_test(
        NAME poseidon_mgpu_copy_insertion_tests
        COMMAND poseidon_mgpu_copy_insertion_tests
    )

    add_test(
        NAME poseidon_mgpu_object_copy_tests
        COMMAND poseidon_mgpu_object_copy_tests
    )

    add_test(
        NAME poseidon_mgpu_materialized_comm_tests
        COMMAND poseidon_mgpu_materialized_comm_tests
    )

    add_test(
        NAME poseidon_mgpu_dacapo_adapter_tests
        COMMAND poseidon_mgpu_dacapo_adapter_tests
    )

    add_test(
        NAME poseidon_mgpu_runtime_tests
        COMMAND poseidon_mgpu_runtime_tests
    )

    add_test(
        NAME poseidon_mgpu_io_binding_tests
        COMMAND poseidon_mgpu_io_binding_tests
    )

    add_test(
        NAME poseidon_mgpu_static_placement_tests
        COMMAND poseidon_mgpu_static_placement_tests
    )

    add_test(
        NAME poseidon_mgpu_static_pipeline_tests
        COMMAND poseidon_mgpu_static_pipeline_tests
    )
endif()

option(POSEIDON_BUILD_MGPU_GPU_RUNTIME "Build optional Poseidon GPU schedule handler for multi-GPU runtime" OFF)
message(STATUS "POSEIDON_BUILD_MGPU_GPU_RUNTIME: ${POSEIDON_BUILD_MGPU_GPU_RUNTIME}")

option(POSEIDON_BUILD_MGPU_GPU_OBJECTS "Build optional GPU object materializer for multi-GPU runtime" OFF)
message(STATUS "POSEIDON_BUILD_MGPU_GPU_OBJECTS: ${POSEIDON_BUILD_MGPU_GPU_OBJECTS}")

set(POSEIDON_MGPU_REQUIRES_RMM FALSE)
if(POSEIDON_BUILD_MGPU_GPU_OBJECTS OR POSEIDON_BUILD_MGPU_GPU_RUNTIME)
    set(POSEIDON_MGPU_REQUIRES_RMM TRUE)
endif()

if(POSEIDON_MGPU_REQUIRES_RMM)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING
            "CUDA architectures for optional multi-GPU CUDA targets" FORCE)
    endif()

    find_package(CUDAToolkit REQUIRED)

    set(POSEIDON_THIRD_PARTY_DIR "${POSEIDON_INCLUDES_DIR}/third_party" CACHE PATH
        "Path to bundled third-party dependencies")
    set(POSEIDON_BUNDLED_RMM_DIR "${POSEIDON_THIRD_PARTY_DIR}/rmm" CACHE PATH
        "Path to bundled RMM source")

    if(EXISTS "${POSEIDON_BUNDLED_RMM_DIR}/CMakeLists.txt")
        set(CPM_SOURCE_CACHE "${POSEIDON_THIRD_PARTY_DIR}/cpm" CACHE PATH
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
        if(NOT TARGET rmm::rmm)
            add_subdirectory(
                "${POSEIDON_BUNDLED_RMM_DIR}"
                "${CMAKE_BINARY_DIR}/third_party/rmm"
                EXCLUDE_FROM_ALL)
        endif()
    else()
        message(FATAL_ERROR
            "Bundled RMM was not found at ${POSEIDON_BUNDLED_RMM_DIR}")
    endif()
endif()

if(POSEIDON_BUILD_MGPU_GPU_OBJECTS)
    add_library(poseidon_mgpu_gpu_objects STATIC
        comm/gpu_object_materializer.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_ciphertext.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_plaintext.cpp
    )

    target_include_directories(poseidon_mgpu_gpu_objects
        PUBLIC
            ${POSEIDON_INCLUDES_DIR}/src
    )

    target_link_libraries(poseidon_mgpu_gpu_objects
        PUBLIC
            poseidon_mgpu
            CUDA::cudart
            rmm::rmm
    )
endif()

if(POSEIDON_BUILD_MGPU_GPU_RUNTIME)
    add_library(poseidon_mgpu_gpu_runtime STATIC
        runtime/poseidon_gpu_schedule_handler.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_ciphertext.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_key.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_parameter.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_plaintext.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_uploader.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_evaluator.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_elementwise_handler.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_keyswitch_handler.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_modswitch_handler.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_ntt_handler.cpp
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/gpu_tensor_core_gemm.cu
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/kernels/gpu_elementwise_kernels.cu
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/kernels/gpu_keyswitch_kernels.cu
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/kernels/gpu_ntt_kernels.cu
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/gpu/kernels/gpu_rescale_kernels.cu
    )

    target_include_directories(poseidon_mgpu_gpu_runtime
        PUBLIC
            ${POSEIDON_INCLUDES_DIR}/src
    )

    target_link_libraries(poseidon_mgpu_gpu_runtime
        PUBLIC
            poseidon_mgpu
            CUDA::cudart
            rmm::rmm
    )

    target_compile_options(poseidon_mgpu_gpu_runtime
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    if(POSEIDON_BUILD_MGPU_TESTS)
        add_executable(poseidon_mgpu_gpu_runtime_smoke_tests
            ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_gpu_runtime_smoke_test.cpp
        )

        target_link_libraries(poseidon_mgpu_gpu_runtime_smoke_tests
            PRIVATE
                poseidon_mgpu_gpu_runtime
        )

        add_test(
            NAME poseidon_mgpu_gpu_runtime_smoke_tests
            COMMAND poseidon_mgpu_gpu_runtime_smoke_tests
        )
        set_tests_properties(poseidon_mgpu_gpu_runtime_smoke_tests
            PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

option(POSEIDON_BUILD_MGPU_CUDA_COMM "Build optional CUDA peer communication backend" OFF)
message(STATUS "POSEIDON_BUILD_MGPU_CUDA_COMM: ${POSEIDON_BUILD_MGPU_CUDA_COMM}")

if(POSEIDON_BUILD_MGPU_CUDA_COMM)
    find_package(CUDAToolkit REQUIRED)

    add_library(poseidon_mgpu_cuda_comm STATIC
        comm/cuda_peer_comm.cpp
    )

    target_include_directories(poseidon_mgpu_cuda_comm
        PUBLIC
            ${POSEIDON_INCLUDES_DIR}/src
    )

    target_link_libraries(poseidon_mgpu_cuda_comm
        PUBLIC
            poseidon_mgpu
            CUDA::cudart
    )

    if(POSEIDON_BUILD_MGPU_TESTS)
        add_executable(poseidon_mgpu_cuda_comm_tests
            ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_cuda_comm_test.cpp
        )

        target_link_libraries(poseidon_mgpu_cuda_comm_tests
            PRIVATE
                poseidon_mgpu_cuda_comm
        )

        add_test(
            NAME poseidon_mgpu_cuda_comm_tests
            COMMAND poseidon_mgpu_cuda_comm_tests
        )
        set_tests_properties(poseidon_mgpu_cuda_comm_tests
            PROPERTIES SKIP_RETURN_CODE 77)
    endif()

    if(POSEIDON_BUILD_MGPU_GPU_RUNTIME AND POSEIDON_BUILD_MGPU_GPU_OBJECTS AND POSEIDON_BUILD_MGPU_TESTS)
        add_executable(poseidon_mgpu_gpu_runtime_copy_smoke_tests
            ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_gpu_runtime_copy_smoke_test.cpp
        )

        target_link_libraries(poseidon_mgpu_gpu_runtime_copy_smoke_tests
            PRIVATE
                poseidon_mgpu_cuda_comm
                poseidon_mgpu_gpu_objects
                poseidon_mgpu_gpu_runtime
        )

        add_test(
            NAME poseidon_mgpu_gpu_runtime_copy_smoke_tests
            COMMAND poseidon_mgpu_gpu_runtime_copy_smoke_tests
        )
        set_tests_properties(poseidon_mgpu_gpu_runtime_copy_smoke_tests
            PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()
