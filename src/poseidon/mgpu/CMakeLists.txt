add_library(poseidon_mgpu STATIC
    comm/execution_preflight.cpp
    comm/gpu_comm.cpp
    comm/gpu_object_copy.cpp
    comm/materialized_gpu_comm.cpp
    comm/topology.cpp
    compiler/copy_insertion.cpp
    compiler/dacapo_artifacts.cpp
    compiler/dacapo_adapter.cpp
    compiler/dacapo_constants.cpp
    compiler/schedule_verifier.cpp
    compiler/static_placement.cpp
    compiler/static_schedule_config.cpp
    compiler/static_schedule_pipeline.cpp
    ir/schedule.cpp
    ir/schedule_json.cpp
    ir/schedule_summary.cpp
    runtime/comm_schedule_handler.cpp
    runtime/gpu_execution_preflight.cpp
    runtime/hevm_artifact_report.cpp
    runtime/hevm_artifact_readiness.cpp
    runtime/hevm_io_binding.cpp
    runtime/hevm_plaintext_encoding.cpp
    runtime/hevm_static_execution_plan.cpp
    runtime/io_binding_handler.cpp
    runtime/object_store.cpp
    runtime/poseidon_gpu_schedule_preflight.cpp
    runtime/schedule_interpreter.cpp
    runtime/static_schedule_executor.cpp
)

target_include_directories(poseidon_mgpu
    PUBLIC
        ${POSEIDON_INCLUDES_DIR}/src
)

target_link_libraries(poseidon_mgpu
    PUBLIC
        poseidon_shared
)

option(POSEIDON_BUILD_MGPU_TOOLS "Build multi-GPU static schedule command-line tools" OFF)
message(STATUS "POSEIDON_BUILD_MGPU_TOOLS: ${POSEIDON_BUILD_MGPU_TOOLS}")

if(POSEIDON_BUILD_MGPU_TOOLS)
    add_executable(poseidon_mgpu_dacapo_hevm_dump
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/mgpu/tools/dacapo_hevm_dump.cpp
    )

    target_link_libraries(poseidon_mgpu_dacapo_hevm_dump
        PRIVATE
            poseidon_mgpu
    )
endif()

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

    add_executable(poseidon_mgpu_topology_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_topology_test.cpp
    )

    add_executable(poseidon_mgpu_communication_execution_preflight_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_communication_execution_preflight_test.cpp
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

    add_executable(poseidon_mgpu_dacapo_artifact_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_dacapo_artifact_test.cpp
    )

    add_executable(poseidon_mgpu_external_hevm_artifact_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_external_hevm_artifact_test.cpp
    )

    add_executable(poseidon_mgpu_dacapo_constants_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_dacapo_constants_test.cpp
    )

    add_executable(poseidon_mgpu_runtime_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_runtime_test.cpp
    )

    add_executable(poseidon_mgpu_io_binding_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_io_binding_test.cpp
    )

    add_executable(poseidon_mgpu_hevm_io_binding_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_hevm_io_binding_test.cpp
    )

    add_executable(poseidon_mgpu_hevm_plaintext_encoding_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_hevm_plaintext_encoding_test.cpp
    )

    add_executable(poseidon_mgpu_hevm_static_execution_plan_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_hevm_static_execution_plan_test.cpp
    )

    add_executable(poseidon_mgpu_hevm_artifact_readiness_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_hevm_artifact_readiness_test.cpp
    )

    add_executable(poseidon_mgpu_hevm_artifact_report_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_hevm_artifact_report_test.cpp
    )

    add_executable(poseidon_mgpu_gpu_execution_preflight_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_gpu_execution_preflight_test.cpp
    )

    add_executable(poseidon_mgpu_poseidon_gpu_preflight_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_poseidon_gpu_preflight_test.cpp
    )

    add_executable(poseidon_mgpu_static_placement_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_static_placement_test.cpp
    )

    add_executable(poseidon_mgpu_static_schedule_config_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_static_schedule_config_test.cpp
    )

    add_executable(poseidon_mgpu_static_pipeline_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_static_pipeline_test.cpp
    )

    add_executable(poseidon_mgpu_static_pipeline_execution_tests
        ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_static_pipeline_execution_test.cpp
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

    target_link_libraries(poseidon_mgpu_topology_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_communication_execution_preflight_tests
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

    target_link_libraries(poseidon_mgpu_dacapo_artifact_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_external_hevm_artifact_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_dacapo_constants_tests
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

    target_link_libraries(poseidon_mgpu_hevm_io_binding_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_hevm_plaintext_encoding_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_hevm_static_execution_plan_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_hevm_artifact_readiness_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_hevm_artifact_report_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_gpu_execution_preflight_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_poseidon_gpu_preflight_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_static_placement_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_static_schedule_config_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_static_pipeline_tests
        PRIVATE
            poseidon_mgpu
    )

    target_link_libraries(poseidon_mgpu_static_pipeline_execution_tests
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
        NAME poseidon_mgpu_topology_tests
        COMMAND poseidon_mgpu_topology_tests
    )

    add_test(
        NAME poseidon_mgpu_communication_execution_preflight_tests
        COMMAND poseidon_mgpu_communication_execution_preflight_tests
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
        NAME poseidon_mgpu_dacapo_artifact_tests
        COMMAND poseidon_mgpu_dacapo_artifact_tests
    )

    add_test(
        NAME poseidon_mgpu_external_hevm_artifact_tests
        COMMAND poseidon_mgpu_external_hevm_artifact_tests
    )
    set_tests_properties(poseidon_mgpu_external_hevm_artifact_tests
        PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME poseidon_mgpu_external_hevm_mock_artifact_tests
        COMMAND poseidon_mgpu_external_hevm_artifact_tests
    )
    set(POSEIDON_MGPU_EXTERNAL_HEVM_MOCK_ENV
        "POSEIDON_MGPU_EXTERNAL_MOCK_ARTIFACT=1"
        "POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT=4"
        "POSEIDON_MGPU_EXTERNAL_UPLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_COMPUTE_DEVICES=2"
        "POSEIDON_MGPU_EXTERNAL_DOWNLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_COMM_AVAILABLE=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_RELIN_KEYS=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_GALOIS_KEYS=1"
        "POSEIDON_MGPU_EXTERNAL_NODES=2"
        "POSEIDON_MGPU_EXTERNAL_DEVICES_PER_NODE=2"
        "POSEIDON_MGPU_EXTERNAL_EXECUTION_CUDA_PEER_AVAILABLE=1"
    )
    set_tests_properties(poseidon_mgpu_external_hevm_mock_artifact_tests
        PROPERTIES
            ENVIRONMENT "${POSEIDON_MGPU_EXTERNAL_HEVM_MOCK_ENV}")

    add_test(
        NAME poseidon_mgpu_external_hevm_rich_mock_artifact_tests
        COMMAND poseidon_mgpu_external_hevm_artifact_tests
    )
    set(POSEIDON_MGPU_EXTERNAL_HEVM_RICH_MOCK_ENV
        "POSEIDON_MGPU_EXTERNAL_RICH_MOCK_ARTIFACT=1"
        "POSEIDON_MGPU_EXTERNAL_REQUIRE_READY=1"
        "POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT=4"
        "POSEIDON_MGPU_EXTERNAL_UPLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_COMPUTE_DEVICES=1,2"
        "POSEIDON_MGPU_EXTERNAL_DOWNLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_COMM_AVAILABLE=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_RELIN_KEYS=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_GALOIS_KEYS=1"
        "POSEIDON_MGPU_EXTERNAL_EXECUTION_CUDA_PEER_AVAILABLE=1"
    )
    set_tests_properties(poseidon_mgpu_external_hevm_rich_mock_artifact_tests
        PROPERTIES
            ENVIRONMENT "${POSEIDON_MGPU_EXTERNAL_HEVM_RICH_MOCK_ENV}")

    add_test(
        NAME poseidon_mgpu_external_hevm_config_mock_artifact_tests
        COMMAND poseidon_mgpu_external_hevm_artifact_tests
    )
    set(POSEIDON_MGPU_EXTERNAL_HEVM_CONFIG_MOCK_JSON
        "{"
        "\"version\":1,"
        "\"device_count\":4,"
        "\"placement\":{"
        "\"default_device\":0,"
        "\"upload_device\":0,"
        "\"compute_devices\":[1,2],"
        "\"download_device\":0"
        "},"
        "\"topology\":{"
        "\"nodes\":2,"
        "\"devices_per_node\":2"
        "},"
        "\"preflight\":{"
        "\"require_ready\":true,"
        "\"comm_available\":true,"
        "\"relin_keys\":true,"
        "\"galois_keys\":true"
        "},"
        "\"execution_backends\":{"
        "\"same_device\":true,"
        "\"cuda_peer\":true,"
        "\"inter_node\":true"
        "}"
        "}")
    string(CONCAT POSEIDON_MGPU_EXTERNAL_HEVM_CONFIG_MOCK_JSON
        ${POSEIDON_MGPU_EXTERNAL_HEVM_CONFIG_MOCK_JSON})
    set(POSEIDON_MGPU_EXTERNAL_HEVM_CONFIG_MOCK_ENV
        "POSEIDON_MGPU_EXTERNAL_RICH_MOCK_ARTIFACT=1"
        "POSEIDON_MGPU_EXTERNAL_CONFIG_JSON=${POSEIDON_MGPU_EXTERNAL_HEVM_CONFIG_MOCK_JSON}"
    )
    set_tests_properties(poseidon_mgpu_external_hevm_config_mock_artifact_tests
        PROPERTIES
            ENVIRONMENT "${POSEIDON_MGPU_EXTERNAL_HEVM_CONFIG_MOCK_ENV}")

    add_test(
        NAME poseidon_mgpu_external_hevm_mock_artifact_require_ready_tests
        COMMAND poseidon_mgpu_external_hevm_artifact_tests
    )
    set(POSEIDON_MGPU_EXTERNAL_HEVM_MOCK_REQUIRE_READY_ENV
        "POSEIDON_MGPU_EXTERNAL_MOCK_ARTIFACT=1"
        "POSEIDON_MGPU_EXTERNAL_REQUIRE_READY=1"
        "POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT=2"
        "POSEIDON_MGPU_EXTERNAL_UPLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_COMPUTE_DEVICES=1"
        "POSEIDON_MGPU_EXTERNAL_DOWNLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_COMM_AVAILABLE=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_RELIN_KEYS=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_GALOIS_KEYS=1"
        "POSEIDON_MGPU_EXTERNAL_EXECUTION_CUDA_PEER_AVAILABLE=1"
    )
    set_tests_properties(poseidon_mgpu_external_hevm_mock_artifact_require_ready_tests
        PROPERTIES
            ENVIRONMENT "${POSEIDON_MGPU_EXTERNAL_HEVM_MOCK_REQUIRE_READY_ENV}")

    add_test(
        NAME poseidon_mgpu_external_hevm_mock_artifact_require_ready_missing_comm_tests
        COMMAND poseidon_mgpu_external_hevm_artifact_tests
    )
    set(POSEIDON_MGPU_EXTERNAL_HEVM_MOCK_REQUIRE_READY_MISSING_COMM_ENV
        "POSEIDON_MGPU_EXTERNAL_MOCK_ARTIFACT=1"
        "POSEIDON_MGPU_EXTERNAL_REQUIRE_READY=1"
        "POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT=2"
        "POSEIDON_MGPU_EXTERNAL_UPLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_COMPUTE_DEVICES=1"
        "POSEIDON_MGPU_EXTERNAL_DOWNLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_COMM_AVAILABLE=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_RELIN_KEYS=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_GALOIS_KEYS=1"
    )
    set_tests_properties(
        poseidon_mgpu_external_hevm_mock_artifact_require_ready_missing_comm_tests
        PROPERTIES
            ENVIRONMENT "${POSEIDON_MGPU_EXTERNAL_HEVM_MOCK_REQUIRE_READY_MISSING_COMM_ENV}"
            WILL_FAIL TRUE)

    add_test(
        NAME poseidon_mgpu_external_hevm_report_mock_artifact_tests
        COMMAND poseidon_mgpu_external_hevm_artifact_tests
    )
    set(POSEIDON_MGPU_EXTERNAL_HEVM_REPORT_MOCK_ENV
        "POSEIDON_MGPU_EXTERNAL_MOCK_ARTIFACT=1"
        "POSEIDON_MGPU_EXTERNAL_DEVICE_COUNT=2"
        "POSEIDON_MGPU_EXTERNAL_UPLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_COMPUTE_DEVICES=1"
        "POSEIDON_MGPU_EXTERNAL_DOWNLOAD_DEVICE=0"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_COMM_AVAILABLE=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_RELIN_KEYS=1"
        "POSEIDON_MGPU_EXTERNAL_PREFLIGHT_GALOIS_KEYS=1"
        "POSEIDON_MGPU_EXTERNAL_EXECUTION_CUDA_PEER_AVAILABLE=1"
        "POSEIDON_MGPU_EXTERNAL_REPORT_JSON=${CMAKE_CURRENT_BINARY_DIR}/external_hevm_report_mock.json"
    )
    set_tests_properties(poseidon_mgpu_external_hevm_report_mock_artifact_tests
        PROPERTIES
            ENVIRONMENT "${POSEIDON_MGPU_EXTERNAL_HEVM_REPORT_MOCK_ENV}")

    add_test(
        NAME poseidon_mgpu_dacapo_constants_tests
        COMMAND poseidon_mgpu_dacapo_constants_tests
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
        NAME poseidon_mgpu_hevm_io_binding_tests
        COMMAND poseidon_mgpu_hevm_io_binding_tests
    )

    add_test(
        NAME poseidon_mgpu_hevm_plaintext_encoding_tests
        COMMAND poseidon_mgpu_hevm_plaintext_encoding_tests
    )

    add_test(
        NAME poseidon_mgpu_hevm_static_execution_plan_tests
        COMMAND poseidon_mgpu_hevm_static_execution_plan_tests
    )

    add_test(
        NAME poseidon_mgpu_hevm_artifact_readiness_tests
        COMMAND poseidon_mgpu_hevm_artifact_readiness_tests
    )

    add_test(
        NAME poseidon_mgpu_hevm_artifact_report_tests
        COMMAND poseidon_mgpu_hevm_artifact_report_tests
    )

    add_test(
        NAME poseidon_mgpu_gpu_execution_preflight_tests
        COMMAND poseidon_mgpu_gpu_execution_preflight_tests
    )

    add_test(
        NAME poseidon_mgpu_poseidon_gpu_preflight_tests
        COMMAND poseidon_mgpu_poseidon_gpu_preflight_tests
    )

    add_test(
        NAME poseidon_mgpu_static_placement_tests
        COMMAND poseidon_mgpu_static_placement_tests
    )

    add_test(
        NAME poseidon_mgpu_static_schedule_config_tests
        COMMAND poseidon_mgpu_static_schedule_config_tests
    )

    add_test(
        NAME poseidon_mgpu_static_pipeline_tests
        COMMAND poseidon_mgpu_static_pipeline_tests
    )

    add_test(
        NAME poseidon_mgpu_static_pipeline_execution_tests
        COMMAND poseidon_mgpu_static_pipeline_execution_tests
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

    if(POSEIDON_BUILD_MGPU_TESTS)
        add_executable(poseidon_mgpu_gpu_object_materializer_tests
            ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_gpu_object_materializer_test.cpp
        )

        target_link_libraries(poseidon_mgpu_gpu_object_materializer_tests
            PRIVATE
                poseidon_mgpu_gpu_objects
        )

        add_test(
            NAME poseidon_mgpu_gpu_object_materializer_tests
            COMMAND poseidon_mgpu_gpu_object_materializer_tests
        )
        set_tests_properties(poseidon_mgpu_gpu_object_materializer_tests
            PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

if(POSEIDON_BUILD_MGPU_GPU_RUNTIME)
    add_library(poseidon_mgpu_gpu_runtime STATIC
        runtime/poseidon_gpu_hevm_executor.cpp
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
        comm/cuda_peer_probe.cpp
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

    if(POSEIDON_BUILD_MGPU_TOOLS)
        add_executable(poseidon_mgpu_cuda_peer_probe
            ${POSEIDON_INCLUDES_DIR}/src/poseidon/mgpu/tools/cuda_peer_probe.cpp
        )

        target_link_libraries(poseidon_mgpu_cuda_peer_probe
            PRIVATE
                poseidon_mgpu_cuda_comm
        )
    endif()

    if(POSEIDON_BUILD_MGPU_TESTS)
        add_executable(poseidon_mgpu_cuda_comm_tests
            ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_cuda_comm_test.cpp
        )

        add_executable(poseidon_mgpu_cuda_peer_probe_tests
            ${POSEIDON_INCLUDES_DIR}/src/poseidon/tests/mgpu/mgpu_cuda_peer_probe_test.cpp
        )

        target_link_libraries(poseidon_mgpu_cuda_comm_tests
            PRIVATE
                poseidon_mgpu_cuda_comm
        )

        target_link_libraries(poseidon_mgpu_cuda_peer_probe_tests
            PRIVATE
                poseidon_mgpu_cuda_comm
        )

        add_test(
            NAME poseidon_mgpu_cuda_comm_tests
            COMMAND poseidon_mgpu_cuda_comm_tests
        )
        set_tests_properties(poseidon_mgpu_cuda_comm_tests
            PROPERTIES SKIP_RETURN_CODE 77)

        add_test(
            NAME poseidon_mgpu_cuda_peer_probe_tests
            COMMAND poseidon_mgpu_cuda_peer_probe_tests
        )
        set_tests_properties(poseidon_mgpu_cuda_peer_probe_tests
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
