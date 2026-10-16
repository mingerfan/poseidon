get_filename_component(r18 "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(native "${r18}/runtime/native_snapshot")
set(app "${native}/benchmark/resnet18_gpu")
set(common "${native}/benchmark/resnet50_gpu")
set(fixture "${native}/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt")

function(resnet18_options target runtime_dir)
  target_link_libraries(${target} PRIVATE poseidon::gpu ${dsl_cupti} ${CMAKE_DL_LIBS} z)
  target_include_directories(${target} PRIVATE "${runtime_dir}" "${app}" "${common}"
    "${r18}/runtime" "${r18}/runtime/include")
  target_compile_definitions(${target} PRIVATE POSEIDON_GPU_RESNET18_SOURCE_DIR="${app}"
    POSEIDON_GPU_RESNET50_SOURCE_DIR="${common}" POSEIDON_RESNET20_RELU_FIXTURE="${fixture}")
  target_compile_options(${target} PRIVATE -O1 -UNDEBUG -ffunction-sections -fdata-sections)
  target_link_options(${target} PRIVATE -Wl,--gc-sections)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

set(model_sources "${app}/gpu_resnet18_inference.cpp" "${app}/resnet18_topology.cpp"
  "${app}/resnet18_weights.cpp" "${app}/gpu_config.cpp")
add_executable(resnet18_native "${r18}/runtime/main_native18_logits.cpp" ${model_sources}
  "${common}/gpu_ckks_runtime.cpp" "${common}/gpu_multiplexed_tensor.cpp"
  "${common}/gpu_relu_s2c_first.cpp" "${common}/resnet50_config.cpp" "${common}/gpu_activity_timing.cpp")
resnet18_options(resnet18_native "${common}")

add_executable(resnet18_dsl "${r18}/runtime/main_imagenet_dsl.cpp"
  "${r18}/runtime/dsl/gpu_ckks_runtime.cpp" "${common}/resnet50_config.cpp" "${common}/gpu_activity_timing.cpp")
resnet18_options(resnet18_dsl "${r18}/runtime/dsl")

set(shared "${r18}/runtime/same_cipher")
add_executable(resnet18_same_cipher "${r18}/runtime/main_shared_cipher18.cpp" ${model_sources}
  "${shared}/gpu_ckks_runtime.cpp" "${shared}/gpu_multiplexed_tensor.cpp"
  "${shared}/gpu_relu_s2c_first.cpp" "${shared}/resnet50_config.cpp" "${shared}/gpu_activity_timing.cpp")
resnet18_options(resnet18_same_cipher "${shared}")
