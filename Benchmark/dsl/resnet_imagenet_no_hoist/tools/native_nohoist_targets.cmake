# Include only after the validated Poseidon GPU target has been configured.
if(NOT TARGET poseidon::gpu)
  message(FATAL_ERROR "Native ImageNet targets require the existing Poseidon GPU core")
endif()
set(shared "${POSEIDON_IMAGENET_NOHOIST_SOURCE}/benchmark/resnet50_gpu")
add_library(imagenet_native_nohoist_common STATIC
  "${shared}/gpu_ckks_runtime.cpp" "${shared}/gpu_multiplexed_tensor.cpp"
  "${shared}/gpu_relu_s2c_first.cpp" "${shared}/resnet50_config.cpp"
  "${shared}/gpu_activity_timing.cpp")
target_link_libraries(imagenet_native_nohoist_common PUBLIC poseidon::gpu ${dsl_cupti} ${CMAKE_DL_LIBS})
target_include_directories(imagenet_native_nohoist_common PUBLIC "${shared}")
target_compile_definitions(imagenet_native_nohoist_common PRIVATE
  POSEIDON_RESNET20_RELU_FIXTURE="${POSEIDON_IMAGENET_REFERENCE_ROOT}/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt")
target_compile_options(imagenet_native_nohoist_common PRIVATE -O1 -UNDEBUG -ffunction-sections -fdata-sections)
foreach(model IN ITEMS 18 50)
  set(app "${POSEIDON_IMAGENET_NOHOIST_SOURCE}/benchmark/resnet${model}_gpu")
  set(sources "${app}/main.cpp" "${app}/gpu_resnet${model}_inference.cpp"
      "${app}/resnet${model}_topology.cpp" "${app}/resnet${model}_weights.cpp")
  if(model EQUAL 18)
    list(APPEND sources "${app}/gpu_config.cpp")
  endif()
  add_executable(resnet${model}_native_nohoist ${sources})
  target_link_libraries(resnet${model}_native_nohoist PRIVATE imagenet_native_nohoist_common)
  target_include_directories(resnet${model}_native_nohoist PRIVATE "${app}" "${shared}")
  target_compile_definitions(resnet${model}_native_nohoist PRIVATE
    POSEIDON_GPU_RESNET18_SOURCE_DIR="${POSEIDON_IMAGENET_REFERENCE_ROOT}/benchmark/resnet18_gpu"
    POSEIDON_GPU_RESNET50_SOURCE_DIR="${POSEIDON_IMAGENET_REFERENCE_ROOT}/benchmark/resnet50_gpu"
    POSEIDON_RESNET20_RELU_FIXTURE="${POSEIDON_IMAGENET_REFERENCE_ROOT}/Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt")
  target_compile_options(resnet${model}_native_nohoist PRIVATE -O1 -UNDEBUG -ffunction-sections -fdata-sections)
  target_link_options(resnet${model}_native_nohoist PRIVATE -Wl,--gc-sections)
  set_target_properties(resnet${model}_native_nohoist PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()
add_executable(imagenet_nohoist_bootstrap_probe "${CMAKE_CURRENT_LIST_DIR}/main_native_nohoist_bootstrap_probe.cpp")
target_link_libraries(imagenet_nohoist_bootstrap_probe PRIVATE imagenet_native_nohoist_common)
target_compile_options(imagenet_nohoist_bootstrap_probe PRIVATE -O1 -UNDEBUG)
set_target_properties(imagenet_nohoist_bootstrap_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
