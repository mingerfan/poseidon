# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Use CPM to find or clone CCCL
function(find_and_configure_cccl)

  if(TARGET CCCL::CCCL)
    return()
  endif()

  set(_poseidon_cccl_dir "${CMAKE_CURRENT_LIST_DIR}/../../../cccl")
  if(EXISTS "${_poseidon_cccl_dir}/libcudacxx/include" AND EXISTS "${_poseidon_cccl_dir}/cub")
    add_library(poseidon_cccl INTERFACE)
    add_library(CCCL::CCCL ALIAS poseidon_cccl)
    add_library(CCCL::CUB ALIAS poseidon_cccl)
    add_library(CCCL::libcudacxx ALIAS poseidon_cccl)
    add_library(CCCL::Thrust ALIAS poseidon_cccl)
    add_library(libcudacxx::libcudacxx ALIAS poseidon_cccl)
    target_include_directories(
      poseidon_cccl
      INTERFACE "${_poseidon_cccl_dir}/libcudacxx/include" "${_poseidon_cccl_dir}/cub"
                "${_poseidon_cccl_dir}/thrust")
    target_compile_definitions(
      poseidon_cccl INTERFACE THRUST_DISABLE_ABI_NAMESPACE THRUST_IGNORE_ABI_NAMESPACE_ERROR)
    set(CCCL_SOURCE_DIR "${_poseidon_cccl_dir}" PARENT_SCOPE)
    set(CCCL_ADDED TRUE PARENT_SCOPE)
    return()
  endif()

  message(
    FATAL_ERROR
      "Bundled CCCL was not found at ${_poseidon_cccl_dir}. "
      "Offline builds require third_party/cccl.")

endfunction()

find_and_configure_cccl()
