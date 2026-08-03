# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

# Use CPM to find or clone NVTX3
function(find_and_configure_nvtx3)

  if(TARGET nvtx3::nvtx3-cpp)
    return()
  endif()

  set(_poseidon_nvtx_dir "${CMAKE_CURRENT_LIST_DIR}/../../../nvtx")
  if(EXISTS "${_poseidon_nvtx_dir}/c/CMakeLists.txt")
    add_subdirectory("${_poseidon_nvtx_dir}/c" "${CMAKE_BINARY_DIR}/third_party/nvtx" EXCLUDE_FROM_ALL)
    if(TARGET nvtx3-c AND NOT TARGET nvtx3::nvtx3-c)
      add_library(nvtx3::nvtx3-c ALIAS nvtx3-c)
    endif()
    if(TARGET nvtx3-cpp AND NOT TARGET nvtx3::nvtx3-cpp)
      add_library(nvtx3::nvtx3-cpp ALIAS nvtx3-cpp)
    endif()
    return()
  endif()

  message(
    FATAL_ERROR
      "Bundled NVTX was not found at ${_poseidon_nvtx_dir}. "
      "Offline builds require third_party/nvtx.")

endfunction()

find_and_configure_nvtx3()
