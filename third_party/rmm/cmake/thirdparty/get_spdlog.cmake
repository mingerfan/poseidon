# =============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

# Use CPM to find or clone speedlog
function(find_and_configure_spdlog)

  if(TARGET spdlog::spdlog_header_only)
    return()
  endif()

  set(_poseidon_fmt_dir "${CMAKE_CURRENT_LIST_DIR}/../../../fmt")
  set(_poseidon_spdlog_dir "${CMAKE_CURRENT_LIST_DIR}/../../../spdlog")

  if(NOT TARGET fmt::fmt-header-only)
    if(EXISTS "${_poseidon_fmt_dir}/CMakeLists.txt")
      set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
      add_subdirectory("${_poseidon_fmt_dir}" "${CMAKE_BINARY_DIR}/third_party/fmt" EXCLUDE_FROM_ALL)
    else()
      message(
        FATAL_ERROR
          "Bundled fmt was not found at ${_poseidon_fmt_dir}. "
          "Offline builds require third_party/fmt.")
    endif()
  endif()

  if(EXISTS "${_poseidon_spdlog_dir}/CMakeLists.txt")
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPDLOG_FMT_EXTERNAL_HO ON CACHE BOOL "" FORCE)
    add_subdirectory(
      "${_poseidon_spdlog_dir}" "${CMAKE_BINARY_DIR}/third_party/spdlog" EXCLUDE_FROM_ALL)
  else()
    message(
      FATAL_ERROR
        "Bundled spdlog was not found at ${_poseidon_spdlog_dir}. "
        "Offline builds require third_party/spdlog.")
  endif()

endfunction()

find_and_configure_spdlog()
