# =============================================================================
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" _rapids_version)
if(_rapids_version MATCHES [[^([0-9][0-9])\.([0-9][0-9])\.([0-9][0-9])]])
  set(RAPIDS_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(RAPIDS_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(RAPIDS_VERSION_PATCH "${CMAKE_MATCH_3}")
  set(RAPIDS_VERSION_MAJOR_MINOR "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}")
  set(RAPIDS_VERSION "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}.${RAPIDS_VERSION_PATCH}")
else()
  string(REPLACE "\n" "\n  " _rapids_version_formatted "  ${_rapids_version}")
  message(
    FATAL_ERROR
      "Could not determine RAPIDS version. Contents of VERSION file:\n${_rapids_version_formatted}")
endif()

set(_poseidon_rapids_cmake_dir "${CMAKE_CURRENT_LIST_DIR}/../rapids-cmake")
if(EXISTS "${_poseidon_rapids_cmake_dir}/rapids-cmake/rapids-cmake.cmake")
  set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_LIST_DIR}/../cpm" CACHE PATH
      "Bundled CPM cache for offline RAPIDS/RMM configuration" FORCE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_0.40.0.cmake" CACHE FILEPATH
      "Bundled CPM.cmake file for offline RAPIDS/RMM configuration" FORCE)
  set(rapids-cmake-dir "${_poseidon_rapids_cmake_dir}/rapids-cmake" CACHE PATH
      "Bundled rapids-cmake module directory" FORCE)
  if(NOT "${rapids-cmake-dir}" IN_LIST CMAKE_MODULE_PATH)
    list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
  endif()
else()
  message(
    FATAL_ERROR
      "Bundled rapids-cmake was not found at ${_poseidon_rapids_cmake_dir}. "
      "Offline builds require third_party/rapids-cmake next to third_party/rmm.")
endif()
unset(_poseidon_rapids_cmake_dir)
