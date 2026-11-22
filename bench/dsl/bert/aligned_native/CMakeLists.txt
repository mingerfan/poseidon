cmake_minimum_required(VERSION 3.26.4)
project(BertAlignedNative LANGUAGES C CXX)
# Compatibility entry point. The maintained top-level build now includes all
# prepared engines without importing a private historical build by default.
set(BERT_BUILD_PREPARED ON CACHE BOOL "Enable prepared Linear/QK/AV")
if(NOT BERT_BUILD_PREPARED)
    message(FATAL_ERROR "aligned_native requires BERT_BUILD_PREPARED=ON")
endif()
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}/bert")
