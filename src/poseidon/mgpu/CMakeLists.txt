add_library(poseidon_mgpu INTERFACE)

target_include_directories(poseidon_mgpu
    INTERFACE
        ${POSEIDON_INCLUDES_DIR}/src
)

target_link_libraries(poseidon_mgpu
    INTERFACE
        poseidon_shared
)
