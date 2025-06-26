set(RAHMY_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_compute_kernel_remix/hello_world_compute_kernel_remix.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/single_core_spmv/single_core_spmv.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_compute_rahmy/add_2_integers_in_compute_rahmy.cpp
)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/SpMV_common)

CREATE_PGM_EXAMPLES_EXE("${RAHMY_SRCS}" "rahmy") # output binaries to build/programming_examples/rahmy

add_custom_target(rahmy DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
