set(RAHMY_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/test_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/profile_block.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/export_to_csv.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/profile_dense.cpp
)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/SpMV_common)

CREATE_PGM_EXAMPLES_EXE("${RAHMY_SRCS}" "rahmy") # output binaries to build/programming_examples/rahmy

add_custom_target(rahmy DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
