cmake_minimum_required(VERSION 3.22...3.30)

add_subdirectory(block_spmm)


set(RAHMY_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/test_block.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/profile_block.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/export_to_csv.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/block_spmm/src/profile_dense.cpp
)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/SpMV_common)

CREATE_PGM_EXAMPLES_EXE("${RAHMY_SRCS}" "rahmy") # output binaries to build/programming_examples/rahmy


foreach(target ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
    if(TARGET ${target})
        target_link_libraries(${target} PRIVATE spmm_host_code)
    endif()
endforeach()

add_custom_target(rahmy DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})