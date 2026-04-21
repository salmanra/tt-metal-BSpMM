cmake_minimum_required(VERSION 3.22...3.30)

add_subdirectory(sparse_common)
add_subdirectory(block_spgemm)
add_subdirectory(SC26_submission)

set(RAHMY_SRCS
    # SpGEMM entry points
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spgemm/src/test_spgemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spgemm/src/run_spgemm.cpp
    # SDDMM entry points
    ${CMAKE_CURRENT_SOURCE_DIR}/SC26_submission/block_sddmm/src/test_sddmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SC26_submission/block_sddmm/src/run_sddmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SC26_submission/block_sddmm/src/export_sddmm.cpp
)

if(ENABLE_TRACY)
    list(APPEND RAHMY_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/SC26_submission/block_sddmm/src/profile_sddmm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/SC26_submission/block_spmm/src/profile_block_spmm.cpp
    )
endif()

list(APPEND RAHMY_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/SC26_submission/block_spmm/src/export_block_spmm_to_csv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SC26_submission/block_spmm/src/run_block_spmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SC26_submission/block_spmm/src/test_block_spmm.cpp
)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/SpMV_common)

CREATE_PGM_EXAMPLES_EXE("${RAHMY_SRCS}" "rahmy") # output binaries to build/programming_examples/rahmy


foreach(target ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
    if(TARGET ${target})
        target_link_libraries(${target} PRIVATE spgemm_host_code sddmm_host_code sparse_common sc26_spmm_host_code)
    endif()
endforeach()

add_custom_target(rahmy DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
