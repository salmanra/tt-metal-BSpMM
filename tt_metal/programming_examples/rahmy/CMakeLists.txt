cmake_minimum_required(VERSION 3.22...3.30)

# Remaining rahmy-level content: block_spgemm (not part of the paper artifact).
# block_spgemm used to share sparse_common with SC26; now the paper's submodule
# (Tenstorrent-WH-BlockSpMM) registers sparse_common as a global CMake target.
# Since programming_examples/CMakeLists.txt adds Tenstorrent-WH-BlockSpMM
# before this directory, the sparse_common target is available here.
add_subdirectory(block_spgemm)

set(RAHMY_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spgemm/src/test_spgemm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/block_spgemm/src/run_spgemm.cpp
)

CREATE_PGM_EXAMPLES_EXE("${RAHMY_SRCS}" "rahmy_spgemm")

foreach(target ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
    if(TARGET ${target})
        target_link_libraries(
            ${target}
            PRIVATE
                spgemm_host_code
                sparse_common
        )
    endif()
endforeach()

add_custom_target(rahmy_spgemm DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
