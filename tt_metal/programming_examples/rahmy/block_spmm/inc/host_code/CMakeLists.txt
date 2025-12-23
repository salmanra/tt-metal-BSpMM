cmake_minimum_required(VERSION 3.22...3.30)

file(GLOB HOST_CODE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(spmm_host_code STATIC ${HOST_CODE_SOURCES})

target_include_directories(spmm_host_code PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(spmm_host_code PUBLIC tt_metal)
