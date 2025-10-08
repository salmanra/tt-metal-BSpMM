<<<<<<< HEAD
set(PROGRAMMING_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_compute/add_2_integers_in_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_binary/eltwise_binary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_sfpu/eltwise_sfpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_compute_kernel/hello_world_compute_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_datamovement_kernel/hello_world_datamovement_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_world_datatypes_kernel/hello_world_datatypes_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loopback/loopback.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multi_core/matmul_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse/matmul_multicore_reuse.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multicore_reuse_profiled/matmul_multicore_reuse_profiled.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul_single_core/matmul_single_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pad/pad_multi_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sharding/shard_data_rm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vecadd_sharding/vecadd_sharding.cpp
)
=======
# All programming example targets should link tt_stl
link_libraries(TT::STL)
>>>>>>> main

add_compile_definitions(OVERRIDE_KERNEL_PREFIX="tt_metal/programming_examples/")
include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/programming_examples")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/profiler)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/distributed)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/contributed)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/rahmy)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_compute)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/add_2_integers_in_riscv)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/eltwise_binary)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/eltwise_sfpu)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hello_world_compute_kernel)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hello_world_datamovement_kernel)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hello_world_datatypes_kernel)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/loopback)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/matmul)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pad_multi_core)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/shard_data_rm)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vecadd_sharding)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vecadd_multi_core)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/NoC_tile_transfer)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sfpu_eltwise_chain)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/custom_sfpi_add)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/custom_sfpi_smoothstep)

add_custom_target(
    programming_examples
    DEPENDS
        profiler_examples
<<<<<<< HEAD
        contributed
        rahmy
=======
>>>>>>> main
        ${PROGRAMMING_EXAMPLES_TEST_TARGETS}
)

install(
    DIRECTORY
        add_2_integers_in_compute
        add_2_integers_in_riscv
        eltwise_binary
        eltwise_sfpu
        hello_world_compute_kernel
        hello_world_datamovement_kernel
        hello_world_datatypes_kernel
        loopback
        matmul
        shard_data_rm
        NoC_tile_transfer
        sfpu_eltwise_chain
        custom_sfpi_add
        custom_sfpi_smoothstep
        tests
    # DESTINATION ${CMAKE_INSTALL_DOCDIR}/examples
    # FIXME(afuller): Something funky is happening when installing files into
    # /usr/share/doc on a default Docker image. Speculation: some dependency for
    # magic doc handling is missing. For now keep it out of 'doc'.
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tt-metalium/examples"
    COMPONENT metalium-examples
    REGEX
    "/kernels/"
        EXCLUDE
)

install(
    DIRECTORY
        add_2_integers_in_compute
        add_2_integers_in_riscv
        eltwise_binary
        eltwise_sfpu
        hello_world_compute_kernel
        hello_world_datamovement_kernel
        hello_world_datatypes_kernel
        loopback
        matmul
        shard_data_rm
        NoC_tile_transfer
        sfpu_eltwise_chain
        custom_sfpi_add
        custom_sfpi_smoothstep
    DESTINATION "${CMAKE_INSTALL_DATADIR}/tenstorrent/kernels"
    COMPONENT metalium-examples
    FILES_MATCHING
    REGEX
    "/kernels/"
)
