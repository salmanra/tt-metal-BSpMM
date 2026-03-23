add_executable(metal_example_matmul_single_core_profile_matrix_engine)
target_sources(metal_example_matmul_single_core_profile_matrix_engine PRIVATE profile_matrix_engine_single_core.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()

target_link_libraries(
    metal_example_matmul_single_core_profile_matrix_engine
    PRIVATE
        TT::Metalium
        Matmul::Common
)
