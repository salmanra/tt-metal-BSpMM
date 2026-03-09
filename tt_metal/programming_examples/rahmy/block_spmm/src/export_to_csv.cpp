#include <common/TracyColor.hpp>
#include <cstdio>
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/profiling_suite.hpp"
#include "../inc/host_code.hpp"
#include "../inc/host_code/spmm_zone_config.hpp"

#include <system_error>
#include <tracy/Tracy.hpp>
#include "hostdevcommon/profiler_common.h"

#include <cstdlib> // required to start ./capture-release listening

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_host_code;
using namespace profiling_suite;

void export_to_csv(int host_code_num, int test_num, ProfileCaseFunctionPtr *Registry, std::string registry_name){
    // get the host code and test case
    HostCodeFunctionPtr host_function = HostCodeRegistryProfiling[host_code_num].first;
    std::string host_function_name = HostCodeRegistryProfiling[host_code_num].second;
    auto [a, b, test_name] = Registry[test_num]();

    auto zone_defines = spmm_zone_config::get_zone_defines();
    
    std::string disabled_zones = zone_defines.empty() ? "" : "_Disable_";
    for (auto it = zone_defines.begin(); it != zone_defines.end(); it++){
        std::string zone_name = it->first;
        disabled_zones += "_" + zone_name;
    }


    std::string test_file_name = test_name + disabled_zones;

    // set up command strings to direct and capture the trace (and its csv file)
    char buf[1000];
    size_t n = sprintf(buf, "/home/user/tt-metal/profiles_bad_noc_full_profiling_suite/bsr/%s/%s/", registry_name.c_str(), host_function_name.c_str());
    std::string trace_directory(buf, n);
    std::string trace_file_location = trace_directory + test_file_name + ".tracy";

    n = sprintf(buf, "/home/user/tt-metal/profiles_bad_noc_full_profiling_suite/csvs/%s/%s/", registry_name.c_str(), host_function_name.c_str());
    std::string csv_directory(buf);
    std::string csv_file_location = csv_directory + test_file_name + ".csv";

    n = sprintf(buf, "mkdir -p %s", csv_directory.c_str());
    std::string csv_mkdir_command(buf, n);

    n = sprintf(buf, "./csvexport-release %s > %s", trace_file_location.c_str(), csv_file_location.c_str());
    std::string csvexport_command(buf);
  
    std::string device_csv_file_location = csv_directory + test_file_name + ".device.csv";
    n = sprintf(buf, "./tracy-csvexport --gpu %s > %s", trace_file_location.c_str(), device_csv_file_location.c_str());
    std::string device_csvexport_command(buf);

    std::system(csv_mkdir_command.c_str());
    std::system(csvexport_command.c_str());
    std::system(device_csvexport_command.c_str());

    // create two output ostreams to two new files in the same dir as the CSV file,
    //  of the same name as the csv file, append {_sparse, _dense} and swap the extension to .log 
    // pipe the output of a.pretty_print() to the sparse file
    // pipe the output of b.pretty_print() to the sparse file
    std::string sparse_log_file = csv_directory + test_file_name + "_sparse.log";
    std::ofstream os_sparse(sparse_log_file);
    
    std::string dense_log_file = csv_directory + test_file_name + "_dense.log";
    std::ofstream os_dense(dense_log_file);

    a.pretty_print(os_sparse);
    b.pretty_print(os_dense);
    os_dense << "Block Size (R x C): " << a.R << " x " << a.C << std::endl;

    // Compute in1_block_w (dense block width in tiles) the same way the host codes do
    uint32_t Nt = b.W / TILE_WIDTH;
    uint32_t Rt = a.R / TILE_HEIGHT;
    uint32_t Ct = a.C / TILE_WIDTH;

    // Indexing buffer tile counts (mirrors host code sizing)
    tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
    uint32_t indexing_tile_size = detail::TileSize(indexing_data_format);
    uint32_t indptr_buf_size = sizeof(int) * a.indptr.size();
    indptr_buf_size = indexing_tile_size * ((indexing_tile_size - 1 + indptr_buf_size) / indexing_tile_size);
    uint32_t col_idx_buf_size = sizeof(int) * a.indices.size();
    col_idx_buf_size = indexing_tile_size * ((indexing_tile_size - 1 + col_idx_buf_size) / indexing_tile_size);
    uint32_t num_tiles_indexing = indptr_buf_size / indexing_tile_size + col_idx_buf_size / indexing_tile_size;

    // Count nonzero block-rows
    uint32_t nnz_rows = 0;
    for (uint32_t i = 0; i + 1 < a.indptr.size(); i++) {
        if (a.indptr[i + 1] - a.indptr[i] > 0) nnz_rows++;
    }

    // Wormhole compute grid: 8×8
    constexpr uint32_t num_cores_x = 8;
    constexpr uint32_t num_cores_y = 8;

    uint32_t in1_block_w = get_Npc_from_BSR_block_size(Nt, Rt, Ct, num_cores_x, num_cores_y, num_tiles_indexing, nnz_rows);
    os_dense << "Dense block width (in1_block_w): " << in1_block_w << " tiles"
             << " (" << in1_block_w * TILE_WIDTH << " columns)" << std::endl;
}

int main(int argc, char** argv) {

    const int num_host_programs = sizeof(HostCodeRegistryProfiling) / sizeof(HostCodeRegistryProfiling[0]);

    const int test_id = 0;
    const int host_code_id = 0;
     
    bool export_all_profiles = argc > 1 ? std::string(argv[1]) == "all" : true;
    bool export_all_host_codes = argc > 2 ? std::string(argv[2]) == "all" : true;
    int test_num = 0;
    int host_code_num = 0;
    // let's make the test registry and test index required arguments
    // and the host code index
    if (!export_all_profiles) 
        test_num = argc > 1 ? std::stoi(argv[1]) : test_id;
    if (!export_all_host_codes)
        host_code_num = argc > 2 ? std::stoi(argv[2]) : host_code_id;
    
    int registry_number = argc > 3 ? std::stoi(argv[3]) : 2;

    ProfileCaseFunctionPtr *Registry = nullptr;
    std::string registry_name = "";
    switch (registry_number) {
        case 0:
            Registry = ProfileCaseRegistry;
            registry_name = "ProfileSuiteSparseVersioning";
            break;
        case 1:
            Registry = ProfileDenseAblationRegistry;
            registry_name = "DenseAblationKProfileSuite";
            break;
        case 2:
            Registry = ProfileLargeSparseRegistry;
            registry_name = "ProfileSuiteLargeSparseVersioning";
            break;
        case 3:
            Registry = ProfileLargeSparseLargeBlocksRegistry;
            registry_name = "ProfileSuiteLargeSparseLargeBlocksVersioning";
            break;
        case 4:
            Registry = ProfileSweepNRegistry;
            registry_name = "ProfileSweepN";
            break;
        case 5:
            Registry = ProfileSweepDensityRegistry;
            registry_name = "ProfileSweepDensity";
            break;
        case 6:
            Registry = ProfileSweepKRegistry;
            registry_name = "ProfileSweepK";
            break;
        case 7:
            Registry = ProfileSweepBlockSizeRegistry;
            registry_name = "ProfileSweepBlockSize";
    }

    int num_profiles = sizeof(Registry) / sizeof(Registry[0]);
    if (export_all_profiles && !export_all_host_codes){
        for (int i = 0; i < num_profiles; i++){
            export_to_csv(host_code_num, i, Registry, registry_name);
        }
    }
    else if (export_all_profiles && export_all_host_codes){
        for (int i = 0; i < num_profiles; i++){
            for (int j = 0; j < num_host_programs; j++){
                export_to_csv(j, i, Registry, registry_name);
            }
        }
    }
    else {
        export_to_csv(host_code_num, test_num, Registry, registry_name);
    }

    return 0;
}
