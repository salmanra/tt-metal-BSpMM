#include <cstdio>
#include <cstdint>
#include <string>
#include <fstream>
#include <filesystem>
#include "../inc/include_me.hpp"
#include "../inc/profiling_suite.hpp"
#include "../inc/host_code.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_host_code;
using namespace profiling_suite;

namespace fs = std::filesystem;

void write_logs(int test_num, ProfileCaseFunctionPtr *Registry, std::string registry_name) {
    auto [a, b, test_name] = Registry[test_num]();

    // Walk all host-code subdirectories under the csvs registry dir
    char buf[1000];
    sprintf(buf, "/home/user/tt-metal/profiles_bad_noc_full_profiling_suite/csvs/%s/", registry_name.c_str());
    std::string csv_registry_dir(buf);

    if (!fs::exists(csv_registry_dir)) {
        printf("  Registry dir not found: %s\n", csv_registry_dir.c_str());
        return;
    }

    for (auto& host_entry : fs::directory_iterator(csv_registry_dir)) {
        if (!host_entry.is_directory()) continue;
        std::string host_dir = host_entry.path().string();

        // Check if a csv for this test case exists in this host-code dir
        std::string csv_path = host_dir + "/" + test_name + ".csv";
        if (!fs::exists(csv_path)) continue;

        // Write sparse log
        std::string sparse_log = host_dir + "/" + test_name + "_sparse.log";
        std::ofstream os_sparse(sparse_log);
        a.pretty_print(os_sparse);
        os_sparse.close();

        // Write dense log
        std::string dense_log = host_dir + "/" + test_name + "_dense.log";
        std::ofstream os_dense(dense_log);
        b.pretty_print(os_dense);
        os_dense << "Block Size (R x C): " << a.R << " x " << a.C << std::endl;

        // Compute in1_block_w
        uint32_t Nt = b.W / TILE_WIDTH;
        uint32_t Rt = a.R / TILE_HEIGHT;
        uint32_t Ct = a.C / TILE_WIDTH;

        tt::DataFormat indexing_data_format = tt::DataFormat::Int32;
        uint32_t indexing_tile_size = detail::TileSize(indexing_data_format);
        uint32_t indptr_buf_size = sizeof(int) * a.indptr.size();
        indptr_buf_size = indexing_tile_size * ((indexing_tile_size - 1 + indptr_buf_size) / indexing_tile_size);
        uint32_t col_idx_buf_size = sizeof(int) * a.indices.size();
        col_idx_buf_size = indexing_tile_size * ((indexing_tile_size - 1 + col_idx_buf_size) / indexing_tile_size);
        uint32_t num_tiles_indexing = indptr_buf_size / indexing_tile_size + col_idx_buf_size / indexing_tile_size;

        uint32_t nnz_rows = 0;
        for (uint32_t i = 0; i + 1 < a.indptr.size(); i++) {
            if (a.indptr[i + 1] - a.indptr[i] > 0) nnz_rows++;
        }

        constexpr uint32_t num_cores_x = 8;
        constexpr uint32_t num_cores_y = 8;

        uint32_t in1_block_w = get_Npc_from_BSR_block_size(Nt, Rt, Ct, num_cores_x, num_cores_y, num_tiles_indexing, nnz_rows);
        os_dense << "Dense block width (in1_block_w): " << in1_block_w << " tiles"
                 << " (" << in1_block_w * TILE_WIDTH << " columns)" << std::endl;
        os_dense.close();

        printf("  wrote %s\n", dense_log.c_str());
    }
}

struct RegistryEntry {
    ProfileCaseFunctionPtr* registry;
    std::string name;
    int num_cases;
};

int main(int argc, char** argv) {
    int registry_number = argc > 1 ? std::stoi(argv[1]) : -1;

    RegistryEntry entries[NUM_REGISTRIES];
    for (int i = 0; i < NUM_REGISTRIES; i++) {
        entries[i] = {Registries[i], RegistryNames[i], -1};
    }

    // Detect number of test cases per registry by scanning existing csv dirs
    for (int r = 0; r < NUM_REGISTRIES; r++) {
        char buf[512];
        sprintf(buf, "/home/user/tt-metal/profiles_bad_noc_full_profiling_suite/csvs/%s/", entries[r].name.c_str());
        std::string csv_reg_dir(buf);
        if (!fs::exists(csv_reg_dir)) {
            entries[r].num_cases = 0;
            continue;
        }
        // Find first host-code subdir and count unique test case csv files
        int count = 0;
        for (auto& host_entry : fs::directory_iterator(csv_reg_dir)) {
            if (!host_entry.is_directory()) continue;
            for (auto& f : fs::directory_iterator(host_entry.path())) {
                std::string fname = f.path().filename().string();
                if (fname.ends_with(".csv") && !fname.ends_with(".device.csv")) {
                    count++;
                }
            }
            break; // only need first host-code dir
        }
        entries[r].num_cases = count;
    }

    int start = 0, end = NUM_REGISTRIES;
    if (registry_number >= 0 && registry_number < NUM_REGISTRIES) {
        start = registry_number;
        end = registry_number + 1;
    }

    for (int r = start; r < end; r++) {
        if (entries[r].num_cases <= 0) {
            printf("Skipping registry %d (%s): no data found\n", r, entries[r].name.c_str());
            continue;
        }
        printf("Registry %d: %s (%d test cases)\n", r, entries[r].name.c_str(), entries[r].num_cases);
        for (int t = 0; t < entries[r].num_cases; t++) {
            write_logs(t, entries[r].registry, entries[r].name);
        }
    }

    printf("Done.\n");
    return 0;
}
