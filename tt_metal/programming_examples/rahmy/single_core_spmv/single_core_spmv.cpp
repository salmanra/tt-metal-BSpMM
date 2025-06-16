#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/test_tiles.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tilize_untilize.hpp>
#include <tt-metalium/device_impl.hpp>
// #include <matmul_common/bmm_op.hpp>

#include <random>  // TODO: does TT have their own rand library?

// extern "C"
// {
// #include "../SpMV_common/input.h"
// #include "../SpMV_common/mmio.h"
// #include "../SpMV_common/config.h"
// }
#include "../SpMV_common/bmm_op.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

// Naive SpMV on COO Matrix

struct coo_matrix {
    std::vector<uint32_t> rows;
    std::vector<uint32_t> cols;
    std::vector<bfloat16> vals;
    int M;    // Number of rows
    int N;    // Number of columns
    int nnz;  // Number of non-zero elements
};

void create_random_coo_matrix(coo_matrix& coo, int M, int N, float density) {
    coo.M = M;
    coo.N = N;
    int nnz = 0;
    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> dis2(0.0, 100.0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (dis(gen) < density) {
                coo.rows.push_back((uint32_t)i);
                coo.cols.push_back((uint32_t)j);
                coo.vals.push_back(bfloat16((float)dis2(gen)));
                nnz++;
            }
        }
    }
    coo.nnz = nnz;
}

struct tilized_coo_matrix {
    std::vector<bfloat16> A_vals;
    std::vector<bfloat16> x_vals;
    std::vector<uint32_t> rows;
    uint32_t M;
    uint32_t N;
    uint32_t nnz;
};

void print_tilized_coo_matrix(tilized_coo_matrix& coo) {
    std::cout << "-- row -- A_val -- x_val --" << std::endl;
    for (int i = 0; i < coo.nnz; i++) {
        uint32_t row = coo.rows[i];
        float A_val = coo.A_vals[i].to_float();
        float x_val = coo.x_vals[i].to_float();

        if (i % 128 == 0) {
            std::cout << "-- " << row << " -- " << A_val << " <--> " << x_val << std::endl;
        }
    }
}

// void coo_to_matvec_page(coo_matrix& coo, std::vector<coo_matvec_page>& tilized_coo, std::vector<bfloat16>&
// input_vector) {
//     int tile_size = TILE_WIDTH * TILE_HEIGHT;
//     int num_tiles = (coo.nnz + tile_size - 1) / tile_size;

//     for (int t = 0; t < num_tiles; t++) {
//         coo_matvec_page page;
//         int start_idx = t * tile_size;
//         int end_idx = std::min(start_idx + tile_size, coo.nnz);

//         for (int i = start_idx; i < end_idx; i++) {
//             int idx = i - start_idx;
//             page.A_vals[idx] = coo.vals[i];
//             page.x_vals[idx] = input_vector[coo.cols[i]];
//             page.row_indices[idx] = coo.rows[i];
//         }

//         tilized_coo.push_back(page);
//     }
// }

void tilize_coo_matrix(coo_matrix& coo, tilized_coo_matrix& tilized_coo, std::vector<bfloat16>& input_vector) {
    // pseudo reduction
    // What we're talking about is a more comprehensive tiliziation which emits intermediate values directly ready
    //  to be reduced on the device (but not completely: some rows will have multiple values after the device reduction,
    //  hence I call it a "pseudo reduction")
    // This will require reintroducing zeros into the tiles for any row with anything other than 32 elements. ... very
    // unsatisfying. This is what the APIs give us! What else is there? ...

    // sorted coo
    // for every elt
    // find the ongoing tile row for that elt
    // if none, start a new tile row and add it there
    // else add it to the existing tile row, updating the count of elts in that tile row
    // add corresponding x elt to corresponding x tile row, starting a new row if necessary
    //
    // for each tile row
    //   store the output row index of that tile row in a Rows vector
    //
    // send elts and x elts to device
    // receive output from device
    // complete the reduction based on the Rows vector on the result from the device

    // what data structures are these three DSs?
    // vectors? hasmaps? multimaps?
    // vectors. A and X go straight to the device in 1-to-1 tiles.
    // When the result is sent back to the host, Rows will be here
    // to inform the last bits of the reduction. Nothing fancy here
    // except for the padding of zeros :(

    // Old tilized coo
    tilized_coo.x_vals.resize(coo.nnz);
    for (int i = 0; i < coo.nnz; i++) {
        tilized_coo.x_vals[i] = input_vector[coo.cols[i]];
    }
    tilized_coo.M = coo.M;
    tilized_coo.N = coo.N;
    tilized_coo.nnz = coo.nnz;
    tilized_coo.A_vals = std::move(coo.vals);
    tilized_coo.rows = std::move(coo.rows);
}

void sequential_spmv(coo_matrix& A, std::vector<bfloat16>& x, std::vector<bfloat16>& output) {
    float float_tmp;
    for (uint32_t i = 0; i < A.nnz; i++) {
        float_tmp = output[A.rows[i]].to_float() + A.vals[i].to_float() * x[A.cols[i]].to_float();
        output[A.rows[i]] = bfloat16(float_tmp);
    }
}

void spmv_single_core(
    tilized_coo_matrix& tilized_coo, std::vector<bfloat16>& output, uint32_t M, uint32_t N, IDevice* device) {
    CommandQueue& cq = device->command_queue();
    Program program{};
    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    uint32_t bfloat_single_tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    assert(bfloat_single_tile_size == 2 * 32 * 32);
    uint32_t num_bfloat_tiles = (tilized_coo.nnz + bfloat_single_tile_size - 1) / bfloat_single_tile_size;
    uint32_t dram_buffer_A_vals_size = bfloat_single_tile_size * num_bfloat_tiles;
    uint32_t dram_buffer_X_vals_size = bfloat_single_tile_size * num_bfloat_tiles;

    assert(dram_buffer_A_vals_size % bfloat_single_tile_size == 0);  // buffer size must be divisible by page size
    assert(dram_buffer_X_vals_size % bfloat_single_tile_size == 0);  // buffer size must be divisible by page size

    uint32_t uint_single_page_size = sizeof(uint32_t) * TILE_WIDTH * TILE_HEIGHT;
    uint32_t num_uint_pages = (tilized_coo.nnz + uint_single_page_size - 1) / uint_single_page_size;
    uint32_t dram_buffer_Rows_size = uint_single_page_size * num_uint_pages;

    assert(dram_buffer_Rows_size % uint_single_page_size == 0);  // buffer size must be divisible by page size

    tt_metal::InterleavedBufferConfig dram_config_A_vals{
        .device = device,
        .size = dram_buffer_A_vals_size,
        .page_size = bfloat_single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_X_vals{
        .device = device,
        .size = dram_buffer_X_vals_size,
        .page_size = bfloat_single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig dram_config_Rows{
        .device = device,
        .size = dram_buffer_Rows_size,
        .page_size = uint_single_page_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    // need the size to be a multiple of bfloat_single_tile_size
    // this should be the minimal multiple of bfloat_single_tile_size which is greater than or equal to M
    int num_bfloats = ((M + bfloat_single_tile_size - 1) / bfloat_single_tile_size) * bfloat_single_tile_size;
    tt_metal::InterleavedBufferConfig dram_y_out_config{
        .device = device,
        .size = sizeof(bfloat16) * num_bfloats,
        .page_size = bfloat_single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    assert((sizeof(bfloat16) * num_bfloats) % bfloat_single_tile_size == 0);

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config_A_vals);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config_X_vals);
    std::shared_ptr<tt::tt_metal::Buffer> src2_dram_buffer = CreateBuffer(dram_config_Rows);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_y_out_config);
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t src2_addr = src2_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    // Configure and create L1 circular buffers
    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_pages = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_pages * bfloat_single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, bfloat_single_tile_size);
    auto cb0_src = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_pages * bfloat_single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, bfloat_single_tile_size);
    auto cb1_src = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t src2_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_src2_config =
        CircularBufferConfig(num_input_pages * uint_single_page_size, {{src2_cb_index, cb_data_format}})
            .set_page_size(src2_cb_index, uint_single_page_size);
    auto cb2_src = tt_metal::CreateCircularBuffer(program, core, cb_src2_config);

    uint32_t inter_cb_index = CBIndex::c_3;
    CircularBufferConfig cb_inter_config =
        CircularBufferConfig(num_input_pages * bfloat_single_tile_size, {{inter_cb_index, cb_data_format}})
            .set_page_size(inter_cb_index, bfloat_single_tile_size);
    auto cb_inter = tt_metal::CreateCircularBuffer(program, core, cb_inter_config);

    uint32_t dst_cb_index = CBIndex::c_16;
    uint32_t num_output_pages = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_pages * bfloat_single_tile_size, {{dst_cb_index, cb_data_format}})
            .set_page_size(dst_cb_index, bfloat_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Compile time arguments
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src2_is_dram = src2_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)src0_is_dram, (uint32_t)src1_is_dram, (uint32_t)src2_is_dram};
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/single_core_spmv/kernels/dataflow/reader_spmv_naive.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/single_core_spmv/kernels/dataflow/writer_spmv_naive.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    std::vector<uint32_t> compute_args = {};
    auto spmv_single_core_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/rahmy/single_core_spmv/kernels/compute/spmv_naive.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args});

    // Runtime arguments
    tt_metal::SetRuntimeArgs(program, reader_id, core, {src0_addr, src1_addr, src2_addr, M, N, tilized_coo.nnz});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, M, N, tilized_coo.nnz});
    tt_metal::SetRuntimeArgs(program, spmv_single_core_kernel_id, core, {M, N, tilized_coo.nnz});

    // Launch program

    // TODO: rework the reduction
    std::vector<bfloat16> intermediate_output(tilized_coo.nnz, 0);

    tt::log_info(tt::LogTest, "Enqueueing DRAM writing");
    EnqueueWriteBuffer(cq, src0_dram_buffer, tilized_coo.A_vals.data(), true);
    EnqueueWriteBuffer(cq, src1_dram_buffer, tilized_coo.x_vals.data(), true);
    EnqueueWriteBuffer(cq, src2_dram_buffer, tilized_coo.rows.data(), true);
    tt::log_info(tt::LogTest, "Enqueueing program");
    EnqueueProgram(cq, program, true);
    tt::log_info(tt::LogTest, "Enqueueing DRAM reading");
    // TODO: why does this hang when nonblocking is set to true?
    EnqueueReadBuffer(cq, dst_dram_buffer, intermediate_output.data(), false);
    tt::log_info(tt::LogTest, "All done-ish");

    // print_golden_metalium_vectors(intermediate_output, intermediate_output);
    // perform reduction:
    float float_tmp;
    for (int i = 0; i < tilized_coo.nnz; i++) {
        float_tmp = output[tilized_coo.rows[i]].to_float() + intermediate_output[i].to_float();
        output[tilized_coo.rows[i]] = bfloat16(float_tmp);
    }
    tt::log_info(tt::LogTest, "Reduction complete");
}

int main(int argc, char** argv) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    try {
        // char *mm_filename = nullptr;
        // if (argc == 1){
        //     printf("Give a MatrixMarket file.\n");
        //     return -1;
        // }
        // else
        //     mm_filename = argv[1];

        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        coo_matrix coo;
        int M = 64;
        int N = 64;
        float density = 0.1;
        create_random_coo_matrix(coo, M, N, density);
        // read_coo_matrix(coo, mm_filename);

        constexpr uint32_t single_tile_size = 2 * 1024;
        uint32_t x_vec_DRAM_buffer_size = single_tile_size * coo.N;  // num_tiles of FP16_B

        // std::vector<bfloat16> x_vec(coo.N, 1.0); // Example input vector
        std::vector<bfloat16> x_vec = create_random_vector_of_bfloat16_native(x_vec_DRAM_buffer_size, 1, 12522);
        std::vector<bfloat16> golden_vec(coo.M, 0);
        sequential_spmv(coo, x_vec, golden_vec);

        tilized_coo_matrix tilized_coo;
        tilize_coo_matrix(coo, tilized_coo, x_vec);

        // print_tilized_coo_matrix(tilized_coo);

        constexpr int num_elts_per_tile = 32 * 32;
        int num_tiles = (tilized_coo.nnz + num_elts_per_tile - 1) / num_elts_per_tile;
        tt::log_info(tt::LogTest, "NNZ: {}, num_tiles: {}", std::to_string(tilized_coo.nnz), num_tiles);

        std::vector<bfloat16> result_vec(coo.M, 0);
        spmv_single_core(tilized_coo, result_vec, coo.M, coo.N, device);

        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        print_golden_metalium_vectors(golden_vec, result_vec);
        log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
        // TT_FATAL(pearson > 0.97, "PCC not high enough. Result PCC: {}, Expected PCC: 0.97", pearson);

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());
        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
