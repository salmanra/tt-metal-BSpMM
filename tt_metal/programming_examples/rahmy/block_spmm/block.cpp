


#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/test_tiles.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <matmul_common/bmm_op.hpp>
#include <tt-metalium/tilize_untilize.hpp>

#include "bsr_matrix.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

void golden_bsr_spmm(bsr_matrix<bfloat16>& bsr, dense_matrix<bfloat16>& B, dense_matrix<bfloat16>& output) {
    output = bsr.spmm(B);
}

void bsr_spmm_multicore_reuse() {

}

int main(int argc, char** argv) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    try {
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);


        // matmul params setup

        // TODO: Commence basic testing on bfloat16's
        bsr_matrix<float> bsr(2048, 2048, 128, 128, 3, RAND);
        dense_matrix<float> dense(2048, 16, RAND);

        bsr_matrix<bfloat16> bsr_bfloat16 = bsr.bfloat16_cast();
        dense_matrix<bfloat16> dense_bfloat16 = dense.bfloat16_cast();
        
        dense_matrix<bfloat16> expected = bsr_bfloat16.to_dense().gemm_bfloat16(dense_bfloat16);
        dense_matrix<bfloat16> result = bsr_bfloat16.tiled_spmm(dense_bfloat16);

        float pearson = check_bfloat16_vector_pcc(expected.data, result.data);
        log_info(tt::LogVerif, "BSR vs Golden -- PCC = {}", pearson);
        TT_FATAL(pearson > 0.99, "PCC not high enough. Result PCC: {}, Expected PCC: 0.99", pearson);

        // create (or read) source data

        // run golden bsr_spmm

        // tilize input data

        // run bsr_spmm_multicore_reuse

        // untile output data

        // check all close

        // close device and check for errors
        pass &= CloseDevice(device);

    }
    catch (const std::exception& e) {
        tt::log_error(tt::LogTest, "Test failed with excpetion!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    }
    else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}