
#include "include_me.hpp"
#include "test_suite.hpp"
#include "host_code.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_test_suite;
using namespace bsr_host_code;

// what if we made the registries arrays of pairs, funcptr:string?
// that way we wouldn't have to pass the strings around, just the index into the registry
namespace bsr_test_suite {
   using TestFunctionPtr = std::tuple<bsr_matrix<bfloat16>, dense_matrix<bfloat16>, std::string> (*)();

    static TestFunctionPtr TestRegistry[] = {
        test_basic, // 0
        test_2_blocks, // 1
        test_2_blocks_col, // 2
        test_2_blocks_col_simplified, // 3
        test_2_blocks_row_simplified, // 4
        test_2_blocks_nonsquare, // 5
        test_many_nonsquare, // 6
        test_nonsquare_diag_blocks, // 7
        test_nonsquare_tall, // 8
        test_2_blocks_nonsquare_tall, // 9
        test_nonsquare, // 10
        test_nonsquare_diag_tall, // 11
        test_nonsquare_stacked, // 12
        test_nonsquare_diag_first_row, // 13
        test_nonsquare_off_diag_first_row, // 14
        test_simplified_off_diag_first_row, // 15
        test_2_blocks_diag, // 16
        test_off_diag_first_row, // 17
        test_diag_first_row, // 18
        test_2_blocks_fill_col, // 19
        test_2_blocks_fill_row, // 20
        test_4_blocks, // 21
        test_diag_times_wide, // 22
        test_simplified_times_wide, // 23
        test_simplified_tall_times_wide, // 24
        test_simplified_tall_times_wide_v2, // 25
        test_big_block_times_wide, // 26
        test_diag_first_row_times_wide, // 27
        test_diag_second_row_times_wide, // 28
        test_big_diag, // 29
        test_checkerboard, // 30
        test_random, // 31
        test_random_wide, // 32
        test_random_tall, // 33
        test_random_tall_blocks, // 34
        test_random_wide_blocks, // 35
        test_big_zero_rows, // 36
        test_big_zero_rows_more, // 37
    };
}

using HostCodeFunctionPtr = void (*)(
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    dense_matrix<bfloat16>& output,
    bool bcast_batch,
    uint32_t nnz_blocks,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t R,
    uint32_t C,
    uint32_t B,
    IDevice* device,
    bool verbose); 

static std::pair<HostCodeFunctionPtr, std::string> HostCodeRegistry[] = {
    {bsr_spmm_multicore_reuse, "bsr_spmm_multicore_reuse"}, // 0
    {bsr_spmm_multicore_reuse_naive, "bsr_spmm_multicore_reuse_naive"}, // 1 
};

struct TestResult {
    std::string test_name;
    float pearson;
    bool all_close;
};

TestResult run_test(
    HostCodeFunctionPtr host_func,
    bsr_matrix<bfloat16>& a,
    dense_matrix<bfloat16>& b,
    std::string& test_name,
    bool verbose = false,
    bool emit_output = false) {

    /*
    Requires: a, b to be initialized on CPU
    Modifies: can modifiy output files and log data
    Effects:

    Returns the PCC between the sequential matmul of a and b and the multicore matmul of a and b.
    */


    // device setup
    constexpr int device_id = 0;
    IDevice* device = CreateDevice(device_id);


    // matmul params setup
    uint32_t M = a.H;
    uint32_t N = b.W;
    uint32_t K = a.W;
    // block params setup
    uint32_t R = a.R;
    uint32_t C = a.C;
    uint32_t nblocks = a.nblocks;
    uint32_t block_matrix_height = M / R;

    uint32_t Rt = R / TILE_HEIGHT;
    uint32_t Ct = C / TILE_WIDTH;


    // initialize output_data
    dense_matrix<float> tmp(M, N, 0.0f);
    dense_matrix<bfloat16> output = tmp.bfloat16_cast();

    // run sequential spmm
    dense_matrix<bfloat16> golden = a.spmm_bfloat16(b);

    // tilize input data
    tilize(a.data, R, C);
    tilize(b.data, K, N);

    // run bsr_spmm_multicore_reuse
    host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device, verbose);


    if (emit_output) {

        // it makes 1000x more sense to print the tilized result. That's what these are!... bruh moment
        // let's write the output vectors to a file
        std::string local_path = "/home/user/tt-metal/tt_metal/programming_examples/rahmy/block_spmm/" + test_name;

        std::filesystem::create_directory(local_path);
        std::string output_file = local_path + "/output.txt";
        std::ofstream out(output_file);
        if (!out.is_open()) {
            TT_THROW("Failed to open output file: {}", output_file);
        }
        for (size_t i = 0; i < output.data.size(); i++) {
            out << output.data[i].to_float() << "\n";
        }
        out.close();

        log_info(tt::LogVerif, "Output written to {}", output_file);
        // let's write the golden vector to a file
        std::string golden_file = local_path + "/golden.txt";
        std::ofstream golden_out(golden_file);
        if (!golden_out.is_open()) {
            TT_THROW("Failed to open golden file: {}", golden_file);
        }

        tilize(golden.data, M, N);
        for (size_t i = 0; i < golden.data.size(); i++) {
            golden_out << golden.data[i].to_float() << "\n";
        }
        untilize(golden.data, M, N);
        golden_out.close();


        // // print bsr matrix. should i tilize?
        // std::string bsr_file = local_path + "/bsr.txt";
        // std::ofstream bsr_out(bsr_file);
        // if (!bsr_out.is_open()) {
        //     TT_THROW("Failed to open bsr file: {}", bsr_file);
        // }
        // untilize(a.data, R, C);
        // for (size_t i = 0; i < a.data.size(); i++) {
        //     bsr_out << a.data[i].to_float() << "\n";
        // }
        // bsr_out.close();
    }

    // untile output data
    untilize(output.data, M, N);

    float pearson = check_bfloat16_vector_pcc(golden.data, output.data);

    // this is useless when matrices are not tiny with tiny elements. I get it now.
    // PCC is faulty and gives false positives for say, equality up to scaling, but
    // all_close is simply not suitable for bfloat16. 
    // surely there is a version of all_close which bases its tolerance on the norm of the input matrices?
    bool all_close = golden.all_close_bfloat16(output);

    CloseDevice(device);

    return TestResult{test_name, pearson, all_close};
}

void add_and_run_test(
        HostCodeFunctionPtr host_func,
        TestFunctionPtr test_case,
        vector<TestResult> &results,
        bool verbose = false,
        bool emit_output = false) {
    auto [a, b, test_name] = test_case();
    results.push_back(run_test(host_func, a, b, test_name, verbose, emit_output));
}

bool print_and_assess_results(std::vector<TestResult> &test_results, std::string& host_code_function_name){
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Test results ----------------------------------------------------------------" << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << "--- Host code function: " << host_code_function_name << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;


    // assume there are <1000 tests.
    std::string spacing = "  ";
    bool all_pass = true;
    char buf[12];
    uint32_t count = 0;
    for (auto &p : test_results) {
        bool pass = true;
        if (p.pearson < 0.99){
            pass = false;
            all_pass = false;
        }

        // counting digits for spacing
        if (count >= 10 && count < 100)
            spacing = " ";
        if (count >= 100)
            spacing = "";

        std::string result = pass ? "✅ PASS " : "❌ FAIL ";
        sprintf(buf, "w/ PCC=%.2f", p.pearson);
        result += std::string(buf);
        result += p.pearson > 0.99 ? " ✅ ": " ❌ ";
        std::cout << "Test #" << count << ": " << spacing << result << " " << count << ' ' << spacing << p.test_name << std::endl;
        count++;
    }

    std::string result = all_pass ? "✅✅✅ PASS ✅✅✅" : "❌❌❌ FAIL ❌❌❌";

    std::cout << "---------------------------------------------------------------------------------" << std::endl;
    std::cout << result << std::endl;
    std::cout << "---------------------------------------------------------------------------------" << std::endl;


    return all_pass;
}

void test_suite(uint32_t host_code_function_index = 0){
    /*
    1. Reserve a vector of <test_name, PCC> pairs.
    2. call run_test(test_func(), verbose, emit_output) for each test, adding to the vector
    3. iter over vector and pretty print passes and fails to the console
    */

    
    auto [host_function_ptr, host_function_name] = HostCodeRegistry[host_code_function_index];
    std::vector<TestResult> test_results;
    size_t num_tests = sizeof(TestRegistry) / sizeof(TestRegistry[0]);
    for (size_t i = 0; i < num_tests; i++) {
        add_and_run_test(host_function_ptr, TestRegistry[i], test_results);
    }
    bool pass = print_and_assess_results(test_results, host_function_name);

    /*
    For later, 
    can we add tests at runtime, ie not having to rebuild this function each time we add a test?
    */
}

void run_verbose_test(int host_code_num, int test_num){
    auto [a, b, test_name] = TestRegistry[test_num]();
    TestResult res = run_test(HostCodeRegistry[host_code_num].first, a, b, test_name, true, true);

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--- Single Test results --------------------------------" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    bool pass = true;
    if (res.pearson < 0.99){
        pass = false;
    }

    char buf[12];
    std::string result = pass ? "✅ PASS " : "❌ FAIL ";
    sprintf(buf, "w/ PCC=%.2f", res.pearson);
    result += std::string(buf);
    result += res.pearson > 0.99 ? " ✅ ": " ❌ ";
    std::cout << "Test #" << test_num << ": " << result << " " << test_num << ' ' << res.test_name << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

int main(int argc, char** argv) {

    // want to provide an opt for which Host code function to use.
    // gonna need the more robust getopts function

    bool test = true;
    if (argc >= 2) {
        test = std::string(argv[1]) == "1";
    }
    if (test) {
        test_suite();
    }
    else {
        int test_num = argc > 2 ? std::stoi(argv[2]) : -1;
        if (test_num == -1) {
            std::cout << "No test specified. Returning." << std::endl;
            return 0;
        }
        run_verbose_test(0, test_num);
    }
}
