
#include <string>
#include "../inc/include_me.hpp"
#include "../inc/test_suite.hpp"
#include "../inc/host_code.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;

using namespace bsr_test_suite;
using namespace bsr_host_code;

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

    // for (int i = 0; i < a.data.size(); i+=32) {
    //     for (int j = 0; j < 32; j++){
    //         std::cout << a.data[i + j] << ' ';
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;

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
    std::cout << "--- Host Code function: " << HostCodeRegistry[host_code_num].second << std::endl;
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

    bool test_all = true;
    int host_code_index = 0;
    if (argc > 1) {
        test_all = std::string(argv[1]) == "all";
    }
    if (argc > 2) {
        host_code_index = std::stoi(argv[2]);
    }


    if (test_all) {
        test_suite(host_code_index);
    }
    else {
        int test_num = argc > 1 ? std::stoi(argv[1]) : -1;
        if (test_num == -1) {
            std::cout << "No test specified. Returning." << std::endl;
            return 0;
        }
        run_verbose_test(host_code_index, test_num);
    }
}
