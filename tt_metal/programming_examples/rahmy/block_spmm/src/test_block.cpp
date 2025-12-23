
#include <cstdio>
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

#define ESC "\033["
#define BLACK_BKG "106"
#define GREEN_TXT "118"
#define RED_TXT "196"
#define RESET "\033[m"

// Print to the *original* console, regardless of where stdout is redirected
void console_printf(const char* fmt, ...) {
    static int console_fd = -1;
    if (console_fd == -1) {
        // If not initialized, we fail-safe to /dev/tty (works when a TTY is present),
        // but you could also inject the saved fd via a setter if you prefer.
        console_fd = ::open("/dev/tty", O_WRONLY | O_CLOEXEC);
        // If /dev/tty isn't available (e.g. no controlling terminal), this will be -1.
    }
    if (console_fd == -1) return; // quietly drop if no console is available

    va_list ap;
    va_start(ap, fmt);
    ::vdprintf(console_fd, fmt, ap);
    va_end(ap);
}

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
    tilize_nfaces(a.data, R, C);
    tilize_nfaces(b.data, K, N);

    // for (int i = 0; i < a.data.size(); i+=32) {
    //     for (int j = 0; j < 32; j++){
    //         console_printf(a.data[i + j] << ' ';
    //     }
    //     console_printf(std::endl;
    // }
    // console_printf(std::endl;
    // console_printf(std::endl;
    // console_printf(std::endl;

    // run bsr_spmm_multicore_reuse
    // console_printf("Do we seg fault before...");
    host_func(a, b, output, false, nblocks, M, N, K, R, C, 1, device, verbose);
    // console_printf("... or after running the program?\n");


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

        tilize_nfaces(golden.data, M, N);
        for (size_t i = 0; i < golden.data.size(); i++) {
            golden_out << golden.data[i].to_float() << "\n";
        }
        untilize_nfaces(golden.data, M, N);
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
    untilize_nfaces(output.data, M, N);

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
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Test results ----------------------------------------------------------------\n");
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Host code function: ");
    console_printf(host_code_function_name.c_str());
    console_printf("\n");
    console_printf("---------------------------------------------------------------------------------\n");


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
        // console_printf("Test #" << count << ": " << spacing << result << " " << count << ' ' << spacing << p.test_name << std::endl;
        console_printf("Test #");
        console_printf(std::to_string(count).c_str());
        console_printf(": ");
        console_printf(spacing.c_str());
        console_printf(result.c_str());
        console_printf(" ");
        console_printf(std::to_string(count).c_str());
        console_printf(" ");
        console_printf(spacing.c_str());
        console_printf(p.test_name.c_str());
        console_printf("\n");
        count++;
    }

    std::string result = all_pass ? "✅✅✅ PASS ✅✅✅" : "❌❌❌ FAIL ❌❌❌";

    console_printf("---------------------------------------------------------------------------------\n");
    console_printf(result.c_str());
    console_printf("\n");
    console_printf("---------------------------------------------------------------------------------\n");


    return all_pass;
}

void test_suite(uint32_t host_code_function_index = 0){
    /*
    1. Reserve a vector of <test_name, PCC> pairs.
    2. call run_test(test_func(), verbose, emit_output) for each test, adding to the vector
    3. iter over vector and pretty print passes and fails to the console
    */

    auto [host_function_ptr, host_function_name] = HostCodeRegistry[host_code_function_index];
    size_t num_tests = sizeof(TestRegistry) / sizeof(TestRegistry[0]);
    // 1. Print Header
    //
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Test results ----------------------------------------------------------------\n");
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Host code function: ");
    console_printf(host_function_name.c_str());
    console_printf("\n");
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("Testing ");
    console_printf(std::to_string(num_tests).c_str());
    console_printf(" tests\n");

    std::vector<TestResult> test_results;
    std::string spacing = "  ";
    char buf[12];
    bool all_pass = true;
    uint32_t count_pass = 0;
    for (size_t i = 0; i < num_tests; i++) {
        if (i >= 10 && i < 100)
            spacing = " ";
        if (i >= 100)
            spacing = "";
        add_and_run_test(host_function_ptr, TestRegistry[i], test_results);
        auto res = test_results[i];
        bool pass = res.pearson >= 0.99;
        count_pass += pass;
        if (!pass){
            all_pass = false;
        }
        // std::string result = pass ? "\033[0;118m PASS \033[m" : "\033[0;196m FAIL \033[m";
        std::string result = pass ? "✅ PASS " : "❌ FAIL ";

        sprintf(buf, "w/ PCC=%.2f", res.pearson);
        result += std::string(buf);
        result += res.pearson > 0.99 ? " ✅ ": " ❌ ";
        console_printf("Test #");
        console_printf(std::to_string(i).c_str());
        console_printf(": ");
        console_printf(spacing.c_str());
        console_printf(result.c_str());
        console_printf(" ");
        console_printf(std::to_string(i).c_str());
        console_printf(" ");
        console_printf(spacing.c_str());
        console_printf(res.test_name.c_str());
        console_printf("\n");
    }
    std::string result = all_pass ? "✅✅✅ PASS ✅✅✅" : "❌❌❌ FAIL ❌❌❌";
    std::string count_result = std::to_string(count_pass) + "/" + std::to_string(num_tests) + " tests passed!\n";
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf(result.c_str());
    console_printf("\n");
    console_printf(count_result.c_str());
    console_printf("---------------------------------------------------------------------------------\n");
    console_printf("--- Host code function: ");
    console_printf(host_function_name.c_str());
    console_printf("\n");
    console_printf("---------------------------------------------------------------------------------\n");
}

void run_verbose_test(int host_code_num, int test_num){
    auto [a, b, test_name] = TestRegistry[test_num]();
    TestResult res = run_test(HostCodeRegistry[host_code_num].first, a, b, test_name, true, true);

    console_printf("--------------------------------------------------------\n");
    console_printf("--- Single Test results --------------------------------\n");
    console_printf("--------------------------------------------------------\n");
    console_printf("--- Host Code function: ");
    console_printf(HostCodeRegistry[host_code_num].second.c_str());
    console_printf("\n");
    console_printf("--------------------------------------------------------\n");

    bool pass = true;
    if (res.pearson < 0.99){
        pass = false;
    }

    char buf[13];
    std::string result = pass ? "✅ PASS " : "❌ FAIL ";
    sprintf(buf, "w/ PCC=%.2f", res.pearson);
    result += std::string(buf);
    result += res.pearson > 0.99 ? " ✅ ": " ❌ ";
    // console_printf("Test #" << test_num << ": " << result << " " << test_num << ' ' << res.test_name << std::endl;
    console_printf("Test #");
    console_printf(std::to_string(test_num).c_str());
    console_printf(": ");
    console_printf(result.c_str());
    console_printf(" ");
    console_printf(std::to_string(test_num).c_str());
    console_printf(" ");
    console_printf(res.test_name.c_str());
    console_printf("\n");
    console_printf("--------------------------------------------------------\n");
    console_printf("--------------------------------------------------------\n");
    console_printf("--------------------------------------------------------\n");
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
        //
        // Redirect TT-Metal output to some file.
        // Let only our print statements go to stdout
        //
        // 1) Save the original stdout (the real console)
        int saved_stdout = ::dup(STDOUT_FILENO);
        if (saved_stdout == -1) {
            std::perror("dup");
            return 1;
        }

        // // 2) Redirect stdout to a log file (affects std::cout and printf)
        int log_fd = ::open("std.out.log", O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (log_fd == -1) {
            std::perror("open");
            return 1;
        }
        if (::dup2(log_fd, STDOUT_FILENO) == -1) {
            std::perror("dup2");
            return 1;
        }
        ::close(log_fd); // not needed after dup2
        //
        //
        test_suite(host_code_index);
    }
    else {
        //
        //
        int test_num = argc > 1 ? std::stoi(argv[1]) : -1;
        if (test_num == -1) {
            console_printf("No test specified. Returning.\n");
            return 0;
        }
        run_verbose_test(host_code_index, test_num);
        console_printf("Leaving the test program\n");

    }
}
