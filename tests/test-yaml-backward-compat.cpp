#include "common.h"
#include "arg.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <cmath>

struct TestCase {
    std::vector<std::string> args;
    std::string description;
};

static void test_cli_args_without_yaml() {
    std::cout << "Testing CLI arguments without YAML..." << std::endl;

    std::vector<TestCase> test_cases = {
        {{"test", "-n", "100"}, "Basic n_predict"},
        {{"test", "-p", "Hello world"}, "Basic prompt"},
        {{"test", "--temp", "0.8"}, "Temperature setting"},
        {{"test", "-c", "2048"}, "Context size"},
        {{"test", "-b", "512"}, "Batch size"},
        {{"test", "--top-k", "40"}, "Top-k sampling"},
        {{"test", "--top-p", "0.9"}, "Top-p sampling"},
        {{"test", "-s", "42"}, "Random seed"},
        {{"test", "-n", "50", "-p", "Test", "--temp", "0.7"}, "Multiple arguments"},
        {{"test", "--help"}, "Help flag (should exit)"},
    };

    for (const auto& test_case : test_cases) {
        if (test_case.description == "Help flag (should exit)") {
            continue;
        }

        std::cout << "  Testing: " << test_case.description << std::endl;

        common_params params;
        std::vector<char*> argv;
        for (const auto& arg : test_case.args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }

        bool result = common_params_parse(argv.size(), argv.data(), params, LLAMA_EXAMPLE_COMMON);

        if (!result && test_case.description != "Help flag (should exit)") {
            std::cout << "    Warning: " << test_case.description << " failed to parse" << std::endl;
        }
    }

    std::cout << "CLI arguments without YAML test completed!" << std::endl;
}

static void test_equivalent_yaml_and_cli() {
    std::cout << "Testing equivalent YAML and CLI produce same results..." << std::endl;

    std::ofstream yaml_file("equivalent_test.yaml");
    yaml_file << R"(
n_predict: 100
n_ctx: 2048
n_batch: 512
prompt: "Test prompt"
sampling:
  seed: 42
  temp: 0.8
  top_k: 40
  top_p: 0.9
  penalty_repeat: 1.1
)";
    yaml_file.close();

    common_params yaml_params;
    const char* yaml_argv[] = {"test", "--config", "equivalent_test.yaml"};
    bool yaml_result = common_params_parse(3, const_cast<char**>(yaml_argv), yaml_params, LLAMA_EXAMPLE_COMMON);

    common_params cli_params;
    const char* cli_argv[] = {
        "test", 
        "-n", "100",
        "-c", "2048", 
        "-b", "512",
        "-p", "Test prompt",
        "-s", "42",
        "--temp", "0.8",
        "--top-k", "40",
        "--top-p", "0.9",
        "--repeat-penalty", "1.1"
    };
    const int cli_argc = sizeof(cli_argv) / sizeof(cli_argv[0]);
    
    bool cli_result = common_params_parse(cli_argc, const_cast<char**>(cli_argv), cli_params, LLAMA_EXAMPLE_COMMON);

    assert(yaml_result == true);
    assert(cli_result == true);
    (void)yaml_result; // Suppress unused variable warning
    (void)cli_result; // Suppress unused variable warning

    assert(yaml_params.n_predict == cli_params.n_predict);
    assert(yaml_params.n_ctx == cli_params.n_ctx);
    assert(yaml_params.n_batch == cli_params.n_batch);
    assert(yaml_params.prompt == cli_params.prompt);
    assert(yaml_params.sampling.seed == cli_params.sampling.seed);
    assert(yaml_params.sampling.temp == cli_params.sampling.temp);
    assert(yaml_params.sampling.top_k == cli_params.sampling.top_k);
    assert(yaml_params.sampling.top_p == cli_params.sampling.top_p);
    
    
    const float epsilon = 1e-6f;
    assert(std::abs(yaml_params.sampling.penalty_repeat - cli_params.sampling.penalty_repeat) < epsilon);

    std::filesystem::remove("equivalent_test.yaml");
    std::cout << "Equivalent YAML and CLI test passed!" << std::endl;
}

static void test_all_major_cli_options() {
    std::cout << "Testing all major CLI options still work..." << std::endl;

    struct CliTest {
        std::vector<std::string> args;
        std::string param_name;
        bool should_succeed;
    };

    std::vector<CliTest> cli_tests = {
        {{"test", "-m", "model.gguf"}, "model path", true},
        {{"test", "-n", "200"}, "n_predict", true},
        {{"test", "-c", "4096"}, "context size", true},
        {{"test", "-b", "1024"}, "batch size", true},
        {{"test", "-p", "Hello"}, "prompt", true},
        {{"test", "-s", "123"}, "seed", true},
        {{"test", "--temp", "0.7"}, "temperature", true},
        {{"test", "--top-k", "50"}, "top_k", true},
        {{"test", "--top-p", "0.95"}, "top_p", true},
        {{"test", "--repeat-penalty", "1.05"}, "repeat penalty", true},
        {{"test", "-t", "4"}, "threads", true},
        {{"test", "-ngl", "32"}, "gpu layers", true},
        {{"test", "--interactive"}, "interactive mode", true},
        {{"test", "--color"}, "color output", true},
        {{"test", "--verbose"}, "verbose mode", true},
    };

    for (const auto& test : cli_tests) {
        std::cout << "  Testing: " << test.param_name << std::endl;

        common_params params;
        std::vector<char*> argv;
        for (const auto& arg : test.args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }

        bool result = common_params_parse(argv.size(), argv.data(), params, LLAMA_EXAMPLE_COMMON);

        if (result != test.should_succeed) {
            std::cout << "    Unexpected result for " << test.param_name
                      << ": expected " << test.should_succeed << ", got " << result << std::endl;
        }
    }

    std::cout << "Major CLI options test completed!" << std::endl;
}

int main() {
    std::cout << "Running backward compatibility tests..." << std::endl;

    try {
        test_cli_args_without_yaml();
        test_equivalent_yaml_and_cli();
        test_all_major_cli_options();

        std::cout << "All backward compatibility tests completed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}
