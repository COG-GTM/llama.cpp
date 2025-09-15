#include "common.h"
#include "arg.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <filesystem>

static void write_test_yaml(const std::string& filename, const std::string& content) {
    std::ofstream file(filename);
    file << content;
    file.close();
}

static void test_basic_yaml_parsing() {
    std::cout << "Testing basic YAML parsing..." << std::endl;

    const std::string yaml_content = R"(
n_predict: 100
n_ctx: 2048
n_batch: 512
prompt: "Hello, world!"
model:
  path: "test-model.gguf"
sampling:
  seed: 42
  temp: 0.7
  top_k: 50
  top_p: 0.9
)";

    write_test_yaml("test_basic.yaml", yaml_content);

    common_params params;
    const char* argv[] = {"test", "--config", "test_basic.yaml"};
    int argc = 3;

    bool result = common_params_parse(argc, const_cast<char**>(argv), params, LLAMA_EXAMPLE_COMMON);
    assert(result == true);
    (void)result; // Suppress unused variable warning
    assert(params.n_predict == 100);
    assert(params.n_ctx == 2048);
    assert(params.n_batch == 512);
    assert(params.prompt == "Hello, world!");
    assert(params.model.path == "test-model.gguf");
    assert(params.sampling.seed == 42);
    assert(params.sampling.temp == 0.7f);
    assert(params.sampling.top_k == 50);
    assert(params.sampling.top_p == 0.9f);

    std::filesystem::remove("test_basic.yaml");
    std::cout << "Basic YAML parsing test passed!" << std::endl;
}

static void test_cli_override_yaml() {
    std::cout << "Testing CLI override of YAML values..." << std::endl;

    const std::string yaml_content = R"(
n_predict: 100
n_ctx: 2048
prompt: "YAML prompt"
sampling:
  temp: 0.7
)";

    write_test_yaml("test_override.yaml", yaml_content);

    common_params params;
    const char* argv[] = {"test", "--config", "test_override.yaml", "-n", "200", "-p", "CLI prompt", "--temp", "0.5"};
    int argc = 8;

    bool result = common_params_parse(argc, const_cast<char**>(argv), params, LLAMA_EXAMPLE_COMMON);
    assert(result == true);
    (void)result; // Suppress unused variable warning
    assert(params.n_predict == 200); // CLI should override YAML
    assert(params.n_ctx == 2048); // YAML value should remain
    assert(params.prompt == "CLI prompt"); // CLI should override YAML
    assert(params.sampling.temp == 0.5f); // CLI should override YAML

    std::filesystem::remove("test_override.yaml");
    std::cout << "CLI override test passed!" << std::endl;
}

static void test_invalid_yaml() {
    std::cout << "Testing invalid YAML handling..." << std::endl;

    const std::string invalid_yaml = R"(
n_predict: 100
invalid_yaml: [unclosed array
)";

    write_test_yaml("test_invalid.yaml", invalid_yaml);

    common_params params;
    const char* argv[] = {"test", "--config", "test_invalid.yaml"};
    int argc = 3;

    bool result = common_params_parse(argc, const_cast<char**>(argv), params, LLAMA_EXAMPLE_COMMON);
    assert(result == false); // Should fail with invalid YAML
    (void)result; // Suppress unused variable warning

    std::filesystem::remove("test_invalid.yaml");
    std::cout << "Invalid YAML test passed!" << std::endl;
}

static void test_missing_config_file() {
    std::cout << "Testing missing config file handling..." << std::endl;

    common_params params;
    const char* argv[] = {"test", "--config", "nonexistent.yaml"};
    int argc = 3;

    bool result = common_params_parse(argc, const_cast<char**>(argv), params, LLAMA_EXAMPLE_COMMON);
    assert(result == false); // Should fail with missing file
    (void)result; // Suppress unused variable warning

    std::cout << "Missing config file test passed!" << std::endl;
}

static void test_backward_compatibility() {
    std::cout << "Testing backward compatibility..." << std::endl;

    common_params params;
    const char* argv[] = {"test", "-n", "150", "-p", "Test prompt", "--temp", "0.8"};
    int argc = 7;

    bool result = common_params_parse(argc, const_cast<char**>(argv), params, LLAMA_EXAMPLE_COMMON);
    assert(result == true);
    (void)result; // Suppress unused variable warning
    assert(params.n_predict == 150);
    assert(params.prompt == "Test prompt");
    assert(params.sampling.temp == 0.8f);

    std::cout << "Backward compatibility test passed!" << std::endl;
}

static void test_complex_yaml_structure() {
    std::cout << "Testing complex YAML structure..." << std::endl;

    const std::string complex_yaml = R"(
n_predict: 200
n_ctx: 4096
model:
  path: "complex-model.gguf"
sampling:
  seed: 123
  temp: 0.6
  top_k: 40
  top_p: 0.95
  penalty_repeat: 1.1
  dry_sequence_breakers:
    - "\n"
    - ":"
    - ";"
speculative:
  n_max: 16
  p_split: 0.1
in_files:
  - "file1.txt"
  - "file2.txt"
antiprompt:
  - "User:"
  - "Assistant:"
)";

    write_test_yaml("test_complex.yaml", complex_yaml);

    common_params params;
    const char* argv[] = {"test", "--config", "test_complex.yaml"};
    int argc = 3;

    bool result = common_params_parse(argc, const_cast<char**>(argv), params, LLAMA_EXAMPLE_COMMON);
    assert(result == true);
    (void)result; // Suppress unused variable warning
    assert(params.n_predict == 200);
    assert(params.n_ctx == 4096);
    assert(params.model.path == "complex-model.gguf");
    assert(params.sampling.seed == 123);
    assert(params.sampling.temp == 0.6f);
    assert(params.sampling.penalty_repeat == 1.1f);
    assert(params.sampling.dry_sequence_breakers.size() == 3);
    assert(params.sampling.dry_sequence_breakers[0] == "\n");
    assert(params.sampling.dry_sequence_breakers[1] == ":");
    assert(params.sampling.dry_sequence_breakers[2] == ";");
    assert(params.speculative.n_max == 16);
    assert(params.speculative.p_split == 0.1f);
    assert(params.in_files.size() == 2);
    assert(params.in_files[0] == "file1.txt");
    assert(params.in_files[1] == "file2.txt");
    assert(params.antiprompt.size() == 2);
    assert(params.antiprompt[0] == "User:");
    assert(params.antiprompt[1] == "Assistant:");

    std::filesystem::remove("test_complex.yaml");
    std::cout << "Complex YAML structure test passed!" << std::endl;
}

int main() {
    std::cout << "Running YAML configuration tests..." << std::endl;

    try {
        test_basic_yaml_parsing();
        test_cli_override_yaml();
        test_invalid_yaml();
        test_missing_config_file();
        test_backward_compatibility();
        test_complex_yaml_structure();

        std::cout << "All YAML configuration tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}
