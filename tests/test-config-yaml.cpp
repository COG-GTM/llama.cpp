#include "common.h"
#include "config.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

static void test_minimal_config() {
    common_params params;
    fs::path temp_dir = fs::temp_directory_path() / "llama_test";
    fs::create_directories(temp_dir);

    std::string config_content = R"(
model:
  path: test_model.gguf
n_ctx: 512
sampling:
  seed: 123
  temp: 0.5
prompt: "Test prompt"
n_predict: 64
simple_io: true
)";

    fs::path config_path = temp_dir / "test_config.yaml";
    std::ofstream config_file(config_path);
    config_file << config_content;
    config_file.close();

    bool result = common_load_yaml_config(config_path.string(), params);
    assert(result);
    (void)result;

    assert(params.model.path == (temp_dir / "test_model.gguf").string());
    assert(params.n_ctx == 512);
    assert(params.sampling.seed == 123);
    assert(params.sampling.temp == 0.5f);
    assert(params.prompt == "Test prompt");
    assert(params.n_predict == 64);
    assert(params.simple_io == true);
    fs::remove_all(temp_dir);

    std::cout << "test_minimal_config: PASSED\n";
}

static void test_unknown_key_error() {
    common_params params;
    fs::path temp_dir = fs::temp_directory_path() / "llama_test";
    fs::create_directories(temp_dir);

    std::string config_content = R"(
model:
  path: test_model.gguf
unknown_key: "should fail"
n_ctx: 512
)";

    fs::path config_path = temp_dir / "test_config.yaml";
    std::ofstream config_file(config_path);
    config_file << config_content;
    config_file.close();

    bool threw_exception = false;
    try {
        common_load_yaml_config(config_path.string(), params);
    } catch (const std::invalid_argument & e) {
        threw_exception = true;
        std::string error_msg = e.what();
        assert(error_msg.find("Unknown YAML keys") != std::string::npos);
        assert(error_msg.find("valid keys are") != std::string::npos);
    }

    assert(threw_exception);
    (void)threw_exception;
    fs::remove_all(temp_dir);

    std::cout << "test_unknown_key_error: PASSED\n";
}

static void test_relative_path_resolution() {
    common_params params;
    fs::path temp_dir = fs::temp_directory_path() / "llama_test";
    fs::path config_dir = temp_dir / "configs";
    fs::create_directories(config_dir);

    std::string config_content = R"(
model:
  path: ../models/test_model.gguf
prompt_file: prompts/test.txt
)";

    fs::path config_path = config_dir / "test_config.yaml";
    std::ofstream config_file(config_path);
    config_file << config_content;
    config_file.close();

    bool result = common_load_yaml_config(config_path.string(), params);
    assert(result);
    (void)result;

    fs::path expected_model = temp_dir / "models" / "test_model.gguf";
    fs::path expected_prompt = config_dir / "prompts" / "test.txt";

    assert(params.model.path == expected_model.lexically_normal().string());
    assert(params.prompt_file == expected_prompt.lexically_normal().string());
    fs::remove_all(temp_dir);

    std::cout << "test_relative_path_resolution: PASSED\n";
}

int main() {
    try {
        test_minimal_config();
        test_unknown_key_error();
        test_relative_path_resolution();

        std::cout << "All tests passed!\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
