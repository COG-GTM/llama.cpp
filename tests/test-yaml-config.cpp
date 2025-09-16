#include "arg.h"
#include "common.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static void test_yaml_config_basic() {
    printf("Testing basic YAML config loading...\n");
    const std::string config_content = R"(
model:
  path: "test-model.gguf"
n_predict: 256
n_ctx: 2048
n_gpu_layers: 16
prompt: "Hello world"
sampling:
  temp: 0.7
  top_k: 50
  top_p: 0.9
  seed: 12345
interactive: true
use_color: false
)";
    std::ofstream config_file("test_config.yaml");
    config_file << config_content;
    config_file.close();
    common_params params;
    bool result = common_params_load_from_yaml("test_config.yaml", params);
    (void)result; // suppress unused variable warning
    assert(result == true);
    assert(params.model.path == "test-model.gguf");
    assert(params.n_predict == 256);
    assert(params.n_ctx == 2048);
    assert(params.n_gpu_layers == 16);
    assert(params.prompt == "Hello world");
    assert(params.sampling.temp == 0.7f);
    assert(params.sampling.top_k == 50);
    assert(params.sampling.top_p == 0.9f);
    assert(params.sampling.seed == 12345);
    assert(params.interactive == true);
    assert(params.use_color == false);
    std::remove("test_config.yaml");
    printf("✓ Basic YAML config loading test passed\n");
}

static void test_yaml_config_cli_override() {
    printf("Testing CLI argument override of YAML config...\n");
    const std::string config_content = R"(
model:
  path: "config-model.gguf"
n_predict: 100
sampling:
  temp: 0.5
interactive: false
)";
    std::ofstream config_file("test_override.yaml");
    config_file << config_content;
    config_file.close();
    const char* argv[] = {
        "test",
        "--config", "test_override.yaml",
        "--model", "cli-model.gguf",
        "--n-predict", "200",
        "--temp", "0.8",
        "--interactive"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    common_params params;
    bool result = common_params_parse(argc, const_cast<char**>(argv), params, LLAMA_EXAMPLE_MAIN, nullptr);
    (void)result; // suppress unused variable warning
    assert(result == true);
    assert(params.model.path == "cli-model.gguf");
    assert(params.n_predict == 200);
    assert(params.sampling.temp == 0.8f);
    assert(params.interactive == true);

    std::remove("test_override.yaml");
    printf("✓ CLI override test passed\n");
}

static void test_yaml_config_invalid_file() {
    printf("Testing invalid YAML file handling...\n");

    common_params params;
    bool result = common_params_load_from_yaml("nonexistent.yaml", params);
    (void)result; // suppress unused variable warning
    assert(result == false);
    const std::string invalid_content = R"(
invalid: yaml: content:
  - missing
    proper: indentation
)";
    std::ofstream invalid_file("invalid.yaml");
    invalid_file << invalid_content;
    invalid_file.close();

    result = common_params_load_from_yaml("invalid.yaml", params);
    assert(result == false);
    std::remove("invalid.yaml");

    printf("✓ Invalid YAML handling test passed\n");
}

static void test_yaml_config_nested_structures() {
    printf("Testing nested structure parsing...\n");
    const std::string config_content = R"(
cpuparams:
  n_threads: 8
  strict_cpu: true
  poll: 100

sampling:
  seed: 42
  top_k: 40
  top_p: 0.95
  temp: 0.8
  penalty_repeat: 1.1
  penalty_freq: 0.1
  penalty_present: 0.1
  dry_sequence_breakers:
    - "\n"
    - ":"
    - "\""

speculative:
  n_max: 16
  n_min: 5
  p_split: 0.1
  model:
    path: "draft-model.gguf"

antiprompt:
  - "User:"
  - "Human:"
  - "\n\n"
)";
    std::ofstream config_file("test_nested.yaml");
    config_file << config_content;
    config_file.close();
    common_params params;
    bool result = common_params_load_from_yaml("test_nested.yaml", params);
    (void)result; // suppress unused variable warning
    assert(result == true);
    assert(params.cpuparams.n_threads == 8);
    assert(params.cpuparams.strict_cpu == true);
    assert(params.cpuparams.poll == 100);
    assert(params.sampling.seed == 42);
    assert(params.sampling.top_k == 40);
    assert(params.sampling.top_p == 0.95f);
    assert(params.sampling.temp == 0.8f);
    assert(params.sampling.penalty_repeat == 1.1f);
    assert(params.sampling.penalty_freq == 0.1f);
    assert(params.sampling.penalty_present == 0.1f);
    assert(params.sampling.dry_sequence_breakers.size() == 3);
    assert(params.sampling.dry_sequence_breakers[0] == "\n");
    assert(params.sampling.dry_sequence_breakers[1] == ":");
    assert(params.sampling.dry_sequence_breakers[2] == "\"");

    assert(params.speculative.n_max == 16);
    assert(params.speculative.n_min == 5);
    assert(params.speculative.p_split == 0.1f);
    assert(params.speculative.model.path == "draft-model.gguf");

    assert(params.antiprompt.size() == 3);
    assert(params.antiprompt[0] == "User:");
    assert(params.antiprompt[1] == "Human:");
    assert(params.antiprompt[2] == "\n\n");
    std::remove("test_nested.yaml");
    printf("✓ Nested structure parsing test passed\n");
}

static void test_backward_compatibility() {
    printf("Testing backward compatibility...\n");
    const char* argv[] = {
        "test",
        "--model", "test-model.gguf",
        "--n-predict", "128",
        "--temp", "0.7",
        "--top-k", "40",
        "--interactive"
    };
    int argc = sizeof(argv) / sizeof(argv[0]);
    common_params params;
    bool result = common_params_parse(argc, const_cast<char**>(argv), params, LLAMA_EXAMPLE_MAIN, nullptr);
    (void)result; // suppress unused variable warning
    assert(result == true);
    assert(params.model.path == "test-model.gguf");
    assert(params.n_predict == 128);
    assert(params.sampling.temp == 0.7f);
    assert(params.sampling.top_k == 40);
    assert(params.interactive == true);
    printf("✓ Backward compatibility test passed\n");
}

static void test_empty_config_file() {
    printf("Testing empty config file handling...\n");
    std::ofstream config_file("empty.yaml");
    config_file << "";
    config_file.close();

    common_params params;
    params.n_predict = 999;
    params.sampling.temp = 0.123f;
    bool result = common_params_load_from_yaml("empty.yaml", params);
    (void)result; // suppress unused variable warning
    assert(result == true);
    assert(params.n_predict == 999);
    assert(params.sampling.temp == 0.123f);
    std::remove("empty.yaml");
    printf("✓ Empty config file test passed\n");
}

int main() {
    printf("Running YAML config tests...\n\n");
    test_yaml_config_basic();
    test_yaml_config_cli_override();
    test_yaml_config_invalid_file();
    test_yaml_config_nested_structures();
    test_backward_compatibility();
    test_empty_config_file();

    printf("\n✓ All YAML config tests passed!\n");
    return 0;
}
