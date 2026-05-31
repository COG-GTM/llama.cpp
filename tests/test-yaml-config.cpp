#include "arg.h"
#include "common.h"

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>

#undef NDEBUG
#include <cassert>

static void write_test_yaml(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    assert(file.is_open());
    file << content;
    file.close();
}

int main(void) {
    printf("test-yaml-config: testing YAML configuration loading\n\n");

    std::filesystem::path test_dir = std::filesystem::temp_directory_path() / "llama_yaml_test";
    std::filesystem::create_directories(test_dir);

    printf("test-yaml-config: test basic YAML loading\n");
    {
        std::string yaml_content = R"(
model: "test-model.gguf"
threads: 4
ctx-size: 1024
predict: 100
seed: 42
temp: 0.8
top-k: 20
prompt: "Hello world"
)";
        std::string yaml_path = (test_dir / "basic.yaml").string();
        write_test_yaml(yaml_path, yaml_content);

        common_params params;
        bool result = common_params_load_yaml_config(yaml_path, params);
        assert(result == true);
        std::string expected_model_path = (test_dir / "test-model.gguf").string();
        assert(params.model.path == expected_model_path);
        assert(params.cpuparams.n_threads == 4);
        assert(params.n_ctx == 1024);
        assert(params.n_predict == 100);
        assert(params.sampling.seed == 42);
        assert(params.sampling.temp == 0.8f);
        assert(params.sampling.top_k == 20);
        assert(params.prompt == "Hello world");
    }

    printf("test-yaml-config: test relative path resolution\n");
    {
        std::filesystem::path subdir = test_dir / "subdir";
        std::filesystem::create_directories(subdir);
        
        std::string model_content = "dummy model content";
        std::string model_path = (subdir / "relative-model.gguf").string();
        write_test_yaml(model_path, model_content);

        std::string yaml_content = R"(
model: "relative-model.gguf"
)";
        std::string yaml_path = (subdir / "relative.yaml").string();
        write_test_yaml(yaml_path, yaml_content);

        common_params params;
        bool result = common_params_load_yaml_config(yaml_path, params);
        assert(result == true);
        assert(params.model.path == model_path);
    }

    printf("test-yaml-config: test unknown key rejection\n");
    {
        std::string yaml_content = R"(
model: "test-model.gguf"
unknown_key: "should fail"
)";
        std::string yaml_path = (test_dir / "unknown.yaml").string();
        write_test_yaml(yaml_path, yaml_content);

        common_params params;
        bool result = common_params_load_yaml_config(yaml_path, params);
        assert(result == false);
    }

    printf("test-yaml-config: test valid keys list\n");
    {
        std::vector<std::string> valid_keys = common_params_get_valid_yaml_keys();
        assert(!valid_keys.empty());
        
        bool found_model = false;
        bool found_threads = false;
        for (const auto& key : valid_keys) {
            if (key == "model") found_model = true;
            if (key == "threads") found_threads = true;
        }
        assert(found_model);
        assert(found_threads);
    }

    printf("test-yaml-config: test boolean values\n");
    {
        std::string yaml_content = R"(
interactive: true
escape: false
color: true
)";
        std::string yaml_path = (test_dir / "booleans.yaml").string();
        write_test_yaml(yaml_path, yaml_content);

        common_params params;
        bool result = common_params_load_yaml_config(yaml_path, params);
        assert(result == true);
        assert(params.interactive == true);
        assert(params.escape == false);
        assert(params.use_color == true);
    }

    std::filesystem::remove_all(test_dir);

    printf("test-yaml-config: all tests passed\n\n");
    return 0;
}
