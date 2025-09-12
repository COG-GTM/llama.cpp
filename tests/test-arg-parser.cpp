#include "arg.h"
#include "common.h"

#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>
#include <fstream>
#include <filesystem>
#include <cstdlib>

#undef NDEBUG
#include <cassert>

int main(void) {
    common_params params;

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.args) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg);
                        exit(1);
                    }
                }
                // check for env var duplications
                if (opt.env) {
                    if (seen_env_vars.find(opt.env) == seen_env_vars.end()) {
                        seen_env_vars.insert(opt.env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", opt.env);
                        exit(1);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));


    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--draft", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.n_max == 123);

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);


    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    if (common_has_curl()) {
        printf("test-arg-parser: test curl-related functions\n\n");
        const char * GOOD_URL = "https://ggml.ai/";
        const char * BAD_URL  = "https://www.google.com/404";
        const char * BIG_FILE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v1.bin";

        {
            printf("test-arg-parser: test good URL\n\n");
            auto res = common_remote_get_content(GOOD_URL, {});
            assert(res.first == 200);
            assert(res.second.size() > 0);
            std::string str(res.second.data(), res.second.size());
            assert(str.find("llama.cpp") != std::string::npos);
        }

        {
            printf("test-arg-parser: test bad URL\n\n");
            auto res = common_remote_get_content(BAD_URL, {});
            assert(res.first == 404);
        }

        {
            printf("test-arg-parser: test max size error\n");
            common_remote_params params;
            params.max_size = 1;
            try {
                common_remote_get_content(GOOD_URL, params);
                assert(false && "it should throw an error");
            } catch (std::exception & e) {
                printf("  expected error: %s\n\n", e.what());
            }
        }

        {
            printf("test-arg-parser: test timeout error\n");
            common_remote_params params;
            params.timeout = 1;
            try {
                common_remote_get_content(BIG_FILE, params);
                assert(false && "it should throw an error");
            } catch (std::exception & e) {
                printf("  expected error: %s\n\n", e.what());
            }
        }
    } else {
        printf("test-arg-parser: no curl, skipping curl-related functions\n");
    }

    printf("test-arg-parser: testing common_arg class methods\n\n");

    {
        common_arg arg({"-t", "--test"}, "test help", [](common_params & params) {
            (void)params;
        });

        arg.set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_SERVER});
        assert(arg.in_example(LLAMA_EXAMPLE_COMMON));
        assert(arg.in_example(LLAMA_EXAMPLE_SERVER));
        assert(!arg.in_example(LLAMA_EXAMPLE_EMBEDDING));

        arg.set_excludes({LLAMA_EXAMPLE_EMBEDDING});
        assert(arg.is_exclude(LLAMA_EXAMPLE_EMBEDDING));
        assert(!arg.is_exclude(LLAMA_EXAMPLE_COMMON));

        arg.set_env("TEST_ENV_VAR");
        std::string output;
        setenv("TEST_ENV_VAR", "test_value", 1);
        assert(arg.get_value_from_env(output));
        assert(output == "test_value");
        assert(arg.has_value_from_env());

        unsetenv("TEST_ENV_VAR");
        assert(!arg.get_value_from_env(output));
        assert(!arg.has_value_from_env());

        arg.set_sparam();
        assert(arg.is_sparam);
    }

    printf("test-arg-parser: testing file I/O functions with temp files\n\n");

    {
        std::string temp_dir = std::filesystem::temp_directory_path();
        std::string test_file = temp_dir + "/test_arg_parser_file.txt";
        std::string test_content = "Hello, World!\nThis is a test file.";

        std::ofstream file(test_file);
        file << test_content;
        file.close();

        std::ifstream read_file(test_file);
        std::string content((std::istreambuf_iterator<char>(read_file)), std::istreambuf_iterator<char>());
        read_file.close();
        assert(content == test_content);

        std::filesystem::remove(test_file);

        try {
            std::ifstream bad_file("/nonexistent/path/file.txt");
            if (!bad_file) {
                printf("  expected: file open failure handled correctly\n");
            }
        } catch (...) {
            printf("  expected: exception handling for bad file paths\n");
        }
    }

    printf("test-arg-parser: testing string processing functions\n\n");

    {
        common_arg arg({"-t", "--test"}, "VALUE", "This is a test argument with a very long help text that should be wrapped properly when displayed to the user.", [](common_params & params, const std::string & value) {
            (void)params;
            (void)value;
        });

        std::string result = arg.to_string();
        assert(!result.empty());
        assert(result.find("-t") != std::string::npos);
        assert(result.find("--test") != std::string::npos);
        assert(result.find("VALUE") != std::string::npos);
        assert(result.find("This is a test") != std::string::npos);
    }

    printf("test-arg-parser: testing edge cases and error conditions\n\n");

    {
        common_arg arg({"-e", "--env-test"}, "test help", [](common_params & params) {
            (void)params;
        });

        std::string empty_output;
        assert(!arg.get_value_from_env(empty_output));
        assert(!arg.has_value_from_env());

        arg.set_env("NONEXISTENT_ENV_VAR_12345");
        assert(!arg.get_value_from_env(empty_output));
        assert(!arg.has_value_from_env());
    }

    printf("test-arg-parser: testing argument parsing with various data types\n\n");

    {
        common_params params;
        std::vector<std::string> argv;
        auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
            std::vector<char *> res;
            for (auto & arg : argv) {
                res.push_back(const_cast<char *>(arg.data()));
            }
            return res;
        };

        argv = {"binary_name", "-c", "512"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
        assert(params.n_ctx == 512);

        argv = {"binary_name", "--seed", "42"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
        assert(params.sampling.seed == 42);

        argv = {"binary_name", "--temp", "0.8"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
        assert(params.sampling.temp == 0.8f);

        argv = {"binary_name", "--top-p", "0.9"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
        assert(params.sampling.top_p == 0.9f);
    }

    printf("test-arg-parser: testing boundary conditions\n\n");

    {
        common_params params;
        std::vector<std::string> argv;
        auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
            std::vector<char *> res;
            for (auto & arg : argv) {
                res.push_back(const_cast<char *>(arg.data()));
            }
            return res;
        };

        argv = {"binary_name", "-c", "0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--temp", "0.0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--temp", "1.0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
        assert(params.sampling.temp == 1.0f);
    }

    printf("test-arg-parser: all tests OK\n\n");
}
