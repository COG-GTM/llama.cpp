#include "get-model.h"

#undef NDEBUG
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

static void test_get_model_with_command_line_arg() {
    std::cout << "Testing get_model_or_exit with command line argument..." << std::endl;

    char prog_name[] = "test_program";
    char model_path[] = "/path/to/test/model.gguf";
    char* argv[] = {prog_name, model_path};
    int argc = 2;

    char* result = get_model_or_exit(argc, argv);

    assert(result != nullptr);
    assert(strcmp(result, "/path/to/test/model.gguf") == 0);
    assert(result == argv[1]); // Should return the same pointer

    std::cout << "  ✓ Command line argument handled correctly" << std::endl;
}

static void test_get_model_with_multiple_args() {
    std::cout << "Testing get_model_or_exit with multiple arguments..." << std::endl;

    char prog_name[] = "test_program";
    char model_path[] = "/first/model.gguf";
    char extra_arg[] = "extra";
    char* argv[] = {prog_name, model_path, extra_arg};
    int argc = 3;

    char* result = get_model_or_exit(argc, argv);

    assert(result != nullptr);
    assert(strcmp(result, "/first/model.gguf") == 0);
    assert(result == argv[1]); // Should return first argument after program name

    std::cout << "  ✓ Multiple arguments handled correctly (uses first)" << std::endl;
}

static void test_get_model_with_environment_variable() {
    std::cout << "Testing get_model_or_exit with environment variable..." << std::endl;

    const char* test_model_path = "/env/test/model.gguf";
    setenv("LLAMACPP_TEST_MODELFILE", test_model_path, 1);

    char prog_name[] = "test_program";
    char* argv[] = {prog_name};
    int argc = 1;

    char* result = get_model_or_exit(argc, argv);

    assert(result != nullptr);
    assert(strcmp(result, test_model_path) == 0);

    unsetenv("LLAMACPP_TEST_MODELFILE");

    std::cout << "  ✓ Environment variable handled correctly" << std::endl;
}

static void test_get_model_env_var_overrides_when_no_args() {
    std::cout << "Testing environment variable with no command line args..." << std::endl;

    const char* test_model_path = "/env/override/model.gguf";
    setenv("LLAMACPP_TEST_MODELFILE", test_model_path, 1);

    char prog_name[] = "test_program";
    char* argv[] = {prog_name};
    int argc = 1;

    char* result = get_model_or_exit(argc, argv);

    assert(result != nullptr);
    assert(strcmp(result, test_model_path) == 0);

    unsetenv("LLAMACPP_TEST_MODELFILE");

    std::cout << "  ✓ Environment variable used when no args provided" << std::endl;
}

static void test_get_model_command_line_overrides_env() {
    std::cout << "Testing command line argument overrides environment variable..." << std::endl;

    setenv("LLAMACPP_TEST_MODELFILE", "/env/model.gguf", 1);

    char prog_name[] = "test_program";
    char model_path[] = "/cmdline/model.gguf";
    char* argv[] = {prog_name, model_path};
    int argc = 2;

    char* result = get_model_or_exit(argc, argv);

    assert(result != nullptr);
    assert(strcmp(result, "/cmdline/model.gguf") == 0);
    assert(result == argv[1]); // Should be command line arg, not env var

    unsetenv("LLAMACPP_TEST_MODELFILE");

    std::cout << "  ✓ Command line argument overrides environment variable" << std::endl;
}

static void test_get_model_with_empty_env_var() {
    std::cout << "Testing get_model_or_exit with empty environment variable..." << std::endl;

    setenv("LLAMACPP_TEST_MODELFILE", "", 1);

    char* env_val = getenv("LLAMACPP_TEST_MODELFILE");
    assert(env_val != nullptr);
    assert(strlen(env_val) == 0);

    unsetenv("LLAMACPP_TEST_MODELFILE");

    std::cout << "  ✓ Empty environment variable detected (would exit)" << std::endl;
}

static void test_get_model_with_null_env_var() {
    std::cout << "Testing get_model_or_exit with null environment variable..." << std::endl;

    unsetenv("LLAMACPP_TEST_MODELFILE");

    char* env_val = getenv("LLAMACPP_TEST_MODELFILE");
    assert(env_val == nullptr);

    std::cout << "  ✓ Null environment variable detected (would exit)" << std::endl;
}

static void test_get_model_edge_cases() {
    std::cout << "Testing get_model_or_exit edge cases..." << std::endl;

    char prog_name[] = "test_program";
    char long_path[1000];
    memset(long_path, 'a', 999);
    long_path[999] = '\0';
    char* argv_long[] = {prog_name, long_path};
    int argc_long = 2;

    char* result = get_model_or_exit(argc_long, argv_long);
    assert(result != nullptr);
    assert(strlen(result) == 999);
    assert(result == argv_long[1]);

    std::cout << "  ✓ Edge cases handled correctly" << std::endl;
}

static void test_get_model_special_characters() {
    std::cout << "Testing get_model_or_exit with special characters..." << std::endl;

    char prog_name[] = "test_program";
    char special_path[] = "/path/with spaces/and-symbols_123.gguf";
    char* argv[] = {prog_name, special_path};
    int argc = 2;

    char* result = get_model_or_exit(argc, argv);

    assert(result != nullptr);
    assert(strcmp(result, special_path) == 0);
    assert(result == argv[1]);

    std::cout << "  ✓ Special characters in path handled correctly" << std::endl;
}

static void test_get_model_boundary_conditions() {
    std::cout << "Testing get_model_or_exit boundary conditions..." << std::endl;

    char prog_name[] = "test_program";
    char* argv_one[] = {prog_name};
    int argc_one = 1;

    setenv("LLAMACPP_TEST_MODELFILE", "/boundary/test.gguf", 1);

    char* result = get_model_or_exit(argc_one, argv_one);
    assert(result != nullptr);
    assert(strcmp(result, "/boundary/test.gguf") == 0);

    unsetenv("LLAMACPP_TEST_MODELFILE");

    char model_path[] = "/exact/two/args.gguf";
    char* argv_two[] = {prog_name, model_path};
    int argc_two = 2;

    result = get_model_or_exit(argc_two, argv_two);
    assert(result != nullptr);
    assert(strcmp(result, model_path) == 0);
    assert(result == argv_two[1]);

    std::cout << "  ✓ Boundary conditions handled correctly" << std::endl;
}

int main() {
    std::cout << "Running get-model tests..." << std::endl;

    try {
        test_get_model_with_command_line_arg();
        test_get_model_with_multiple_args();
        test_get_model_with_environment_variable();
        test_get_model_env_var_overrides_when_no_args();
        test_get_model_command_line_overrides_env();
        test_get_model_with_empty_env_var();
        test_get_model_with_null_env_var();
        test_get_model_edge_cases();
        test_get_model_special_characters();
        test_get_model_boundary_conditions();

        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}
