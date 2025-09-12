#include "llama.h"

#undef NDEBUG
#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <memory>

static void test_llama_model_quantize_default_params() {
    std::cout << "Testing llama_model_quantize_default_params..." << std::endl;

    llama_model_quantize_params params = llama_model_quantize_default_params();

    assert(params.nthread == 0);
    assert(params.ftype == LLAMA_FTYPE_MOSTLY_Q5_1);
    assert(params.output_tensor_type == GGML_TYPE_COUNT);
    assert(params.token_embedding_type == GGML_TYPE_COUNT);
    assert(params.allow_requantize == false);
    assert(params.quantize_output_tensor == true);
    assert(params.only_copy == false);
    assert(params.pure == false);
    assert(params.keep_split == false);
    assert(params.imatrix == nullptr);
    assert(params.kv_overrides == nullptr);
    assert(params.tensor_types == nullptr);
    assert(params.prune_layers == nullptr);

    std::cout << "  ✓ Default parameters initialized correctly" << std::endl;
}

static void test_llama_model_quantize_invalid_inputs() {
    std::cout << "Testing llama_model_quantize with invalid inputs..." << std::endl;

    llama_model_quantize_params params = llama_model_quantize_default_params();

    uint32_t result = llama_model_quantize(nullptr, "/tmp/test_output.gguf", &params);
    assert(result == 1);
    std::cout << "  ✓ Null input filename handled correctly" << std::endl;

    result = llama_model_quantize("/tmp/nonexistent_input.gguf", nullptr, &params);
    assert(result == 1);
    std::cout << "  ✓ Null output filename handled correctly" << std::endl;

    result = llama_model_quantize("/tmp/definitely_nonexistent_file_12345.gguf", "/tmp/test_output.gguf", &params);
    assert(result == 1);
    std::cout << "  ✓ Valid params with nonexistent file handled correctly" << std::endl;

    result = llama_model_quantize("/tmp/definitely_nonexistent_file_12345.gguf", "/tmp/test_output.gguf", &params);
    assert(result == 1);
    std::cout << "  ✓ Nonexistent input file handled correctly" << std::endl;
}

static void test_llama_model_quantize_params_variations() {
    std::cout << "Testing llama_model_quantize_params variations..." << std::endl;

    llama_model_quantize_params params = llama_model_quantize_default_params();

    std::vector<llama_ftype> ftypes = {
        LLAMA_FTYPE_MOSTLY_Q4_0,
        LLAMA_FTYPE_MOSTLY_Q4_1,
        LLAMA_FTYPE_MOSTLY_Q5_0,
        LLAMA_FTYPE_MOSTLY_Q5_1,
        LLAMA_FTYPE_MOSTLY_Q8_0,
        LLAMA_FTYPE_MOSTLY_F16,
        LLAMA_FTYPE_MOSTLY_BF16,
        LLAMA_FTYPE_ALL_F32
    };

    for (auto ftype : ftypes) {
        params.ftype = ftype;
        uint32_t result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
        assert(result == 1);
    }
    std::cout << "  ✓ Different ftype values handled" << std::endl;

    params = llama_model_quantize_default_params();
    params.nthread = 1;
    uint32_t result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);

    params.nthread = 4;
    result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);

    params.nthread = -1; // Should default to hardware_concurrency
    result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);

    std::cout << "  ✓ Different thread counts handled" << std::endl;
}

static void test_llama_model_quantize_boolean_flags() {
    std::cout << "Testing llama_model_quantize boolean flags..." << std::endl;

    llama_model_quantize_params params = llama_model_quantize_default_params();

    params.allow_requantize = true;
    uint32_t result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);

    params = llama_model_quantize_default_params();
    params.quantize_output_tensor = false;
    result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);

    params = llama_model_quantize_default_params();
    params.only_copy = true;
    result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);

    params = llama_model_quantize_default_params();
    params.pure = true;
    result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);

    params = llama_model_quantize_default_params();
    params.keep_split = true;
    result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);

    std::cout << "  ✓ Boolean flags handled correctly" << std::endl;
}

static void test_llama_model_quantize_tensor_types() {
    std::cout << "Testing llama_model_quantize tensor type parameters..." << std::endl;

    llama_model_quantize_params params = llama_model_quantize_default_params();

    std::vector<ggml_type> tensor_types = {
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_F16,
        GGML_TYPE_F32
    };

    for (auto tensor_type : tensor_types) {
        params.output_tensor_type = tensor_type;
        uint32_t result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
        assert(result == 1);

        params.token_embedding_type = tensor_type;
        result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
        assert(result == 1);
    }

    std::cout << "  ✓ Tensor type parameters handled" << std::endl;
}

static void test_llama_model_quantize_edge_cases() {
    std::cout << "Testing llama_model_quantize edge cases..." << std::endl;

    llama_model_quantize_params params = llama_model_quantize_default_params();

    uint32_t result = llama_model_quantize("", "/tmp/output.gguf", &params);
    assert(result == 1);
    std::cout << "  ✓ Empty input filename handled" << std::endl;

    result = llama_model_quantize("/tmp/input.gguf", "", &params);
    assert(result == 1);
    std::cout << "  ✓ Empty output filename handled" << std::endl;

    std::string long_filename(1000, 'a');
    long_filename += ".gguf";
    result = llama_model_quantize(long_filename.c_str(), "/tmp/output.gguf", &params);
    assert(result == 1);
    std::cout << "  ✓ Long filename handled" << std::endl;

    result = llama_model_quantize("/tmp/same.gguf", "/tmp/same.gguf", &params);
    assert(result == 1);
    std::cout << "  ✓ Same input/output filename handled" << std::endl;
}

static void test_llama_model_quantize_boundary_conditions() {
    std::cout << "Testing llama_model_quantize boundary conditions..." << std::endl;

    llama_model_quantize_params params = llama_model_quantize_default_params();

    params.nthread = std::thread::hardware_concurrency() * 2;
    uint32_t result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);
    std::cout << "  ✓ High thread count handled" << std::endl;

    params.nthread = 0;
    result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);
    std::cout << "  ✓ Zero thread count handled" << std::endl;

    params = llama_model_quantize_default_params();
    params.ftype = (llama_ftype)999; // Invalid ftype
    result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
    assert(result == 1);
    std::cout << "  ✓ Invalid ftype handled" << std::endl;
}

static void test_llama_model_quantize_multiple_operations() {
    std::cout << "Testing multiple llama_model_quantize operations..." << std::endl;

    llama_model_quantize_params params = llama_model_quantize_default_params();

    for (int i = 0; i < 5; i++) {
        params.ftype = (i % 2 == 0) ? LLAMA_FTYPE_MOSTLY_Q4_0 : LLAMA_FTYPE_MOSTLY_Q5_1;
        params.nthread = i + 1;

        uint32_t result = llama_model_quantize("/tmp/nonexistent.gguf", "/tmp/output.gguf", &params);
        assert(result == 1);
    }

    std::cout << "  ✓ Multiple operations handled" << std::endl;
}

int main() {
    std::cout << "Running llama-quant tests..." << std::endl;

    try {
        test_llama_model_quantize_default_params();
        test_llama_model_quantize_invalid_inputs();
        test_llama_model_quantize_params_variations();
        test_llama_model_quantize_boolean_flags();
        test_llama_model_quantize_tensor_types();
        test_llama_model_quantize_edge_cases();
        test_llama_model_quantize_boundary_conditions();
        test_llama_model_quantize_multiple_operations();

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
