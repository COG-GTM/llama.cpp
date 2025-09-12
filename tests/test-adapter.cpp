#include "../src/llama-adapter.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstring>

static ggml_tensor * create_mock_tensor(int ne0, int ne1 = 1, int ne2 = 1, int ne3 = 1, const char* name = nullptr) {
    static std::vector<ggml_tensor> mock_tensors;
    mock_tensors.emplace_back();
    ggml_tensor* tensor = &mock_tensors.back();
    
    tensor->ne[0] = ne0;
    tensor->ne[1] = ne1;
    tensor->ne[2] = ne2;
    tensor->ne[3] = ne3;
    
    if (name) {
        strncpy(tensor->name, name, sizeof(tensor->name) - 1);
        tensor->name[sizeof(tensor->name) - 1] = '\0';
    } else {
        tensor->name[0] = '\0';
    }
    
    return tensor;
}

static void test_lora_weight_get_scale() {
    std::cout << "Testing llama_adapter_lora_weight::get_scale..." << std::endl;
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(16);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 32.0f;
        float adapter_scale = 1.0f;
        float expected_scale = adapter_scale * alpha / 16.0f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::abs(actual_scale - expected_scale) < 1e-6f);
        std::cout << "  ✓ Basic scale calculation with alpha" << std::endl;
    }
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(8);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 0.0f;
        float adapter_scale = 0.5f;
        float expected_scale = adapter_scale;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::abs(actual_scale - expected_scale) < 1e-6f);
        std::cout << "  ✓ Scale calculation without alpha" << std::endl;
    }
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(64);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 16.0f;
        float adapter_scale = 2.0f;
        float expected_scale = adapter_scale * alpha / 64.0f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::abs(actual_scale - expected_scale) < 1e-6f);
        std::cout << "  ✓ Different rank values" << std::endl;
    }
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(1);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 1.0f;
        float adapter_scale = 1.0f;
        float expected_scale = adapter_scale * alpha / 1.0f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::abs(actual_scale - expected_scale) < 1e-6f);
        std::cout << "  ✓ Edge case - rank = 1" << std::endl;
    }
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(1024);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 512.0f;
        float adapter_scale = 1.0f;
        float expected_scale = adapter_scale * alpha / 1024.0f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::abs(actual_scale - expected_scale) < 1e-6f);
        std::cout << "  ✓ Large rank value" << std::endl;
    }
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(16);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 32.0f;
        float adapter_scale = 0.0f;
        float expected_scale = 0.0f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::abs(actual_scale - expected_scale) < 1e-6f);
        std::cout << "  ✓ Zero adapter_scale" << std::endl;
    }
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(16);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 32.0f;
        float adapter_scale = -1.0f;
        float expected_scale = adapter_scale * alpha / 16.0f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::abs(actual_scale - expected_scale) < 1e-6f);
        std::cout << "  ✓ Negative adapter_scale" << std::endl;
    }
}

static void test_lora_weight_constructors() {
    std::cout << "Testing llama_adapter_lora_weight constructors..." << std::endl;
    
    {
        llama_adapter_lora_weight weight;
        assert(weight.a == nullptr);
        assert(weight.b == nullptr);
        std::cout << "  ✓ Default constructor" << std::endl;
    }
    
    {
        ggml_tensor * tensor_a = create_mock_tensor(16, 32);
        ggml_tensor * tensor_b = create_mock_tensor(32, 64);
        llama_adapter_lora_weight weight(tensor_a, tensor_b);
        
        assert(weight.a == tensor_a);
        assert(weight.b == tensor_b);
        std::cout << "  ✓ Parameterized constructor" << std::endl;
    }
}

static void test_lora_adapter_basic() {
    std::cout << "Testing llama_adapter_lora basic functionality..." << std::endl;
    
    {
        llama_adapter_lora adapter;
        assert(adapter.ab_map.empty());
        assert(adapter.gguf_kv.empty());
        std::cout << "  ✓ Default constructor" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        ggml_tensor * tensor_a = create_mock_tensor(16, 32);
        ggml_tensor * tensor_b = create_mock_tensor(32, 64);
        llama_adapter_lora_weight weight(tensor_a, tensor_b);
        
        adapter.ab_map["test_weight"] = weight;
        assert(adapter.ab_map.size() == 1);
        assert(adapter.ab_map["test_weight"].a == tensor_a);
        assert(adapter.ab_map["test_weight"].b == tensor_b);
        std::cout << "  ✓ Adding entries to ab_map" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.alpha = 16.0f;
        assert(adapter.alpha == 16.0f);
        std::cout << "  ✓ Alpha value assignment" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["model_name"] = "test_model";
        adapter.gguf_kv["version"] = "1.0";
        
        assert(adapter.gguf_kv.size() == 2);
        assert(adapter.gguf_kv["model_name"] == "test_model");
        assert(adapter.gguf_kv["version"] == "1.0");
        std::cout << "  ✓ GGUF metadata" << std::endl;
    }
}

static void test_lora_adapter_get_weight() {
    std::cout << "Testing llama_adapter_lora::get_weight..." << std::endl;
    
    {
        llama_adapter_lora adapter;
        ggml_tensor * tensor_a = create_mock_tensor(16, 32, 1, 1, "test.lora_a");
        ggml_tensor * tensor_b = create_mock_tensor(32, 64, 1, 1, "test.lora_b");
        llama_adapter_lora_weight weight(tensor_a, tensor_b);
        
        adapter.ab_map["test"] = weight;
        
        ggml_tensor * query_tensor = create_mock_tensor(1, 1, 1, 1, "test");
        llama_adapter_lora_weight * found_weight = adapter.get_weight(query_tensor);
        
        assert(found_weight != nullptr);
        assert(found_weight->a == tensor_a);
        assert(found_weight->b == tensor_b);
        std::cout << "  ✓ Found existing weight" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        ggml_tensor * query_tensor = create_mock_tensor(1, 1, 1, 1, "nonexistent");
        llama_adapter_lora_weight * found_weight = adapter.get_weight(query_tensor);
        
        assert(found_weight == nullptr);
        std::cout << "  ✓ Returns nullptr for nonexistent weight" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        ggml_tensor * query_tensor = create_mock_tensor(1, 1, 1, 1, "");
        llama_adapter_lora_weight * found_weight = adapter.get_weight(query_tensor);
        
        assert(found_weight == nullptr);
        std::cout << "  ✓ Returns nullptr for empty name" << std::endl;
    }
}

static void test_cvec_tensor_for() {
    std::cout << "Testing llama_adapter_cvec::tensor_for..." << std::endl;
    
    {
        llama_adapter_cvec cvec;
        
        ggml_tensor * result = cvec.tensor_for(-1);
        assert(result == nullptr);
        std::cout << "  ✓ Returns nullptr for negative layer" << std::endl;
    }
    
    {
        llama_adapter_cvec cvec;
        
        ggml_tensor * result = cvec.tensor_for(0);
        assert(result == nullptr);
        std::cout << "  ✓ Returns nullptr for uninitialized cvec" << std::endl;
    }
}

static void test_cvec_apply_to() {
    std::cout << "Testing llama_adapter_cvec::apply_to..." << std::endl;
    
    {
        llama_adapter_cvec cvec;
        ggml_tensor * input_tensor = create_mock_tensor(512);
        
        ggml_tensor * result = cvec.apply_to(nullptr, input_tensor, 0);
        assert(result == input_tensor);
        std::cout << "  ✓ Returns input tensor when no layer tensor available" << std::endl;
    }
}

static void test_metadata_functions() {
    std::cout << "Testing metadata functions..." << std::endl;
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["key1"] = "value1";
        adapter.gguf_kv["key2"] = "value2";
        adapter.gguf_kv["key3"] = "value3";
        
        int32_t count = llama_adapter_meta_count(&adapter);
        assert(count == 3);
        std::cout << "  ✓ llama_adapter_meta_count returns correct count" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        
        int32_t count = llama_adapter_meta_count(&adapter);
        assert(count == 0);
        std::cout << "  ✓ llama_adapter_meta_count returns 0 for empty adapter" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["test_key"] = "test_value";
        
        char buf[256];
        int32_t result = llama_adapter_meta_val_str(&adapter, "test_key", buf, sizeof(buf));
        
        assert(result > 0);
        assert(strcmp(buf, "test_value") == 0);
        std::cout << "  ✓ llama_adapter_meta_val_str retrieves existing key" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        
        char buf[256];
        int32_t result = llama_adapter_meta_val_str(&adapter, "nonexistent", buf, sizeof(buf));
        
        assert(result == -1);
        assert(buf[0] == '\0');
        std::cout << "  ✓ llama_adapter_meta_val_str returns -1 for nonexistent key" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["key1"] = "value1";
        adapter.gguf_kv["key2"] = "value2";
        
        char buf[256];
        int32_t result = llama_adapter_meta_key_by_index(&adapter, 0, buf, sizeof(buf));
        
        assert(result > 0);
        assert(strlen(buf) > 0);
        std::cout << "  ✓ llama_adapter_meta_key_by_index retrieves valid index" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        
        char buf[256];
        int32_t result = llama_adapter_meta_key_by_index(&adapter, 0, buf, sizeof(buf));
        
        assert(result == -1);
        assert(buf[0] == '\0');
        std::cout << "  ✓ llama_adapter_meta_key_by_index returns -1 for invalid index" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["key1"] = "value1";
        
        char buf[256];
        int32_t result = llama_adapter_meta_key_by_index(&adapter, -1, buf, sizeof(buf));
        
        assert(result == -1);
        assert(buf[0] == '\0');
        std::cout << "  ✓ llama_adapter_meta_key_by_index handles negative index" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["key1"] = "value1";
        adapter.gguf_kv["key2"] = "value2";
        
        char buf[256];
        int32_t result = llama_adapter_meta_val_str_by_index(&adapter, 0, buf, sizeof(buf));
        
        assert(result > 0);
        assert(strlen(buf) > 0);
        std::cout << "  ✓ llama_adapter_meta_val_str_by_index retrieves valid index" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        
        char buf[256];
        int32_t result = llama_adapter_meta_val_str_by_index(&adapter, 0, buf, sizeof(buf));
        
        assert(result == -1);
        assert(buf[0] == '\0');
        std::cout << "  ✓ llama_adapter_meta_val_str_by_index returns -1 for invalid index" << std::endl;
    }
}

static void test_lora_free() {
    std::cout << "Testing llama_adapter_lora_free..." << std::endl;
    
    {
        llama_adapter_lora * adapter = new llama_adapter_lora();
        adapter->alpha = 1.0f;
        adapter->gguf_kv["test"] = "value";
        
        llama_adapter_lora_free(adapter);
        std::cout << "  ✓ llama_adapter_lora_free completes without error" << std::endl;
    }
    
    {
        llama_adapter_lora_free(nullptr);
        std::cout << "  ✓ llama_adapter_lora_free handles nullptr" << std::endl;
    }
}

static void test_buffer_edge_cases() {
    std::cout << "Testing buffer edge cases..." << std::endl;
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["test_key"] = "test_value";
        
        char buf[5];
        int32_t result = llama_adapter_meta_val_str(&adapter, "test_key", buf, sizeof(buf));
        
        assert(result > 0);
        assert(strlen(buf) < sizeof(buf));
        std::cout << "  ✓ llama_adapter_meta_val_str handles small buffer" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["test_key"] = "test_value";
        
        int32_t result = llama_adapter_meta_val_str(&adapter, "test_key", nullptr, 0);
        
        assert(result > 0);
        std::cout << "  ✓ llama_adapter_meta_val_str handles null buffer" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["key1"] = "value1";
        
        char buf[5];
        int32_t result = llama_adapter_meta_key_by_index(&adapter, 0, buf, sizeof(buf));
        
        assert(result > 0);
        assert(strlen(buf) < sizeof(buf));
        std::cout << "  ✓ llama_adapter_meta_key_by_index handles small buffer" << std::endl;
    }
}

static void test_cvec_boundary_conditions() {
    std::cout << "Testing llama_adapter_cvec boundary conditions..." << std::endl;
    
    {
        llama_adapter_cvec cvec;
        
        ggml_tensor * result = cvec.tensor_for(0);
        assert(result == nullptr);
        std::cout << "  ✓ Returns nullptr for uninitialized cvec at layer 0" << std::endl;
    }
    
    {
        llama_adapter_cvec cvec;
        
        ggml_tensor * result = cvec.tensor_for(100);
        assert(result == nullptr);
        std::cout << "  ✓ Returns nullptr for uninitialized cvec at high layer" << std::endl;
    }
    
    {
        llama_adapter_cvec cvec;
        ggml_tensor * input_tensor = create_mock_tensor(512);
        
        ggml_tensor * result = cvec.apply_to(nullptr, input_tensor, 0);
        assert(result == input_tensor);
        std::cout << "  ✓ apply_to returns input tensor when cvec uninitialized" << std::endl;
    }
    
    {
        llama_adapter_cvec cvec;
        ggml_tensor * input_tensor = create_mock_tensor(512);
        
        ggml_tensor * result = cvec.apply_to(nullptr, input_tensor, 50);
        assert(result == input_tensor);
        std::cout << "  ✓ apply_to returns input tensor for high layer index" << std::endl;
    }
}

static void test_cvec_apply_functionality() {
    std::cout << "Testing llama_adapter_cvec::apply functionality..." << std::endl;
    
    {
        llama_adapter_cvec cvec;
        
        bool result = cvec.apply(*(llama_model*)nullptr, nullptr, 0, 0, 0, 0);
        assert(result == true);
        std::cout << "  ✓ apply with nullptr data returns true" << std::endl;
    }
}

static void test_lora_weight_edge_cases() {
    std::cout << "Testing llama_adapter_lora_weight edge cases..." << std::endl;
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(0);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 32.0f;
        float adapter_scale = 1.0f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::isinf(actual_scale) || std::isnan(actual_scale));
        std::cout << "  ✓ Division by zero rank handled" << std::endl;
    }
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(1);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 0.0f;
        float adapter_scale = 2.5f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(actual_scale == adapter_scale);
        std::cout << "  ✓ Zero alpha defaults to adapter_scale" << std::endl;
    }
}

static void test_lora_adapter_advanced() {
    std::cout << "Testing llama_adapter_lora advanced functionality..." << std::endl;
    
    {
        llama_adapter_lora adapter;
        
        ggml_tensor * tensor_with_long_name = create_mock_tensor(1, 1, 1, 1, "very_long_tensor_name_that_exceeds_normal_limits");
        llama_adapter_lora_weight * result = adapter.get_weight(tensor_with_long_name);
        
        assert(result == nullptr);
        std::cout << "  ✓ get_weight handles long tensor names" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["key_with_special_chars"] = "value with spaces and symbols !@#$%";
        adapter.gguf_kv["unicode_key"] = "value_with_unicode_αβγ";
        adapter.gguf_kv["empty_value"] = "";
        
        assert(adapter.gguf_kv.size() == 3);
        assert(adapter.gguf_kv["empty_value"] == "");
        std::cout << "  ✓ GGUF metadata handles special characters and empty values" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        for (int i = 0; i < 1000; ++i) {
            adapter.gguf_kv["key_" + std::to_string(i)] = "value_" + std::to_string(i);
        }
        
        assert(adapter.gguf_kv.size() == 1000);
        assert(llama_adapter_meta_count(&adapter) == 1000);
        std::cout << "  ✓ Large number of metadata entries handled" << std::endl;
    }
}

static void test_metadata_advanced() {
    std::cout << "Testing metadata functions advanced cases..." << std::endl;
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["key1"] = "value1";
        adapter.gguf_kv["key2"] = "value2";
        adapter.gguf_kv["key3"] = "value3";
        
        char buf[256];
        for (int i = 0; i < 3; ++i) {
            int32_t result = llama_adapter_meta_key_by_index(&adapter, i, buf, sizeof(buf));
            assert(result > 0);
            assert(strlen(buf) > 0);
        }
        
        int32_t result = llama_adapter_meta_key_by_index(&adapter, 3, buf, sizeof(buf));
        assert(result == -1);
        std::cout << "  ✓ meta_key_by_index boundary testing" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["very_long_key_name_that_might_cause_buffer_issues"] = "short_value";
        
        char small_buf[10];
        int32_t result = llama_adapter_meta_key_by_index(&adapter, 0, small_buf, sizeof(small_buf));
        
        assert(result > 0);
        assert(strlen(small_buf) < sizeof(small_buf));
        std::cout << "  ✓ Long key names with small buffers handled" << std::endl;
    }
    
    {
        llama_adapter_lora adapter;
        adapter.gguf_kv["key"] = std::string(1000, 'x');
        
        char buf[256];
        int32_t result = llama_adapter_meta_val_str(&adapter, "key", buf, sizeof(buf));
        
        assert(result > 0);
        assert(strlen(buf) < sizeof(buf));
        std::cout << "  ✓ Very long values truncated properly" << std::endl;
    }
}

static void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(16);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 1e-10f;
        float adapter_scale = 1e-10f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::isfinite(actual_scale));
        std::cout << "  ✓ Very small floating point values" << std::endl;
    }
    
    {
        ggml_tensor * tensor_b = create_mock_tensor(1);
        llama_adapter_lora_weight weight(nullptr, tensor_b);
        
        float alpha = 1e6f;
        float adapter_scale = 1e6f;
        float actual_scale = weight.get_scale(alpha, adapter_scale);
        
        assert(std::isfinite(actual_scale));
        std::cout << "  ✓ Large floating point values" << std::endl;
    }
    
    {
        llama_adapter_cvec cvec;
        
        ggml_tensor * result = cvec.tensor_for(1000000);
        assert(result == nullptr);
        std::cout << "  ✓ Very large layer index" << std::endl;
    }
}

int main() {
    std::cout << "Running llama-adapter tests..." << std::endl;
    
    try {
        test_lora_weight_get_scale();
        test_lora_weight_constructors();
        test_lora_adapter_basic();
        test_lora_adapter_get_weight();
        test_cvec_tensor_for();
        test_cvec_apply_to();
        test_metadata_functions();
        test_lora_free();
        test_buffer_edge_cases();
        test_cvec_boundary_conditions();
        test_cvec_apply_functionality();
        test_lora_weight_edge_cases();
        test_lora_adapter_advanced();
        test_metadata_advanced();
        test_edge_cases();
        
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
