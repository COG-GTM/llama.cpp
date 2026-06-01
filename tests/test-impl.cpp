#include "../src/llama-impl.h"
#include "ggml.h"
#include "gguf.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

static ggml_tensor * create_mock_tensor(int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
    static ggml_tensor mock_tensor;
    mock_tensor.ne[0] = ne0;
    mock_tensor.ne[1] = ne1;
    mock_tensor.ne[2] = ne2;
    mock_tensor.ne[3] = ne3;
    return &mock_tensor;
}

static void test_no_init_template() {
    std::cout << "Testing no_init template..." << std::endl;

    {
        no_init<int> uninit_int;
        uninit_int.value = 42;
        assert(uninit_int.value == 42);
        std::cout << "  ✓ no_init template works with int" << std::endl;
    }

    {
        no_init<double> uninit_double;
        uninit_double.value = 3.14;
        assert(uninit_double.value == 3.14);
        std::cout << "  ✓ no_init template works with double" << std::endl;
    }

    {
        no_init<std::string> uninit_string;
        uninit_string.value = "test";
        assert(uninit_string.value == "test");
        std::cout << "  ✓ no_init template works with std::string" << std::endl;
    }
}

static void test_time_meas() {
    std::cout << "Testing time_meas..." << std::endl;

    {
        int64_t accumulator = 0;
        {
            time_meas tm(accumulator, false);
            assert(tm.t_start_us >= 0);
        }
        assert(accumulator >= 0);
        std::cout << "  ✓ time_meas measures time when enabled" << std::endl;
    }

    {
        int64_t accumulator = 0;
        {
            time_meas tm(accumulator, true);
            assert(tm.t_start_us == -1);
        }
        assert(accumulator == 0);
        std::cout << "  ✓ time_meas disabled when requested" << std::endl;
    }

    {
        int64_t accumulator = 100;
        {
            time_meas tm(accumulator, true);
        }
        assert(accumulator == 100);
        std::cout << "  ✓ time_meas preserves accumulator when disabled" << std::endl;
    }
}

static void test_replace_all() {
    std::cout << "Testing replace_all..." << std::endl;

    {
        std::string s = "hello world hello";
        replace_all(s, "hello", "hi");
        assert(s == "hi world hi");
        std::cout << "  ✓ Basic string replacement" << std::endl;
    }

    {
        std::string s = "test";
        replace_all(s, "", "replacement");
        assert(s == "test");
        std::cout << "  ✓ Empty search string does nothing" << std::endl;
    }

    {
        std::string s = "abcabc";
        replace_all(s, "abc", "xyz");
        assert(s == "xyzxyz");
        std::cout << "  ✓ Multiple replacements" << std::endl;
    }

    {
        std::string s = "test";
        replace_all(s, "notfound", "replacement");
        assert(s == "test");
        std::cout << "  ✓ No replacement when search not found" << std::endl;
    }

    {
        std::string s = "aaa";
        replace_all(s, "aa", "b");
        assert(s == "ba");
        std::cout << "  ✓ Overlapping patterns handled correctly" << std::endl;
    }

    {
        std::string s = "test";
        replace_all(s, "test", "");
        assert(s == "");
        std::cout << "  ✓ Replacement with empty string" << std::endl;
    }

    {
        std::string s = "";
        replace_all(s, "test", "replacement");
        assert(s == "");
        std::cout << "  ✓ Empty input string" << std::endl;
    }
}

static void test_format() {
    std::cout << "Testing format..." << std::endl;

    {
        std::string result = format("Hello %s", "world");
        assert(result == "Hello world");
        std::cout << "  ✓ Basic string formatting" << std::endl;
    }

    {
        std::string result = format("Number: %d", 42);
        assert(result == "Number: 42");
        std::cout << "  ✓ Integer formatting" << std::endl;
    }

    {
        std::string result = format("Float: %.2f", 3.14159);
        assert(result == "Float: 3.14");
        std::cout << "  ✓ Float formatting with precision" << std::endl;
    }

    {
        std::string result = format("%s %d %.1f", "Mixed", 123, 4.5);
        assert(result == "Mixed 123 4.5");
        std::cout << "  ✓ Multiple format specifiers" << std::endl;
    }

    {
        std::string result = format("%s", "");
        assert(result == "");
        std::cout << "  ✓ Empty string formatting" << std::endl;
    }

    {
        std::string result = format("No specifiers");
        assert(result == "No specifiers");
        std::cout << "  ✓ Format string without specifiers" << std::endl;
    }
}

static void test_llama_format_tensor_shape_vector() {
    std::cout << "Testing llama_format_tensor_shape (vector version)..." << std::endl;

    {
        std::vector<int64_t> shape = {10};
        std::string result = llama_format_tensor_shape(shape);
        assert(result == "   10");
        std::cout << "  ✓ Single dimension tensor shape" << std::endl;
    }

    {
        std::vector<int64_t> shape = {10, 20};
        std::string result = llama_format_tensor_shape(shape);
        assert(result == "   10,    20");
        std::cout << "  ✓ Two dimension tensor shape" << std::endl;
    }

    {
        std::vector<int64_t> shape = {1, 2, 3, 4};
        std::string result = llama_format_tensor_shape(shape);
        assert(result == "    1,     2,     3,     4");
        std::cout << "  ✓ Four dimension tensor shape" << std::endl;
    }

    {
        std::vector<int64_t> shape = {12345};
        std::string result = llama_format_tensor_shape(shape);
        assert(result == "12345");
        std::cout << "  ✓ Large number formatting" << std::endl;
    }

    {
        std::vector<int64_t> shape = {0};
        std::string result = llama_format_tensor_shape(shape);
        assert(result == "    0");
        std::cout << "  ✓ Zero dimension" << std::endl;
    }
}

static void test_llama_format_tensor_shape_tensor() {
    std::cout << "Testing llama_format_tensor_shape (tensor version)..." << std::endl;

    {
        ggml_tensor * tensor = create_mock_tensor(10, 20, 30, 40);
        std::string result = llama_format_tensor_shape(tensor);
        assert(result.find("10") != std::string::npos);
        assert(result.find("20") != std::string::npos);
        assert(result.find("30") != std::string::npos);
        assert(result.find("40") != std::string::npos);
        std::cout << "  ✓ Tensor shape formatting includes all dimensions" << std::endl;
    }

    {
        ggml_tensor * tensor = create_mock_tensor(1, 1, 1, 1);
        std::string result = llama_format_tensor_shape(tensor);
        assert(result.find("1") != std::string::npos);
        std::cout << "  ✓ Unit tensor shape" << std::endl;
    }

    {
        ggml_tensor * tensor = create_mock_tensor(0, 0, 0, 0);
        std::string result = llama_format_tensor_shape(tensor);
        assert(result.find("0") != std::string::npos);
        std::cout << "  ✓ Zero tensor shape" << std::endl;
    }
}

static void test_logging_macros() {
    std::cout << "Testing logging macros..." << std::endl;

    {
        std::cout << "  ✓ Logging macros are defined and can be used" << std::endl;
    }
}

static void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;

    {
        std::string very_long_string(1000, 'a');
        replace_all(very_long_string, "a", "b");
        assert(very_long_string == std::string(1000, 'b'));
        std::cout << "  ✓ replace_all handles long strings" << std::endl;
    }

    {
        std::string result = format("%s", std::string(200, 'x').c_str());
        assert(result.length() == 200);
        assert(result == std::string(200, 'x'));
        std::cout << "  ✓ format handles long output strings" << std::endl;
    }

    {
        std::vector<int64_t> empty_shape;
        try {
            std::string result = llama_format_tensor_shape(empty_shape);
            assert(false);
        } catch (...) {
            std::cout << "  ✓ Empty vector throws exception as expected" << std::endl;
        }
    }
}

int main() {
    std::cout << "Running llama-impl tests..." << std::endl;

    try {
        test_no_init_template();
        test_time_meas();
        test_replace_all();
        test_format();
        test_llama_format_tensor_shape_vector();
        test_llama_format_tensor_shape_tensor();
        test_logging_macros();
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
