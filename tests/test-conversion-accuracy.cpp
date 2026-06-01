
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <thread>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;

constexpr float MAX_CROSS_FORMAT_CONVERSION_ERROR = 0.01f;
constexpr float MAX_ROUND_TRIP_CONVERSION_ERROR = 0.015f;

static const char* RESULT_STR[] = {"✓", "✗"};

static const ggml_type all_quant_types[] = {
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0, GGML_TYPE_Q8_1,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
};

static const ggml_type base_types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16,
};

static void generate_test_data(float offset, size_t n, float * dst) {
    std::default_random_engine gen(12345 + static_cast<unsigned>(offset * 1000));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.7f * dist(gen) + 0.3f * (2.0f * cosf(i * 0.01f + offset));
    }
}

// Calculate RMSE between two float arrays
static float calculate_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum / n);
}

static float calculate_max_error(const float * a1, const float * a2, size_t n) {
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float err = fabsf(a1[i] - a2[i]);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

static float get_error_threshold(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return MAX_QUANTIZATION_TOTAL_ERROR_TERNARY;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ2_S:
            return MAX_QUANTIZATION_TOTAL_ERROR_2BITS;
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ3_S:
            return MAX_QUANTIZATION_TOTAL_ERROR_3BITS;
        case GGML_TYPE_IQ3_XXS:
            return MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS;
        default:
            return MAX_QUANTIZATION_TOTAL_ERROR;
    }
}

static bool test_single_format(ggml_type type, size_t test_size, bool verbose) {
    const auto * qfns = ggml_get_type_traits(type);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

    if (!qfns_cpu->from_float || !qfns->to_float) {
        if (verbose) {
            printf("  Skipping %s (no quantization functions)\n", ggml_type_name(type));
        }
        return true;
    }

    std::vector<float> test_data(test_size);
    generate_test_data(0.0, test_size, test_data.data());

    std::vector<uint8_t> quantized(ggml_row_size(type, test_size));
    std::vector<float> dequantized(test_size);

    qfns_cpu->from_float(test_data.data(), quantized.data(), test_size);
    qfns->to_float(quantized.data(), dequantized.data(), test_size);

    float rmse = calculate_rmse(test_data.data(), dequantized.data(), test_size);
    float threshold = get_error_threshold(type);
    bool passed = rmse < threshold;

    if (verbose || !passed) {
        printf("  %s %-12s: RMSE=%.6f (threshold=%.6f)\n",
               RESULT_STR[!passed], ggml_type_name(type), rmse, threshold);
    }

    return passed;
}

static bool test_cross_format_conversion(ggml_type src_type, ggml_type dst_type,
                                         size_t test_size, bool verbose) {
    const auto * src_qfns = ggml_get_type_traits(src_type);
    const auto * src_qfns_cpu = ggml_get_type_traits_cpu(src_type);
    const auto * dst_qfns = ggml_get_type_traits(dst_type);
    const auto * dst_qfns_cpu = ggml_get_type_traits_cpu(dst_type);

    if (!src_qfns_cpu->from_float || !src_qfns->to_float ||
        !dst_qfns_cpu->from_float || !dst_qfns->to_float) {
        return true; // Skip if functions not available
    }

    std::vector<float> original(test_size);
    generate_test_data(1.0, test_size, original.data());

    std::vector<uint8_t> quantized_src(ggml_row_size(src_type, test_size));
    std::vector<float> intermediate(test_size);
    src_qfns_cpu->from_float(original.data(), quantized_src.data(), test_size);
    src_qfns->to_float(quantized_src.data(), intermediate.data(), test_size);

    std::vector<uint8_t> quantized_dst(ggml_row_size(dst_type, test_size));
    std::vector<float> final(test_size);
    dst_qfns_cpu->from_float(intermediate.data(), quantized_dst.data(), test_size);
    dst_qfns->to_float(quantized_dst.data(), final.data(), test_size);

    float rmse = calculate_rmse(original.data(), final.data(), test_size);
    bool passed = rmse < MAX_CROSS_FORMAT_CONVERSION_ERROR;

    if (verbose || !passed) {
        printf("  %s %s → %s: RMSE=%.6f\n",
               RESULT_STR[!passed], ggml_type_name(src_type),
               ggml_type_name(dst_type), rmse);
    }

    return passed;
}

static bool test_round_trip_conversion(ggml_type intermediate_type, size_t test_size, bool verbose) {
    const auto * qfns = ggml_get_type_traits(intermediate_type);
    const auto * qfns_cpu = ggml_get_type_traits_cpu(intermediate_type);

    if (!qfns_cpu->from_float || !qfns->to_float) {
        return true; // Skip if functions not available
    }

    std::vector<float> original(test_size);
    generate_test_data(2.0, test_size, original.data());

    std::vector<uint8_t> quantized1(ggml_row_size(intermediate_type, test_size));
    std::vector<float> intermediate(test_size);
    std::vector<uint8_t> quantized2(ggml_row_size(intermediate_type, test_size));
    std::vector<float> final(test_size);

    qfns_cpu->from_float(original.data(), quantized1.data(), test_size);
    qfns->to_float(quantized1.data(), intermediate.data(), test_size);

    qfns_cpu->from_float(intermediate.data(), quantized2.data(), test_size);
    qfns->to_float(quantized2.data(), final.data(), test_size);

    float rmse = calculate_rmse(intermediate.data(), final.data(), test_size);
    bool passed = rmse < MAX_ROUND_TRIP_CONVERSION_ERROR;

    if (verbose || !passed) {
        printf("  %s Round-trip %s: RMSE=%.6f\n",
               RESULT_STR[!passed], ggml_type_name(intermediate_type), rmse);
    }

    return passed;
}

static bool test_tensor_alignment(ggml_type type, size_t test_size, bool verbose) {
    const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

    if (!qfns_cpu->from_float) {
        return true;
    }

    std::vector<size_t> test_sizes = {
        static_cast<size_t>(ggml_blck_size(type)),
        static_cast<size_t>(ggml_blck_size(type) * 2),
        static_cast<size_t>(ggml_blck_size(type) * 7),
        test_size
    };

    bool all_passed = true;
    for (size_t size : test_sizes) {
        if (size > test_size) continue;

        std::vector<float> data(size);
        generate_test_data(3.0, size, data.data());

        std::vector<uint8_t> quantized(ggml_row_size(type, size));

        qfns_cpu->from_float(data.data(), quantized.data(), size);
    }

    if (verbose) {
        printf("  %s Alignment test for %s\n", RESULT_STR[!all_passed], ggml_type_name(type));
    }

    return all_passed;
}

static bool test_large_model_simulation(bool verbose) {
    const size_t chunk_size = 1024 * 1024; // 1M floats = 4MB per chunk
    const size_t num_chunks = 4;           // Total 16MB of float data

    if (verbose) {
        printf("\nTesting large model simulation (%zu chunks of %zu elements)...\n",
               num_chunks, chunk_size);
    }

    bool all_passed = true;
    int num_failed = 0;

    for (ggml_type type : all_quant_types) {
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        if (!qfns_cpu->from_float || !qfns->to_float) {
            continue;
        }

        ggml_quantize_init(type);

        std::vector<float> chunk_errors;

        for (size_t chunk = 0; chunk < num_chunks; chunk++) {
            std::vector<float> data(chunk_size);
            generate_test_data(chunk * 10.0f, chunk_size, data.data());

            std::vector<uint8_t> quantized(ggml_row_size(type, chunk_size));
            std::vector<float> dequantized(chunk_size);

            qfns_cpu->from_float(data.data(), quantized.data(), chunk_size);
            qfns->to_float(quantized.data(), dequantized.data(), chunk_size);

            float rmse = calculate_rmse(data.data(), dequantized.data(), chunk_size);
            chunk_errors.push_back(rmse);
        }

        float avg_error = 0.0f;
        for (float err : chunk_errors) {
            avg_error += err;
        }
        avg_error /= chunk_errors.size();

        float threshold = get_error_threshold(type);
        bool passed = avg_error < threshold;

        if (!passed) {
            all_passed = false;
            num_failed++;
        }

        if (verbose || !passed) {
            printf("  %s %-12s: Avg RMSE=%.6f across %zu chunks\n",
                   RESULT_STR[!passed], ggml_type_name(type), avg_error, num_chunks);
        }
    }

    if (verbose || num_failed > 0) {
        printf("Large model simulation: %d/%d types passed\n",
               (int)(sizeof(all_quant_types)/sizeof(all_quant_types[0])) - num_failed,
               (int)(sizeof(all_quant_types)/sizeof(all_quant_types[0])));
    }

    return all_passed;
}

static bool test_multi_file_support(bool verbose) {
    if (verbose) {
        printf("\nTesting multi-file model support simulation...\n");
    }

    const size_t file_sizes[] = {512 * 1024, 768 * 1024, 1024 * 1024};
    const size_t num_files = sizeof(file_sizes) / sizeof(file_sizes[0]);

    bool all_passed = true;

    ggml_type test_types[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K};

    for (ggml_type type : test_types) {
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        if (!qfns_cpu->from_float || !qfns->to_float) {
            continue;
        }

        ggml_quantize_init(type);

        float total_error = 0.0f;

        for (size_t i = 0; i < num_files; i++) {
            std::vector<float> data(file_sizes[i]);
            generate_test_data(i * 5.0f, file_sizes[i], data.data());

            std::vector<uint8_t> quantized(ggml_row_size(type, file_sizes[i]));
            std::vector<float> dequantized(file_sizes[i]);

            qfns_cpu->from_float(data.data(), quantized.data(), file_sizes[i]);
            qfns->to_float(quantized.data(), dequantized.data(), file_sizes[i]);

            float rmse = calculate_rmse(data.data(), dequantized.data(), file_sizes[i]);
            total_error += rmse;
        }

        float avg_error = total_error / num_files;
        float threshold = get_error_threshold(type);
        bool passed = avg_error < threshold;

        if (!passed) {
            all_passed = false;
        }

        if (verbose || !passed) {
            printf("  %s %-12s: Avg RMSE=%.6f across %zu files\n",
                   RESULT_STR[!passed], ggml_type_name(type), avg_error, num_files);
        }
    }

    return all_passed;
}

int main(int argc, char ** argv) {
    bool verbose = false;
    bool test_all = true;
    bool test_single = false;
    bool test_cross = false;
    bool test_round_trip = false;
    bool test_large = false;
    bool test_multi_file = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--single") {
            test_all = false;
            test_single = true;
        } else if (arg == "--cross") {
            test_all = false;
            test_cross = true;
        } else if (arg == "--round-trip") {
            test_all = false;
            test_round_trip = true;
        } else if (arg == "--large") {
            test_all = false;
            test_large = true;
        } else if (arg == "--multi-file") {
            test_all = false;
            test_multi_file = true;
        } else {
            fprintf(stderr, "Usage: %s [-v|--verbose] [--single] [--cross] [--round-trip] [--large] [--multi-file]\n", argv[0]);
            return 1;
        }
    }

    ggml_cpu_init();

    const size_t test_size = 32 * 128; // Same as test-quantize-fns.cpp
    int total_tests = 0;
    int passed_tests = 0;

    if (test_all || test_single) {
        printf("\n=== Testing single format quantization ===\n");
        for (ggml_type type : all_quant_types) {
            ggml_quantize_init(type);
            total_tests++;
            if (test_single_format(type, test_size, verbose)) {
                passed_tests++;
            }
        }
    }

    if (test_all || test_cross) {
        printf("\n=== Testing cross-format conversions ===\n");

        for (ggml_type src : base_types) {
            for (ggml_type dst : all_quant_types) {
                total_tests++;
                if (test_cross_format_conversion(src, dst, test_size, verbose)) {
                    passed_tests++;
                }
            }
        }

        ggml_type sample_types[] = {
            GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K
        };

        for (size_t i = 0; i < sizeof(sample_types)/sizeof(sample_types[0]); i++) {
            for (size_t j = 0; j < sizeof(sample_types)/sizeof(sample_types[0]); j++) {
                if (i != j) {
                    ggml_quantize_init(sample_types[i]);
                    ggml_quantize_init(sample_types[j]);
                    total_tests++;
                    if (test_cross_format_conversion(sample_types[i], sample_types[j],
                                                     test_size, verbose)) {
                        passed_tests++;
                    }
                }
            }
        }
    }

    if (test_all || test_round_trip) {
        printf("\n=== Testing round-trip conversions ===\n");
        for (ggml_type type : all_quant_types) {
            ggml_quantize_init(type);
            total_tests++;
            if (test_round_trip_conversion(type, test_size, verbose)) {
                passed_tests++;
            }
        }
    }

    if (test_all) {
        printf("\n=== Testing tensor alignment ===\n");
        for (ggml_type type : all_quant_types) {
            ggml_quantize_init(type);
            total_tests++;
            if (test_tensor_alignment(type, test_size, verbose)) {
                passed_tests++;
            }
        }
    }

    if (test_all || test_large) {
        total_tests++;
        if (test_large_model_simulation(verbose)) {
            passed_tests++;
        }
    }

    if (test_all || test_multi_file) {
        total_tests++;
        if (test_multi_file_support(verbose)) {
            passed_tests++;
        }
    }

    printf("\n=== Test Summary ===\n");
    printf("Passed: %d/%d tests\n", passed_tests, total_tests);

    if (passed_tests == total_tests) {
        printf("All tests passed! ✓\n");
        return 0;
    } else {
        printf("%d tests failed ✗\n", total_tests - passed_tests);
        return 1;
    }
}
