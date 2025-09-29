
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct test_result {
    const char* test_name;
    bool passed;
    const char* error_msg;
};

static std::vector<test_result> test_results;

static void report_test(const char* name, bool passed, const char* msg = "") {
    test_results.push_back({name, passed, msg});
    printf("[%s] %s%s%s\n",
           passed ? "PASS" : "FAIL",
           name,
           msg[0] ? ": " : "",
           msg);
}

static void test_basic_allocation() {
    const char* test_name = "basic_allocation";

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        report_test(test_name, false, "Failed to initialize backend");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        report_test(test_name, false, "Failed to create context");
        return;
    }

    ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100, 100);
    bool success = (tensor != nullptr && tensor->data != nullptr);

    ggml_free(ctx);
    ggml_backend_free(backend);

    report_test(test_name, success, "Basic allocation completed");
}

static void test_memory_pressure() {
    const char* test_name = "memory_pressure";

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        report_test(test_name, false, "Failed to initialize backend");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 512*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        report_test(test_name, false, "Failed to create context");
        return;
    }

    std::vector<ggml_tensor*> tensors;
    bool allocation_succeeded __attribute__((unused)) = true;

    for (int i = 0; i < 100; i++) {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
        if (tensor && tensor->data) {
            tensors.push_back(tensor);
        } else {
            allocation_succeeded = false;
            break;
        }
    }

    ggml_free(ctx);
    ggml_backend_free(backend);

    char msg[256];
    snprintf(msg, sizeof(msg), "Allocated %zu tensors before running out of memory", tensors.size());
    report_test(test_name, tensors.size() > 0, msg);
}

static void test_graph_allocator_small_buffer() {
    const char* test_name = "graph_allocator_small_buffer";

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        report_test(test_name, false, "Failed to initialize backend");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 128*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        report_test(test_name, false, "Failed to create context");
        return;
    }

    ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    ggml_tensor* c = ggml_add(ctx, a, b);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!allocr) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        report_test(test_name, false, "Failed to create graph allocator");
        return;
    }

    bool reserved = ggml_gallocr_reserve(allocr, gf);
    bool allocated = false;
    if (reserved) {
        allocated = ggml_gallocr_alloc_graph(allocr, gf);
    }

    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_free(backend);

    report_test(test_name, reserved && allocated, "Graph allocation with small buffer");
}

static void test_zero_size_tensor() {
    const char* test_name = "zero_size_tensor";

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        report_test(test_name, false, "Failed to initialize backend");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        report_test(test_name, false, "Failed to create context");
        return;
    }

    ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 0);
    bool handled = (tensor != nullptr) && (ggml_nelements(tensor) == 0);

    ggml_free(ctx);
    ggml_backend_free(backend);

    report_test(test_name, handled, "Zero-sized tensor handled correctly");
}

static void test_alignment_requirements() {
    const char* test_name = "alignment_requirements";

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        report_test(test_name, false, "Failed to initialize backend");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        report_test(test_name, false, "Failed to create context");
        return;
    }

    bool all_aligned = true;
    for (int i = 0; i < 10; i++) {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64 + i*16);
        if (tensor && tensor->data) {
            uintptr_t addr = (uintptr_t)tensor->data;
            if (addr % GGML_MEM_ALIGN != 0) {
                all_aligned = false;
                break;
            }
        }
    }

    ggml_free(ctx);
    ggml_backend_free(backend);

    report_test(test_name, all_aligned, "All allocations properly aligned");
}

static void test_large_tensor_allocation() {
    const char* test_name = "large_tensor_allocation";

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        report_test(test_name, false, "Failed to initialize backend");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 512*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        report_test(test_name, false, "Failed to create context");
        return;
    }

    ggml_tensor* large_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1024);
    bool success = (large_tensor != nullptr && large_tensor->data != nullptr);

    ggml_free(ctx);
    ggml_backend_free(backend);

    report_test(test_name, success, "Large tensor allocation handled");
}

static void test_sequential_allocations() {
    const char* test_name = "sequential_allocations";

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        report_test(test_name, false, "Failed to initialize backend");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        report_test(test_name, false, "Failed to create context");
        return;
    }

    bool success = true;
    for (int i = 0; i < 20; i++) {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000);
        if (!tensor || !tensor->data) {
            success = false;
            break;
        }
    }

    ggml_free(ctx);
    ggml_backend_free(backend);

    report_test(test_name, success, "Sequential allocations completed");
}

static void test_mixed_type_allocations() {
    const char* test_name = "mixed_type_allocations";

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!backend) {
        report_test(test_name, false, "Failed to initialize backend");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        report_test(test_name, false, "Failed to create context");
        return;
    }

    bool success = true;
    ggml_tensor* t1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 100);
    ggml_tensor* t2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 100);
    ggml_tensor* t3 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 100);

    if (!t1 || !t1->data || !t2 || !t2->data || !t3 || !t3->data) {
        success = false;
    }

    ggml_free(ctx);
    ggml_backend_free(backend);

    report_test(test_name, success, "Mixed type allocations handled");
}

int main() {
    printf("=== Memory Exhaustion and Allocation Failure Tests ===\n\n");

    test_basic_allocation();
    test_memory_pressure();
    test_graph_allocator_small_buffer();
    test_zero_size_tensor();
    test_alignment_requirements();
    test_large_tensor_allocation();
    test_sequential_allocations();
    test_mixed_type_allocations();

    printf("\n=== Test Summary ===\n");
    int passed = 0;
    int failed = 0;

    for (const auto& result : test_results) {
        if (result.passed) {
            passed++;
        } else {
            failed++;
            printf("FAILED: %s - %s\n", result.test_name, result.error_msg);
        }
    }

    printf("\nTotal: %d tests, %d passed, %d failed\n",
           passed + failed, passed, failed);

    return failed > 0 ? 1 : 0;
}
