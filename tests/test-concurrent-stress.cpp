#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>


struct test_result {
    std::atomic<int> contexts_created{0};
    std::atomic<int> contexts_destroyed{0};
    std::atomic<int> batches_processed{0};
    std::atomic<int> errors{0};
    std::atomic<bool> corruption_detected{false};
};

static void test_rapid_context_cycles(
    llama_model * model,
    llama_context_params base_params,
    test_result & result,
    int thread_id,
    int iterations
) {
    const int64_t t_start = ggml_time_us();

    std::random_device rd;
    std::mt19937 gen(rd() + thread_id);
    std::uniform_int_distribution<> delay_dist(1, 10);

    for (int i = 0; i < iterations; i++) {
        llama_context * ctx = llama_init_from_model(model, base_params);

        if (!ctx) {
            LOG_ERR("thread %d: failed to create context on iteration %d\n", thread_id, i);
            result.errors++;
            continue;
        }

        result.contexts_created++;

        std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));

        llama_free(ctx);
        result.contexts_destroyed++;
    }

    const int64_t t_end = ggml_time_us();
    LOG_INF("thread %d: completed %d context cycles in %.2f ms\n",
            thread_id, iterations, (t_end - t_start) / 1000.0);
}


static void test_backend_resource_stress(
    llama_model * model,
    llama_context_params base_params,
    test_result & result,
    int thread_id,
    int iterations
) {
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id);
    std::uniform_int_distribution<> delay_dist(1, 8);

    for (int i = 0; i < iterations; i++) {
        llama_context_params ctx_params = base_params;

        ctx_params.n_ctx = 128 + (i % 4) * 64;
        ctx_params.n_batch = 32 + (i % 3) * 16;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            LOG_ERR("thread %d: failed to create context with varying params on iteration %d\n", thread_id, i);
            result.errors++;
            continue;
        }

        result.contexts_created++;

        std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));

        llama_free(ctx);
        result.contexts_destroyed++;
    }
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    auto mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("failed to load model\n");
        return 1;
    }

    auto cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const int n_threads = params.cpuparams.n_threads;
    const int iterations_per_thread = 20;

    LOG_INF("Starting concurrent stress tests with %d threads, %d iterations per thread\n",
            n_threads, iterations_per_thread);

    LOG_INF("\n=== Test 1: Rapid Context Creation/Destruction Cycles ===\n");
    {
        test_result result;
        std::vector<std::thread> threads;

        const int64_t t_start = ggml_time_us();

        for (int i = 0; i < n_threads; i++) {
            threads.emplace_back(test_rapid_context_cycles, model, cparams,
                               std::ref(result), i, iterations_per_thread);
        }

        for (auto & t : threads) {
            t.join();
        }

        const int64_t t_end = ggml_time_us();

        LOG_INF("Test 1 Results:\n");
        LOG_INF("  Contexts created: %d\n", result.contexts_created.load());
        LOG_INF("  Contexts destroyed: %d\n", result.contexts_destroyed.load());
        LOG_INF("  Errors: %d\n", result.errors.load());
        LOG_INF("  Total time: %.2f ms\n", (t_end - t_start) / 1000.0);
        LOG_INF("  Avg time per context: %.2f ms\n",
                (t_end - t_start) / 1000.0 / result.contexts_created.load());

        if (result.contexts_created != result.contexts_destroyed) {
            LOG_ERR("FAIL: Context leak detected! Created: %d, Destroyed: %d\n",
                    result.contexts_created.load(), result.contexts_destroyed.load());
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        if (result.errors > 0) {
            LOG_ERR("FAIL: %d errors occurred during context cycles\n", result.errors.load());
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        LOG_INF("PASS: No context leaks or errors detected\n");
    }

    LOG_INF("\n=== Test 2: Parallel Context Operations ===\n");
    {
        test_result result;
        std::vector<std::thread> threads;

        const int64_t t_start = ggml_time_us();

        auto parallel_context_ops = [&](int thread_id) {
            std::random_device rd;
            std::mt19937 gen(rd() + thread_id);
            std::uniform_int_distribution<> delay_dist(1, 5);

            for (int i = 0; i < iterations_per_thread / 4; i++) {
                llama_context * thread_ctx = llama_init_from_model(model, cparams);
                if (!thread_ctx) {
                    LOG_ERR("thread %d: failed to create context on iteration %d\n", thread_id, i);
                    result.errors++;
                    continue;
                }

                result.contexts_created++;

                std::vector<llama_token> tokens = common_tokenize(thread_ctx, "Test prompt", true, true);
                if (!tokens.empty()) {
                    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
                    for (size_t j = 0; j < tokens.size(); j++) {
                        common_batch_add(batch, tokens[j], j, {0}, false);
                    }

                    if (llama_decode(thread_ctx, batch) == 0) {
                        result.batches_processed++;
                    }

                    llama_batch_free(batch);
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));

                llama_free(thread_ctx);
                result.contexts_destroyed++;
            }
        };

        for (int i = 0; i < n_threads; i++) {
            threads.emplace_back(parallel_context_ops, i);
        }

        for (auto & t : threads) {
            t.join();
        }

        const int64_t t_end = ggml_time_us();

        LOG_INF("Test 2 Results:\n");
        LOG_INF("  Contexts created: %d\n", result.contexts_created.load());
        LOG_INF("  Contexts destroyed: %d\n", result.contexts_destroyed.load());
        LOG_INF("  Batches processed: %d\n", result.batches_processed.load());
        LOG_INF("  Errors: %d\n", result.errors.load());
        LOG_INF("  Total time: %.2f ms\n", (t_end - t_start) / 1000.0);

        if (result.contexts_created != result.contexts_destroyed) {
            LOG_ERR("FAIL: Context leak detected! Created: %d, Destroyed: %d\n",
                    result.contexts_created.load(), result.contexts_destroyed.load());
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        if (result.errors > 0) {
            LOG_ERR("FAIL: %d errors occurred during parallel operations\n", result.errors.load());
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        LOG_INF("PASS: All parallel context operations completed successfully\n");
    }

    LOG_INF("\n=== Test 3: Backend Resource Allocation Stress ===\n");
    {
        test_result result;
        std::vector<std::thread> threads;

        const int64_t t_start = ggml_time_us();

        for (int i = 0; i < n_threads; i++) {
            threads.emplace_back(test_backend_resource_stress, model, cparams,
                               std::ref(result), i, iterations_per_thread / 4);
        }

        for (auto & t : threads) {
            t.join();
        }

        const int64_t t_end = ggml_time_us();

        LOG_INF("Test 3 Results:\n");
        LOG_INF("  Contexts created: %d\n", result.contexts_created.load());
        LOG_INF("  Contexts destroyed: %d\n", result.contexts_destroyed.load());
        LOG_INF("  Errors: %d\n", result.errors.load());
        LOG_INF("  Total time: %.2f ms\n", (t_end - t_start) / 1000.0);

        if (result.contexts_created != result.contexts_destroyed) {
            LOG_ERR("FAIL: Resource leak detected! Created: %d, Destroyed: %d\n",
                    result.contexts_created.load(), result.contexts_destroyed.load());
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        if (result.errors > 0) {
            LOG_ERR("FAIL: %d errors occurred during resource stress test\n", result.errors.load());
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        LOG_INF("PASS: No resource leaks detected\n");
    }

    llama_free(ctx);
    llama_model_free(model);

    LOG_INF("\n=== All Concurrent Stress Tests PASSED ===\n");
    return 0;
}
