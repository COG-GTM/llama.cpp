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
    std::atomic<int> prepare_success{0};
    std::atomic<int> update_success{0};
    std::atomic<int> seq_ops_success{0};
    std::atomic<int> errors{0};
};

static void test_concurrent_kv_prepare(
    llama_model * model,
    llama_context_params cparams,
    const std::vector<llama_token> & tokens,
    test_result & result,
    int thread_id,
    int iterations
) {
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id);
    std::uniform_int_distribution<> delay_dist(1, 5);

    for (int i = 0; i < iterations; i++) {
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            LOG_ERR("thread %d: failed to create context on iteration %d\n", thread_id, i);
            result.errors++;
            continue;
        }

        result.contexts_created++;

        const int n_batch = llama_n_batch(ctx);
        llama_batch batch = llama_batch_init(n_batch, 0, 1);

        const int n_tokens = std::min((int)tokens.size(), n_batch);
        for (int j = 0; j < n_tokens; j++) {
            common_batch_add(batch, tokens[j], j, {0}, false);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));

        if (llama_decode(ctx, batch) == 0) {
            result.prepare_success++;
        } else {
            result.errors++;
        }

        llama_batch_free(batch);
        llama_free(ctx);
        result.contexts_destroyed++;
    }
}

static void test_concurrent_kv_update(
    llama_model * model,
    llama_context_params base_params,
    const std::vector<llama_token> & tokens,
    test_result & result,
    int thread_id,
    int iterations
) {
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id);
    std::uniform_int_distribution<> delay_dist(1, 5);

    for (int i = 0; i < iterations; i++) {
        llama_context_params cparams = base_params;
        cparams.n_ctx = 128 + (i % 3) * 64;

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            LOG_ERR("thread %d: failed to create context on iteration %d\n", thread_id, i);
            result.errors++;
            continue;
        }

        result.contexts_created++;

        const int n_batch = llama_n_batch(ctx);
        llama_batch batch = llama_batch_init(std::min((int)tokens.size(), n_batch), 0, 1);

        for (size_t j = 0; j < std::min(tokens.size(), (size_t)n_batch); j++) {
            common_batch_add(batch, tokens[j], j, {0}, false);
        }

        if (llama_decode(ctx, batch) == 0) {
            result.update_success++;
        } else {
            result.errors++;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));

        llama_batch_free(batch);
        llama_free(ctx);
        result.contexts_destroyed++;
    }
}

static void test_concurrent_seq_operations(
    llama_model * model,
    llama_context_params cparams,
    const std::vector<llama_token> & tokens,
    test_result & result,
    int thread_id,
    int iterations
) {
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id);
    std::uniform_int_distribution<> delay_dist(1, 5);

    for (int i = 0; i < iterations; i++) {
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            LOG_ERR("thread %d: failed to create context on iteration %d\n", thread_id, i);
            result.errors++;
            continue;
        }

        result.contexts_created++;

        const int n_batch = llama_n_batch(ctx);
        llama_batch batch = llama_batch_init(std::min((int)tokens.size(), n_batch), 0, 1);

        for (size_t j = 0; j < std::min(tokens.size(), (size_t)n_batch); j++) {
            common_batch_add(batch, tokens[j], j, {0}, false);
        }

        if (llama_decode(ctx, batch) == 0) {
            llama_memory_seq_cp(llama_get_memory(ctx), 0, 1, -1, -1);
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);
            result.seq_ops_success++;
        } else {
            result.errors++;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));

        llama_batch_free(batch);
        llama_free(ctx);
        result.contexts_destroyed++;
    }
}

static void test_concurrent_mixed_operations(
    llama_model * model,
    llama_context_params cparams,
    const std::vector<llama_token> & tokens,
    test_result & result,
    int thread_id,
    int iterations
) {
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id);
    std::uniform_int_distribution<> delay_dist(1, 3);

    for (int i = 0; i < iterations; i++) {
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            LOG_ERR("thread %d: failed to create context on iteration %d\n", thread_id, i);
            result.errors++;
            continue;
        }

        result.contexts_created++;

        const int n_batch = llama_n_batch(ctx);
        llama_batch batch = llama_batch_init(std::min((int)tokens.size(), n_batch), 0, 1);

        for (size_t j = 0; j < std::min(tokens.size(), (size_t)n_batch); j++) {
            common_batch_add(batch, tokens[j], j, {0}, false);
        }

        if (llama_decode(ctx, batch) == 0) {
            result.prepare_success++;
            result.update_success++;
            result.seq_ops_success++;
        } else {
            result.errors++;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));

        llama_batch_free(batch);
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

    std::vector<llama_token> tokens;
    const char * test_prompt = "Once upon a time in a distant galaxy, there was a brave explorer";
    tokens = common_tokenize(ctx, test_prompt, true, true);

    if (tokens.empty()) {
        LOG_ERR("failed to tokenize test prompt\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    LOG_INF("Test prompt tokenized to %zu tokens\n", tokens.size());

    llama_free(ctx);

    const int n_threads = std::min(params.cpuparams.n_threads, 8);
    const int iterations_per_thread = 15;

    LOG_INF("Starting KV cache concurrent tests with %d threads, %d iterations per thread\n",
            n_threads, iterations_per_thread);

    LOG_INF("\n=== Test 1: Concurrent KV Cache Prepare Operations ===\n");
    {
        test_result result;
        std::vector<std::thread> threads;

        const int64_t t_start = ggml_time_us();

        for (int i = 0; i < n_threads; i++) {
            threads.emplace_back(test_concurrent_kv_prepare, model, cparams, std::cref(tokens),
                               std::ref(result), i, iterations_per_thread);
        }

        for (auto & t : threads) {
            t.join();
        }

        const int64_t t_end = ggml_time_us();

        LOG_INF("Test 1 Results:\n");
        LOG_INF("  Contexts created: %d\n", result.contexts_created.load());
        LOG_INF("  Contexts destroyed: %d\n", result.contexts_destroyed.load());
        LOG_INF("  Successful prepare operations: %d\n", result.prepare_success.load());
        LOG_INF("  Errors: %d\n", result.errors.load());
        LOG_INF("  Total time: %.2f ms\n", (t_end - t_start) / 1000.0);

        if (result.contexts_created != result.contexts_destroyed) {
            LOG_ERR("FAIL: Context leak detected! Created: %d, Destroyed: %d\n",
                    result.contexts_created.load(), result.contexts_destroyed.load());
            llama_model_free(model);
            return 1;
        }

        if (result.errors > 0) {
            LOG_ERR("FAIL: %d errors occurred during concurrent prepare\n", result.errors.load());
            llama_model_free(model);
            return 1;
        }

        LOG_INF("PASS: All concurrent prepare operations successful\n");
    }

    LOG_INF("\n=== Test 2: Concurrent KV Cache Update Operations ===\n");
    {
        test_result result;
        std::vector<std::thread> threads;

        const int64_t t_start = ggml_time_us();

        for (int i = 0; i < n_threads; i++) {
            threads.emplace_back(test_concurrent_kv_update, model, cparams, std::cref(tokens),
                               std::ref(result), i, iterations_per_thread);
        }

        for (auto & t : threads) {
            t.join();
        }

        const int64_t t_end = ggml_time_us();

        LOG_INF("Test 2 Results:\n");
        LOG_INF("  Contexts created: %d\n", result.contexts_created.load());
        LOG_INF("  Contexts destroyed: %d\n", result.contexts_destroyed.load());
        LOG_INF("  Successful update operations: %d\n", result.update_success.load());
        LOG_INF("  Errors: %d\n", result.errors.load());
        LOG_INF("  Total time: %.2f ms\n", (t_end - t_start) / 1000.0);

        if (result.contexts_created != result.contexts_destroyed) {
            LOG_ERR("FAIL: Context leak detected! Created: %d, Destroyed: %d\n",
                    result.contexts_created.load(), result.contexts_destroyed.load());
            llama_model_free(model);
            return 1;
        }

        if (result.errors > 0) {
            LOG_ERR("FAIL: %d errors occurred during concurrent update\n", result.errors.load());
            llama_model_free(model);
            return 1;
        }

        LOG_INF("PASS: All concurrent update operations successful\n");
    }

    LOG_INF("\n=== Test 3: Concurrent Sequence Operations ===\n");
    {
        test_result result;
        std::vector<std::thread> threads;

        const int64_t t_start = ggml_time_us();

        for (int i = 0; i < n_threads; i++) {
            threads.emplace_back(test_concurrent_seq_operations, model, cparams, std::cref(tokens),
                               std::ref(result), i, iterations_per_thread);
        }

        for (auto & t : threads) {
            t.join();
        }

        const int64_t t_end = ggml_time_us();

        LOG_INF("Test 3 Results:\n");
        LOG_INF("  Contexts created: %d\n", result.contexts_created.load());
        LOG_INF("  Contexts destroyed: %d\n", result.contexts_destroyed.load());
        LOG_INF("  Successful sequence operations: %d\n", result.seq_ops_success.load());
        LOG_INF("  Errors: %d\n", result.errors.load());
        LOG_INF("  Total time: %.2f ms\n", (t_end - t_start) / 1000.0);

        if (result.contexts_created != result.contexts_destroyed) {
            LOG_ERR("FAIL: Context leak detected! Created: %d, Destroyed: %d\n",
                    result.contexts_created.load(), result.contexts_destroyed.load());
            llama_model_free(model);
            return 1;
        }

        if (result.errors > 0) {
            LOG_ERR("FAIL: %d errors occurred during concurrent sequence ops\n", result.errors.load());
            llama_model_free(model);
            return 1;
        }

        LOG_INF("PASS: All concurrent sequence operations successful\n");
    }

    LOG_INF("\n=== Test 4: Mixed Concurrent Operations ===\n");
    {
        test_result result;
        std::vector<std::thread> threads;

        const int64_t t_start = ggml_time_us();

        for (int i = 0; i < n_threads; i++) {
            threads.emplace_back(test_concurrent_mixed_operations, model, cparams, std::cref(tokens),
                               std::ref(result), i, iterations_per_thread / 2);
        }

        for (auto & t : threads) {
            t.join();
        }

        const int64_t t_end = ggml_time_us();

        LOG_INF("Test 4 Results:\n");
        LOG_INF("  Contexts created: %d\n", result.contexts_created.load());
        LOG_INF("  Contexts destroyed: %d\n", result.contexts_destroyed.load());
        LOG_INF("  Successful prepare operations: %d\n", result.prepare_success.load());
        LOG_INF("  Successful update operations: %d\n", result.update_success.load());
        LOG_INF("  Successful sequence operations: %d\n", result.seq_ops_success.load());
        LOG_INF("  Errors: %d\n", result.errors.load());
        LOG_INF("  Total time: %.2f ms\n", (t_end - t_start) / 1000.0);

        if (result.contexts_created != result.contexts_destroyed) {
            LOG_ERR("FAIL: Context leak detected! Created: %d, Destroyed: %d\n",
                    result.contexts_created.load(), result.contexts_destroyed.load());
            llama_model_free(model);
            return 1;
        }

        if (result.errors > 0) {
            LOG_ERR("FAIL: %d errors occurred during mixed operations\n", result.errors.load());
            llama_model_free(model);
            return 1;
        }

        LOG_INF("PASS: All mixed concurrent operations successful\n");
    }

    llama_model_free(model);

    LOG_INF("\n=== All KV Cache Concurrent Tests PASSED ===\n");
    return 0;
}
