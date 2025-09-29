
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include "llama.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"

static std::atomic<int> g_contexts_created{0};
static std::atomic<int> g_contexts_destroyed{0};
static std::atomic<int> g_decode_operations{0};
static std::atomic<int> g_errors{0};

struct stress_test_result {
    int contexts_created = 0;
    int contexts_destroyed = 0;
    int decode_operations = 0;
    int errors = 0;
    double duration_seconds = 0.0;
};

static void rapid_context_lifecycle_test(
    llama_model * model,
    const llama_context_params & cparams,
    const common_params & params,
    int iterations) {
    
    for (int i = 0; i < iterations; ++i) {
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (ctx == NULL) {
            LOG_ERR("failed to create context in rapid lifecycle test\n");
            g_errors++;
            continue;
        }
        g_contexts_created++;
        
        std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler {
            common_sampler_init(model, params.sampling), common_sampler_free
        };
        if (sampler == NULL) {
            LOG_ERR("failed to create sampler in rapid lifecycle test\n");
            g_errors++;
            llama_free(ctx);
            continue;
        }
        
        auto prompt = common_tokenize(ctx, params.prompt, true);
        if (!prompt.empty()) {
            llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
            if (llama_decode(ctx, batch) == 0) {
                g_decode_operations++;
            } else {
                g_errors++;
            }
        }
        
        llama_free(ctx);
        g_contexts_destroyed++;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

static void sustained_inference_test(
    llama_model * model,
    const llama_context_params & cparams,
    const common_params & params,
    int num_iterations) {
    
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        LOG_ERR("failed to create context in sustained inference test\n");
        g_errors++;
        return;
    }
    g_contexts_created++;
    
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler {
        common_sampler_init(model, params.sampling), common_sampler_free
    };
    if (sampler == NULL) {
        LOG_ERR("failed to create sampler in sustained inference test\n");
        g_errors++;
        llama_free(ctx);
        return;
    }
    
    const auto * vocab = llama_model_get_vocab(model);
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        auto prompt = common_tokenize(ctx, params.prompt, true);
        if (prompt.empty()) {
            g_errors++;
            continue;
        }
        
        llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
        if (llama_decode(ctx, batch)) {
            g_errors++;
            continue;
        }
        g_decode_operations++;
        
        for (int i = 0; i < 10; i++) {
            llama_token token;
            if (batch.n_tokens > 0) {
                token = common_sampler_sample(sampler.get(), ctx, batch.n_tokens - 1);
            } else {
                token = llama_vocab_bos(vocab);
            }
            
            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }
            
            batch = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, batch)) {
                g_errors++;
                break;
            }
            g_decode_operations++;
        }
        
        llama_memory_clear(llama_get_memory(ctx), false);
    }
    
    llama_free(ctx);
    g_contexts_destroyed++;
}

static void concurrent_sequence_test(
    llama_model * model,
    const llama_context_params & cparams,
    const common_params & params,
    int num_sequences) {
    
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        LOG_ERR("failed to create context in concurrent sequence test\n");
        g_errors++;
        return;
    }
    g_contexts_created++;
    
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler {
        common_sampler_init(model, params.sampling), common_sampler_free
    };
    if (sampler == NULL) {
        LOG_ERR("failed to create sampler in concurrent sequence test\n");
        g_errors++;
        llama_free(ctx);
        return;
    }
    
    const auto * vocab = llama_model_get_vocab(model);
    
    for (int seq_id = 0; seq_id < num_sequences; ++seq_id) {
        auto prompt = common_tokenize(ctx, params.prompt, true);
        if (prompt.empty()) {
            g_errors++;
            continue;
        }
        
        llama_batch batch = llama_batch_init(prompt.size(), 0, 1);
        for (size_t i = 0; i < prompt.size(); ++i) {
            batch.token[i] = prompt[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i] = (i == prompt.size() - 1);
        }
        batch.n_tokens = prompt.size();
        
        if (llama_decode(ctx, batch)) {
            g_errors++;
            llama_batch_free(batch);
            continue;
        }
        g_decode_operations++;
        
        for (int i = 0; i < 5; i++) {
            llama_token token = common_sampler_sample(sampler.get(), ctx, batch.n_tokens - 1);
            
            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }
            
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = prompt.size() + i;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = seq_id;
            batch.logits[0] = true;
            
            if (llama_decode(ctx, batch)) {
                g_errors++;
                break;
            }
            g_decode_operations++;
        }
        
        llama_batch_free(batch);
        llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
    }
    
    llama_free(ctx);
    g_contexts_destroyed++;
}

static void memory_stress_test(
    llama_model * model,
    const llama_context_params & cparams,
    const common_params & params,
    int num_operations) {
    
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        LOG_ERR("failed to create context in memory stress test\n");
        g_errors++;
        return;
    }
    g_contexts_created++;
    
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler {
        common_sampler_init(model, params.sampling), common_sampler_free
    };
    if (sampler == NULL) {
        LOG_ERR("failed to create sampler in memory stress test\n");
        g_errors++;
        llama_free(ctx);
        return;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> seq_dist(0, 15);
    
    for (int op = 0; op < num_operations; ++op) {
        int seq_id = seq_dist(gen);
        
        auto prompt = common_tokenize(ctx, params.prompt, true);
        if (!prompt.empty()) {
            llama_batch batch = llama_batch_init(prompt.size(), 0, 1);
            for (size_t i = 0; i < prompt.size(); ++i) {
                batch.token[i] = prompt[i];
                batch.pos[i] = i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = seq_id;
                batch.logits[i] = (i == prompt.size() - 1);
            }
            batch.n_tokens = prompt.size();
            
            if (llama_decode(ctx, batch) == 0) {
                g_decode_operations++;
            } else {
                g_errors++;
            }
            
            llama_batch_free(batch);
        }
        
        if (op % 3 == 0) {
            llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
        } else if (op % 3 == 1) {
            int target_seq = (seq_id + 1) % 16;
            llama_memory_seq_cp(llama_get_memory(ctx), seq_id, target_seq, -1, -1);
        } else {
            llama_memory_clear(llama_get_memory(ctx), false);
        }
    }
    
    llama_free(ctx);
    g_contexts_destroyed++;
}

int main(int argc, char ** argv) {
    common_params params;
    
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    
    common_init();
    
    llama_backend_init();
    llama_numa_init(params.numa);
    
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("Starting concurrent stress tests...\n");
    
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }
    
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = std::max(16u, cparams.n_seq_max);
    
    const int num_threads = std::max(1, params.n_parallel);
    const int iterations_per_thread = 5;
    
    g_contexts_created = 0;
    g_contexts_destroyed = 0;
    g_decode_operations = 0;
    g_errors = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    LOG_INF("\n=== Test 1: Rapid Context Lifecycle (%d threads, %d iterations each) ===\n", 
            num_threads, iterations_per_thread);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(rapid_context_lifecycle_test, model, cparams, params, iterations_per_thread);
        }
        for (auto & thread : threads) {
            thread.join();
        }
    }
    LOG_INF("Contexts created: %d, destroyed: %d, decode ops: %d, errors: %d\n",
            g_contexts_created.load(), g_contexts_destroyed.load(), 
            g_decode_operations.load(), g_errors.load());
    
    g_contexts_created = 0;
    g_contexts_destroyed = 0;
    g_decode_operations = 0;
    int errors_after_test1 = g_errors.load();
    
    LOG_INF("\n=== Test 2: Sustained Concurrent Inference (%d threads, %d iterations each) ===\n", 
            num_threads, iterations_per_thread * 2);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(sustained_inference_test, model, cparams, params, iterations_per_thread * 2);
        }
        for (auto & thread : threads) {
            thread.join();
        }
    }
    LOG_INF("Contexts created: %d, destroyed: %d, decode ops: %d, errors: %d\n",
            g_contexts_created.load(), g_contexts_destroyed.load(), 
            g_decode_operations.load(), g_errors.load());
    
    g_contexts_created = 0;
    g_contexts_destroyed = 0;
    g_decode_operations = 0;
    int errors_after_test2 = g_errors.load();
    
    LOG_INF("\n=== Test 3: Concurrent Sequence Operations (%d threads, %d sequences each) ===\n", 
            num_threads / 2, 8);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < std::max(1, num_threads / 2); ++t) {
            threads.emplace_back(concurrent_sequence_test, model, cparams, params, 8);
        }
        for (auto & thread : threads) {
            thread.join();
        }
    }
    LOG_INF("Contexts created: %d, destroyed: %d, decode ops: %d, errors: %d\n",
            g_contexts_created.load(), g_contexts_destroyed.load(), 
            g_decode_operations.load(), g_errors.load());
    
    g_contexts_created = 0;
    g_contexts_destroyed = 0;
    g_decode_operations = 0;
    int errors_after_test3 = g_errors.load();
    
    LOG_INF("\n=== Test 4: Memory Operations Stress (%d threads, %d operations each) ===\n", 
            num_threads, iterations_per_thread * 3);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(memory_stress_test, model, cparams, params, iterations_per_thread * 3);
        }
        for (auto & thread : threads) {
            thread.join();
        }
    }
    LOG_INF("Contexts created: %d, destroyed: %d, decode ops: %d, errors: %d\n",
            g_contexts_created.load(), g_contexts_destroyed.load(), 
            g_decode_operations.load(), g_errors.load());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    int total_errors = g_errors.load();
    
    LOG_INF("\n=== Stress Test Summary ===\n");
    LOG_INF("Total duration: %.2f seconds\n", duration.count() / 1000.0);
    LOG_INF("Total errors: %d\n", total_errors);
    LOG_INF("  After test 1: %d\n", errors_after_test1);
    LOG_INF("  After test 2: %d\n", errors_after_test2);
    LOG_INF("  After test 3: %d\n", errors_after_test3);
    LOG_INF("  After test 4: %d\n", total_errors);
    
    llama_model_free(model);
    
    if (total_errors > 0) {
        LOG_ERR("Stress tests completed with %d errors\n", total_errors);
        return 1;
    }
    
    LOG_INF("All stress tests passed successfully!\n");
    return 0;
}
