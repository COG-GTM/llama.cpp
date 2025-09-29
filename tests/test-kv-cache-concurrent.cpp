
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>
#include "llama.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"

static std::atomic<int> g_cache_operations{0};
static std::atomic<int> g_slot_allocations{0};
static std::atomic<int> g_slot_deallocations{0};
static std::atomic<int> g_errors{0};
static std::atomic<bool> g_stop_flag{false};

static void concurrent_cache_alloc_dealloc_test(
    llama_model * model,
    const llama_context_params & cparams,
    const common_params & params,
    int num_iterations,
    int thread_id) {
    
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        LOG_ERR("Thread %d: failed to create context\n", thread_id);
        g_errors++;
        return;
    }
    
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler {
        common_sampler_init(model, params.sampling), common_sampler_free
    };
    if (sampler == NULL) {
        LOG_ERR("Thread %d: failed to create sampler\n", thread_id);
        g_errors++;
        llama_free(ctx);
        return;
    }
    
    const auto * vocab = llama_model_get_vocab(model);
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id);
    std::uniform_int_distribution<> seq_dist(thread_id * 4, thread_id * 4 + 3);
    
    for (int iter = 0; iter < num_iterations && !g_stop_flag; ++iter) {
        int seq_id = seq_dist(gen);
        
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
        g_cache_operations++;
        g_slot_allocations++;
        
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
            g_cache_operations++;
        }
        
        llama_batch_free(batch);
        
        llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
        g_slot_deallocations++;
        
        if (iter % 10 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    
    llama_free(ctx);
}

static void concurrent_sequence_copy_test(
    llama_model * model,
    const llama_context_params & cparams,
    const common_params & params,
    int num_iterations,
    int thread_id) {
    
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        LOG_ERR("Thread %d: failed to create context for sequence copy test\n", thread_id);
        g_errors++;
        return;
    }
    
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler {
        common_sampler_init(model, params.sampling), common_sampler_free
    };
    if (sampler == NULL) {
        LOG_ERR("Thread %d: failed to create sampler for sequence copy test\n", thread_id);
        g_errors++;
        llama_free(ctx);
        return;
    }
    
    const auto * vocab = llama_model_get_vocab(model);
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id + 1000);
    std::uniform_int_distribution<> seq_dist(thread_id * 3, thread_id * 3 + 2);
    
    for (int iter = 0; iter < num_iterations && !g_stop_flag; ++iter) {
        int src_seq = seq_dist(gen);
        int dst_seq = (src_seq + 1) % (thread_id * 3 + 3);
        
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
            batch.seq_id[i][0] = src_seq;
            batch.logits[i] = (i == prompt.size() - 1);
        }
        batch.n_tokens = prompt.size();
        
        if (llama_decode(ctx, batch)) {
            g_errors++;
            llama_batch_free(batch);
            continue;
        }
        g_cache_operations++;
        
        llama_token token = common_sampler_sample(sampler.get(), ctx, batch.n_tokens - 1);
        if (!llama_vocab_is_eog(vocab, token)) {
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = prompt.size();
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = src_seq;
            batch.logits[0] = true;
            
            if (llama_decode(ctx, batch)) {
                g_errors++;
            } else {
                g_cache_operations++;
            }
        }
        
        llama_batch_free(batch);
        
        llama_memory_seq_cp(llama_get_memory(ctx), src_seq, dst_seq, -1, -1);
        g_cache_operations++;
        
        batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = prompt.size() + 1;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = dst_seq;
        batch.logits[0] = true;
        
        if (llama_decode(ctx, batch)) {
            g_errors++;
        } else {
            g_cache_operations++;
        }
        
        llama_batch_free(batch);
        
        llama_memory_seq_rm(llama_get_memory(ctx), src_seq, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx), dst_seq, -1, -1);
        
        if (iter % 5 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    
    llama_free(ctx);
}

static void concurrent_cache_clear_test(
    llama_model * model,
    const llama_context_params & cparams,
    const common_params & params,
    int num_iterations,
    int thread_id) {
    
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        LOG_ERR("Thread %d: failed to create context for cache clear test\n", thread_id);
        g_errors++;
        return;
    }
    
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler {
        common_sampler_init(model, params.sampling), common_sampler_free
    };
    if (sampler == NULL) {
        LOG_ERR("Thread %d: failed to create sampler for cache clear test\n", thread_id);
        g_errors++;
        llama_free(ctx);
        return;
    }
    
    for (int iter = 0; iter < num_iterations && !g_stop_flag; ++iter) {
        for (int seq_id = thread_id * 2; seq_id < thread_id * 2 + 2; ++seq_id) {
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
            g_cache_operations++;
            
            llama_batch_free(batch);
        }
        
        llama_memory_clear(llama_get_memory(ctx), false);
        g_cache_operations++;
        
        if (iter % 3 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
    
    llama_free(ctx);
}

static void concurrent_mixed_operations_test(
    llama_model * model,
    const llama_context_params & cparams,
    const common_params & params,
    int num_iterations,
    int thread_id) {
    
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == NULL) {
        LOG_ERR("Thread %d: failed to create context for mixed operations test\n", thread_id);
        g_errors++;
        return;
    }
    
    std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler {
        common_sampler_init(model, params.sampling), common_sampler_free
    };
    if (sampler == NULL) {
        LOG_ERR("Thread %d: failed to create sampler for mixed operations test\n", thread_id);
        g_errors++;
        llama_free(ctx);
        return;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd() + thread_id + 2000);
    std::uniform_int_distribution<> op_dist(0, 3);
    std::uniform_int_distribution<> seq_dist(thread_id * 2, thread_id * 2 + 1);
    
    for (int iter = 0; iter < num_iterations && !g_stop_flag; ++iter) {
        int operation = op_dist(gen);
        int seq_id = seq_dist(gen);
        
        switch (operation) {
            case 0: {
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
                        g_cache_operations++;
                    } else {
                        g_errors++;
                    }
                    
                    llama_batch_free(batch);
                }
                break;
            }
            case 1: {
                int target_seq = (seq_id + 1) % (thread_id * 2 + 2);
                llama_memory_seq_cp(llama_get_memory(ctx), seq_id, target_seq, -1, -1);
                g_cache_operations++;
                break;
            }
            case 2: {
                llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
                g_cache_operations++;
                break;
            }
            case 3: {
                llama_pos min_pos = llama_memory_seq_pos_min(llama_get_memory(ctx), seq_id);
                llama_pos max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
                if (min_pos >= 0 && max_pos >= min_pos) {
                    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, min_pos, max_pos / 2);
                    g_cache_operations++;
                }
                break;
            }
        }
        
        if (iter % 20 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(150));
        }
    }
    
    llama_free(ctx);
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
    LOG_INF("Starting KV cache concurrent tests...\n");
    
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }
    
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = std::max(32u, cparams.n_seq_max);
    
    const int num_threads = std::max(2, params.n_parallel);
    const int iterations_per_test = 20;
    
    g_cache_operations = 0;
    g_slot_allocations = 0;
    g_slot_deallocations = 0;
    g_errors = 0;
    g_stop_flag = false;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    LOG_INF("\n=== Test 1: Concurrent Cache Allocation/Deallocation (%d threads, %d iterations) ===\n", 
            num_threads, iterations_per_test);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(concurrent_cache_alloc_dealloc_test, model, cparams, params, 
                                 iterations_per_test, t);
        }
        for (auto & thread : threads) {
            thread.join();
        }
    }
    LOG_INF("Cache operations: %d, allocations: %d, deallocations: %d, errors: %d\n",
            g_cache_operations.load(), g_slot_allocations.load(), 
            g_slot_deallocations.load(), g_errors.load());
    
    int errors_after_test1 = g_errors.load();
    g_cache_operations = 0;
    g_slot_allocations = 0;
    g_slot_deallocations = 0;
    
    LOG_INF("\n=== Test 2: Concurrent Sequence Copy Operations (%d threads, %d iterations) ===\n", 
            num_threads, iterations_per_test);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(concurrent_sequence_copy_test, model, cparams, params, 
                                 iterations_per_test, t);
        }
        for (auto & thread : threads) {
            thread.join();
        }
    }
    LOG_INF("Cache operations: %d, errors: %d\n",
            g_cache_operations.load(), g_errors.load());
    
    int errors_after_test2 = g_errors.load();
    g_cache_operations = 0;
    
    LOG_INF("\n=== Test 3: Concurrent Cache Clear Operations (%d threads, %d iterations) ===\n", 
            num_threads, iterations_per_test * 2);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(concurrent_cache_clear_test, model, cparams, params, 
                                 iterations_per_test * 2, t);
        }
        for (auto & thread : threads) {
            thread.join();
        }
    }
    LOG_INF("Cache operations: %d, errors: %d\n",
            g_cache_operations.load(), g_errors.load());
    
    int errors_after_test3 = g_errors.load();
    g_cache_operations = 0;
    
    LOG_INF("\n=== Test 4: Mixed Concurrent Operations (%d threads, %d iterations) ===\n", 
            num_threads, iterations_per_test * 3);
    {
        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back(concurrent_mixed_operations_test, model, cparams, params, 
                                 iterations_per_test * 3, t);
        }
        for (auto & thread : threads) {
            thread.join();
        }
    }
    LOG_INF("Cache operations: %d, errors: %d\n",
            g_cache_operations.load(), g_errors.load());
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    int total_errors = g_errors.load();
    
    LOG_INF("\n=== KV Cache Concurrent Test Summary ===\n");
    LOG_INF("Total duration: %.2f seconds\n", duration.count() / 1000.0);
    LOG_INF("Total errors: %d\n", total_errors);
    LOG_INF("  After test 1: %d\n", errors_after_test1);
    LOG_INF("  After test 2: %d\n", errors_after_test2);
    LOG_INF("  After test 3: %d\n", errors_after_test3);
    LOG_INF("  After test 4: %d\n", total_errors);
    
    llama_model_free(model);
    
    if (total_errors > 0) {
        LOG_ERR("KV cache concurrent tests completed with %d errors\n", total_errors);
        return 1;
    }
    
    LOG_INF("All KV cache concurrent tests passed successfully!\n");
    return 0;
}
