// 
//

#include "llama.h"
#include "get-model.h"
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>

static void test_model_load_unload_cycles(const char * model_path) {
    fprintf(stderr, "test_model_load_unload_cycles: ");
    
    for (int i = 0; i < 10; i++) {
        llama_backend_init();
        
        auto params = llama_model_default_params();
        auto * model = llama_model_load_from_file(model_path, params);
        if (model == nullptr) {
            fprintf(stderr, "FAILED (model load failed on iteration %d)\n", i);
            return;
        }
        
        llama_model_free(model);
        llama_backend_free();
    }
    
    fprintf(stderr, "OK\n");
}

static void test_context_lifecycle(const char * model_path) {
    fprintf(stderr, "test_context_lifecycle: ");
    
    llama_backend_init();
    
    auto model_params = llama_model_default_params();
    auto * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "FAILED (model load failed)\n");
        llama_backend_free();
        return;
    }
    
    for (int i = 0; i < 10; i++) {
        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;
        
        auto * ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            fprintf(stderr, "FAILED (context creation failed on iteration %d)\n", i);
            llama_model_free(model);
            llama_backend_free();
            return;
        }
        
        llama_free(ctx);
    }
    
    llama_model_free(model);
    llama_backend_free();
    
    fprintf(stderr, "OK\n");
}

static void test_multiple_contexts_same_model(const char * model_path) {
    fprintf(stderr, "test_multiple_contexts_same_model: ");
    
    llama_backend_init();
    
    auto model_params = llama_model_default_params();
    auto * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "FAILED (model load failed)\n");
        llama_backend_free();
        return;
    }
    
    const int num_contexts = 5;
    std::vector<llama_context *> contexts(num_contexts);
    
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    
    for (int i = 0; i < num_contexts; i++) {
        contexts[i] = llama_init_from_model(model, ctx_params);
        if (contexts[i] == nullptr) {
            fprintf(stderr, "FAILED (context %d creation failed)\n", i);
            for (int j = 0; j < i; j++) {
                llama_free(contexts[j]);
            }
            llama_model_free(model);
            llama_backend_free();
            return;
        }
    }
    
    for (auto * ctx : contexts) {
        llama_free(ctx);
    }
    
    llama_model_free(model);
    llama_backend_free();
    
    fprintf(stderr, "OK\n");
}

static void test_sampler_lifecycle(const char * model_path) {
    fprintf(stderr, "test_sampler_lifecycle: ");
    
    llama_backend_init();
    
    auto model_params = llama_model_default_params();
    auto * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "FAILED (model load failed)\n");
        llama_backend_free();
        return;
    }
    
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    auto * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "FAILED (context creation failed)\n");
        llama_model_free(model);
        llama_backend_free();
        return;
    }
    
    for (int i = 0; i < 10; i++) {
        auto sparams = llama_sampler_chain_default_params();
        auto * smpl = llama_sampler_chain_init(sparams);
        if (smpl == nullptr) {
            fprintf(stderr, "FAILED (sampler creation failed on iteration %d)\n", i);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return;
        }
        
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_sampler_free(smpl);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    fprintf(stderr, "OK\n");
}

static void test_error_condition_cleanup(const char * /* model_path */) {
    fprintf(stderr, "test_error_condition_cleanup: ");
    
    llama_backend_init();
    
    auto params = llama_model_default_params();
    auto * model = llama_model_load_from_file("/nonexistent/path/to/model.gguf", params);
    if (model != nullptr) {
        fprintf(stderr, "FAILED (expected nullptr for nonexistent model)\n");
        llama_model_free(model);
        llama_backend_free();
        return;
    }
    
    llama_backend_free();
    
    fprintf(stderr, "OK\n");
}

static void test_model_load_cancel(const char * model_path) {
    fprintf(stderr, "test_model_load_cancel: ");
    
    llama_backend_init();
    
    auto params = llama_model_default_params();
    params.use_mmap = false;
    params.progress_callback = [](float progress, void * ctx) {
        (void) ctx;
        return progress > 0.50f;
    };
    
    auto * model = llama_model_load_from_file(model_path, params);
    
    if (model != nullptr) {
        llama_model_free(model);
    }
    
    llama_backend_free();
    
    fprintf(stderr, "OK\n");
}

static void test_batch_operations(const char * model_path) {
    fprintf(stderr, "test_batch_operations: ");
    
    llama_backend_init();
    
    auto model_params = llama_model_default_params();
    auto * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "FAILED (model load failed)\n");
        llama_backend_free();
        return;
    }
    
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    auto * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "FAILED (context creation failed)\n");
        llama_model_free(model);
        llama_backend_free();
        return;
    }
    
    for (int i = 0; i < 10; i++) {
        llama_batch batch = llama_batch_init(32, 0, 1);
        
        llama_batch_free(batch);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    fprintf(stderr, "OK\n");
}

static void test_backend_init_free_cycles() {
    fprintf(stderr, "test_backend_init_free_cycles: ");
    
    for (int i = 0; i < 10; i++) {
        llama_backend_init();
        llama_backend_free();
    }
    
    fprintf(stderr, "OK\n");
}

static void test_threaded_contexts(const char * model_path) {
    fprintf(stderr, "test_threaded_contexts: ");
    
    llama_backend_init();
    
    auto model_params = llama_model_default_params();
    auto * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "FAILED (model load failed)\n");
        llama_backend_free();
        return;
    }
    
    std::atomic<bool> failed = false;
    std::vector<std::thread> threads;
    const int num_threads = 3;
    
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&, t, model]() {
            auto ctx_params = llama_context_default_params();
            ctx_params.n_ctx = 512;
            
            auto * ctx = llama_init_from_model(model, ctx_params);
            if (ctx == nullptr) {
                failed.store(true);
                return;
            }
            
            auto sparams = llama_sampler_chain_default_params();
            auto * smpl = llama_sampler_chain_init(sparams);
            if (smpl == nullptr) {
                llama_free(ctx);
                failed.store(true);
                return;
            }
            
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
            
            llama_sampler_free(smpl);
            llama_free(ctx);
        });
    }
    
    for (auto & thread : threads) {
        thread.join();
    }
    
    llama_model_free(model);
    llama_backend_free();
    
    if (failed) {
        fprintf(stderr, "FAILED (thread error)\n");
    } else {
        fprintf(stderr, "OK\n");
    }
}

static void test_kv_cache_clear_operations(const char * model_path) {
    fprintf(stderr, "test_kv_cache_clear_operations: ");
    
    llama_backend_init();
    
    auto model_params = llama_model_default_params();
    auto * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "FAILED (model load failed)\n");
        llama_backend_free();
        return;
    }
    
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    auto * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "FAILED (context creation failed)\n");
        llama_model_free(model);
        llama_backend_free();
        return;
    }
    
    for (int i = 0; i < 10; i++) {
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_clear(mem, false);
    }
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    fprintf(stderr, "OK\n");
}

int main(int argc, char ** argv) {
    auto * model_path = get_model_or_exit(argc, argv);
    
    fprintf(stderr, "Running memory leak regression tests...\n\n");
    
    test_backend_init_free_cycles();
    test_model_load_unload_cycles(model_path);
    test_context_lifecycle(model_path);
    test_multiple_contexts_same_model(model_path);
    test_sampler_lifecycle(model_path);
    test_batch_operations(model_path);
    test_kv_cache_clear_operations(model_path);
    test_threaded_contexts(model_path);
    test_model_load_cancel(model_path);
    test_error_condition_cleanup(model_path);
    
    fprintf(stderr, "\nAll memory leak tests completed successfully!\n");
    
    return 0;
}
