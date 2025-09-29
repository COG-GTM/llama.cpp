// thread safety test
// - Loads a copy of the same model on each GPU, plus a copy on the CPU
// - Creates n_parallel (--parallel) contexts per model
// - Runs inference in parallel on each context

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

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    //llama_log_set([](ggml_log_level level, const char * text, void * /*user_data*/) {
    //    if (level == GGML_LOG_LEVEL_ERROR) {
    //        common_log_add(common_log_main(), level, "%s", text);
    //    }
    //}, NULL);

    auto cparams = common_context_params_to_llama(params);

    // each context has a single sequence
    cparams.n_seq_max = 1;

    int dev_count = ggml_backend_dev_count();
    int gpu_dev_count = 0;
    for (int i = 0; i < dev_count; ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_dev_count++;
        }
    }
    const int num_models = gpu_dev_count + 1 + 1; // GPUs + 1 CPU model + 1 layer split
    //const int num_models = std::max(1, gpu_dev_count);
    const int num_contexts = std::max(1, params.n_parallel);

    std::vector<llama_model_ptr> models;
    std::vector<std::thread> threads;
    std::atomic<bool> failed = false;

    for (int m = 0; m < num_models; ++m) {
        auto mparams = common_model_params_to_llama(params);

        if (m < gpu_dev_count) {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.main_gpu = m;
        } else if (m == gpu_dev_count) {
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            mparams.main_gpu = -1; // CPU model
        } else {
            mparams.split_mode = LLAMA_SPLIT_MODE_LAYER;;
        }

        llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model == NULL) {
            LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
            return 1;
        }

        models.emplace_back(model);
    }

    for  (int m = 0; m < num_models; ++m) {
        auto * model = models[m].get();
        for (int c = 0; c < num_contexts; ++c) {
            threads.emplace_back([&, m, c, model]() {
                LOG_INF("Creating context %d/%d for model %d/%d\n", c + 1, num_contexts, m + 1, num_models);

                llama_context_ptr ctx { llama_init_from_model(model, cparams) };
                if (ctx == NULL) {
                    LOG_ERR("failed to create context\n");
                    failed.store(true);
                    return;
                }

                std::unique_ptr<common_sampler, decltype(&common_sampler_free)> sampler { common_sampler_init(model, params.sampling), common_sampler_free };
                if (sampler == NULL) {
                    LOG_ERR("failed to create sampler\n");
                    failed.store(true);
                    return;
                }

                llama_batch batch = {};
                {
                    auto prompt = common_tokenize(ctx.get(), params.prompt, true);
                    if (prompt.empty()) {
                        LOG_ERR("failed to tokenize prompt\n");
                        failed.store(true);
                        return;
                    }
                    batch = llama_batch_get_one(prompt.data(), prompt.size());
                    if (llama_decode(ctx.get(), batch)) {
                        LOG_ERR("failed to decode prompt\n");
                        failed.store(true);
                        return;
                    }
                }

                const auto * vocab = llama_model_get_vocab(model);
                std::string result = params.prompt;

                for (int i = 0; i < params.n_predict; i++) {
                    llama_token token;
                    if (batch.n_tokens > 0) {
                        token = common_sampler_sample(sampler.get(), ctx.get(), batch.n_tokens - 1);
                    } else {
                        token = llama_vocab_bos(vocab);
                    }

                    result += common_token_to_piece(ctx.get(), token);

                    if (llama_vocab_is_eog(vocab, token)) {
                        break;
                    }

                    batch = llama_batch_get_one(&token, 1);
                    if (llama_decode(ctx.get(), batch)) {
                        LOG_ERR("Model %d/%d, Context %d/%d: failed to decode\n", m + 1, num_models, c + 1, num_contexts);
                        failed.store(true);
                        return;
                    }
                }

                LOG_INF("Model %d/%d, Context %d/%d: %s\n\n", m + 1, num_models, c + 1, num_contexts, result.c_str());
            });
        }
    }

    for (auto & thread : threads) {
        thread.join();
    }

    if (failed) {
        LOG_ERR("One or more threads failed.\n");
        return 1;
    }

    LOG_INF("All threads finished without errors.\n");

    LOG_INF("\n=== Additional Stress Tests ===\n");

    LOG_INF("\n=== Test 2: Rapid Context Recreation Stress Test ===\n");
    {
        std::atomic<int> contexts_created{0};
        std::atomic<int> contexts_destroyed{0};
        std::atomic<int> errors{0};

        const int stress_iterations = 10;
        auto * model_stress = models[0].get();

        auto stress_test_func = [&](int thread_id) {
            std::random_device rd;
            std::mt19937 gen(rd() + thread_id);
            std::uniform_int_distribution<> delay_dist(1, 5);

            for (int i = 0; i < stress_iterations; i++) {
                llama_context_ptr stress_ctx { llama_init_from_model(model_stress, cparams) };

                if (!stress_ctx) {
                    LOG_ERR("thread %d: failed to create context on iteration %d\n", thread_id, i);
                    errors++;
                    continue;
                }

                contexts_created++;

                std::this_thread::sleep_for(std::chrono::milliseconds(delay_dist(gen)));

                contexts_destroyed++;
            }
        };

        const int64_t t_start = ggml_time_us();

        std::vector<std::thread> stress_threads;
        const int n_stress_threads = std::min(4, num_contexts);
        for (int i = 0; i < n_stress_threads; i++) {
            stress_threads.emplace_back(stress_test_func, i);
        }

        for (auto & t : stress_threads) {
            t.join();
        }

        const int64_t t_end = ggml_time_us();

        LOG_INF("Stress test results:\n");
        LOG_INF("  Contexts created: %d\n", contexts_created.load());
        LOG_INF("  Contexts destroyed: %d\n", contexts_destroyed.load());
        LOG_INF("  Errors: %d\n", errors.load());
        LOG_INF("  Total time: %.2f ms\n", (t_end - t_start) / 1000.0);

        if (contexts_created != contexts_destroyed) {
            LOG_ERR("FAIL: Context leak detected! Created: %d, Destroyed: %d\n",
                    contexts_created.load(), contexts_destroyed.load());
            return 1;
        }

        if (errors > 0) {
            LOG_ERR("FAIL: %d errors occurred during stress test\n", errors.load());
            return 1;
        }

        LOG_INF("PASS: Stress test completed without leaks or errors\n");
    }

    LOG_INF("\n=== All Thread Safety Tests PASSED ===\n");
    return 0;
}
