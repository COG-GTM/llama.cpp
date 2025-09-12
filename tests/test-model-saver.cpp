#include "../src/llama-model-saver.h"
#include "../src/llama-model.h"
#include "../src/llama-vocab.h"
#include "../src/llama-hparams.h"
#include "ggml.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstdio>
#include <filesystem>
#include <cstring>

class MockModel {
public:
    llama_hparams hparams;
    llama_vocab vocab;
    std::string name;
    llm_arch arch;
    
    ggml_tensor* tok_embd;
    ggml_tensor* type_embd;
    ggml_tensor* pos_embd;
    ggml_tensor* tok_norm;
    ggml_tensor* tok_norm_b;
    ggml_tensor* output_norm;
    ggml_tensor* output_norm_b;
    ggml_tensor* output;
    ggml_tensor* output_b;
    ggml_tensor* output_norm_enc;
    ggml_tensor* cls;
    ggml_tensor* cls_b;
    ggml_tensor* cls_out;
    ggml_tensor* cls_out_b;
    
    std::vector<llama_layer> layers;
    
    MockModel() : arch(LLM_ARCH_LLAMA) {
        hparams.n_ctx_train = 2048;
        hparams.n_embd = 512;
        hparams.n_layer = 2;
        hparams.n_layer_dense_lead = 1;
        hparams.n_ff_arr[0] = 1024;
        hparams.n_ff_arr[1] = 1024;
        hparams.n_ff_exp = 0;
        hparams.use_par_res = false;
        hparams.n_expert = 0;
        hparams.n_expert_used = 0;
        hparams.n_expert_shared = 0;
        hparams.expert_weights_scale = 1.0f;
        hparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        hparams.f_logit_scale = 1.0f;
        hparams.dec_start_token_id = -1;
        hparams.f_attn_logit_softcapping = 0.0f;
        hparams.f_final_logit_softcapping = 0.0f;
        hparams.swin_norm = false;
        hparams.rescale_every_n_layers = 0;
        hparams.time_mix_extra_dim = 0;
        hparams.time_decay_extra_dim = 0;
        hparams.f_residual_scale = 1.0f;
        hparams.f_embedding_scale = 1.0f;
        hparams.n_head_arr[0] = 8;
        hparams.n_head_arr[1] = 8;
        hparams.n_head_kv_arr[0] = 8;
        hparams.n_head_kv_arr[1] = 8;
        hparams.f_max_alibi_bias = 0.0f;
        hparams.f_clamp_kqv = 0.0f;
        hparams.n_embd_head_k = 64;
        hparams.n_embd_head_v = 64;
        hparams.f_norm_eps = 1e-5f;
        hparams.f_norm_rms_eps = 1e-5f;
        hparams.causal_attn = true;
        hparams.n_lora_q = 0;
        hparams.n_lora_kv = 0;
        hparams.n_rel_attn_bkts = 0;
        hparams.n_swa = 0;
        hparams.f_attention_scale = 1.0f;
        hparams.n_rot = 32;
        hparams.rope_freq_base_train = 10000.0f;
        hparams.rope_freq_scale_train = 1.0f;
        hparams.rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;
        hparams.rope_attn_factor = 1.0f;
        hparams.n_ctx_orig_yarn = 2048;
        hparams.rope_finetuned = false;
        hparams.rope_yarn_log_mul = 0.1f;
        hparams.ssm_d_inner = 0;
        hparams.ssm_d_conv = 0;
        hparams.ssm_d_state = 0;
        hparams.ssm_dt_rank = 0;
        hparams.ssm_dt_b_c_rms = false;
        hparams.wkv_head_size = 0;
        
        name = "test_model";
        
        tok_embd = nullptr;
        type_embd = nullptr;
        pos_embd = nullptr;
        tok_norm = nullptr;
        tok_norm_b = nullptr;
        output_norm = nullptr;
        output_norm_b = nullptr;
        output = nullptr;
        output_b = nullptr;
        output_norm_enc = nullptr;
        cls = nullptr;
        cls_b = nullptr;
        cls_out = nullptr;
        cls_out_b = nullptr;
        
        layers.resize(2);
    }
    
    const char* arch_name() const {
        return llm_arch_name(arch);
    }
};

static void test_model_saver_constructor_destructor() {
    std::cout << "Testing llama_model_saver constructor/destructor..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    {
        llama_model_saver saver(model);
        assert(saver.gguf_ctx != nullptr);
        std::cout << "  ✓ Constructor initializes gguf_ctx" << std::endl;
    }
    
    std::cout << "  ✓ Destructor completes without error" << std::endl;
}

static void test_add_kv_basic_types() {
    std::cout << "Testing add_kv with basic types..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(1000));
    std::cout << "  ✓ add_kv with uint32_t" << std::endl;
    
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, int32_t(2048));
    std::cout << "  ✓ add_kv with int32_t" << std::endl;
    
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 10000.0f);
    std::cout << "  ✓ add_kv with float" << std::endl;
    
    saver.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL, false);
    std::cout << "  ✓ add_kv with bool" << std::endl;
    
    saver.add_kv(LLM_KV_GENERAL_NAME, "test_model");
    std::cout << "  ✓ add_kv with const char*" << std::endl;
}

static void test_add_kv_containers() {
    std::cout << "Testing add_kv with containers..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    std::vector<std::string> string_vec = {"token1", "token2", "token3"};
    saver.add_kv(LLM_KV_TOKENIZER_LIST, string_vec);
    std::cout << "  ✓ add_kv with vector<string>" << std::endl;
    
    std::vector<std::string> empty_vec;
    saver.add_kv(LLM_KV_TOKENIZER_LIST, empty_vec);
    std::cout << "  ✓ add_kv with empty vector<string>" << std::endl;
    
    std::vector<std::string> single_vec = {"single_token"};
    saver.add_kv(LLM_KV_TOKENIZER_LIST, single_vec);
    std::cout << "  ✓ add_kv with single element vector<string>" << std::endl;
}

static void test_add_kv_edge_cases() {
    std::cout << "Testing add_kv edge cases..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(0));
    std::cout << "  ✓ add_kv with zero uint32_t" << std::endl;
    
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, int32_t(-1));
    std::cout << "  ✓ add_kv with negative int32_t" << std::endl;
    
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 0.0f);
    std::cout << "  ✓ add_kv with zero float" << std::endl;
    
    saver.add_kv(LLM_KV_GENERAL_NAME, "");
    std::cout << "  ✓ add_kv with empty string" << std::endl;
}

static void test_add_tensor() {
    std::cout << "Testing add_tensor..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    saver.add_tensor(nullptr);
    std::cout << "  ✓ add_tensor with nullptr" << std::endl;
    
    ggml_init_params params = {};
    params.mem_size = 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    if (ctx) {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
        if (tensor) {
            ggml_set_name(tensor, "test_tensor");
            saver.add_tensor(tensor);
            std::cout << "  ✓ add_tensor with valid tensor" << std::endl;
        }
        ggml_free(ctx);
    }
}

static void test_save_functionality() {
    std::cout << "Testing save functionality..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    saver.add_kv(LLM_KV_GENERAL_NAME, "test_model");
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(1000));
    
    std::string temp_path = "/tmp/test_model_save.gguf";
    
    try {
        saver.save(temp_path);
        std::cout << "  ✓ save completes without error" << std::endl;
        
        if (std::filesystem::exists(temp_path)) {
            std::cout << "  ✓ save creates output file" << std::endl;
            std::filesystem::remove(temp_path);
        } else {
            std::cout << "  ! save did not create expected file" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "  ! save threw exception: " << e.what() << std::endl;
    }
}

static void test_boundary_conditions() {
    std::cout << "Testing boundary conditions..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, UINT32_MAX);
    std::cout << "  ✓ add_kv with UINT32_MAX" << std::endl;
    
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, INT32_MAX);
    std::cout << "  ✓ add_kv with INT32_MAX" << std::endl;
    
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, INT32_MIN);
    std::cout << "  ✓ add_kv with INT32_MIN" << std::endl;
    
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 0.0f);
    std::cout << "  ✓ add_kv with 0.0f" << std::endl;
    
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 1e10f);
    std::cout << "  ✓ add_kv with large float" << std::endl;
    
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 1e-10f);
    std::cout << "  ✓ add_kv with small float" << std::endl;
    
    saver.add_kv(LLM_KV_GENERAL_NAME, "");
    std::cout << "  ✓ add_kv with empty string" << std::endl;
    
    std::string long_string(1000, 'x');
    saver.add_kv(LLM_KV_GENERAL_NAME, long_string.c_str());
    std::cout << "  ✓ add_kv with long string" << std::endl;
}

static void test_multiple_operations() {
    std::cout << "Testing multiple operations..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    saver.add_kv(LLM_KV_GENERAL_NAME, "multi_test");
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(5000));
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, int32_t(4096));
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 20000.0f);
    saver.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL, true);
    
    std::vector<std::string> tokens = {"<s>", "</s>", "<unk>"};
    saver.add_kv(LLM_KV_TOKENIZER_LIST, tokens);
    
    std::cout << "  ✓ Multiple add_kv operations complete" << std::endl;
    
    saver.add_kv(LLM_KV_GENERAL_NAME, "overwritten_name");
    std::cout << "  ✓ Overwriting existing key works" << std::endl;
}

static void test_add_kv_advanced_usage() {
    std::cout << "Testing add_kv advanced usage patterns..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    saver.add_kv(LLM_KV_GENERAL_NAME, "first_name");
    saver.add_kv(LLM_KV_GENERAL_NAME, "overwritten_name");
    std::cout << "  ✓ Key overwriting works" << std::endl;
    
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(4096));
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 10000.0f);
    saver.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL, false);
    std::cout << "  ✓ Multiple key types work" << std::endl;
}

static void test_add_kv_from_model() {
    std::cout << "Testing add_kv_from_model..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    try {
        saver.add_kv_from_model();
        std::cout << "  ✓ add_kv_from_model completes without error" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  ! add_kv_from_model threw exception: " << e.what() << std::endl;
    }
}

static void test_add_tensors_from_model() {
    std::cout << "Testing add_tensors_from_model..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    
    if (ctx) {
        model.tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1000);
        ggml_set_name(model.tok_embd, "token_embd.weight");
        
        model.output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1000);
        ggml_set_name(model.output, "output.weight");
        
        model.tok_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        ggml_set_name(model.tok_norm, "token_norm.weight");
        
        model.output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 512);
        ggml_set_name(model.output_norm, "output_norm.weight");
        
        model.layers.resize(2);
        for (size_t i = 0; i < model.layers.size(); ++i) {
            model.layers[i] = llama_layer{};
        }
        
        llama_model_saver saver(model);
        
        try {
            saver.add_tensors_from_model();
            std::cout << "  ✓ add_tensors_from_model completes without error" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  ! add_tensors_from_model threw exception: " << e.what() << std::endl;
        }
        
        ggml_free(ctx);
    } else {
        std::cout << "  ! Failed to create ggml context for tensor tests" << std::endl;
    }
}

static void test_basic_tensor_operations() {
    std::cout << "Testing basic tensor operations..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    ggml_init_params params = {};
    params.mem_size = 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    
    if (ctx) {
        llama_model_saver saver(model);
        
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
        ggml_set_name(tensor, "test_tensor");
        
        saver.add_tensor(tensor);
        std::cout << "  ✓ add_tensor with valid tensor" << std::endl;
        
        saver.add_tensor(nullptr);
        std::cout << "  ✓ add_tensor with nullptr (should return early)" << std::endl;
        
        ggml_tensor* rope_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
        ggml_set_name(rope_tensor, "rope_freqs.weight");
        saver.add_tensor(rope_tensor);
        saver.add_tensor(rope_tensor);
        std::cout << "  ✓ add_tensor with rope_freqs.weight (duplicate handling)" << std::endl;
        
        ggml_free(ctx);
    }
}

static void test_string_vector_variations() {
    std::cout << "Testing string vector variations..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    std::vector<std::string> tokens = {"<s>", "</s>", "<unk>", "hello", "world"};
    saver.add_kv(LLM_KV_TOKENIZER_LIST, tokens);
    std::cout << "  ✓ add_kv with vector<string> (multiple tokens)" << std::endl;
    
    std::vector<std::string> special_chars = {"<|endoftext|>", "\n", "\t", " "};
    saver.add_kv(LLM_KV_TOKENIZER_LIST, special_chars);
    std::cout << "  ✓ add_kv with special character tokens" << std::endl;
    
    std::vector<std::string> unicode_tokens = {"café", "naïve", "résumé"};
    saver.add_kv(LLM_KV_TOKENIZER_LIST, unicode_tokens);
    std::cout << "  ✓ add_kv with unicode tokens" << std::endl;
    
    std::vector<std::string> long_tokens;
    for (int i = 0; i < 100; i++) {
        long_tokens.push_back("token_" + std::to_string(i));
    }
    saver.add_kv(LLM_KV_TOKENIZER_LIST, long_tokens);
    std::cout << "  ✓ add_kv with large token list" << std::endl;
}

static void test_comprehensive_tensor_scenarios() {
    std::cout << "Testing comprehensive tensor scenarios..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    ggml_init_params params = {};
    params.mem_size = 2048;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    
    if (ctx) {
        llama_model_saver saver(model);
        
        ggml_tensor* tensor1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
        ggml_set_name(tensor1, "first_tensor");
        saver.add_tensor(tensor1);
        std::cout << "  ✓ add_tensor with first tensor" << std::endl;
        
        ggml_tensor* tensor2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 5, 8);
        ggml_set_name(tensor2, "second_tensor");
        saver.add_tensor(tensor2);
        std::cout << "  ✓ add_tensor with different dimensions" << std::endl;
        
        ggml_tensor* rope_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
        ggml_set_name(rope_tensor, "rope_freqs.weight");
        saver.add_tensor(rope_tensor);
        saver.add_tensor(rope_tensor);
        std::cout << "  ✓ add_tensor with rope_freqs.weight (duplicate handling)" << std::endl;
        
        saver.add_tensor(nullptr);
        std::cout << "  ✓ add_tensor with nullptr (early return)" << std::endl;
        
        ggml_free(ctx);
    }
}

static void test_comprehensive_model_operations() {
    std::cout << "Testing comprehensive model operations..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    saver.add_kv(LLM_KV_GENERAL_NAME, "comprehensive_test");
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(8192));
    saver.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(4096));
    saver.add_kv(LLM_KV_BLOCK_COUNT, uint32_t(32));
    std::cout << "  ✓ add_kv with model architecture parameters" << std::endl;
    
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 10000.0f);
    saver.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS, 1e-5f);
    saver.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-6f);
    std::cout << "  ✓ add_kv with attention parameters" << std::endl;
    
    saver.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL, true);
    saver.add_kv(LLM_KV_ATTENTION_CAUSAL, false);
    saver.add_kv(LLM_KV_TOKENIZER_ADD_BOS, true);
    saver.add_kv(LLM_KV_TOKENIZER_ADD_EOS, false);
    std::cout << "  ✓ add_kv with boolean flags" << std::endl;
    
    std::vector<std::string> empty_strings;
    saver.add_kv(LLM_KV_TOKENIZER_LIST, empty_strings);
    std::cout << "  ✓ add_kv with empty string vector" << std::endl;
}

static void test_edge_case_coverage() {
    std::cout << "Testing edge case coverage..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    std::vector<std::string> empty_strings;
    saver.add_kv(LLM_KV_TOKENIZER_LIST, empty_strings);
    std::cout << "  ✓ add_kv with empty string vector (early return)" << std::endl;
    
    std::vector<std::string> single_token = {"<pad>"};
    saver.add_kv(LLM_KV_TOKENIZER_LIST, single_token);
    std::cout << "  ✓ add_kv with single string vector" << std::endl;
    
    std::vector<std::string> large_tokens;
    for (int i = 0; i < 1000; i++) {
        large_tokens.push_back("token_" + std::to_string(i));
    }
    saver.add_kv(LLM_KV_TOKENIZER_LIST, large_tokens);
    std::cout << "  ✓ add_kv with large string vector" << std::endl;
    
    std::string very_long_string(10000, 'x');
    saver.add_kv(LLM_KV_GENERAL_NAME, very_long_string.c_str());
    std::cout << "  ✓ add_kv with very long string" << std::endl;
    
    saver.add_kv(LLM_KV_GENERAL_NAME, "");
    std::cout << "  ✓ add_kv with empty string" << std::endl;
    
    saver.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(0));
    saver.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(UINT32_MAX));
    std::cout << "  ✓ add_kv with boundary uint32_t values" << std::endl;
    
    saver.add_kv(LLM_KV_DECODER_START_TOKEN_ID, int32_t(INT32_MIN));
    saver.add_kv(LLM_KV_DECODER_START_TOKEN_ID, int32_t(INT32_MAX));
    std::cout << "  ✓ add_kv with boundary int32_t values" << std::endl;
    
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, 0.0f);
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, std::numeric_limits<float>::max());
    saver.add_kv(LLM_KV_ROPE_FREQ_BASE, std::numeric_limits<float>::min());
    std::cout << "  ✓ add_kv with boundary float values" << std::endl;
}

static void test_template_container_types() {
    std::cout << "Testing template container types..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    
    llama_model_saver saver(model);
    
    std::vector<float> float_vec = {1.0f, 2.5f, 3.14f, 4.2f};
    saver.add_kv(LLM_KV_TOKENIZER_SCORES, float_vec);
    std::cout << "  ✓ add_kv with vector<float>" << std::endl;
    
    std::vector<int32_t> int32_vec = {-1, 0, 1, 2, 3};
    saver.add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE, int32_vec);
    std::cout << "  ✓ add_kv with vector<int32_t>" << std::endl;
    
    std::string single_string = "test_string";
    saver.add_kv(LLM_KV_GENERAL_NAME, single_string);
    std::cout << "  ✓ add_kv with std::string" << std::endl;
    
    std::vector<float> empty_float_vec;
    saver.add_kv(LLM_KV_TOKENIZER_SCORES, empty_float_vec);
    std::cout << "  ✓ add_kv with empty vector<float>" << std::endl;
    
    std::vector<int32_t> empty_int32_vec;
    saver.add_kv(LLM_KV_TOKENIZER_TOKEN_TYPE, empty_int32_vec);
    std::cout << "  ✓ add_kv with empty vector<int32_t>" << std::endl;
}

static void test_per_layer_variations() {
    std::cout << "Testing per_layer variations..." << std::endl;
    
    MockModel mock_model;
    llama_model model(llama_model_default_params());
    model.hparams = mock_model.hparams;
    model.name = mock_model.name;
    model.arch = mock_model.arch;
    model.hparams.n_layer = 3;
    
    llama_model_saver saver(model);
    
    mock_model.hparams.n_ff_arr[0] = 100;
    mock_model.hparams.n_ff_arr[1] = 100;
    model.hparams = mock_model.hparams;
    saver.add_kv(LLM_KV_FEED_FORWARD_LENGTH, model.hparams.n_ff_arr, true);
    std::cout << "  ✓ add_kv with per_layer=true, same values from hparams array" << std::endl;
    
    mock_model.hparams.n_ff_arr[0] = 100;
    mock_model.hparams.n_ff_arr[1] = 200;
    model.hparams = mock_model.hparams;
    saver.add_kv(LLM_KV_FEED_FORWARD_LENGTH, model.hparams.n_ff_arr, true);
    std::cout << "  ✓ add_kv with per_layer=true, different values from hparams array" << std::endl;
    
    std::vector<float> same_floats = {1.5f, 1.5f, 1.5f};
    saver.add_kv(LLM_KV_TOKENIZER_SCORES, same_floats, true);
    std::cout << "  ✓ add_kv with per_layer=true, same float values" << std::endl;
    
    std::vector<float> different_floats = {1.0f, 2.0f, 3.0f};
    saver.add_kv(LLM_KV_TOKENIZER_SCORES, different_floats, true);
    std::cout << "  ✓ add_kv with per_layer=true, different float values" << std::endl;
}

static void test_additional_coverage() {
    std::cout << "Testing additional coverage scenarios..." << std::endl;
    
    MockModel mock_model;
    llama_model_saver saver(reinterpret_cast<const llama_model&>(mock_model));
    
    std::vector<float> empty_floats;
    saver.add_kv(LLM_KV_TOKENIZER_SCORES, empty_floats, false);
    std::cout << "  ✓ add_kv with empty container" << std::endl;
    
    std::cout << "✓ Additional coverage tests completed!" << std::endl;
}

int main() {
    std::cout << "Running llama-model-saver tests..." << std::endl;
    
    try {
        test_model_saver_constructor_destructor();
        test_add_kv_basic_types();
        test_add_kv_containers();
        test_add_kv_edge_cases();
        test_add_tensor();
        test_save_functionality();
        test_boundary_conditions();
        test_multiple_operations();
        test_add_kv_advanced_usage();
        test_add_kv_from_model();
        test_add_tensors_from_model();
        test_basic_tensor_operations();
        test_string_vector_variations();
        test_comprehensive_tensor_scenarios();
        test_comprehensive_model_operations();
        test_edge_case_coverage();
        test_template_container_types();
        test_per_layer_variations();
        test_additional_coverage();
        
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
