#include "../src/llama-cparams.h"

#include <cassert>
#include <iostream>

static void test_llama_max_parallel_sequences() {
    std::cout << "Testing llama_max_parallel_sequences..." << std::endl;

    {
        size_t result = llama_max_parallel_sequences();
        assert(result == LLAMA_MAX_SEQ);
        assert(result == 64);
        std::cout << "  ✓ Returns correct constant value (64)" << std::endl;
    }

    {
        size_t result1 = llama_max_parallel_sequences();
        size_t result2 = llama_max_parallel_sequences();
        assert(result1 == result2);
        std::cout << "  ✓ Consistent return value across multiple calls" << std::endl;
    }

    {
        size_t result = llama_max_parallel_sequences();
        assert(result > 0);
        assert(result <= 1024);
        std::cout << "  ✓ Returns reasonable value within expected range" << std::endl;
    }
}

static void test_llama_max_seq_constant() {
    std::cout << "Testing LLAMA_MAX_SEQ constant..." << std::endl;

    {
        assert(LLAMA_MAX_SEQ == 64);
        std::cout << "  ✓ LLAMA_MAX_SEQ has expected value" << std::endl;
    }

    {
        assert(LLAMA_MAX_SEQ > 0);
        assert(LLAMA_MAX_SEQ <= 1024);
        std::cout << "  ✓ LLAMA_MAX_SEQ is within reasonable bounds" << std::endl;
    }
}

static void test_llama_cparams_struct() {
    std::cout << "Testing llama_cparams struct..." << std::endl;

    {
        llama_cparams params = {};
        assert(params.n_ctx == 0);
        assert(params.n_batch == 0);
        assert(params.n_ubatch == 0);
        assert(params.n_seq_max == 0);
        assert(params.n_threads == 0);
        assert(params.n_threads_batch == 0);
        std::cout << "  ✓ Default initialization sets numeric fields to zero" << std::endl;
    }

    {
        llama_cparams params = {};
        params.n_ctx = 2048;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.n_seq_max = LLAMA_MAX_SEQ;
        params.n_threads = 4;
        params.n_threads_batch = 4;

        assert(params.n_ctx == 2048);
        assert(params.n_batch == 512);
        assert(params.n_ubatch == 512);
        assert(params.n_seq_max == 64);
        assert(params.n_threads == 4);
        assert(params.n_threads_batch == 4);
        std::cout << "  ✓ Field assignment works correctly" << std::endl;
    }

    {
        llama_cparams params = {};
        params.rope_freq_base = 10000.0f;
        params.rope_freq_scale = 1.0f;
        params.yarn_ext_factor = 1.0f;
        params.yarn_attn_factor = 1.0f;
        params.yarn_beta_fast = 32.0f;
        params.yarn_beta_slow = 1.0f;

        assert(params.rope_freq_base == 10000.0f);
        assert(params.rope_freq_scale == 1.0f);
        assert(params.yarn_ext_factor == 1.0f);
        assert(params.yarn_attn_factor == 1.0f);
        assert(params.yarn_beta_fast == 32.0f);
        assert(params.yarn_beta_slow == 1.0f);
        std::cout << "  ✓ Float field assignment works correctly" << std::endl;
    }

    {
        llama_cparams params = {};
        params.embeddings = true;
        params.causal_attn = false;
        params.offload_kqv = true;
        params.flash_attn = false;
        params.no_perf = true;
        params.warmup = false;
        params.op_offload = true;
        params.kv_unified = false;

        assert(params.embeddings == true);
        assert(params.causal_attn == false);
        assert(params.offload_kqv == true);
        assert(params.flash_attn == false);
        assert(params.no_perf == true);
        assert(params.warmup == false);
        assert(params.op_offload == true);
        assert(params.kv_unified == false);
        std::cout << "  ✓ Boolean field assignment works correctly" << std::endl;
    }
}

int main() {
    std::cout << "Running llama-cparams tests..." << std::endl;

    try {
        test_llama_max_parallel_sequences();
        test_llama_max_seq_constant();
        test_llama_cparams_struct();

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
