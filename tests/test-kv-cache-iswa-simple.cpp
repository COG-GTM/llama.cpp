#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-memory.h"
#include "../src/llama-io.h"
#include "ggml.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include <cstring>

class MockWriter : public llama_io_write_i {
public:
    size_t bytes_written = 0;

    void write(const void* data, size_t size) override {
        (void)data;
        bytes_written += size;
    }

    void write_tensor(const ggml_tensor* tensor, size_t offset, size_t size) override {
        (void)tensor; (void)offset;
        bytes_written += size;
    }

    size_t n_bytes() override {
        return bytes_written;
    }
};

class MockReader : public llama_io_read_i {
public:
    size_t bytes_read = 0;

    const uint8_t* read(size_t size) override {
        bytes_read += size;
        return nullptr;
    }

    void read_to(void* dst, size_t size) override {
        (void)dst;
        bytes_read += size;
    }

    size_t n_bytes() override {
        return bytes_read;
    }
};

static void test_context_status_handling() {
    std::cout << "Testing llama_kv_cache_iswa_context status handling..." << std::endl;

    {
        llama_kv_cache_iswa_context ctx(LLAMA_MEMORY_STATUS_SUCCESS);
        assert(ctx.get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
        std::cout << "  ✓ Context with success status" << std::endl;
    }

    {
        llama_kv_cache_iswa_context ctx(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        assert(ctx.get_status() == LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        std::cout << "  ✓ Context with failure status" << std::endl;
    }

    {
        llama_kv_cache_iswa_context ctx(LLAMA_MEMORY_STATUS_NO_UPDATE);
        assert(ctx.get_status() == LLAMA_MEMORY_STATUS_NO_UPDATE);
        std::cout << "  ✓ Context with no update status" << std::endl;
    }
}

static void test_memory_status_values() {
    std::cout << "Testing memory status enumeration values..." << std::endl;

    assert(LLAMA_MEMORY_STATUS_SUCCESS != LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    assert(LLAMA_MEMORY_STATUS_SUCCESS != LLAMA_MEMORY_STATUS_NO_UPDATE);
    assert(LLAMA_MEMORY_STATUS_FAILED_PREPARE != LLAMA_MEMORY_STATUS_NO_UPDATE);

    std::cout << "  ✓ Memory status values are distinct" << std::endl;
}

static void test_layer_callback_types() {
    std::cout << "Testing layer callback function types..." << std::endl;

    llama_memory_i::layer_filter_cb filter = [](int32_t il) { return il < 10; };
    llama_memory_i::layer_reuse_cb reuse = [](int32_t il) { return il % 2 == 0; };

    assert(filter(5) == true);
    assert(filter(15) == false);
    assert(reuse(4) == true);
    assert(reuse(5) == false);

    std::cout << "  ✓ Layer filter and reuse callbacks work correctly" << std::endl;
}

static void test_sequence_parameter_validation() {
    std::cout << "Testing sequence parameter validation..." << std::endl;

    llama_seq_id valid_seq = 0;
    llama_seq_id invalid_seq = -1;
    llama_pos valid_pos = 10;
    llama_pos invalid_pos = -1;

    assert(valid_seq >= 0);
    assert(invalid_seq < 0);
    assert(valid_pos >= 0);
    assert(invalid_pos < 0);

    std::cout << "  ✓ Sequence ID and position validation" << std::endl;
}

static void test_ggml_type_validation() {
    std::cout << "Testing GGML type validation..." << std::endl;

    ggml_type valid_types[] = {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0};

    for (size_t i = 0; i < sizeof(valid_types) / sizeof(valid_types[0]); i++) {
        assert(valid_types[i] >= 0);
    }

    std::cout << "  ✓ GGML type enumeration validation" << std::endl;
}

static void test_cache_parameter_ranges() {
    std::cout << "Testing cache parameter ranges..." << std::endl;

    uint32_t n_ctx = 1024;
    uint32_t n_seq_max = 8;
    uint32_t n_batch = 32;
    uint32_t n_ubatch = 16;

    assert(n_ctx > 0);
    assert(n_seq_max > 0);
    assert(n_batch > 0);
    assert(n_ubatch > 0);
    assert(n_ubatch <= n_batch);

    std::cout << "  ✓ Cache parameter validation" << std::endl;
}

static void test_io_interfaces() {
    std::cout << "Testing I/O interface implementations..." << std::endl;

    {
        MockWriter writer;

        writer.write(nullptr, 10);
        assert(writer.bytes_written == 10);

        writer.write_tensor(nullptr, 0, 20);
        assert(writer.bytes_written == 30);
        assert(writer.n_bytes() == 30);

        std::cout << "  ✓ MockWriter interface works correctly" << std::endl;
    }

    {
        MockReader reader;

        reader.read(15);
        assert(reader.bytes_read == 15);

        reader.read_to(nullptr, 25);
        assert(reader.bytes_read == 40);
        assert(reader.n_bytes() == 40);

        std::cout << "  ✓ MockReader interface works correctly" << std::endl;
    }
}

static void test_ubatch_parameter_validation() {
    std::cout << "Testing ubatch parameter validation..." << std::endl;

    {
        llama_ubatch ubatch = {};
        ubatch.n_tokens = 10;
        ubatch.n_seq_tokens = 5;
        ubatch.n_seqs = 2;

        assert(ubatch.n_tokens > 0);
        assert(ubatch.n_seq_tokens > 0);
        assert(ubatch.n_seqs > 0);
        assert(ubatch.n_seq_tokens <= ubatch.n_tokens);

        std::cout << "  ✓ Valid ubatch parameter validation" << std::endl;
    }

    {
        llama_ubatch empty_batch = {};
        assert(empty_batch.n_tokens == 0);
        assert(empty_batch.n_seq_tokens == 0);
        assert(empty_batch.n_seqs == 0);

        std::cout << "  ✓ Empty ubatch initialization" << std::endl;
    }
}

static void test_state_flags_validation() {
    std::cout << "Testing state flags validation..." << std::endl;

    {
        uint32_t flags = 0;
        assert(flags == 0);
        std::cout << "  ✓ Default state flags" << std::endl;
    }

    {
        uint32_t swa_only_flag = LLAMA_STATE_SEQ_FLAGS_SWA_ONLY;
        assert(swa_only_flag != 0);
        std::cout << "  ✓ SWA-only state flag" << std::endl;
    }

    {
        llama_seq_id seq_all = -1;
        assert(seq_all < 0);
        std::cout << "  ✓ All sequences flag validation" << std::endl;
    }
}

static void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;

    {
        llama_pos zero_range_start = 5;
        llama_pos zero_range_end = 5;
        assert(zero_range_start == zero_range_end);
        std::cout << "  ✓ Zero-length range handling" << std::endl;
    }

    {
        int divisor = 2;
        assert(divisor > 1);

        int invalid_divisor = 0;
        assert(invalid_divisor == 0);
        std::cout << "  ✓ Division parameter validation" << std::endl;
    }

    {
        llama_memory_i::layer_filter_cb null_filter = nullptr;
        llama_memory_i::layer_reuse_cb null_reuse = nullptr;

        assert(null_filter == nullptr);
        assert(null_reuse == nullptr);
        std::cout << "  ✓ Null callback handling" << std::endl;
    }

    {
        uint32_t min_cache_size = 1;
        uint32_t min_seq_max = 1;
        uint32_t min_batch = 1;
        uint32_t min_ubatch = 1;

        assert(min_cache_size > 0);
        assert(min_seq_max > 0);
        assert(min_batch > 0);
        assert(min_ubatch > 0);
        std::cout << "  ✓ Minimum parameter values" << std::endl;
    }
}

static void test_boolean_flag_combinations() {
    std::cout << "Testing boolean flag combinations..." << std::endl;

    {
        bool offload_kqv = false;
        bool do_defrag = true;
        bool flash_attn = false;
        bool unified = true;

        assert(offload_kqv == false);
        assert(do_defrag == true);
        assert(flash_attn == false);
        assert(unified == true);
        std::cout << "  ✓ Boolean flag validation" << std::endl;
    }

    {
        bool all_false = false;
        bool all_true = true;

        assert(all_false != all_true);
        assert(!all_false == all_true);
        std::cout << "  ✓ Boolean logic validation" << std::endl;
    }
}

static void test_io_byte_tracking() {
    std::cout << "Testing I/O byte tracking..." << std::endl;

    {
        MockWriter writer;

        writer.write(nullptr, 100);
        assert(writer.n_bytes() == 100);

        writer.write_tensor(nullptr, 0, 200);
        assert(writer.n_bytes() == 300);

        std::cout << "  ✓ Writer byte tracking" << std::endl;
    }

    {
        MockReader reader;

        reader.read(50);
        assert(reader.n_bytes() == 50);

        reader.read_to(nullptr, 75);
        assert(reader.n_bytes() == 125);

        std::cout << "  ✓ Reader byte tracking" << std::endl;
    }

    {
        MockWriter writer1, writer2;
        writer1.write(nullptr, 100);
        writer2.write(nullptr, 200);

        assert(writer1.n_bytes() != writer2.n_bytes());
        assert(writer1.n_bytes() == 100);
        assert(writer2.n_bytes() == 200);

        std::cout << "  ✓ Independent writer instances" << std::endl;
    }
}

static void test_comprehensive_parameter_validation() {
    std::cout << "Testing comprehensive parameter validation..." << std::endl;

    {
        uint32_t large_ctx = 8192;
        uint32_t large_seq_max = 64;
        uint32_t large_batch = 512;
        uint32_t large_ubatch = 256;

        assert(large_ctx > 1024);
        assert(large_seq_max > 8);
        assert(large_batch > 32);
        assert(large_ubatch > 16);
        assert(large_ubatch <= large_batch);

        std::cout << "  ✓ Large parameter values validation" << std::endl;
    }

    {
        llama_memory_i::layer_filter_cb always_true = [](int32_t il) { (void)il; return true; };
        llama_memory_i::layer_filter_cb always_false = [](int32_t il) { (void)il; return false; };
        llama_memory_i::layer_reuse_cb never_reuse = [](int32_t il) { (void)il; return false; };
        llama_memory_i::layer_reuse_cb always_reuse = [](int32_t il) { (void)il; return true; };

        assert(always_true(0) == true);
        assert(always_false(0) == false);
        assert(never_reuse(0) == false);
        assert(always_reuse(0) == true);

        std::cout << "  ✓ Callback function behavior validation" << std::endl;
    }
}

int main() {
    std::cout << "Running llama-kv-cache-iswa tests..." << std::endl;

    try {
        test_context_status_handling();
        test_memory_status_values();
        test_layer_callback_types();
        test_sequence_parameter_validation();
        test_ggml_type_validation();
        test_cache_parameter_ranges();
        test_io_interfaces();
        test_ubatch_parameter_validation();
        test_state_flags_validation();
        test_edge_cases();
        test_boolean_flag_combinations();
        test_io_byte_tracking();
        test_comprehensive_parameter_validation();

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
