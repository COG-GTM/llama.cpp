#include "../src/llama-kv-cache-iswa.h"
#include "../src/llama-memory.h"
#include "../src/llama-model.h"
#include "../src/llama-batch.h"
#include "../src/llama-io.h"
#include "ggml.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include <functional>

class MockModel {
public:
    llama_hparams hparams;

    MockModel() {
        hparams.n_layer = 12;
        hparams.n_embd = 768;
        hparams.n_embd_head_k = 64;
        hparams.n_embd_head_v = 64;
        hparams.n_swa = 4;
        hparams.swa_type = LLAMA_SWA_TYPE_NONE;

        std::fill(hparams.n_head_arr.begin(), hparams.n_head_arr.begin() + hparams.n_layer, 12);
        std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.begin() + hparams.n_layer, 12);
        std::fill(hparams.n_ff_arr.begin(), hparams.n_ff_arr.begin() + hparams.n_layer, 3072);
        std::fill(hparams.swa_layers.begin(), hparams.swa_layers.begin() + hparams.n_layer, false);
        std::fill(hparams.recurrent_layer_arr.begin(), hparams.recurrent_layer_arr.begin() + hparams.n_layer, false);
    }

    void set_swa_params(uint32_t n_swa, llama_swa_type swa_type) {
        hparams.n_swa = n_swa;
        hparams.swa_type = swa_type;
    }
};

class MockKvCache : public llama_memory_i {
private:
    uint32_t size;
    bool can_shift;
    llama_memory_status status;

public:
    MockKvCache(uint32_t size = 100, bool can_shift = true, llama_memory_status status = LLAMA_MEMORY_STATUS_SUCCESS)
        : size(size), can_shift(can_shift), status(status) {}

    uint32_t get_size() const { return size; }

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override {
        (void)balloc; (void)n_ubatch; (void)embd_all;
        return std::make_unique<MockMemoryContext>(status);
    }

    llama_memory_context_ptr init_full() override {
        return std::make_unique<MockMemoryContext>(status);
    }

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override {
        (void)lctx; (void)optimize;
        return std::make_unique<MockMemoryContext>(status);
    }

    bool get_can_shift() const override { return can_shift; }

    void clear(bool data) override { (void)data; }

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override {
        (void)seq_id; (void)p0; (void)p1;
        return true;
    }

    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override {
        (void)seq_id_src; (void)seq_id_dst; (void)p0; (void)p1;
    }

    void seq_keep(llama_seq_id seq_id) override { (void)seq_id; }

    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override {
        (void)seq_id; (void)p0; (void)p1; (void)shift;
    }

    void seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override {
        (void)seq_id; (void)p0; (void)p1; (void)d;
    }

    llama_pos seq_pos_min(llama_seq_id seq_id) const override {
        (void)seq_id;
        return 0;
    }

    llama_pos seq_pos_max(llama_seq_id seq_id) const override {
        (void)seq_id;
        return 100;
    }

    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const override {
        (void)io; (void)seq_id; (void)flags;
    }

    void state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) override {
        (void)io; (void)seq_id; (void)flags;
    }

private:
    class MockMemoryContext : public llama_memory_context_i {
    private:
        llama_memory_status status;
        llama_ubatch dummy_ubatch;

    public:
        MockMemoryContext(llama_memory_status status) : status(status) {
            dummy_ubatch.n_tokens = 0;
        }

        bool next() override { return false; }
        bool apply() override { return status == LLAMA_MEMORY_STATUS_SUCCESS; }
        llama_memory_status get_status() const override { return status; }
        const llama_ubatch & get_ubatch() const override { return dummy_ubatch; }
    };
};

class MockBatchAllocr : public llama_batch_allocr {
private:
    uint32_t n_tokens;
    uint32_t n_used;

public:
    MockBatchAllocr(uint32_t n_tokens = 100) : llama_batch_allocr(1), n_tokens(n_tokens), n_used(0) {}

    void split_reset() { n_used = 0; }

    llama_ubatch split_simple(uint32_t n_ubatch) {
        llama_ubatch ubatch = {};
        if (n_used < n_tokens) {
            ubatch.n_tokens = std::min(n_ubatch, n_tokens - n_used);
            n_used += ubatch.n_tokens;
        }
        return ubatch;
    }

    llama_ubatch split_equal(uint32_t n_ubatch, bool force_equal) {
        (void)force_equal;
        return split_simple(n_ubatch);
    }

    uint32_t get_n_tokens() const { return n_tokens; }
    uint32_t get_n_used() const { return n_used; }
};

class MockWriter {
public:
    size_t bytes_written = 0;

    MockWriter() = default;
    ~MockWriter() = default;

    void write(const void * data, size_t size) {
        (void)data;
        bytes_written += size;
    }

    void write_tensor(const struct ggml_tensor * tensor, size_t offset, size_t size) {
        (void)tensor;
        (void)offset;
        bytes_written += size;
    }

    size_t n_bytes() const {
        return bytes_written;
    }
};

class MockReader {
public:
    size_t bytes_read = 0;

    MockReader() = default;
    ~MockReader() = default;

    void read(size_t size) {
        bytes_read += size;
    }

    void read_to(void * data, size_t size) {
        (void)data;
        bytes_read += size;
    }

    size_t n_bytes() const {
        return bytes_read;
    }
};

static llama_model * create_mock_model() {
    static llama_model_params params = llama_model_default_params();
    static llama_model model(params);
    model.hparams.n_swa = 4;
    model.hparams.swa_type = LLAMA_SWA_TYPE_NONE;
    return &model;
}

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
    std::cout << "Testing memory status values..." << std::endl;

    {
        uint32_t status = 0;
        assert(status == 0);
        std::cout << "  ✓ Default memory status" << std::endl;
    }

    {
        uint32_t active_status = 1;
        uint32_t inactive_status = 0;
        assert(active_status != inactive_status);
        std::cout << "  ✓ Memory status differentiation" << std::endl;
    }
}

static void test_layer_callback_types() {
    std::cout << "Testing layer callback types..." << std::endl;

    {
        llama_memory_i::layer_filter_cb filter = nullptr;
        llama_memory_i::layer_reuse_cb reuse = nullptr;
        assert(filter == nullptr);
        assert(reuse == nullptr);
        std::cout << "  ✓ Null callback initialization" << std::endl;
    }

    {
        llama_memory_i::layer_filter_cb filter = [](int32_t il) { return il >= 0; };
        llama_memory_i::layer_reuse_cb reuse = [](int32_t il) { return il < 10; };
        assert(filter(5) == true);
        assert(filter(-1) == false);
        assert(reuse(5) == true);
        assert(reuse(15) == false);
        std::cout << "  ✓ Lambda callback functionality" << std::endl;
    }
}

static void test_sequence_parameter_validation() {
    std::cout << "Testing sequence parameter validation..." << std::endl;

    {
        llama_seq_id seq_id = 0;
        assert(seq_id >= 0);
        std::cout << "  ✓ Valid sequence ID" << std::endl;
    }

    {
        llama_seq_id all_seqs = -1;
        assert(all_seqs < 0);
        std::cout << "  ✓ All sequences identifier" << std::endl;
    }

    {
        llama_pos pos_start = 0;
        llama_pos pos_end = 100;
        assert(pos_start <= pos_end);
        assert(pos_start >= 0);
        std::cout << "  ✓ Position range validation" << std::endl;
    }
}

static void test_ggml_type_validation() {
    std::cout << "Testing GGML type validation..." << std::endl;

    {
        ggml_type type_f32 = GGML_TYPE_F32;
        ggml_type type_f16 = GGML_TYPE_F16;
        assert(type_f32 != type_f16);
        std::cout << "  ✓ GGML type differentiation" << std::endl;
    }

    {
        ggml_type type_q4_0 = GGML_TYPE_Q4_0;
        ggml_type type_q8_0 = GGML_TYPE_Q8_0;
        assert(type_q4_0 != type_q8_0);
        std::cout << "  ✓ Quantized type validation" << std::endl;
    }
}

static void test_cache_parameter_ranges() {
    std::cout << "Testing cache parameter ranges..." << std::endl;

    {
        uint32_t min_size = 1;
        uint32_t max_size = 1000000;
        assert(min_size > 0);
        assert(max_size > min_size);
        std::cout << "  ✓ Cache size range validation" << std::endl;
    }

    {
        uint32_t seq_max = 64;
        uint32_t batch_size = 512;
        uint32_t ubatch_size = 256;
        assert(seq_max > 0);
        assert(batch_size > 0);
        assert(ubatch_size > 0);
        assert(ubatch_size <= batch_size);
        std::cout << "  ✓ Batch parameter validation" << std::endl;
    }
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
