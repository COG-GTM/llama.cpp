#include "../src/llama-io.h"
#include "ggml.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

class MockWriter : public llama_io_write_i {
private:
    std::vector<uint8_t> buffer;
    size_t bytes_written = 0;

public:
    void write(const void * src, size_t size) override {
        const uint8_t * data = static_cast<const uint8_t *>(src);
        buffer.insert(buffer.end(), data, data + size);
        bytes_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        (void)tensor;
        (void)offset;
        std::vector<uint8_t> dummy_data(size, 0x42);
        write(dummy_data.data(), size);
    }

    size_t n_bytes() override {
        return bytes_written;
    }

    const std::vector<uint8_t>& get_buffer() const {
        return buffer;
    }

    void clear() {
        buffer.clear();
        bytes_written = 0;
    }
};

class MockReader : public llama_io_read_i {
private:
    std::vector<uint8_t> buffer;
    size_t read_pos = 0;
    size_t bytes_read = 0;

public:
    void set_buffer(const std::vector<uint8_t>& data) {
        buffer = data;
        read_pos = 0;
        bytes_read = 0;
    }

    const uint8_t * read(size_t size) override {
        if (read_pos + size > buffer.size()) {
            return nullptr;
        }
        const uint8_t * result = buffer.data() + read_pos;
        read_pos += size;
        bytes_read += size;
        return result;
    }

    void read_to(void * dst, size_t size) override {
        if (read_pos + size > buffer.size()) {
            return;
        }
        std::memcpy(dst, buffer.data() + read_pos, size);
        read_pos += size;
        bytes_read += size;
    }

    size_t n_bytes() override {
        return bytes_read;
    }

    void reset() {
        read_pos = 0;
        bytes_read = 0;
    }
};

static void test_write_string_basic() {
    std::cout << "Testing write_string basic functionality..." << std::endl;
    
    {
        MockWriter writer;
        std::string test_str = "hello";
        
        writer.write_string(test_str);
        
        const auto& buffer = writer.get_buffer();
        assert(buffer.size() == sizeof(uint32_t) + test_str.size());
        assert(writer.n_bytes() == sizeof(uint32_t) + test_str.size());
        
        uint32_t stored_size;
        std::memcpy(&stored_size, buffer.data(), sizeof(uint32_t));
        assert(stored_size == test_str.size());
        
        std::string stored_str(buffer.begin() + sizeof(uint32_t), buffer.end());
        assert(stored_str == test_str);
        
        std::cout << "  ✓ Basic string writing" << std::endl;
    }
    
    {
        MockWriter writer;
        std::string empty_str = "";
        
        writer.write_string(empty_str);
        
        const auto& buffer = writer.get_buffer();
        assert(buffer.size() == sizeof(uint32_t));
        assert(writer.n_bytes() == sizeof(uint32_t));
        
        uint32_t stored_size;
        std::memcpy(&stored_size, buffer.data(), sizeof(uint32_t));
        assert(stored_size == 0);
        
        std::cout << "  ✓ Empty string writing" << std::endl;
    }
    
    {
        MockWriter writer;
        std::string long_str(1000, 'x');
        
        writer.write_string(long_str);
        
        const auto& buffer = writer.get_buffer();
        assert(buffer.size() == sizeof(uint32_t) + long_str.size());
        assert(writer.n_bytes() == sizeof(uint32_t) + long_str.size());
        
        uint32_t stored_size;
        std::memcpy(&stored_size, buffer.data(), sizeof(uint32_t));
        assert(stored_size == long_str.size());
        
        std::string stored_str(buffer.begin() + sizeof(uint32_t), buffer.end());
        assert(stored_str == long_str);
        
        std::cout << "  ✓ Long string writing" << std::endl;
    }
}

static void test_read_string_basic() {
    std::cout << "Testing read_string basic functionality..." << std::endl;
    
    {
        MockReader reader;
        std::string original = "hello";
        
        std::vector<uint8_t> buffer;
        uint32_t size = original.size();
        buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(&size) + sizeof(size));
        buffer.insert(buffer.end(), original.begin(), original.end());
        
        reader.set_buffer(buffer);
        
        std::string result;
        reader.read_string(result);
        
        assert(result == original);
        assert(reader.n_bytes() == sizeof(uint32_t) + original.size());
        
        std::cout << "  ✓ Basic string reading" << std::endl;
    }
    
    {
        MockReader reader;
        std::string original = "";
        
        std::vector<uint8_t> buffer;
        uint32_t size = 0;
        buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(&size) + sizeof(size));
        
        reader.set_buffer(buffer);
        
        std::string result;
        reader.read_string(result);
        
        assert(result == original);
        assert(reader.n_bytes() == sizeof(uint32_t));
        
        std::cout << "  ✓ Empty string reading" << std::endl;
    }
    
    {
        MockReader reader;
        std::string original(500, 'y');
        
        std::vector<uint8_t> buffer;
        uint32_t size = original.size();
        buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&size), reinterpret_cast<uint8_t*>(&size) + sizeof(size));
        buffer.insert(buffer.end(), original.begin(), original.end());
        
        reader.set_buffer(buffer);
        
        std::string result;
        reader.read_string(result);
        
        assert(result == original);
        assert(reader.n_bytes() == sizeof(uint32_t) + original.size());
        
        std::cout << "  ✓ Long string reading" << std::endl;
    }
}

static void test_write_read_roundtrip() {
    std::cout << "Testing write/read roundtrip..." << std::endl;
    
    std::vector<std::string> test_strings = {
        "",
        "a",
        "hello world",
        "special chars: !@#$%^&*()",
        std::string(100, 'z'),
        "unicode: 你好世界",
        "newlines\nand\ttabs",
        std::string(1, '\0') + "null byte test"
    };
    
    for (const auto& original : test_strings) {
        MockWriter writer;
        writer.write_string(original);
        
        MockReader reader;
        reader.set_buffer(writer.get_buffer());
        
        std::string result;
        reader.read_string(result);
        
        assert(result == original);
        assert(writer.n_bytes() == reader.n_bytes());
    }
    
    std::cout << "  ✓ All roundtrip tests passed" << std::endl;
}

static void test_multiple_strings() {
    std::cout << "Testing multiple string operations..." << std::endl;
    
    {
        MockWriter writer;
        std::vector<std::string> strings = {"first", "second", "third"};
        
        for (const auto& str : strings) {
            writer.write_string(str);
        }
        
        MockReader reader;
        reader.set_buffer(writer.get_buffer());
        
        for (const auto& expected : strings) {
            std::string result;
            reader.read_string(result);
            assert(result == expected);
        }
        
        assert(writer.n_bytes() == reader.n_bytes());
        
        std::cout << "  ✓ Multiple string write/read" << std::endl;
    }
    
    {
        MockWriter writer;
        
        writer.write_string("first");
        size_t bytes_after_first = writer.n_bytes();
        
        writer.write_string("second");
        size_t bytes_after_second = writer.n_bytes();
        
        assert(bytes_after_second > bytes_after_first);
        
        std::cout << "  ✓ Byte counting with multiple writes" << std::endl;
    }
}

static void test_mock_interfaces() {
    std::cout << "Testing mock interface implementations..." << std::endl;
    
    {
        MockWriter writer;
        assert(writer.n_bytes() == 0);
        
        uint32_t test_data = 0x12345678;
        writer.write(&test_data, sizeof(test_data));
        
        assert(writer.n_bytes() == sizeof(test_data));
        
        const auto& buffer = writer.get_buffer();
        assert(buffer.size() == sizeof(test_data));
        
        uint32_t read_back;
        std::memcpy(&read_back, buffer.data(), sizeof(read_back));
        assert(read_back == test_data);
        
        std::cout << "  ✓ MockWriter basic functionality" << std::endl;
    }
    
    {
        MockReader reader;
        assert(reader.n_bytes() == 0);
        
        uint32_t test_data = 0x87654321;
        std::vector<uint8_t> buffer(reinterpret_cast<uint8_t*>(&test_data), 
                                   reinterpret_cast<uint8_t*>(&test_data) + sizeof(test_data));
        reader.set_buffer(buffer);
        
        uint32_t read_back;
        reader.read_to(&read_back, sizeof(read_back));
        
        assert(read_back == test_data);
        assert(reader.n_bytes() == sizeof(test_data));
        
        std::cout << "  ✓ MockReader basic functionality" << std::endl;
    }
    
    {
        MockWriter writer;
        static ggml_tensor dummy_tensor;
        
        writer.write_tensor(&dummy_tensor, 0, 10);
        
        assert(writer.n_bytes() == 10);
        const auto& buffer = writer.get_buffer();
        assert(buffer.size() == 10);
        
        for (uint8_t byte : buffer) {
            assert(byte == 0x42);
        }
        
        std::cout << "  ✓ MockWriter tensor writing" << std::endl;
    }
}

static void test_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    {
        MockWriter writer;
        std::string binary_str;
        for (int i = 0; i < 256; ++i) {
            binary_str += static_cast<char>(i);
        }
        
        writer.write_string(binary_str);
        
        MockReader reader;
        reader.set_buffer(writer.get_buffer());
        
        std::string result;
        reader.read_string(result);
        
        assert(result == binary_str);
        assert(result.size() == 256);
        
        std::cout << "  ✓ Binary data in strings" << std::endl;
    }
    
    {
        MockWriter writer;
        writer.clear();
        assert(writer.n_bytes() == 0);
        assert(writer.get_buffer().empty());
        
        std::cout << "  ✓ Writer clear functionality" << std::endl;
    }
    
    {
        MockReader reader;
        reader.reset();
        assert(reader.n_bytes() == 0);
        
        std::cout << "  ✓ Reader reset functionality" << std::endl;
    }
}

int main() {
    std::cout << "Running llama-io tests..." << std::endl;
    
    try {
        test_write_string_basic();
        test_read_string_basic();
        test_write_read_roundtrip();
        test_multiple_strings();
        test_mock_interfaces();
        test_edge_cases();
        
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
