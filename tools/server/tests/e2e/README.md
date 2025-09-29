# End-to-End Test Suite

This directory contains comprehensive end-to-end (E2E) tests for llama.cpp, extending beyond unit-focused API testing to validate complete user workflows and component integration.

## Overview

The E2E test suite provides comprehensive coverage of:

1. **Pipeline Workflows** - Complete model download, loading, and inference workflows
2. **Tool Integration** - CLI tool testing (llama-cli, llama-bench)
3. **Multimodal Workflows** - Vision + text processing coordination
4. **Concurrent Scenarios** - Multi-user simulation and parallel request handling

## Test Files

### test_pipeline_workflows.py

Tests complete pipeline workflows from model acquisition to inference:

- **Model Download & Loading**: Validates HuggingFace model download and loading
- **State Transitions**: Tracks server state progression (INITIAL → LOADING_MODEL → READY → GENERATING)
- **Context Management**: Tests extended inference sessions with context preservation
- **KV Cache Behavior**: Validates cache utilization during workflows
- **Streaming Pipeline**: Tests streaming inference through complete pipeline
- **Embedding Models**: Validates embedding model pipelines

**Example:**
```bash
./tests.sh e2e/test_pipeline_workflows.py::test_basic_pipeline_workflow
```

### test_tool_integration.py

Tests CLI tool integration and coordination:

- **llama-cli Execution**: Basic and advanced CLI usage patterns
- **llama-bench Testing**: Performance benchmark execution
- **Embedding Generation**: CLI-based embedding workflows
- **Parameter Validation**: Error handling and validation
- **Server/CLI Coordination**: Resource sharing between tools

**Example:**
```bash
./tests.sh e2e/test_tool_integration.py::test_cli_basic_execution
```

### test_multimodal_workflows.py

Tests multimodal (vision + text) processing:

- **Model Loading**: Multimodal model initialization with vision projection
- **Image Processing**: Image input handling with text completion
- **Context Preservation**: Cross-modal context management
- **Sequential Requests**: Mixed text-only and multimodal requests
- **Streaming**: Multimodal streaming responses
- **Error Handling**: Invalid input handling

**Example:**
```bash
./tests.sh e2e/test_multimodal_workflows.py::test_multimodal_chat_with_image
```

### test_concurrent_scenarios.py

Tests concurrent request handling and real-world scenarios:

- **Concurrent Requests**: Multiple simultaneous completion/chat requests
- **Multi-turn Conversations**: Context preservation across conversation turns
- **Slot Management**: Request queuing and slot allocation under load
- **Streaming Concurrency**: Multiple streaming sessions
- **LoRA Switching**: Adapter loading/switching during active sessions
- **Mixed Workloads**: Different request types running concurrently

**Example:**
```bash
./tests.sh e2e/test_concurrent_scenarios.py::test_concurrent_completion_requests
```

## Framework Extensions

### PipelineTestProcess Class

The `PipelineTestProcess` class extends `ServerProcess` with E2E testing capabilities:

```python
from utils import PipelineTestProcess

# Create pipeline test instance
pipeline = PipelineTestProcess()

# Test complete pipeline workflow
results = pipeline.test_full_pipeline({
    "model_hf_repo": "ggml-org/models",
    "model_hf_file": "tinyllamas/stories260K.gguf",
    "n_ctx": 512,
})

# Run CLI commands
result = pipeline.run_cli_command(["-m", model_path, "-p", "Hello", "-n", "16"])

# Run benchmarks
bench_results = pipeline.run_bench_command(model_path, ["-p", "8", "-n", "8"])
```

**Key Methods:**

- `test_full_pipeline(model_config)` - Execute complete pipeline workflow
- `run_cli_command(args, input_text, timeout)` - Execute llama-cli
- `run_bench_command(model_path, args, timeout)` - Execute llama-bench
- `test_context_management(prompts, max_context)` - Test context handling
- `validate_kv_cache_behavior(context_size, tokens)` - Validate cache usage

### Test Fixtures

New pytest fixtures in `conftest.py`:

- **`pipeline_process`** - PipelineTestProcess instance with automatic cleanup
- **`e2e_small_model_config`** - Small model config for fast E2E tests
- **`e2e_embedding_model_config`** - Embedding model configuration
- **`e2e_multimodal_model_config`** - Multimodal model configuration
- **`concurrent_test_prompts`** - Prompts for concurrent testing

## Running E2E Tests

### Run All E2E Tests

```bash
./tests.sh e2e/
```

### Run Specific Test File

```bash
./tests.sh e2e/test_pipeline_workflows.py
```

### Run Single Test

```bash
./tests.sh e2e/test_pipeline_workflows.py::test_basic_pipeline_workflow
```

### Run with Verbose Output

```bash
DEBUG=1 ./tests.sh e2e/ -s -v
```

### Run Slow Tests

Some tests are marked as slow and require the `SLOW_TESTS` environment variable:

```bash
SLOW_TESTS=1 ./tests.sh e2e/
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLAMA_CLI_BIN_PATH` | Path to llama-cli binary | `../../../build/bin/llama-cli` |
| `LLAMA_BENCH_BIN_PATH` | Path to llama-bench binary | `../../../build/bin/llama-bench` |
| `LLAMA_CACHE` | Model cache directory | `tmp` |
| `SLOW_TESTS` | Enable slow tests | `0` |
| `DEBUG` | Enable verbose output | `0` |

### Model Selection

E2E tests use smaller models for CI compatibility:

- **Text Generation**: tinyllama (stories260K.gguf) - Fast, small footprint
- **Embeddings**: bert-bge-small - Efficient embedding generation
- **Multimodal**: tinygemma3 - Compact vision+text model

For local testing with larger models, modify the fixture configurations in `conftest.py`.

## Writing New E2E Tests

### Example Test Structure

```python
def test_my_e2e_workflow(pipeline_process, e2e_small_model_config):
    """
    Test description here.
    
    Validates:
    - Point 1
    - Point 2
    """
    # Configure pipeline
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    # Start server
    pipeline_process.start()
    
    # Test workflow
    res = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "Test",
        "n_predict": 8,
    })
    
    # Assertions
    assert res.status_code == 200
    assert "content" in res.body
```

### Best Practices

1. **Use Fixtures**: Leverage existing fixtures for model configs and test data
2. **Small Models**: Use small models for fast execution in CI
3. **Resource Cleanup**: Fixtures handle cleanup automatically
4. **Test Isolation**: Each test should be independent
5. **Descriptive Names**: Use clear, descriptive test names
6. **Documentation**: Include docstrings explaining what is validated
7. **Slow Tests**: Mark expensive tests with `@pytest.mark.skipif(not is_slow_test_allowed())`

## CI Integration

E2E tests are designed to run in CI environments with:

- 4 vCPU GitHub runners
- Limited memory footprint
- Fast model downloads from HuggingFace
- Reasonable timeout configurations

Tests automatically skip slow scenarios unless `SLOW_TESTS=1` is set.

## Troubleshooting

### Tests Timeout

- Increase timeout in test: `pipeline_process.start(timeout_seconds=120)`
- Use smaller models in CI
- Check network connectivity for model downloads

### Model Download Issues

- Set `LLAMA_CACHE` to a persistent directory
- Pre-download models before running tests
- Check HuggingFace availability

### CLI Tool Not Found

- Ensure binaries are built: `cmake --build build --target llama-cli llama-bench`
- Set `LLAMA_CLI_BIN_PATH` and `LLAMA_BENCH_BIN_PATH`
- Check binary permissions

### Concurrent Test Failures

- Increase `n_slots` for higher concurrency
- Adjust timing expectations for slower systems
- Enable `server_continuous_batching` for better scheduling

## Contributing

When adding new E2E tests:

1. Place tests in appropriate file based on category
2. Use existing fixtures when possible
3. Add new fixtures to `conftest.py` if needed
4. Update this README with new test descriptions
5. Ensure tests pass in CI environment
6. Document special requirements or configurations

## Related Documentation

- [Main Test README](../README.md) - General testing documentation
- [Server Documentation](../../README.md) - llama-server documentation
- [Contributing Guide](../../../../CONTRIBUTING.md) - Project contribution guidelines
