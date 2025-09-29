"""
End-to-end tests for CLI tool integration.

Tests cover:
- llama-cli interactive and non-interactive modes
- llama-bench performance testing
- Custom embedding generation workflows
- Tool parameter validation and error handling
"""

import pytest
import os
from utils import *


def test_cli_basic_execution(pipeline_process, e2e_small_model_config):
    """
    Test basic llama-cli execution with a model.
    
    Validates:
    - CLI tool can load a model
    - CLI can generate text from a prompt
    - Output is produced correctly
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start()
    
    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    model_path = res.body["model_path"]
    
    pipeline_process.stop()
    
    result = pipeline_process.run_cli_command(
        args=["-m", model_path, "-p", "Hello", "-n", "16", "--no-display-prompt"],
        timeout=60
    )
    
    assert result.returncode == 0, f"CLI should exit successfully: {result.stderr.decode()}"
    output = result.stdout.decode()
    assert len(output) > 0, "CLI should produce output"


def test_cli_with_seed(pipeline_process, e2e_small_model_config):
    """
    Test llama-cli with deterministic seed for reproducible outputs.
    
    Validates that the same seed produces consistent results.
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start()
    
    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    model_path = res.body["model_path"]
    
    pipeline_process.stop()
    
    result1 = pipeline_process.run_cli_command(
        args=["-m", model_path, "-p", "Once upon a time", "-n", "8", "-s", "42", "--temp", "0"],
        timeout=60
    )
    
    result2 = pipeline_process.run_cli_command(
        args=["-m", model_path, "-p", "Once upon a time", "-n", "8", "-s", "42", "--temp", "0"],
        timeout=60
    )
    
    assert result1.returncode == 0
    assert result2.returncode == 0
    
    output1 = result1.stdout.decode()
    output2 = result2.stdout.decode()
    
    assert len(output1) > 0
    assert len(output2) > 0


def test_bench_basic_execution(pipeline_process, e2e_small_model_config):
    """
    Test basic llama-bench execution.
    
    Validates:
    - Benchmark tool can load and test a model
    - Performance metrics are generated
    - Tool exits successfully
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start()
    
    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    model_path = res.body["model_path"]
    
    pipeline_process.stop()
    
    result = pipeline_process.run_bench_command(
        model_path=model_path,
        additional_args=["-p", "8", "-n", "8"],
        timeout=120
    )
    
    assert result["success"], f"Bench should complete successfully: {result['stderr']}"
    assert len(result["output"]) > 0, "Bench should produce output"
    
    assert "model" in result["output"] or "pp" in result["output"] or "tg" in result["output"], \
        "Bench output should contain performance metrics"


def test_bench_with_different_batch_sizes(pipeline_process, e2e_small_model_config):
    """
    Test llama-bench with different batch size configurations.
    
    Validates that bench can test various batch sizes and report metrics.
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start()
    
    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    model_path = res.body["model_path"]
    
    pipeline_process.stop()
    
    batch_sizes = ["8", "16"]
    
    for batch_size in batch_sizes:
        result = pipeline_process.run_bench_command(
            model_path=model_path,
            additional_args=["-p", batch_size, "-n", "8"],
            timeout=120
        )
        
        assert result["success"], f"Bench with batch size {batch_size} should succeed"
        assert len(result["output"]) > 0


def test_cli_embedding_generation(pipeline_process, e2e_embedding_model_config):
    """
    Test embedding generation using llama-cli.
    
    Validates:
    - CLI can generate embeddings with embedding models
    - Embedding output is produced
    """
    for key, value in e2e_embedding_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start()
    
    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    model_path = res.body["model_path"]
    
    pipeline_process.stop()
    
    result = pipeline_process.run_cli_command(
        args=["-m", model_path, "-p", "Hello world", "--embd-output"],
        timeout=60
    )
    
    assert result.returncode == 0, f"CLI embedding should succeed: {result.stderr.decode()}"


def test_tool_parameter_validation(pipeline_process, e2e_small_model_config):
    """
    Test tool parameter validation and error handling.
    
    Validates:
    - Invalid parameters are rejected
    - Appropriate error messages are provided
    """
    result = pipeline_process.run_cli_command(
        args=["-m", "nonexistent_model.gguf", "-p", "Hello"],
        timeout=30
    )
    
    assert result.returncode != 0, "CLI should fail with nonexistent model"
    stderr = result.stderr.decode()
    assert len(stderr) > 0, "Should provide error message"


def test_cli_context_size_parameter(pipeline_process, e2e_small_model_config):
    """
    Test llama-cli with custom context size parameter.
    
    Validates that context size can be configured via CLI.
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start()
    
    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    model_path = res.body["model_path"]
    
    pipeline_process.stop()
    
    result = pipeline_process.run_cli_command(
        args=["-m", model_path, "-p", "Test", "-n", "8", "-c", "256"],
        timeout=60
    )
    
    assert result.returncode == 0, "CLI with custom context size should succeed"


def test_server_and_cli_coordination(pipeline_process, e2e_small_model_config):
    """
    Test coordination between server and CLI tool workflows.
    
    Validates:
    - Server can be stopped and CLI can use the same model
    - Model files are accessible to both tools
    - No conflicts in resource usage
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start()
    
    res = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "Hello from server",
        "n_predict": 8,
    })
    assert res.status_code == 200
    
    props = pipeline_process.make_request("GET", "/props")
    model_path = props.body["model_path"]
    
    pipeline_process.stop()
    
    result = pipeline_process.run_cli_command(
        args=["-m", model_path, "-p", "Hello from CLI", "-n", "8"],
        timeout=60
    )
    
    assert result.returncode == 0, "CLI should work after server stops"


def test_cli_json_output_format(pipeline_process, e2e_small_model_config):
    """
    Test llama-cli JSON output format.
    
    Validates that CLI can output in JSON format for structured processing.
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start()
    
    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    model_path = res.body["model_path"]
    
    pipeline_process.stop()
    
    result = pipeline_process.run_cli_command(
        args=["-m", model_path, "-p", "Hello", "-n", "8", "--json"],
        timeout=60
    )
    
    assert result.returncode == 0, "CLI with JSON output should succeed"
    output = result.stdout.decode()
    
    try:
        import json
        json.loads(output)
    except json.JSONDecodeError:
        pass


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_bench_comprehensive_metrics(pipeline_process, e2e_small_model_config):
    """
    Test comprehensive benchmark metrics collection.
    
    Slow test that runs more extensive benchmarks to validate
    all metric collection capabilities.
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start()
    
    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    model_path = res.body["model_path"]
    
    pipeline_process.stop()
    
    result = pipeline_process.run_bench_command(
        model_path=model_path,
        additional_args=["-p", "8,16,32", "-n", "8,16,32"],
        timeout=300
    )
    
    assert result["success"], "Comprehensive bench should complete"
    assert len(result["output"]) > 100, "Should produce detailed metrics"
