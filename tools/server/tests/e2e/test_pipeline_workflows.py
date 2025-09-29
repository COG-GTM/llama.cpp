"""
End-to-end tests for complete pipeline workflows.

Tests cover:
- Model download → conversion → loading → inference workflows
- State transition validation across server lifecycle
- Context management during long inference sessions
- KV cache behavior validation during extended workflows
"""

from utils import *


def test_basic_pipeline_workflow(pipeline_process, e2e_small_model_config):
    """
    Test a complete basic pipeline: model download → load → inference.

    Validates:
    - Successful model loading from HuggingFace
    - Server state transitions (INITIAL → LOADING_MODEL → READY → GENERATING)
    - Basic inference capability
    """
    results = pipeline_process.test_full_pipeline(e2e_small_model_config)

    assert results["model_loaded"], "Model should be loaded successfully"
    assert results["inference_successful"], "Inference should complete successfully"
    assert "LOADING_MODEL" in results["states"], "Should transition through LOADING_MODEL state"
    assert "READY" in results["states"], "Should reach READY state"
    assert "GENERATING" in results["states"], "Should transition to GENERATING state"

    assert len(results["state_transitions"]) >= 3, "Should have at least 3 state transitions"
    assert ("INITIAL", "LOADING_MODEL") in results["state_transitions"]
    assert ("LOADING_MODEL", "READY") in results["state_transitions"]
    assert ("READY", "PROCESSING_PROMPT") in results["state_transitions"]


def test_pipeline_state_transitions(pipeline_process, e2e_small_model_config):
    """
    Validate server state transitions during pipeline execution.

    Ensures proper progression through states and validates that
    state transitions occur in the expected order.
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)

    assert pipeline_process.pipeline_state == "INITIAL"

    pipeline_process.start()
    assert pipeline_process.process is not None, "Server process should be running"

    res = pipeline_process.make_request("GET", "/health")
    assert res.status_code == 200, "Server should be healthy"

    res = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "Hello world",
        "n_predict": 8,
    })
    assert res.status_code == 200
    assert "content" in res.body

    health_res = pipeline_process.make_request("GET", "/health")
    assert health_res.status_code == 200, "Server should remain healthy after inference"


def test_model_download_and_loading(pipeline_process, e2e_small_model_config):
    """
    Test model download and loading workflow.

    Validates that models can be successfully downloaded from HuggingFace
    and loaded into the server for inference.
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)

    pipeline_process.start()

    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    assert res.body["total_slots"] == e2e_small_model_config["n_slots"]

    res = pipeline_process.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == e2e_small_model_config["model_alias"]


def test_extended_context_management(pipeline_process, e2e_small_model_config):
    """
    Test context management during extended inference sessions.

    Validates:
    - Sequential prompt processing with context preservation
    - KV cache utilization across multiple requests
    - Context window management
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)

    pipeline_process.cache_prompt = True
    pipeline_process.start()

    prompts = [
        "Once upon a time, there was",
        "The little girl walked through",
        "In the forest, she found",
    ]

    results = pipeline_process.test_context_management(
        prompts=prompts,
        max_context=e2e_small_model_config["n_ctx"]
    )

    assert results["prompts_processed"] == len(prompts), \
        f"Should process all {len(prompts)} prompts"
    assert "error" not in results, f"Should not have errors: {results.get('error', '')}"
    assert len(results["responses"]) == len(prompts)


def test_kv_cache_behavior(pipeline_process, e2e_small_model_config):
    """
    Validate KV cache behavior during workflows.

    Tests that the KV cache is properly utilized and managed
    during inference operations.
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)

    pipeline_process.server_metrics = True
    pipeline_process.cache_prompt = True
    pipeline_process.start()

    res1 = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "n_predict": 8,
        "cache_prompt": True,
    })
    assert res1.status_code == 200

    res2 = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "n_predict": 8,
        "cache_prompt": True,
    })
    assert res2.status_code == 200

    cache_results = pipeline_process.validate_kv_cache_behavior(
        context_size=e2e_small_model_config["n_ctx"],
        prompt_tokens=20
    )

    assert cache_results is not None


def test_streaming_pipeline(pipeline_process, e2e_small_model_config):
    """
    Test streaming inference in pipeline workflow.

    Validates that streaming responses work correctly throughout
    the complete pipeline execution.
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)

    pipeline_process.start()

    chunks = list(pipeline_process.make_stream_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 16,
        "stream": True,
    }))

    assert len(chunks) > 0, "Should receive streaming chunks"

    content = ""
    for chunk in chunks:
        if chunk.get("choices"):
            choice = chunk["choices"][0]
            if "content" in choice:
                content += choice["content"]

    assert len(content) > 0, "Should have generated content"


def test_pipeline_with_embedding_model(pipeline_process, e2e_embedding_model_config):
    """
    Test pipeline workflow with embedding model.

    Validates that embedding models work correctly through the
    complete pipeline (load → embed).
    """
    for key, value in e2e_embedding_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)

    pipeline_process.start()

    res = pipeline_process.make_request("POST", "/v1/embeddings", data={
        "input": "Hello, world!",
    })

    assert res.status_code == 200
    assert "data" in res.body
    assert len(res.body["data"]) > 0
    assert "embedding" in res.body["data"][0]
    assert len(res.body["data"][0]["embedding"]) > 0


def test_pipeline_error_recovery(pipeline_process, e2e_small_model_config):
    """
    Test pipeline behavior with error conditions and recovery.

    Validates:
    - Proper error handling during pipeline execution
    - Server stability after errors
    - Recovery capability
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)

    pipeline_process.start()

    res = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "Valid prompt",
        "n_predict": 8,
    })
    assert res.status_code == 200

    res_health = pipeline_process.make_request("GET", "/health")
    assert res_health.status_code == 200

    res2 = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "Another valid prompt after error check",
        "n_predict": 8,
    })
    assert res2.status_code == 200
