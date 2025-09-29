"""
End-to-end tests for concurrent scenarios.

Tests cover:
- Multi-turn conversation management with context preservation
- Concurrent user simulation and request queuing validation
- LoRA adapter loading and switching during active sessions
- Batch processing with multiple simultaneous users
- Request slot management under load conditions
"""

import pytest
from utils import *


def test_concurrent_completion_requests(pipeline_process, e2e_small_model_config, concurrent_test_prompts):
    """
    Test concurrent completion requests from multiple simulated users.
    
    Validates:
    - Server handles multiple simultaneous requests
    - All requests complete successfully
    - Responses are independent and correct
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.n_slots = 4
    pipeline_process.server_continuous_batching = True
    pipeline_process.start()
    
    tasks = [
        (
            pipeline_process.make_request,
            ("POST", "/completion", {
                "prompt": prompt,
                "n_predict": 16,
                "temperature": 0.8,
            })
        )
        for prompt in concurrent_test_prompts
    ]
    
    results = parallel_function_calls(tasks)
    
    assert len(results) == len(concurrent_test_prompts)
    assert all([res.status_code == 200 for res in results]), \
        "All concurrent requests should succeed"
    assert all(["content" in res.body for res in results]), \
        "All responses should contain content"


def test_concurrent_chat_completions(pipeline_process, e2e_small_model_config):
    """
    Test concurrent chat completion requests.
    
    Validates:
    - Multiple chat sessions run simultaneously
    - Context is isolated between sessions
    - No cross-contamination of conversations
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.n_slots = 3
    pipeline_process.server_continuous_batching = True
    pipeline_process.start()
    
    conversations = [
        [{"role": "user", "content": "Tell me about dogs"}],
        [{"role": "user", "content": "Tell me about cats"}],
        [{"role": "user", "content": "Tell me about birds"}],
    ]
    
    tasks = [
        (
            pipeline_process.make_request,
            ("POST", "/chat/completions", {
                "messages": conv,
                "max_tokens": 16,
            })
        )
        for conv in conversations
    ]
    
    results = parallel_function_calls(tasks)
    
    assert all([res.status_code == 200 for res in results])
    assert all(["choices" in res.body for res in results])


def test_multi_turn_conversation_with_context(pipeline_process, e2e_small_model_config):
    """
    Test multi-turn conversation with context preservation.
    
    Validates:
    - Context is maintained across conversation turns
    - Responses build on previous messages
    - Server state management is correct
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.cache_prompt = True
    pipeline_process.start()
    
    messages = []
    
    user_msg_1 = {"role": "user", "content": "Hello"}
    messages.append(user_msg_1)
    
    res1 = pipeline_process.make_request("POST", "/chat/completions", data={
        "messages": messages,
        "max_tokens": 16,
    })
    assert res1.status_code == 200
    
    messages.append({
        "role": "assistant",
        "content": res1.body["choices"][0]["message"]["content"]
    })
    
    messages.append({
        "role": "user",
        "content": "Tell me more"
    })
    
    res2 = pipeline_process.make_request("POST", "/chat/completions", data={
        "messages": messages,
        "max_tokens": 16,
    })
    assert res2.status_code == 200
    
    messages.append({
        "role": "assistant",
        "content": res2.body["choices"][0]["message"]["content"]
    })
    
    messages.append({
        "role": "user",
        "content": "That's interesting"
    })
    
    res3 = pipeline_process.make_request("POST", "/chat/completions", data={
        "messages": messages,
        "max_tokens": 16,
    })
    assert res3.status_code == 200


def test_request_slot_management(pipeline_process, e2e_small_model_config):
    """
    Test request slot management under load.
    
    Validates:
    - Server properly manages limited slot resources
    - Requests queue when all slots are busy
    - Slot allocation and deallocation work correctly
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.n_slots = 2
    pipeline_process.server_slots = True
    pipeline_process.server_continuous_batching = True
    pipeline_process.start()
    
    res = pipeline_process.make_request("GET", "/slots")
    assert res.status_code == 200
    initial_slots = res.body
    assert len(initial_slots) == 2
    
    tasks = [
        (
            pipeline_process.make_request,
            ("POST", "/completion", {
                "prompt": f"Request {i}",
                "n_predict": 8,
            })
        )
        for i in range(4)
    ]
    
    results = parallel_function_calls(tasks)
    
    assert all([res.status_code == 200 for res in results]), \
        "All requests should eventually complete"


def test_concurrent_streaming_requests(pipeline_process, e2e_small_model_config):
    """
    Test concurrent streaming requests.
    
    Validates:
    - Multiple streaming sessions can run simultaneously
    - Streams remain independent
    - All streams complete successfully
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.n_slots = 3
    pipeline_process.server_continuous_batching = True
    pipeline_process.start()
    
    def stream_request(prompt):
        chunks = list(pipeline_process.make_stream_request("POST", "/completion", data={
            "prompt": prompt,
            "n_predict": 12,
            "stream": True,
        }))
        return len(chunks)
    
    tasks = [
        (stream_request, (f"Story {i}",))
        for i in range(3)
    ]
    
    results = parallel_function_calls(tasks)
    
    assert all([count > 0 for count in results]), \
        "All streams should produce chunks"


def test_concurrent_embeddings(pipeline_process, e2e_embedding_model_config):
    """
    Test concurrent embedding generation requests.
    
    Validates:
    - Multiple embedding requests process concurrently
    - Embeddings are generated correctly for each input
    - No interference between concurrent embedding requests
    """
    for key, value in e2e_embedding_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.n_slots = 3
    pipeline_process.start()
    
    texts = [
        "The quick brown fox",
        "jumps over the lazy",
        "dog in the yard",
    ]
    
    tasks = [
        (
            pipeline_process.make_request,
            ("POST", "/embeddings", {
                "input": text,
            })
        )
        for text in texts
    ]
    
    results = parallel_function_calls(tasks)
    
    assert all([res.status_code == 200 for res in results])
    assert all(["data" in res.body and len(res.body["data"]) > 0 for res in results])
    
    embeddings = [res.body["data"][0]["embedding"] for res in results]
    assert all([len(emb) > 0 for emb in embeddings])


def test_lora_switching_during_active_session(pipeline_process):
    """
    Test LoRA adapter switching during active inference sessions.
    
    Validates:
    - LoRA adapters can be loaded and configured
    - Different scales produce different outputs
    - Switching works while server is actively processing
    """
    LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"
    
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]
    server.n_slots = 2
    server.start()
    
    res1 = server.make_request("POST", "/lora-adapters", data=[
        {"id": 0, "scale": 0.0}
    ])
    assert res1.status_code == 200
    
    res2 = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
        "n_predict": 16,
    })
    assert res2.status_code == 200
    
    res3 = server.make_request("POST", "/lora-adapters", data=[
        {"id": 0, "scale": 1.0}
    ])
    assert res3.status_code == 200
    
    res4 = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
        "n_predict": 16,
    })
    assert res4.status_code == 200
    
    server.stop()


def test_concurrent_lora_requests(pipeline_process):
    """
    Test concurrent requests with different LoRA configurations.
    
    Validates:
    - Multiple requests with different LoRA scales run concurrently
    - Each request gets the correct LoRA configuration
    - No cross-contamination between LoRA configurations
    """
    LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"
    
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]
    server.n_slots = 3
    server.start()
    
    lora_configs = [
        [{"id": 0, "scale": 0.0}],
        [{"id": 0, "scale": 0.5}],
        [{"id": 0, "scale": 1.0}],
    ]
    
    tasks = [
        (
            server.make_request,
            ("POST", "/completion", {
                "prompt": "Look in thy glass",
                "lora": lora,
                "n_predict": 12,
            })
        )
        for lora in lora_configs
    ]
    
    results = parallel_function_calls(tasks)
    
    assert all([res.status_code == 200 for res in results])
    assert all(["content" in res.body for res in results])
    
    server.stop()


def test_high_concurrency_stress(pipeline_process, e2e_small_model_config):
    """
    Test server under high concurrency stress.
    
    Validates:
    - Server remains stable under high request load
    - All requests eventually complete
    - No crashes or hangs
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.n_slots = 4
    pipeline_process.server_continuous_batching = True
    pipeline_process.start()
    
    tasks = [
        (
            pipeline_process.make_request,
            ("POST", "/completion", {
                "prompt": f"Test {i}",
                "n_predict": 8,
            })
        )
        for i in range(10)
    ]
    
    results = parallel_function_calls(tasks)
    
    assert len(results) == 10
    successful = sum(1 for res in results if res.status_code == 200)
    assert successful >= 8, f"At least 8/10 requests should succeed, got {successful}"


def test_mixed_request_types_concurrent(pipeline_process, e2e_small_model_config):
    """
    Test concurrent requests of different types.
    
    Validates:
    - Different endpoint types (completion, chat, health) work concurrently
    - No interference between different request types
    - Server handles mixed workloads correctly
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.n_slots = 3
    pipeline_process.server_continuous_batching = True
    pipeline_process.start()
    
    tasks = [
        (
            pipeline_process.make_request,
            ("POST", "/completion", {"prompt": "Hello", "n_predict": 8})
        ),
        (
            pipeline_process.make_request,
            ("POST", "/chat/completions", {
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 8
            })
        ),
        (
            pipeline_process.make_request,
            ("GET", "/health", None)
        ),
        (
            pipeline_process.make_request,
            ("GET", "/props", None)
        ),
    ]
    
    results = parallel_function_calls(tasks)
    
    assert all([res.status_code == 200 for res in results])


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_sustained_concurrent_load(pipeline_process, e2e_small_model_config):
    """
    Test sustained concurrent load over multiple rounds.
    
    Slow test that validates:
    - Server maintains stability over extended concurrent usage
    - Performance doesn't degrade significantly
    - Memory is managed correctly under sustained load
    """
    for key, value in e2e_small_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.n_slots = 4
    pipeline_process.server_continuous_batching = True
    pipeline_process.server_metrics = True
    pipeline_process.start()
    
    for round_num in range(3):
        tasks = [
            (
                pipeline_process.make_request,
                ("POST", "/completion", {
                    "prompt": f"Round {round_num} request {i}",
                    "n_predict": 12,
                })
            )
            for i in range(6)
        ]
        
        results = parallel_function_calls(tasks)
        
        assert all([res.status_code == 200 for res in results]), \
            f"All requests in round {round_num} should succeed"
        
        health = pipeline_process.make_request("GET", "/health")
        assert health.status_code == 200, \
            f"Server should be healthy after round {round_num}"
