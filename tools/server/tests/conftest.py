import pytest
from utils import *


@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    yield
    instances = set(
        server_instances
    )
    for server in instances:
        server.stop()


@pytest.fixture
def pipeline_process():
    """
    Fixture providing a PipelineTestProcess instance for E2E testing.
    Automatically cleaned up after test completion.
    """
    process = PipelineTestProcess()
    yield process
    if process.process is not None:
        process.stop()


@pytest.fixture
def e2e_small_model_config():
    """
    Fixture providing configuration for a small model suitable for E2E testing.
    Uses tinyllama for fast execution in CI environments.
    """
    return {
        "model_hf_repo": "ggml-org/models",
        "model_hf_file": "tinyllamas/stories260K.gguf",
        "model_alias": "tinyllama-e2e",
        "n_ctx": 512,
        "n_batch": 32,
        "n_slots": 2,
        "n_predict": 32,
        "seed": 42,
        "temperature": 0.8,
    }


@pytest.fixture
def e2e_embedding_model_config():
    """
    Fixture providing configuration for embedding model E2E testing.
    """
    return {
        "model_hf_repo": "ggml-org/models",
        "model_hf_file": "bert-bge-small/ggml-model-f16.gguf",
        "model_alias": "bert-e2e",
        "n_ctx": 512,
        "n_batch": 128,
        "n_ubatch": 128,
        "n_slots": 2,
        "seed": 42,
        "server_embeddings": True,
    }


@pytest.fixture
def e2e_multimodal_model_config():
    """
    Fixture providing configuration for multimodal model E2E testing.
    """
    return {
        "model_hf_repo": "ggml-org/tinygemma3-GGUF",
        "model_hf_file": "tinygemma3-Q8_0.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/mmproj-tinygemma3.gguf",
        "model_alias": "tinygemma3-e2e",
        "n_ctx": 1024,
        "n_batch": 32,
        "n_slots": 2,
        "n_predict": 16,
        "seed": 42,
    }


@pytest.fixture
def concurrent_test_prompts():
    """
    Fixture providing a list of prompts for concurrent testing scenarios.
    """
    return [
        "Once upon a time",
        "In a distant land",
        "There was a brave knight",
        "The dragon soared",
        "Magic filled the air",
    ]
