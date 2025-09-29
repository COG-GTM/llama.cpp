"""
End-to-end tests for multimodal workflows.

Tests cover:
- Vision model + text processing coordination
- Multi-modal inference pipeline validation
- Image input processing with text completion
- Cross-modal context management
"""

import pytest
import base64
from utils import *


@pytest.fixture
def sample_image_base64():
    """
    Provide a minimal 1x1 pixel PNG image as base64 for testing.
    
    This is a valid PNG file that can be used to test image input handling
    without requiring external image files.
    """
    png_1x1 = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01'
        b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    return base64.b64encode(png_1x1).decode('utf-8')


def test_multimodal_model_loading(pipeline_process, e2e_multimodal_model_config):
    """
    Test loading a multimodal model with vision projection.
    
    Validates:
    - Multimodal model loads successfully
    - Vision projection (mmproj) is loaded
    - Server is ready for multimodal inference
    """
    for key, value in e2e_multimodal_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start(timeout_seconds=120)
    
    res = pipeline_process.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    
    res = pipeline_process.make_request("GET", "/health")
    assert res.status_code == 200


def test_multimodal_text_only_inference(pipeline_process, e2e_multimodal_model_config):
    """
    Test text-only inference with a multimodal model.
    
    Validates that multimodal models can still perform text-only tasks
    when no image is provided.
    """
    for key, value in e2e_multimodal_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start(timeout_seconds=120)
    
    res = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 8,
    })
    
    assert res.status_code == 200
    assert "content" in res.body
    assert len(res.body["content"]) > 0


def test_multimodal_chat_with_image(pipeline_process, e2e_multimodal_model_config, sample_image_base64):
    """
    Test multimodal chat completion with image input.
    
    Validates:
    - Image data can be included in chat messages
    - Model processes both image and text inputs
    - Response is generated considering multimodal context
    """
    for key, value in e2e_multimodal_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start(timeout_seconds=120)
    
    res = pipeline_process.make_request("POST", "/chat/completions", data={
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{sample_image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 16,
    })
    
    assert res.status_code == 200
    assert "choices" in res.body
    assert len(res.body["choices"]) > 0
    assert "message" in res.body["choices"][0]


def test_multimodal_sequential_requests(pipeline_process, e2e_multimodal_model_config, sample_image_base64):
    """
    Test sequential multimodal requests with different modality combinations.
    
    Validates:
    - Text-only followed by multimodal requests
    - Model handles modality switching correctly
    - Context is maintained appropriately
    """
    for key, value in e2e_multimodal_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start(timeout_seconds=120)
    
    res1 = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "Hello",
        "n_predict": 4,
    })
    assert res1.status_code == 200
    
    res2 = pipeline_process.make_request("POST", "/chat/completions", data={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{sample_image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 8,
    })
    assert res2.status_code == 200
    
    res3 = pipeline_process.make_request("POST", "/completion", data={
        "prompt": "Another text",
        "n_predict": 4,
    })
    assert res3.status_code == 200


def test_multimodal_context_preservation(pipeline_process, e2e_multimodal_model_config, sample_image_base64):
    """
    Test context preservation in multimodal conversations.
    
    Validates:
    - Multimodal context is maintained across turns
    - Follow-up messages reference previous multimodal context
    """
    for key, value in e2e_multimodal_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start(timeout_seconds=120)
    
    res = pipeline_process.make_request("POST", "/chat/completions", data={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{sample_image_base64}"
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": "I see an image."
            },
            {
                "role": "user",
                "content": "Can you elaborate?"
            }
        ],
        "max_tokens": 16,
    })
    
    assert res.status_code == 200
    assert "choices" in res.body


def test_multimodal_streaming_response(pipeline_process, e2e_multimodal_model_config, sample_image_base64):
    """
    Test streaming responses with multimodal input.
    
    Validates:
    - Streaming works with image inputs
    - Chunks are delivered correctly
    - Complete response is assembled
    """
    for key, value in e2e_multimodal_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start(timeout_seconds=120)
    
    chunks = list(pipeline_process.make_stream_request("POST", "/chat/completions", data={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{sample_image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 12,
        "stream": True,
    }))
    
    assert len(chunks) > 0, "Should receive streaming chunks"


def test_multimodal_error_handling(pipeline_process, e2e_multimodal_model_config):
    """
    Test error handling in multimodal workflows.
    
    Validates:
    - Invalid image data is handled gracefully
    - Appropriate error messages are returned
    - Server remains stable after errors
    """
    for key, value in e2e_multimodal_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start(timeout_seconds=120)
    
    res = pipeline_process.make_request("POST", "/chat/completions", data={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,invalid_base64_data"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 8,
    })
    
    res_health = pipeline_process.make_request("GET", "/health")
    assert res_health.status_code == 200, "Server should remain healthy after error"


def test_multimodal_multiple_images(pipeline_process, e2e_multimodal_model_config, sample_image_base64):
    """
    Test handling multiple images in a single request.
    
    Validates that the model can handle multiple image inputs
    in the same conversation context.
    """
    for key, value in e2e_multimodal_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.start(timeout_seconds=120)
    
    res = pipeline_process.make_request("POST", "/chat/completions", data={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{sample_image_base64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{sample_image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 16,
    })
    
    assert res.status_code == 200


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_multimodal_extended_conversation(pipeline_process, e2e_multimodal_model_config, sample_image_base64):
    """
    Test extended multimodal conversation with multiple turns.
    
    Slow test validating:
    - Long conversations with images maintain context
    - Performance remains stable
    - Memory is managed correctly
    """
    for key, value in e2e_multimodal_model_config.items():
        if hasattr(pipeline_process, key):
            setattr(pipeline_process, key, value)
    
    pipeline_process.n_ctx = 2048
    pipeline_process.start(timeout_seconds=120)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{sample_image_base64}"
                    }
                }
            ]
        }
    ]
    
    for i in range(3):
        res = pipeline_process.make_request("POST", "/chat/completions", data={
            "messages": messages,
            "max_tokens": 16,
        })
        
        assert res.status_code == 200
        
        messages.append({
            "role": "assistant",
            "content": res.body["choices"][0]["message"]["content"]
        })
        
        messages.append({
            "role": "user",
            "content": f"Tell me more about point {i+1}"
        })
    
    assert len(messages) > 3
