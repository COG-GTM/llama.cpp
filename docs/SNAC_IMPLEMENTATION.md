# SNAC Decoder Implementation for Orpheus TTS

## Overview

This document describes the implementation of SNAC (Multi-Scale Neural Audio Codec) decoder support in llama.cpp for Orpheus TTS models.

## Current Status

### ✅ Completed

1. **Architecture Infrastructure**
   - Added `LLM_ARCH_SNAC_DEC` architecture enum
   - Registered "snac-dec" architecture name
   - Defined 31 SNAC-specific tensor types
   - Added tensor name mappings for decoder, quantizer, and encoder components

2. **GGUF Constants**
   - Added `MODEL_ARCH.SNAC_DEC` to gguf constants
   - Defined tensor enums for all SNAC components
   - Added tensor name format strings

3. **Model Conversion**
   - Implemented `SnacDecModel` class in `convert_hf_to_gguf.py`
   - Handles weight_norm parameters (skips _g and _v suffixes)
   - Configures SNAC-specific hyperparameters

### 🚧 In Progress / TODO

1. **Model Loading (llama-model.cpp)**
   - Need to implement SNAC decoder model loading
   - Load decoder convolution layers
   - Load vector quantizer components (in_proj, out_proj, codebook)
   - Load attention layers if present
   - Handle Snake activation parameters

2. **Forward Pass Implementation (llama.cpp)**
   - Implement SNAC decoder forward pass
   - Vector quantization decoding (from_codes)
   - Decoder blocks with:
     - Transposed convolutions (upsampling)
     - Residual units with dilated convolutions
     - Snake activation function
   - Local multi-head attention (if present)
   - Output convolution and tanh activation

3. **TTS Tool Integration (tools/tts/tts.cpp)**
   - Add SNAC decoder option to TTS tool
   - Support for multi-scale code input
   - Audio generation from hierarchical codes
   - Integration with Orpheus TTS models

4. **Testing**
   - Download and convert SNAC models from HuggingFace
   - Test with Orpheus TTS models
   - Validate audio quality
   - Performance benchmarking

## SNAC Architecture

### Components

1. **Encoder** (not needed for TTS, only for training)
   - Input convolution
   - Encoder blocks with strided convolutions
   - Local attention (optional)
   - Output convolution

2. **Vector Quantizer** (needed for decoding)
   - 4 quantization levels with different strides [8, 4, 2, 1]
   - Each level has:
     - `in_proj`: Projects latent to codebook dimension
     - `codebook`: Embedding table (4096 x 8)
     - `out_proj`: Projects back to latent dimension
   - Residual quantization across levels

3. **Decoder** (main component needed)
   - Input convolution (or direct from quantizer output)
   - Local attention (optional)
   - Decoder blocks (4 blocks for standard config):
     - Transposed convolution for upsampling
     - 3 residual units with dilations [1, 3, 9]
     - Snake activation
   - Output convolution + tanh

### Snake Activation

Formula: `x + (1/alpha) * sin^2(alpha * x)`

Can be implemented using existing ggml operations:
```c
// x_scaled = x * alpha
// sin_x = sin(x_scaled)
// sin2_x = sin_x * sin_x
// result = x + sin2_x / alpha
```

### Tensor Naming Convention

Decoder tensors:
- `decoder.conv_in` - Input convolution
- `decoder.attn_norm`, `decoder.attn_q/k/v/out` - Attention (if present)
- `decoder.block.{i}.conv_up` - Upsampling transposed conv
- `decoder.block.{i}.conv1/2/3` - Residual unit convolutions
- `decoder.block.{i}.snake_alpha` - Snake activation parameters
- `decoder.conv_out` - Output convolution

Quantizer tensors:
- `quantizer.{i}.in_proj` - Input projection for level i
- `quantizer.{i}.out_proj` - Output projection for level i
- `quantizer.{i}.codebook` - Codebook embeddings for level i

## Model Conversion

### Converting SNAC Models

```bash
# Download SNAC model
git clone https://huggingface.co/hubertsiuzdak/snac_24khz

# Convert to GGUF
python convert_hf_to_gguf.py snac_24khz \
    --outfile snac-24khz-f16.gguf \
    --outtype f16
```

### Expected Hyperparameters

From SNAC config.json:
```json
{
  "sampling_rate": 24000,
  "encoder_dim": 64,
  "encoder_rates": [3, 3, 7, 7],
  "latent_dim": 1344,
  "decoder_dim": 1536,
  "decoder_rates": [7, 7, 3, 3],
  "attn_window_size": 32,
  "codebook_size": 4096,
  "codebook_dim": 8,
  "vq_strides": [8, 4, 2, 1]
}
```

## Integration with Orpheus TTS

Orpheus TTS uses a two-model architecture:
1. **Text-to-Codes Model**: LLM that generates hierarchical audio codes
2. **Codes-to-Speech Model**: SNAC decoder that converts codes to audio

Usage flow:
```
Text → Orpheus LLM → Multi-scale codes → SNAC Decoder → Audio waveform
```

## References

- SNAC Paper: https://arxiv.org/abs/2410.14411
- SNAC GitHub: https://github.com/hubertsiuzdak/snac
- Orpheus Models: https://huggingface.co/collections/canopylabs/orpheus-tts-67d9ea3f6c05a941c06ad9d2
- OuteTTS Reference: PR #10784 in llama.cpp

## Implementation Notes

### Key Differences from WavTokenizer

1. **Multi-scale Quantization**: SNAC uses 4 levels with different temporal resolutions
2. **Snake Activation**: Custom activation function (WavTokenizer uses standard activations)
3. **Simpler Architecture**: No PosNet or ConvNext blocks
4. **Hierarchical Codes**: Variable-length codes at different scales

### Performance Considerations

- SNAC is designed for low bitrate (0.98-2.6 kbps)
- Decoder is relatively lightweight
- Main computation in transposed convolutions and residual blocks
- Attention is optional and can be disabled for faster inference

## Next Steps

1. Implement model loading in `llama-model.cpp`
2. Implement forward pass in `llama.cpp`
3. Add SNAC support to TTS tool
4. Test with Orpheus models
5. Add documentation and examples
6. Performance optimization
