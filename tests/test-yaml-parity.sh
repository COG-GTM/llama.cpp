#!/bin/bash

set -e

LLAMA_CLI="./llama-cli"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_ROOT/configs/minimal.yaml"
MODEL_PATH="$REPO_ROOT/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Running with YAML config..."
YAML_OUTPUT=$($LLAMA_CLI --config "$CONFIG_FILE" -no-cnv 2>/dev/null | tail -n +2)

echo "Running with equivalent flags..."
FLAGS_OUTPUT=$($LLAMA_CLI -m "$MODEL_PATH" -n 16 -s 42 -c 256 --temp 0.0 -p "Hello from YAML" --simple-io -no-cnv 2>/dev/null | tail -n +2)

if [ "$YAML_OUTPUT" = "$FLAGS_OUTPUT" ]; then
    echo "PARITY TEST PASSED: YAML and flags produce identical output"
    exit 0
else
    echo "PARITY TEST FAILED: Outputs differ"
    echo "YAML output:"
    echo "$YAML_OUTPUT"
    echo "Flags output:"
    echo "$FLAGS_OUTPUT"
    exit 1
fi
