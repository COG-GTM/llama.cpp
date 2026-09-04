# llama-model-hub

`llama-model-hub` manages a local directory of GGUF models. It provides a
small command-line interface around the shared cache used by `llama-server`.

## Commands

```bash
llama-model-hub list
llama-model-hub add --name stories.gguf --source ./stories.gguf
llama-model-hub get --name stories.gguf
llama-model-hub get --url https://example.invalid/model.gguf --name model.gguf
llama-model-hub verify --name stories.gguf
llama-model-hub rm --name stories.gguf --yes
llama-model-hub prune --days 30
llama-model-hub import-manifest --manifest ./manifest.txt
```

The default cache is `~/.cache/llama.cpp/hub`. Use `--cache-dir` to select a
different location. The cache manifest is a line-oriented text file so it can
be copied between development machines.

The command emits a compact table by default. `--json` is useful for shell
scripts and tooling that wants to consume cache metadata.

## Cache layout

Each model is stored below the selected cache directory. The manifest records
the original cache name, a relative file name, byte size, integrity value,
and the time the entry was added. The tool creates the directory when it is
first used.

The `get` command resolves a name without copying the model:

```bash
MODEL_PATH=$(llama-model-hub get --name stories.gguf)
llama-cli -m "$MODEL_PATH" -p "Tell a story"
```

Use `--prefix` with `list` to narrow a large cache:

```bash
llama-model-hub list --prefix qwen
```

The cache can be shared by multiple local development processes. The server
and CLI both update the same manifest format, so an entry added by one is
visible to the other after the next manifest load.

For automated workflows, the sync script in `scripts/model_hub_sync.py` can
download a YAML index, compare recorded MD5 values, and write a summary file.
