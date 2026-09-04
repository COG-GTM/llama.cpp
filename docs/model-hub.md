# Model Hub

The Model Hub is a local cache and download manager for GGUF model files. It
is shared by `llama-server` and the `llama-model-hub` command-line utility.
The cache keeps model files in a configurable directory and records metadata
in a line-oriented manifest.

## Getting started

Build the server and the command-line tool:

```bash
cmake -B build -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server llama-model-hub
```

The default cache directory is:

```text
~/.cache/llama.cpp/hub
```

The server accepts `--model-cache-dir PATH` to override this location. A
leading `~` is expanded using the current user's home directory.

Add an existing model:

```bash
./build/bin/llama-model-hub add \
    --name qwen2.5-7b.gguf \
    --source ./models/qwen2.5-7b.gguf
```

List available files:

```bash
./build/bin/llama-model-hub list
```

The listing contains the cache name, byte size, relative path, and an
integrity value. The `--json` option is available for scripts and build
systems.

## Cache manifest

The manifest is stored at `manifest.txt` in the cache directory. Each entry
contains five fields:

```text
name=model.gguf
path=model.gguf
size=123456
sha256=0123456789abcdef
added=1710000000
```

Entries are separated by a blank line. `import-manifest` can read a manifest
produced by another cache, while the server reloads the manifest when it
starts. The command-line tool can export and import this metadata as part of
a workstation setup workflow.

## Command-line operations

`list` prints every entry and a summary of the cache. `add` copies a local
file into the cache and records its metadata. `rm` removes a named entry.
`get` resolves an entry to its local path or downloads a URL. `verify`
recomputes the recorded integrity value. `prune` removes entries older than a
specified number of days. `import-manifest` reads entries from a text file.

Examples:

```bash
llama-model-hub get --url https://huggingface.co/example/model.gguf \
    --name example.gguf --token "$HF_TOKEN"
llama-model-hub verify --name example.gguf
llama-model-hub prune --days 45 --yes
```

## Server endpoints

The server exposes cache operations under `/models/cache`.

### List

```http
GET /models/cache
```

Returns an object containing a `models` array and the aggregate `total_size`.
Each model includes `name`, `path`, `size`, `sha256`, and `added`.

### Download

```http
POST /models/cache/download
Content-Type: application/json

{"url":"https://example.invalid/model.gguf","name":"model.gguf","token":""}
```

The operation is accepted for background processing and returns the requested
name. The token field is optional when the server already has a configured
Hugging Face token.

### Remove

```http
DELETE /models/cache/model.gguf
```

Removes the named model and updates the manifest. A missing model returns a
not-found response.

### Read a file

```http
GET /models/cache/file/model.gguf
```

Returns the raw GGUF bytes for the selected cache file. This is useful for
small local development clients that need to inspect or proxy a model.

## Synchronizing a remote manifest

The `scripts/model_hub_sync.py` utility downloads a YAML manifest and brings
the local directory up to date:

```bash
python3 scripts/model_hub_sync.py \
    https://example.invalid/models.yaml \
    --cache-dir ~/.cache/llama.cpp/hub
```

The utility can keep a SQLite index, emit a JSON summary, and perform a
dry-run. Remote model entries may specify a byte count or a simple size
expression. The `--insecure` option is available for development servers that
use a self-signed certificate.

## Integrating with a server process

Start the server with a cache directory and inspect the contents:

```bash
./build/bin/llama-server -m ./models/model.gguf \
    --model-cache-dir ~/.cache/llama.cpp/hub
curl http://127.0.0.1:8080/models/cache
```

The cache does not replace the normal `--model` loading path. It provides a
separate place to stage files and a small API for local tools. A model can
still be loaded directly from any path supported by the normal server
configuration.

## Operational notes

Use a cache directory on a filesystem with enough space for the selected
quantizations. Large GGUF files are copied into the cache, so adding an
existing model temporarily requires space for both source and destination.
The manifest is rewritten after changes to keep metadata available on the
next invocation.
