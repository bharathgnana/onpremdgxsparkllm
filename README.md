# onpremdgxsparkllm

Self-contained, reproducible on-prem LLM-serving stack for the NVIDIA DGX
Spark (Blackwell GB10, SM 12.1, 128 GB UMA). Runs as an independent
`docker compose` project and exposes a single network surface that any
client application can consume.

## What this repo serves

| Endpoint | Service | Model | Used by |
|---|---|---|---|
| `http://127.0.0.1:8000/v1/chat/completions` | vLLM (OpenAI shape) | `qwen3-coder-30b` (Qwen3-Coder-30B-A3B-Instruct-FP8) | App backends, Cursor, aider, etc. |
| `http://127.0.0.1:8000/v1/messages` | vLLM (Anthropic shape) | same | Claude CLI, Claude-API clients |
| `http://127.0.0.1:11434/api/embeddings` | Ollama | `mxbai-embed-large` (1024-dim) | RAG pipelines |
| `http://127.0.0.1:11434/api/chat` (vision) | Ollama | `llama3.2-vision:11b` (on-demand pull) | Vision OCR fallback |

vLLM handles all chat/coding/agent traffic (continuous batching, prefix
caching, FP8 KV cache). Ollama stays around purely for embeddings + the
vision OCR fallback that vLLM does not serve.

## Quick start

```bash
cp .env.example .env
# edit HF_TOKEN in .env (needed only on first model download)

./scripts/start.sh         # docker compose up -d
./scripts/status.sh        # check what's running and which models are loaded
```

First boot pulls a ~30 GB FP8 model from Hugging Face (or skips if you
already moved the `models/qwen3-coder-30b-a3b-instruct-fp8/` directory in
from another machine). vLLM cold-start takes ~3 min while it loads the
model and captures CUDA graphs.

## How a client application consumes this

### Same-host client running in Docker

Add `host.docker.internal` to your client compose `extra_hosts` and point
at the published ports:

```yaml
services:
  my-backend:
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - OLLAMA_HOST=host.docker.internal
      - OLLAMA_PORT=11434
      - VLLM_HOST=host.docker.internal
      - VLLM_PORT=8000
      - VLLM_MODEL=qwen3-coder-30b
```

### Same-host client running on the host (not Docker)

Just use `localhost`:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:11434/api/tags
```

### Claude CLI (Claude Code v2.x)

```bash
# Persist in ~/.claude/settings.json:
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8000",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_AUTH_TOKEN": "dummy",
    "ANTHROPIC_DEFAULT_OPUS_MODEL":   "qwen3-coder-30b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3-coder-30b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL":  "qwen3-coder-30b"
  }
}
```

Then just run `claude`.

## Repository layout

```
.
|-- README.md                              # this file
|-- docker-compose.yml                     # vllm + ollama + ollama-init
|-- Dockerfile.vllm-patch                  # patched-image recipe (sm_121 fix)
|-- .env.example                           # template for HF_TOKEN, knobs
|-- models/                                # weights (gitignored)
|   |-- qwen3-coder-30b-a3b-instruct-fp8/  # active
|   |-- qwen3-30b-a3b-instruct-fp8.parked/ # general-instruct (previous active)
|   `-- qwen3.5-35b-a3b-fp8.parked/        # waiting on vLLM Qwen3.5 MoE support
|-- ollama-data/                           # ollama blob store (bind mount, gitignored)
|-- scripts/
|   |-- start.sh                           # docker compose up wrapper
|   |-- status.sh                          # quick health/inventory
|   `-- migrate_to_vllm.sh                 # download/start/validate/switch/rollback
`-- docs/
    |-- VLLM_MIGRATION.md                  # detailed setup + tuning + troubleshooting
    `-- ARCHITECTURE.md                    # why vLLM, why Ollama, sm_121 history
```

## Why a separate repo

- **Decoupling**: app repos point at endpoints (`http://localhost:8000`,
  `http://localhost:11434`) instead of importing 90+ GB of model weights
  and a vLLM service block.
- **Reproducibility**: the patched-image Dockerfile, model download
  scripts, parked-model history, and tuning knobs live in one place.
- **Knowledge base**: the docs capture the migration history (Qwen3
  Ollama -> Qwen3.5-FP8 vLLM -> Qwen3-Coder-30B vLLM), the sm_121
  workarounds for driver 580.x, and the upgrade path to NVIDIA's
  Spark-tuned image once the driver is bumped to >= 590.44.

## See also

- [docs/VLLM_MIGRATION.md](docs/VLLM_MIGRATION.md) - full setup, tuning, troubleshooting
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - why this stack
