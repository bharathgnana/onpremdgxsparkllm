# onpremdgxsparkllm

Self-contained on-prem inference for NVIDIA DGX Spark (Blackwell GB10,
SM 12.1, 119 GiB usable unified memory). The supported agentic-coding profile
runs Qwen3-Coder as the interactive lane and GLM-5.2 as a bounded,
asynchronous verifier.

## Services

| Endpoint | Service | Model | Purpose |
|---|---|---|---|
| `http://127.0.0.1:8000/v1/chat/completions` | vLLM | `qwen3-coder-30b` FP8 | OpenAI clients |
| `http://127.0.0.1:8002/v1/messages` | Claude proxy → vLLM | `qwen3-coder-30b` FP8 | Claude Code |
| `http://127.0.0.1:8001/v1/messages` | Colibrì | `glm-5.2-colibri` int4 | Async review |
| `http://127.0.0.1:11434` | Ollama | embeddings and vision | RAG/OCR |

Claude Code must use port `8002`, not direct vLLM port `8000`. The proxy
normalizes hook-injected Anthropic system messages. GLM advice never blocks
the editing loop.

## Quick start

```bash
cp .env.example .env
./scripts/migrate_to_vllm.sh download
./scripts/install-hybrid-services.sh
./scripts/start-hybrid.sh
./scripts/hybrid-status.sh
```

The Qwen download is approximately 31 GB. GLM is a separate roughly 359 GB
int4 download; see [docs/COLIBRI_GLM52.md](docs/COLIBRI_GLM52.md).
`start-hybrid.sh` applies safe memory budgets, starts the proxy, waits for both
lanes, and rolls back if Qwen cannot initialize.

## Claude Code

```bash
./scripts/claude-hybrid.sh
```

This verified launcher selects `qwen3-coder-30b`, uses only
`ANTHROPIC_AUTH_TOKEN`, and installs asynchronous GLM review hooks.
Interactive `/init`, tool use, file creation, and a second prompt in the same
session have been tested successfully.

For another laptop, follow
[docs/REMOTE_CLAUDE_QWEN.md](docs/REMOTE_CLAUDE_QWEN.md).

OpenAI-compatible clients use port `8000` directly:

```bash
curl http://127.0.0.1:8000/v1/models
```

## Asynchronous GLM review

The Claude `Stop` hook snapshots a bounded, redacted Git diff. The worker waits
for Qwen to become idle and for at least 10 GiB available memory, compacts the
packet with Qwen, asks GLM for an advisory review, and detects stale results.

```bash
./scripts/glm-review status
./scripts/glm-review show JOB_ID
./scripts/glm-review wait JOB_ID
```

Reports under `.claude/glm-reviews/` are ignored by Git.

## Measured hybrid performance

| Lane | Case | TTFT | Decode | Available memory |
|---|---:|---:|---:|---:|
| Qwen | approximate 20k | 3.535 s | 35.272 tok/s | 29.797 GiB |
| GLM | bounded 2k | 213.452 s | 0.831 tok/s | 18.338 GiB |

Qwen passed the interactive latency, memory, and 50/50 tool-call checks.
GLM passed its ten-minute TTFT and memory bounds but missed the 3 tok/s stretch
target, so it remains asynchronous.

## Repository layout

```text
.
|-- docker-compose.yml                     # vLLM and Ollama
|-- Dockerfile.vllm-patch                  # driver-580 / SM-121 fallback
|-- models/
|   |-- qwen3-coder-30b-a3b-instruct-fp8/  # interactive model
|   `-- glm52-colibri-int4/                # async verifier
|-- deploy/systemd/                        # user services and profiles
|-- scripts/
|   |-- start-hybrid.sh                    # safe co-resident startup
|   |-- claude-hybrid.sh                   # verified Claude launcher
|   |-- anthropic-vllm-proxy.py            # request normalizer
|   `-- glm_review.py                      # async queue
`-- docs/                                  # architecture and operations
```

## Operations

```bash
./scripts/hybrid-status.sh
./scripts/bench-edge.py --lane both --quick
./scripts/quality-tools.py --count 50
./scripts/stop-hybrid.sh
```

## Further reading

- [plan_ed.md](plan_ed.md) — design, gates, and measured outcome.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — hybrid and hardware architecture.
- [docs/REMOTE_CLAUDE_QWEN.md](docs/REMOTE_CLAUDE_QWEN.md) — remote laptop setup.
- [docs/COLIBRI_GLM52.md](docs/COLIBRI_GLM52.md) — GLM deployment.
- [docs/COLIBRI_GPU_AND_DIAGNOSTICS.md](docs/COLIBRI_GPU_AND_DIAGNOSTICS.md) — GLM tuning.
- [docs/VLLM_MIGRATION.md](docs/VLLM_MIGRATION.md) — standalone vLLM history.
