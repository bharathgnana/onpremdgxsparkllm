# vLLM Coding Stack (DGX Spark / Blackwell GB10)

This guide documents the **single-engine, single-model** LLM stack: vLLM
running `Qwen3-Coder-30B-A3B-Instruct-FP8` on a DGX Spark, with continuous
batching and prefix caching tuned for 2-16 concurrent coding agents.

Ollama is still installed but only handles two auxiliary jobs:
- `mxbai-embed-large` (1024-dim embeddings for RAG)
- `llama3.2-vision:11b` (vision OCR fallback)

Everything chat / coding / RAG / Claude CLI flows through vLLM.

---

## Target architecture

```
Claude CLI / Cursor / aider ─┐
FastAPI backend (chat/RAG)   ├──> vLLM :8000   (qwen3-coder-30b, OpenAI + Anthropic API)
Any OpenAI-compatible agent  ┘

FastAPI backend (embeddings) ───> Ollama :11434 (mxbai-embed-large)
FastAPI backend (vision OCR) ───> Ollama :11434 (llama3.2-vision:11b)
```

## Why this model

`Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` won the June 2026 survey of
coder-capable models that fit on a single GB10 (128 GB UMA):

| Model | Type | Disk (FP8) | SWE-Bench Verified | Verdict |
|---|---|---|---|---|
| **Qwen3-Coder-30B-A3B-Instruct-FP8** (active) | MoE 30B/3B | ~30 GB | ~60% | Fast single-stream (~80 tok/s theoretical), Apache 2.0, native `qwen3_coder` tool calls, 256K context, headroom for 8-16 concurrent agents |
| Devstral-Small-2-24B-Instruct-2512 (FP8) | Dense 24B | ~24 GB | 68.0% | +8 SWE-Bench points but dense → ~11 tok/s single-stream on bandwidth-bound Spark |
| Qwen3-Next-80B-A3B-Instruct-FP8 | MoE 80B/3B | ~80 GB | (generalist) | Tight fit; ~45 tok/s; not coder-tuned |

The 2026 frontier (DeepSeek V4-Pro 1.6 T, Kimi K2.6 ~1 T, GLM-5.1 744 B,
Qwen3-Coder-Plus 480 B, Devstral-2 123 B, Qwen3-Coder-Next 80 B BF16-only)
all require 200 GB+ or multi-GPU and are off the table for DGX Spark.

We chose the MoE 30B over the +8% denser Devstral because Spark is memory-
bandwidth-bound at 273 GB/s. With only 3 B parameters scanned per token,
the MoE has ~8x the theoretical single-stream throughput, which dominates
the interactive feel when 2-3 coding agents are active at once.

---

## Which image: driver gates the choice

Two viable images for this stack, picked by the host NVIDIA driver:

| Host driver | Image | Notes |
|---|---|---|
| **>= 590.44** | `nvcr.io/nvidia/vllm:25.12.post1-py3` | NVIDIA's Spark-tuned build. Ships precompiled sm_121 binaries for CUTLASS FP8 GEMM, FlashInfer, and FlashAttention. V1 engine + CUDA-graph capture works natively (no `--enforce-eager`). Enables FlashInfer MoE FP8 + FLASH_ATTN. |
| **< 590.44** (current host: 580.95.05) | `vllm-patched:qwen3.5` | Upstream `vllm/vllm-openai` + locally-applied sm_121 source patch (`cutlass_block_fp8_supported -> False`) and runtime flags (`--enforce-eager`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`). Single-stream throughput is ~30% lower but the stack works on the older driver. |

**Current setup** uses the patched image. To migrate to the NVIDIA image
once the driver is upgraded:

1. `sudo apt install -y nvidia-driver-590` (or NVIDIA's run-file equivalent) and reboot.
2. In `docker-compose.yml`'s `vllm` service block:
   - `image: ${VLLM_IMAGE:-nvcr.io/nvidia/vllm:25.12.post1-py3}`
   - Replace `VLLM_ATTENTION_BACKEND=TRITON_ATTN` with `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
   - Add `VLLM_USE_FLASHINFER_MOE_FP8=1`, `VLLM_FLASHINFER_MOE_BACKEND=latency`, `VLLM_USE_DEEP_GEMM=0`
   - Remove `--enforce-eager` from `command:`
3. `docker compose --profile vllm up -d --force-recreate vllm`

If you ever need to rebuild the patched image, the recipe is preserved
in git history (see `/tmp/Dockerfile.vllm-patch` in the migration PR).

---

## Repo files involved

| File | Purpose |
|---|---|
| [`backend/app/config.py`](../backend/app/config.py) | `LLM_PROVIDER=vllm` default, `VLLM_MODEL=qwen3-coder-30b` default |
| [`backend/app/llm_service.py`](../backend/app/llm_service.py) | `LLMProvider.VLLM`, `_vllm_generate`, vLLM streaming, `get_chat_llm` |
| [`docker-compose.yml`](../docker-compose.yml) | `vllm` service under the `vllm` profile, FlashInfer MoE FP8 + FLASH_ATTN env, qwen3_coder tool-call parser |
| [`scripts/migrate_to_vllm.sh`](../scripts/migrate_to_vllm.sh) | `download` / `start` / `validate` / `switch` / `rollback` / `status` |
| `models/qwen3-coder-30b-a3b-instruct-fp8/` | Active model (~30 GB) |
| `models/qwen3-30b-a3b-instruct-fp8.parked/` | Previous general-instruct model |
| `models/qwen3.5-35b-a3b-fp8.parked/` | Pre-release Qwen3.5 weights (waiting for vLLM arch support) |

---

## First-time setup

> All commands assume you are in the repo root.

### 1. Pre-flight

```bash
df -h .                          # need ~50 GB free for model
docker info | grep -iE 'runtime|driver'
grep -E '^HF_TOKEN=' .env        # avoid anonymous HF rate limits
```

### 2. Pull the NVIDIA vLLM image (~12 GB)

```bash
docker pull nvcr.io/nvidia/vllm:25.12.post1-py3
```

### 3. Download the model (~30 GB)

```bash
scripts/migrate_to_vllm.sh download
```

The script reads `HF_TOKEN` from `.env`, runs `snapshot_download` inside
a one-shot Python container, and writes `.download_complete` when done.
It is resumable.

### 4. Start vLLM

```bash
scripts/migrate_to_vllm.sh start
```

First boot takes 3-5 minutes — CUDA-graph capture under the V1 engine
runs once and then caches results.

### 5. Validate

```bash
scripts/migrate_to_vllm.sh validate
```

Healthy output:

```
Prompt tokens:     32
Completion tokens: 108
Wall time:         1.6s
Output speed:      67.5 tok/s
```

### 6. Confirm the backend is on vLLM

The compose default is `LLM_PROVIDER=vllm`. If `.env` overrides it to
`ollama`, run:

```bash
scripts/migrate_to_vllm.sh switch
```

Look for this in `docker logs defense-backend`:

```
INFO:app.llm_service:vLLM initialized at http://vllm:8000 as primary LLM
INFO:app.main:✓ LLM service initialized using local (vllm) provider
```

### 7. Rollback if needed

```bash
scripts/migrate_to_vllm.sh rollback
```

Resets `LLM_PROVIDER=ollama` and recreates the backend. The vLLM container
keeps running — stop it with `docker compose --profile vllm stop vllm` to
free VRAM. The previous Instruct model lives in
`models/qwen3-30b-a3b-instruct-fp8.parked/` if you want to swap back to it.

---

## Claude CLI / agent endpoints

vLLM serves **both** `/v1/chat/completions` (OpenAI shape) and
`/v1/messages` (Anthropic shape), so you can point any agent at one URL.

### Claude CLI (Claude Code v2.x)

For the current hybrid deployment, prefer `./scripts/claude-hybrid.sh`; it routes through the required port-8002 compatibility proxy. The direct vLLM recipe below applies only to standalone vLLM without review-hook system-message injection.

Claude Code v2.x no longer accepts arbitrary names via `--model` when an
`ANTHROPIC_BASE_URL` is set; the CLI always routes traffic through the
Opus/Sonnet/Haiku tier names and rejects `qwen3-coder-30b` client-side
("There's an issue with the selected model"). Alias all three tiers to
the local model:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_AUTH_TOKEN=local-qwen \
ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3-coder-30b \
ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3-coder-30b \
ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3-coder-30b \
claude
```

Optional: persist the config in `~/.claude/settings.json` so you don't
have to set the env vars each time:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8002",
    "ANTHROPIC_AUTH_TOKEN": "local-qwen",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3-coder-30b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3-coder-30b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3-coder-30b"
  }
}
```

Notes:
- The `served-model-name` (`qwen3-coder-30b`) cannot contain `/`, so we
  use the short form — vLLM is already configured this way.
- `ANTHROPIC_AUTH_TOKEN` must be non-empty in v2.x (any string works).
- If you want the `/model` picker inside Claude to also show the local
  model, set `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`. Claude Code
  will hit `/v1/models` on vLLM, but it only picks up entries whose IDs
  start with `claude` or `anthropic`, so the env-var aliasing above is
  still the simpler path.

### Cursor / OpenAI-compatible CLIs (aider, codex, etc.)

```
Base URL : http://localhost:8000/v1
API key  : not-needed
Model    : qwen3-coder-30b
```

### Spinning up more concurrent agents

vLLM continuously batches up to `--max-num-seqs` requests automatically.
Default is 16 (compose: `VLLM_MAX_NUM_SEQS=16`). To go higher, set
`VLLM_MAX_NUM_SEQS` in `.env` and recreate the container:

```bash
echo 'VLLM_MAX_NUM_SEQS=24' >> .env
docker compose --profile vllm up -d --no-deps --force-recreate vllm
```

Each concurrent sequence at the configured `MAX_MODEL_LEN` consumes
proportional KV cache. With the default 64K context + 8 seqs vLLM
reports ~53 GB usable KV cache (~1.15M tokens of buffer total), on top
of ~30 GB model weights. The `--gpu-memory-utilization 0.70` setting
reserves ~90 GB out of 128 GB UMA for vLLM, leaving room for Ollama,
postgres, elasticsearch, etc.

---

## Tuning knobs

All settable via `.env` (compose defaults shown):

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `vllm` | `ollama` to roll back |
| `VLLM_IMAGE` | `nvcr.io/nvidia/vllm:25.12.post1-py3` | Spark-tuned NVIDIA build |
| `VLLM_MODEL_PATH` | `/models/qwen3-coder-30b-a3b-instruct-fp8` | Path inside container |
| `VLLM_MODEL` | `qwen3-coder-30b` | Logical name advertised by `/v1/models` |
| `VLLM_MAX_MODEL_LEN` | `65536` | Context window. Claude Code's system prompt + tool schemas can be ~28K tokens with `max_tokens=32000` for the response; 32K isn't enough. Qwen3-Coder supports up to 256K if you need more. |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.70` | Fraction of GPU mem reserved for vLLM |
| `VLLM_MAX_NUM_SEQS` | `8` | Max concurrent in-flight sequences. Drop to 4 if you bump `MAX_MODEL_LEN` to 131072; raise to 16 if you lower it to 32768. |
| `VLLM_KV_CACHE_DTYPE` | `fp8` | FP8 KV cache halves memory vs BF16 |

### Reclaim Ollama VRAM (optional, after validation)

The Ollama `qwen3-coder:30b` is no longer used now that vLLM serves the
CLI. Remove it to reclaim ~18 GB:

```bash
docker exec defense-ollama ollama rm qwen3-coder:30b
```

Ollama keeps `mxbai-embed-large` (~700 MB) and downloads
`llama3.2-vision:11b` on demand for OCR.

---

## Measured performance (current patched-image setup)

Tested on DGX Spark (GB10, 128 GB UMA, driver 580.95.05) with
Qwen3-Coder-30B-A3B-Instruct-FP8 + `--enforce-eager` + Triton attn:

| Scenario | Throughput |
|---|---|
| Single-stream, warm cache | ~38 tok/s |
| 4 concurrent requests | ~74 tok/s aggregate (1.93x) → continuous batching working |
| Model load + warmup | ~3 min (cold), ~30s (warm) |

Projected gains once the host moves to driver >= 590.44 and the
`nvcr.io/nvidia/vllm:25.12.post1-py3` image:

| Metric | Patched image (now) | nvcr Spark image (post driver upgrade) |
|---|---|---|
| Single-stream output | ~38 tok/s | ~70-90 tok/s (CUDA-graphs + FlashInfer MoE FP8 + FlashAttention) |
| Time-to-first-token | ~400-600 ms | ~150-300 ms |
| Concurrent agents | 16 (batched, single-stream bound) | 16 (batched, faster per-stream) |
| sm_121 patches | 3 (cutlass src patch, TRITON_ATTN, --enforce-eager) | 0 |

Numbers depend on prompt length, prefix-cache hit rate, and the exact
concurrency mix.

---

## Troubleshooting

- **`/v1/models` returns empty list** — model didn't load. Check
  `docker logs defense-vllm --tail 120` for tokenizer / weight errors.
- **`Tool parser qwen3_coder not found`** — confirm `VLLM_IMAGE` is the
  nvcr.io image; the qwen3_coder parser was added upstream in 2025-Q4.
- **OOM during model load** — lower `VLLM_GPU_MEMORY_UTILIZATION` (try
  `0.55`) or temporarily stop Ollama:
  `docker compose stop ollama`.
- **Claude CLI returns 404** — confirm you're hitting `:8000` (vLLM),
  not `:11434` (Ollama). vLLM serves `/v1/messages` natively.
- **Claude CLI says "There's an issue with the selected model
  (qwen3-coder-30b). It may not exist..."** — Claude Code v2.x ignores
  custom names passed via `--model` when `ANTHROPIC_BASE_URL` is set;
  use the three `ANTHROPIC_DEFAULT_*_MODEL` env vars instead. Also
  remember that `~/.claude/settings.json` overrides shell env vars; if
  your old Ollama config is still there (pointing at port 11434), edit
  the file's `env` block to match the recipe above.
- **vLLM returns 400 "max_tokens too large"** — Claude Code asks for
  `max_tokens=32000` and prepends a ~28K-token system prompt. Bump
  `VLLM_MAX_MODEL_LEN` to at least `65536` (the new default) and reduce
  `VLLM_MAX_NUM_SEQS` proportionally so KV cache fits.
- **Backend logs say `(ollama) provider`** — `.env` overrides the
  compose default. Run `scripts/migrate_to_vllm.sh switch`.
- **Container restarts with `RuntimeError: Error Internal`** — you
  reverted to an upstream image without the sm_121 binaries. Switch
  back to `nvcr.io/nvidia/vllm:25.12.post1-py3` or check git history
  for the legacy patch set.
