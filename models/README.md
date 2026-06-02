# Models

This directory holds the on-disk weights served by vLLM. Subdirectories
are gitignored (weights total ~95 GB) but their existence is part of the
contract with `docker-compose.yml`, which bind-mounts the whole directory
into the vLLM container at `/models` (read-only).

## Active model

| Directory | Hugging Face ID | Size | Served as | Used for |
|---|---|---|---|---|
| `qwen3-coder-30b-a3b-instruct-fp8/` | [`Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8) | ~30 GB | `qwen3-coder-30b` | Chat, coding, agent tool calls, RAG. MoE 30B/3B-active. Native `qwen3_coder` tool-call format. |

## Parked models (kept for fallback / future use)

| Directory | Reason it's parked |
|---|---|
| `qwen3-30b-a3b-instruct-fp8.parked/` | The previous active general-instruct model (`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`). Replaced by the coder-tuned variant. Move back if a future task requires the plain instruct model. |
| `qwen3.5-35b-a3b-fp8.parked/` | `Qwen/Qwen3.5-35B-A3B-FP8` was downloaded before vLLM added support for the `Qwen3_5MoeForConditionalGeneration` architecture. Park it here until vLLM ships native support; the rest of the file system stays clean. |

## Downloading a new model

```bash
scripts/migrate_to_vllm.sh download
```

The script:
1. Reads `HF_TOKEN` from the repo's `.env`.
2. Runs a one-shot `python:3.12-slim` container with `huggingface_hub.snapshot_download(..., max_workers=8, resume_download=True)`.
3. Writes a `.download_complete` sentinel into the target directory so subsequent runs are idempotent.

To swap the active model, change `VLLM_MODEL_PATH` and `VLLM_MODEL` in `.env` and recreate vLLM:

```bash
docker compose up -d --force-recreate vllm
```

## Reactivating a parked model

```bash
mv qwen3-30b-a3b-instruct-fp8.parked qwen3-30b-a3b-instruct-fp8
# (or rename whichever you want)
```

Then set `VLLM_MODEL_PATH=/models/qwen3-30b-a3b-instruct-fp8` and
`VLLM_MODEL=qwen3-30b-a3b-instruct` in `.env` and recreate vLLM.
