# GLM-5.2 via Colibrì (DGX Spark)

The supported hybrid profile runs Colibrì on the host alongside a memory-bounded Qwen vLLM lane. Colibrì is the asynchronous verifier; use `./scripts/start-hybrid.sh` for co-residency. The standalone instructions below remain useful for GLM-only diagnostics.

## Layout

| Path | Purpose |
|------|---------|
| `/home/test1/demos/colibri` | Engine checkout (tag **v1.1.1**), built with `c/setup.sh` |
| `models/glm52-colibri-int4/` | int4 weights + **int8 MTP** heads (~372 GB) |
| `scripts/download_glm52_colibri.sh` | Resumable HF download (Docker + `huggingface_hub`) |
| `scripts/start-colibri.sh` | OpenAI-compatible API on **:8001** with GB10-oriented env defaults |

## Monitor download

```bash
tail -f models/glm52-colibri-int4/.download.log
du -sh models/glm52-colibri-int4
test -f models/glm52-colibri-int4/.download_complete && echo ready
```

## When download completes

```bash
COLI_MODEL=$PWD/models/glm52-colibri-int4 /home/test1/demos/colibri/c/coli doctor
COLI_MODEL=$PWD/models/glm52-colibri-int4 /home/test1/demos/colibri/c/coli plan --policy quality
./scripts/start-colibri.sh
# or interactive: COLI_MODEL=... DIRECT=1 PIPE=1 /home/test1/demos/colibri/c/coli chat
```

Claude Code / Cursor: point `ANTHROPIC_BASE_URL` or OpenAI base URL to `http://localhost:8001`.

### Boot at login / startup

```bash
chmod +x scripts/install-colibri-service.sh
./scripts/install-colibri-service.sh
systemctl --user start colibri-glm52.service

# Optional: start at boot without logging in (once):
sudo loginctl enable-linger "$(whoami)"
```

Service unit: `deploy/systemd/colibri-glm52.service` (binds **127.0.0.1:8001**).

### Claude Code settings

Merge into `~/.claude/settings.json` (or copy from `docs/claude-settings-glm52.json`):

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8001",
    "ANTHROPIC_AUTH_TOKEN": "same-as-COLI_API_KEY",
    "ANTHROPIC_MODEL": "glm-5.2-colibri",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.2-colibri",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2-colibri",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.2-colibri"
  }
}
```

Use **`ANTHROPIC_AUTH_TOKEN` only** (do not set `ANTHROPIC_API_KEY` in settings). Claude Code
2.1.x warns when both are set; Cursor’s integrated terminal may still export
`ANTHROPIC_API_KEY` — launch via `./scripts/claude-glm52.sh` or run `unset ANTHROPIC_API_KEY`
before `claude`.

If you set `COLI_API_KEY` in `.env` for the server, use the same value for
`ANTHROPIC_AUTH_TOKEN`.

Do not run both services at their full standalone memory budgets. Use `./scripts/start-hybrid.sh` for safe co-resident profiles, or stop vLLM when benchmarking GLM at full budget.

## Deployed Colibrì revision

The host runs branch **`deploy/glm52-spark`** at `/home/test1/demos/colibri` (upstream
`dev` + [PR #625](https://github.com/JustVugg/colibri/pull/625) Anthropic `StopFilter`).
GPU tuning, diagnostics, and A/B benchmark: [COLIBRI_GPU_AND_DIAGNOSTICS.md](COLIBRI_GPU_AND_DIAGNOSTICS.md).

Rebuild and restart:

```bash
chmod +x scripts/redeploy-colibri.sh
./scripts/redeploy-colibri.sh
```

The v1.1.1-only patch `deploy/patches/colibri-role-marker-stop.patch` is **not** used on
this branch — stop handling comes from upstream `StopFilter` + the Anthropic wiring.

## Restore vLLM stack (optional)

```bash
./scripts/migrate_to_vllm.sh download   # re-pull Qwen if weights were removed
docker compose up -d
```

Do **not** run vLLM and Colibrì GLM at full RAM budget simultaneously on 128 GB UMA.

## Role-marker stop leak

In serve mode the engine arms only `<|endoftext|>` and filters the other 17 stop ids
(`[stop] serve mode: filtered 17 non-EOS stop tokens`). Upstream #401 did this on
purpose — as hard stops, role markers cut generation the moment int4 argmax noise picks
a stop id while the model is opening a `<tool_call>` — which makes the role-marker
boundary the Python gateway's job. On v1.1.1 the gateway never enforced it, so `<|user|>`
was detokenized into replies and generation ran past the end of the turn to `max_tokens`.

**Upstream state.** `dev` (post-v1.1.1) fixed this for `/v1/chat/completions` with a
`StopFilter` plus `DEFAULT_CHAT_STOP_SEQUENCES`, and added `COLI_SERVE_ALL_STOPS=1` (#549)
to re-arm the full engine stop set for users who are not tool-calling. It left
`/v1/messages` — the Claude Code path — unfixed; that is
[PR #625](https://github.com/JustVugg/colibri/pull/625), branch worktree at
`/home/test1/demos/colibri-pr`.

**This deployment** runs v1.1.1 plus a local equivalent:
`deploy/patches/colibri-role-marker-stop.patch` against
`/home/test1/demos/colibri/c/openai_server.py`. It covers all four response paths (OpenAI
and Anthropic, streaming and not), truncating at the first role marker and cancelling
generation early; holdback is prefix-aware so prose still streams token by token.
Upstream's 51 + 14 gateway tests pass. **Re-apply after updating Colibrì**, or drop it
when moving to a release containing both the `dev` StopFilter and #625.

```bash
cd /home/test1/demos/colibri && git apply /home/test1/demos/onpremdgxsparkllm/deploy/patches/colibri-role-marker-stop.patch
systemctl --user restart colibri-glm52.service
```

## Quantization

Only the Colibrì **int4** container with **int8 MTP** heads is supported here.
FP8 vLLM paths do not apply to GLM-5.2 on Spark.
