# Colibrì GPU tuning and diagnostics (DGX Spark / GB10)

## Problem

On a single GB10 host, Colibrì was **CPU- and disk-bound**: only ~528 experts (~10 GB) lived in the CUDA tier while ~95% of MoE lookups streamed from NVMe. Claude Code sends **20k+ token** prefills, so turns time out before the first streamed chunk.

## Tuning (launcher defaults)

Configured in `scripts/start-colibri.sh` (override via `.env`):

| Area | Knobs | Intent |
|------|--------|--------|
| GPU compute | `CUDA_DENSE=1`, `COLI_CUDA_ATTN=1`, `COLI_CUDA_PIPE=2`, `COLI_CUDA_TC_W4A16=1`, `COLI_CUDA_SHARED_W4A16=1` | Run dense/attention/shared work on GPU |
| Expert residency | `CUDA_EXPERT_GB=50`, `PIN_FILL=1`, `PIN_GB=all`, `CUDA_RELEASE_HOST=1`, `REPIN=64` | Fill VRAM tier; use RAM budget for warm tier |
| Speculation | `COLI_CUDA_MTP=0`, `DRAFT=0` | MTP hurts when hit rate is low (#467) |
| Disk overlap | `URING=1` (with existing `DIRECT=1`, `PIPE=1`, `PILOT=*`) | Batch expert I/O |
| Diagnostics | `PROF=1` | Per-turn stderr profile → `journalctl` |

**Limit:** ~373 GB of expert weights cannot fully fit in ~119 GB UMA. Expect better short prompts, not fast `/init`.

## GLM-5.2 (Colibrì) vs Qwen (vLLM) on DGX Spark

Qwen on this repo’s **vLLM** stack usually feels much more responsive. That is expected: different model scale, memory layout, and inference engine.

### Why Qwen felt faster

| | **Qwen3-Coder-30B FP8 (vLLM)** | **GLM-5.2 int4 (Colibrì)** |
|---|-------------------------------|------------------------------|
| Role | MoE ~30B active / fits in UMA | 744B MoE; **~370 GB** routed experts on disk |
| Serving | GPU-centric, continuous batching | Stream experts from NVMe when not resident |
| Spark numbers (this repo) | ~**38 tok/s** single-stream warm; ~**74 tok/s** aggregate (4 seq) — see [VLLM_MIGRATION.md](VLLM_MIGRATION.md) | Decode often **~0.5–2 tok/s** API class; agent **20k+ token** prefills take **many minutes** |
| Agent clients | Large prefill still heavy but tractable | Prefill can exceed Claude Code **stream idle** before first text chunk |

Colibrì’s public story ([Show HN](https://news.ycombinator.com/item?id=48842459), [project site](https://justvugg.github.io/colibri/)) targets **~25 GB RAM laptops** at **~0.05–0.1 tok/s** cold — “frontier model locally,” not “Claude Code at GPU speed.”

### Community / upstream numbers (GB10 included)

| Source | Hardware | Reported decode (honest context) |
|--------|----------|----------------------------------|
| [colibrì `benchmarks.md`](https://github.com/JustVugg/colibri/blob/main/docs/benchmarks.md) | **DGX Spark GB10**, aarch64, ~121 GB UMA | **~2.4 tok/s** full top-8 warm; **~3.33 tok/s** with `CACHE_ROUTE=1` ([#136](https://github.com/JustVugg/colibri/issues/136), [#161](https://github.com/JustVugg/colibri/issues/161)) |
| [Vincent Marquez — `glm52-gb10-colibri`](https://github.com/VincentMarquez/glm52-gb10-colibri) | **Same: 1× DGX Spark GB10** | **~11.1+ tok/s** decode at full top-8 — **lab `./glm` protocol**, not stock `coli serve` + 22k agent prompts ([NUMBERS.md](https://github.com/VincentMarquez/glm52-gb10-colibri/blob/main/docs/NUMBERS.md)) |
| [6× RTX 5090 experiment](https://github.com/JustVugg/colibri/blob/main/docs/experiments/glm52-6x5090-2026-07-12.md) | Full VRAM+RAM residency | **~5.8–6.8 tok/s** decode |
| [GB10 issue #76](https://github.com/JustVugg/colibri/issues/76) | GB10 | ARM64/CUDA OK; **0/20 oracle** was tiny-model self-test mismatch, not a generation bug |

Reddit/HN threads are mostly **laptop + disk streaming** narratives; the actionable **Spark** references are **upstream benchmarks**, **#161 / #136**, and **Marquez’s repo**.

### How ~11 tok/s on Spark is achieved (reference stack)

Not enabled by default in `scripts/start-colibri.sh`. Documented on GB10 by [glm52-gb10-colibri](https://github.com/VincentMarquez/glm52-gb10-colibri):

- **~99.9% expert hit**, **~64 GB** CUDA expert tier (`CUDA_EXPERT_GB` ~70–85).
- **`CACHE_ROUTE=1`** ([#199](https://github.com/JustVugg/colibri/pull/199)) — cache-aware routing; default upstream is **off**. See [CACHE_ROUTE.md](https://github.com/JustVugg/colibri/blob/main/docs/CACHE_ROUTE.md) in colibrì.
- **GPU MLA + fused CUDA decode** (`CUDA_FUSE`, etc.) — experimental vs this repo’s serve launcher.
- **PLD / n-gram drafts** (`FORCE_PLD`, `DRAFT=3`), **MTP off** — differs from MTP-on experiments.
- **Short prompts** (~40 tokens), warm discard, **timed decode only** — not Claude Code’s **~20k-token** first turn.

At ~11 tok/s, profile on that stack is **~80% attention (MLA)**; experts/disk are small once pinned.

### Practical recommendations on this host

1. **Dual stack (typical):** Keep **vLLM + Qwen** for Claude Code / Cursor latency; use **Colibrì + GLM** when you accept slower turns or want that model specifically.
2. **Push GLM speed (engineering):** Reproduce Marquez flags on a colibrì checkout; validate with **`./glm` timed decode** and `[PROF]` before trusting `coli serve`. Start with `CACHE_ROUTE=1` and residency tuning; read his `docs/HARDWARE.md` / scripts.
3. **Claude Code on GLM:** `./scripts/claude-glm52.sh`, `--bare`, short messages; avoid `/init` until a one-line reply works. See [Claude Code](#claude-code) below.

## Diagnostics

```bash
./scripts/diagnose-colibri.sh
```

Reports: service health, in-flight KV/prefill, HTTP codes (429/400), placement lines, last `[PROF]` block (hit rate, I/O vs matmul vs attention, verdict), GPU/CPU samples, live engine env from `/proc`, Claude settings hints.

Live logs:

```bash
journalctl --user -u colibri-glm52.service -f
```

Look for `[PROF] verdict:` after each completed turn.

## A/B benchmark

```bash
./scripts/bench-colibri.sh baseline   # systemd drop-in = old CPU-heavy profile
./scripts/bench-colibri.sh tuned      # remove drop-in; use launcher defaults
```

Saves `.coli_usage` snapshot under `.bench/` for fair pin profiles; clears `.coli_kv` each run. Probes: short decode + ~4k-token prefill (adjust `BENCH_PREFILL`, `BENCH_TIMEOUT`).

## Claude Code

- Launch: `./scripts/claude-glm52.sh` (unsets `ANTHROPIC_API_KEY`, sets `CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000`).
- Skip `/init` until a one-line reply works.
- Settings: `docs/claude-settings-glm52.json`.

### Why Claude Code shows “Stream idle timeout”

1. **Huge input:** interactive Claude sends **~19–23k tokens** (tools + system); `--bare` is **~2k** but still slow.
2. **No content until prefill ends:** Colibrì sends SSE `ping` during prefill, but Claude Code counts **content deltas**, not pings.
3. **Default ~5 min idle:** prefill for 2k tokens measured **~7+ min** (78 layers); 22k is much longer.

Mitigations: `CLAUDE_STREAM_IDLE_TIMEOUT_MS` (max **600000**), `claude --bare`, short prompts, `./scripts/diagnose-colibri.sh` after each turn. Full log: `/tmp/claude-code-monitor.log`.
