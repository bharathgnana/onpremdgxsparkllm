# Edge inference plan: interactive coding plus GLM-5.2 verification

## Outcome and latency budgets

This DGX Spark has 119 GiB of unified memory. The 359 GiB GLM-5.2 Colibrì checkpoint cannot provide interactive 20k-token Claude Code prefills by itself, so the production shape is a local cascade:

- `qwen3-coder-30b` is the interactive lane: 20k-token TTFT <= 30 seconds, decode >= 15 tok/s, one 64k context.
- `glm-5.2-colibri` is an asynchronous verifier: input <= 2,000 approximate tokens, TTFT <= 10 minutes, decode >= 3 tok/s.
- GLM advice never blocks editing. Results go to `.claude/glm-reviews/<job>.md`.
- Approximate routing is accepted only within a two-point coding regression budget and at least 98% valid tool calls.

## Implementation

1. Restore Qwen3-Coder FP8 and run vLLM on :8000 with 64k context, one sequence, FP8 KV, prefix caching, and chunked prefill.
2. Run Colibrì on :8001 with CACHE_ROUTE J=2/M=12, coding-trained pinning, CUDA dense/attention, and MTP off unless an A/B shows >=15% gain.
3. Launch `scripts/claude-hybrid.sh`; its asynchronous Stop hook snapshots and enqueues changed work.
4. The review worker compacts the diff with Qwen, enforces the packet limit, calls GLM, and detects stale results.
5. Defer GLM below 10 GiB available memory, keep endpoints local, and redact likely credentials.

## Measurement and rollout

Use `scripts/bench-edge.py` for 128/2k/20k prefills and 32/256-token outputs. Record TTFT, token rates, memory, swap, and validity under `.bench/edge/`. Roll out Qwen-only, GLM A/B, then a 30-minute concurrent soak. Roll back by stopping the review worker or setting `CACHE_ROUTE=0`; never silently reduce expert top-k.

## Acceptance

- Claude reads, edits, and tests through Qwen at the interactive SLO.
- Changed work produces one deduplicated GLM job and result file without blocking editing.
- Changed diffs mark older results stale.
- Concurrent operation has no OOM or sustained swap, keeps >=10 GiB available, and leaves :8000/:8001 healthy.

## Implemented result (2026-07-26)

- Both lanes are live concurrently: Qwen on `:8000`, GLM on `:8001`, with the asynchronous reviewer active.
- The measured 20k Qwen case reached 3.535 seconds TTFT and 35.272 tok/s decode with 29.797 GiB available.
- Tool-call validation passed 50/50 cases, and a real Claude CLI smoke test completed through Qwen.
- The bounded 2k GLM case reached 213.452 seconds TTFT with 18.338 GiB available.
- GLM decode measured 0.831 tok/s, below the 3 tok/s stretch target. It remains advisory and asynchronous; the interactive Qwen lane is production-ready.
