#!/usr/bin/env bash
# Run Colibrì OpenAI-compatible API for GLM-5.2 (host-native; vLLM stack should be stopped).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
MODEL_DIR="${COLI_MODEL:-${ROOT}/models/glm52-colibri-int4}"
COLIBRI_ROOT="${COLIBRI_ROOT:-/home/test1/demos/colibri}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a && source "${ENV_FILE}" && set +a
fi

PORT="${COLIBRI_PORT:-8001}"
BIND="${COLIBRI_BIND:-127.0.0.1}"
MODEL_DIR="${COLI_MODEL:-${MODEL_DIR}}"
COLIBRI_ROOT="${COLIBRI_ROOT:-/home/test1/demos/colibri}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: model dir missing: ${MODEL_DIR}"
  echo "Run: ${ROOT}/scripts/download_glm52_colibri.sh"
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/.download_complete" ]]; then
  echo "WARN: ${MODEL_DIR}/.download_complete not found — download may still be in progress."
fi

if [[ ! -x "${COLIBRI_ROOT}/c/colibri" ]] && [[ ! -f "${COLIBRI_ROOT}/c/colibri" ]]; then
  echo "ERROR: Colibrì engine not built at ${COLIBRI_ROOT}/c/colibri"
  echo "Clone https://github.com/JustVugg/colibri and run: cd c && ./setup.sh"
  exit 1
fi

export COLI_MODEL="${MODEL_DIR}"
export PATH="${COLIBRI_ROOT}/c:${HOME}/.local/bin:${PATH}"

COLI_BIN="${COLIBRI_ROOT}/c/coli"
if [[ ! -x "${COLI_BIN}" ]]; then
  COLI_BIN="$(command -v coli || true)"
fi
if [[ -z "${COLI_BIN}" || ! -x "${COLI_BIN}" ]]; then
  echo "ERROR: coli launcher not found (expected ${COLIBRI_ROOT}/c/coli)"
  exit 1
fi

# GB10 / DGX Spark community tuning baseline (re-measure on your box)
export DIRECT="${DIRECT:-1}"
export PIPE="${PIPE:-1}"
export PILOT="${PILOT:-1}"
export PILOT_REAL="${PILOT_REAL:-1}"
export PILOT_TWO="${PILOT_TWO:-1}"

# CUDA expert tier (engine built with `make glm CUDA=1 CUDA_ARCH=native`).
# GB10 is unified memory: the "VRAM" tier and RAM cache share the same
# 128 GB LPDDR5x pool, so the budgets below must sum well under ~119 GB
# (dense ~11 GB + runtime ~6 GB also come out of the same pool).
export COLI_CUDA="${COLI_CUDA:-1}"
export COLI_GPU="${COLI_GPU:-0}"
export CUDA_EXPERT_GB="${CUDA_EXPERT_GB:-50}"   # GPU-computed hot expert tier
export RAM_GB="${RAM_GB:-56}"                   # warm experts + KV (raise if [RAM_GB] warns over-budget)

# Unified memory makes the default host-copy retention a double charge: the
# engine only auto-releases host slabs on multi-GPU hosts, so a single-GPU box
# paid for every pinned expert twice (once in the VRAM tier, once in RAM).
# Releasing also flips PIN_FILL on and makes the VRAM prefix additive to the
# RAM pin budget instead of stealing from it.
export CUDA_RELEASE_HOST="${CUDA_RELEASE_HOST:-1}"

# Dense (non-expert) work on the GPU: attention projections, MLA, router GEMMs,
# shared expert and lm_head. Without these the GPU only ever runs the resident
# expert tier and the 20 ARM cores carry the rest of every layer.
export CUDA_DENSE="${CUDA_DENSE:-1}"
export COLI_CUDA_ATTN="${COLI_CUDA_ATTN:-1}"    # S<=4 attention on GPU (decode)
export COLI_CUDA_PIPE="${COLI_CUDA_PIPE:-2}"    # residual stays on-device (#273/#274)

# Tensor-Core int4xfp16 dispatch. Engages above the per-expert row threshold,
# so it pays on agentic prefills (Claude Code sends >20k tokens) and is inert
# at S=1 decode.
export COLI_CUDA_TC_W4A16="${COLI_CUDA_TC_W4A16:-1}"
export COLI_CUDA_SHARED_W4A16="${COLI_CUDA_SHARED_W4A16:-1}"

# MTP speculative decoding (requires int8 MTP heads in the model container).
# Measured (#467) to cost ~32% at ~85% expert-hit and to pay only near-full
# residency; this box holds ~20% of a 373 GB expert set, so it stays off.
# Set COLI_CUDA_MTP=1 DRAFT=-1 in .env to re-test once residency improves.
export MTP="${MTP:-1}"
export COLI_CUDA_MTP="${COLI_CUDA_MTP:-0}"
export SPEC_PIN="${SPEC_PIN:-1}"
export COLI_KV_SLOTS="${COLI_KV_SLOTS:-1}"
export DRAFT="${DRAFT:-0}"

# Linux batched expert I/O (implies PIPE=1). Disk stays the floor here: the
# routed working set is far larger than the machine, so every turn streams.
export URING="${URING:-1}"

# Seed the VRAM/RAM hot tiers from the learned usage history (.coli_usage
# next to the model). Without this the CUDA expert tier stays empty.
#
# PIN_FILL ranks the measured-hot experts first and then fills the remaining
# budget with zero-heat ones. Without it the tier is capped by how many experts
# the profile has ever seen, not by the budget: a 60 GB budget loaded 528
# experts (10 GB) because .coli_usage only covered ~234 tokens of history.
# PIN_GB defaults to 10 GB inside pin_load(); "all" clamps to the RAM budget.
export PIN="${PIN:-auto}"
export PIN_FILL="${PIN_FILL:-1}"
export PIN_GB="${PIN_GB:-all}"
export REPIN="${REPIN:-64}"

# GB10 cache-aware routing profile from Vincent Marquez / upstream #199.
# Preserve the router's true top-2 experts, then fill the other six top-8
# slots from resident experts ranked inside the true top-12 candidates.
# This changes routing (unlike PILOT), so expose every quality lever for easy
# A/B rollback with CACHE_ROUTE=0 and keep overlap/KL telemetry enabled.
export CACHE_ROUTE="${CACHE_ROUTE:-1}"
export ROUTE_J="${ROUTE_J:-2}"
export ROUTE_M="${ROUTE_M:-12}"
export ROUTE_P="${ROUTE_P:-0}"
export ROUTE_ALPHA="${ROUTE_ALPHA:-1}"
export ROUTE_AGREE="${ROUTE_AGREE:-1}"

# Diagnostics: PROF prints a per-turn block on stderr (-> journal) with tier
# residency, expert-I/O totals, CPU-vs-GPU execution split and a verdict naming
# the knob most likely to help. Additive only; set PROF=0 to silence.
export PROF="${PROF:-1}"
export COLI_CUDA_PROFILE="${COLI_CUDA_PROFILE:-0}"
export COLI_MODEL_ID="${COLI_MODEL_ID:-glm-5.2-colibri}"

# Context window. The engine default (4096) is too small for agentic clients:
# Claude Code sends a multi-thousand-token preamble before your first word.
# KV is ~180 KB/token and comes out of the RAM budget, so raising CTX shrinks
# the warm expert cache — 32K/45G keeps ~19 GB of experts resident.
COLI_CTX="${COLI_CTX:-32768}"

is_local_bind() {
  case "${BIND}" in
    127.0.0.1|localhost|::1) return 0 ;;
    *) return 1 ;;
  esac
}

if ! is_local_bind; then
  if [[ -z "${COLI_API_KEY:-}" ]]; then
    echo "ERROR: COLIBRI_BIND=${BIND} requires COLI_API_KEY in ${ENV_FILE}"
    echo "       Colibrì refuses LAN bind without a bearer token."
    exit 1
  fi
fi

API_KEY_ARGS=()
if [[ -n "${COLI_API_KEY:-}" ]]; then
  API_KEY_ARGS=(--api-key "${COLI_API_KEY}")
fi

cd "${COLIBRI_ROOT}"
exec "${COLI_BIN}" serve --model "${MODEL_DIR}" --host "${BIND}" --port "${PORT}" \
  --model-id "${COLI_MODEL_ID}" --ctx "${COLI_CTX}" --ram "${RAM_GB}" \
  --kv-slots "${COLI_KV_SLOTS}" \
  "${API_KEY_ARGS[@]}"
