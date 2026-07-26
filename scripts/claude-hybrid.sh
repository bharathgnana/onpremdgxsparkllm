#!/usr/bin/env bash
# Interactive Qwen Claude Code lane with asynchronous GLM verification.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "${ROOT}/.env" ]] && { set -a; source "${ROOT}/.env"; set +a; }

if ! curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null; then
  echo "ERROR: Qwen vLLM is not healthy on :8000." >&2
  echo "Run: ${ROOT}/scripts/start-hybrid.sh" >&2
  exit 1
fi

"${ROOT}/scripts/anthropic-vllm-proxy.py" --ensure
export ANTHROPIC_BASE_URL="http://127.0.0.1:8002"
export ANTHROPIC_AUTH_TOKEN="${QWEN_API_KEY:-local-qwen}"
unset ANTHROPIC_API_KEY
export ANTHROPIC_MODEL="${QWEN_MODEL_ID:-qwen3-coder-30b}"
export ANTHROPIC_DEFAULT_OPUS_MODEL="${ANTHROPIC_MODEL}"
export ANTHROPIC_DEFAULT_SONNET_MODEL="${ANTHROPIC_MODEL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="${ANTHROPIC_MODEL}"
exec claude --settings "${ROOT}/docs/claude-settings-hybrid.json" "$@"
