#!/usr/bin/env bash
# Claude Code against local Colibrì.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "${ROOT}/.env" && set +a
fi

# --bare removes plugin, MCP, hook, memory and project-discovery overhead.
# Bare mode accepts ANTHROPIC_API_KEY, so mirror the local bearer token.
if [[ -z "${ANTHROPIC_API_KEY:-}" && -n "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  export ANTHROPIC_API_KEY="${ANTHROPIC_AUTH_TOKEN}"
fi
if [[ -z "${ANTHROPIC_API_KEY:-}" && -n "${COLI_API_KEY:-}" ]]; then
  export ANTHROPIC_API_KEY="${COLI_API_KEY}"
fi
unset ANTHROPIC_AUTH_TOKEN
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://127.0.0.1:${COLIBRI_PORT:-8001}}"
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-${COLI_MODEL_ID:-glm-5.2-colibri}}"
# Colibrì may not emit content deltas until prefill finishes (minutes on GLM-5.2).
# Claude Code defaults ~5m stream idle; engine allows up to 600s (10m).
export CLAUDE_STREAM_IDLE_TIMEOUT_MS="${CLAUDE_STREAM_IDLE_TIMEOUT_MS:-600000}"
exec claude --bare "$@"
