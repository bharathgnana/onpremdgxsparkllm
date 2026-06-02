#!/usr/bin/env bash
# Thin wrapper around `scripts/migrate_to_vllm.sh status` for quick health checks.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DIR}/migrate_to_vllm.sh" status "$@"
