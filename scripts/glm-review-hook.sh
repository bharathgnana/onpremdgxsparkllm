#!/usr/bin/env bash
# Claude Stop hook: enqueue changed work without delaying the coding session.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
payload="$(cat)"
cwd="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("cwd",""))' <<<"${payload}" 2>/dev/null)"
[[ -n "${cwd}" ]] || exit 0
repo="$(git -C "${cwd}" rev-parse --show-toplevel 2>/dev/null)" || exit 0
"${ROOT}/scripts/glm-review" enqueue --repo "${repo}" --quiet >/dev/null 2>&1 || true
exit 0
