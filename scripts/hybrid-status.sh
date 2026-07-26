#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Hybrid inference status"
printf '  Qwen :8000  '; curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null && echo healthy || echo down
printf '  Proxy :8002 '; curl -sf --max-time 3 http://127.0.0.1:8002/health >/dev/null && echo healthy || echo down
printf '  GLM  :8001  '; curl -sf --max-time 3 http://127.0.0.1:8001/health >/dev/null && echo healthy || echo down
printf '  reviewer   '; systemctl --user is-active glm-review-worker.service 2>/dev/null || true
free -h | sed -n '1,3p'
echo
"${ROOT}/scripts/glm-review" status
