#!/usr/bin/env bash
# Stop Qwen and restore the full-memory GLM profile.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
systemctl --user stop glm-review-worker.service 2>/dev/null || true
cd "${ROOT}"
docker compose stop vllm
DROPIN="${HOME}/.config/systemd/user/colibri-glm52.service.d/hybrid.conf"
if [[ -f "${DROPIN}" ]]; then
  rm -f "${DROPIN}"
  rm -f "${HOME}/.config/systemd/user/colibri-glm52.service.d/resource-profile.conf"
  systemctl --user daemon-reload
  systemctl --user restart colibri-glm52.service
fi
echo "Hybrid stopped; full-memory GLM profile restored."
