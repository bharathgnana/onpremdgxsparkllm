#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_UNIT_DIR="${HOME}/.config/systemd/user"
mkdir -p "${USER_UNIT_DIR}"
install -m 0644 "${ROOT}/deploy/systemd/glm-review-worker.service" \
  "${USER_UNIT_DIR}/glm-review-worker.service"
systemctl --user daemon-reload
systemctl --user enable glm-review-worker.service
echo "Installed glm-review-worker.service"
echo "Start after both endpoints are ready: systemctl --user start glm-review-worker.service"
