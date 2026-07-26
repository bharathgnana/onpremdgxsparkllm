#!/usr/bin/env bash
# Select a reproducible GLM resource profile and restart the user service.
set -euo pipefail
PROFILE="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DROPIN_DIR="${HOME}/.config/systemd/user/colibri-glm52.service.d"
TARGET="${DROPIN_DIR}/resource-profile.conf"

case "${PROFILE}" in
  full)
    rm -f "${TARGET}"
    ;;
  gpu-hot|hybrid)
    mkdir -p "${DROPIN_DIR}"
    install -m 0644 \
      "${ROOT}/deploy/systemd/colibri-glm52.service.d/${PROFILE}.conf" \
      "${TARGET}"
    ;;
  *)
    echo "usage: $0 full|gpu-hot|hybrid" >&2
    exit 2
    ;;
esac

systemctl --user daemon-reload
systemctl --user restart colibri-glm52.service
echo "GLM profile '${PROFILE}' selected; waiting for health..."
for _ in $(seq 1 180); do
  curl -sf --max-time 3 http://127.0.0.1:8001/health >/dev/null && exit 0
  sleep 2
done
echo "GLM health timeout" >&2
exit 1
