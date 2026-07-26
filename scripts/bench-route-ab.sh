#!/usr/bin/env bash
# Run one GLM benchmark arm with exact or CACHE_ROUTE routing.
set -euo pipefail
PROFILE="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT="colibri-glm52.service"
DROPIN_DIR="${HOME}/.config/systemd/user/${UNIT}.d"
DROPIN="${DROPIN_DIR}/zz-route-ab.conf"

case "${PROFILE}" in
  exact) cache=0 ;;
  cache) cache=1 ;;
  *) echo "usage: $0 exact|cache" >&2; exit 2 ;;
esac

mkdir -p "${DROPIN_DIR}" "${ROOT}/.bench/edge"
printf '[Service]\nEnvironment=CACHE_ROUTE=%s\nEnvironment=ROUTE_J=2\nEnvironment=ROUTE_M=12\n' \
  "${cache}" > "${DROPIN}"
systemctl --user daemon-reload
systemctl --user restart "${UNIT}"

for _ in $(seq 1 180); do
  curl -sf --max-time 3 http://127.0.0.1:8001/health >/dev/null && break
  sleep 2
done
curl -sf --max-time 3 http://127.0.0.1:8001/health >/dev/null

timestamp="$(date +%s)"
exec "${ROOT}/scripts/bench-edge.py" --lane glm --quick \
  --output "${ROOT}/.bench/edge/route-${PROFILE}-${timestamp}.json"
