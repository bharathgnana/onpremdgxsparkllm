#!/usr/bin/env bash
# Rebuild Colibrì from deploy/glm52-spark (upstream dev + Anthropic StopFilter) and restart.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLIBRI_ROOT="${COLIBRI_ROOT:-/home/test1/demos/colibri}"
DEPLOY_BRANCH="${COLIBRI_DEPLOY_BRANCH:-deploy/glm52-spark}"
DEPLOY_REMOTE="${COLIBRI_DEPLOY_REMOTE:-https://github.com/bharathgnana/colibri.git}"
DEPLOY_REF="${COLIBRI_DEPLOY_REF:-fix/anthropic-stop-sequences}"

if [[ ! -d "${COLIBRI_ROOT}/.git" ]]; then
  echo "ERROR: ${COLIBRI_ROOT} is not a colibri git checkout"
  exit 1
fi

echo "==> Colibrì @ ${COLIBRI_ROOT}"
cd "${COLIBRI_ROOT}"
git fetch "${DEPLOY_REMOTE}" "${DEPLOY_REF}:${DEPLOY_BRANCH}"
git checkout "${DEPLOY_BRANCH}"
git reset --hard "${DEPLOY_BRANCH}"

echo "==> Build engine (CUDA)"
make -C c glm CUDA=1 CUDA_ARCH=native

echo "==> Install coli launcher"
python3 -m pip install -e . --break-system-packages -q

echo "==> Restart systemd user service"
mkdir -p "${HOME}/.config/systemd/user"
cp "${ROOT}/deploy/systemd/colibri-glm52.service" "${HOME}/.config/systemd/user/colibri-glm52.service"
systemctl --user daemon-reload
systemctl --user restart colibri-glm52.service

echo "Waiting for health (up to 5 min)..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo "OK: http://127.0.0.1:8001/health"
    systemctl --user is-active colibri-glm52.service
    exit 0
  fi
  sleep 5
done
echo "WARN: health not ready yet — check: journalctl --user -u colibri-glm52.service -f"
exit 1
