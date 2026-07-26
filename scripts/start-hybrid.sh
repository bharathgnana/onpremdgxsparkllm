#!/usr/bin/env bash
# Start co-resident GLM verifier + Qwen interactive coding lane.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${ROOT}/models/qwen3-coder-30b-a3b-instruct-fp8"
DROPIN_DIR="${HOME}/.config/systemd/user/colibri-glm52.service.d"

rollback() {
  code=$?
  if (( code != 0 )); then
    echo "Hybrid start failed; restoring full-memory GLM." >&2
    docker stop onprem-vllm >/dev/null 2>&1 || true
    rm -f "${DROPIN_DIR}/hybrid.conf"
    systemctl --user daemon-reload
    systemctl --user restart colibri-glm52.service || true
  fi
  exit "${code}"
}
if [[ ! -f "${MODEL}/.download_complete" ]]; then
  echo "ERROR: Qwen weights are missing. Run: ${ROOT}/scripts/migrate_to_vllm.sh download" >&2
  exit 1
fi

trap rollback EXIT

mkdir -p "${DROPIN_DIR}"
rm -f "${DROPIN_DIR}/resource-profile.conf"
install -m 0644 "${ROOT}/deploy/systemd/colibri-glm52.service.d/hybrid.conf" \
  "${DROPIN_DIR}/hybrid.conf"
systemctl --user daemon-reload
systemctl --user restart colibri-glm52.service

echo "Waiting for hybrid-budget GLM..."
for _ in $(seq 1 180); do
  curl -sf --max-time 3 http://127.0.0.1:8001/health >/dev/null && break
  systemctl --user is-active --quiet colibri-glm52.service || {
    journalctl --user -u colibri-glm52.service -n 40 --no-pager
    exit 1
  }
  sleep 2
done
curl -sf --max-time 3 http://127.0.0.1:8001/health >/dev/null

cd "${ROOT}"
export VLLM_MAX_MODEL_LEN="${HYBRID_VLLM_MAX_MODEL_LEN:-65536}"
export VLLM_MAX_NUM_SEQS="${HYBRID_VLLM_MAX_NUM_SEQS:-1}"
export VLLM_GPU_MEMORY_UTILIZATION="${HYBRID_VLLM_GPU_MEMORY_UTILIZATION:-0.30}"
docker compose up -d vllm

echo "Waiting for Qwen vLLM..."
for _ in $(seq 1 180); do
  curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null && break
  docker ps --format '{{.Names}}' | grep -qx onprem-vllm || {
    docker logs onprem-vllm --tail 80
    exit 1
  }
  sleep 2
done
curl -sf --max-time 3 http://127.0.0.1:8000/health >/dev/null

"${ROOT}/scripts/anthropic-vllm-proxy.py" --ensure
systemctl --user restart glm-review-worker.service 2>/dev/null || true
trap - EXIT
echo "Hybrid endpoints ready: Qwen :8000, Claude proxy :8002, GLM :8001"
echo "Launch Claude: ${ROOT}/scripts/claude-hybrid.sh"
