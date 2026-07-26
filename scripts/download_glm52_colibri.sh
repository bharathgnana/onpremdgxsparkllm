#!/usr/bin/env bash
# Download GLM-5.2 Colibrì int4 container (int8 MTP heads) into ./models/glm52-colibri-int4/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
MODEL_DIR="${ROOT}/models/glm52-colibri-int4"
REPO_ID="mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp"
LOG="${MODEL_DIR}/.download.log"

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_DIR}/.download_complete" ]]; then
  echo "Already complete: ${MODEL_DIR} (.download_complete present)"
  exit 0
fi

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a && source "${ENV_FILE}" && set +a
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN not set in ${ENV_FILE}; anonymous HF rate limits may apply."
fi

echo "Downloading ${REPO_ID} (~372 GB) -> ${MODEL_DIR}"
echo "Log: ${LOG}"

ENV_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_ARGS+=(-e "HF_TOKEN=${HF_TOKEN}")
fi

docker run --rm "${ENV_ARGS[@]}" \
  -v "${MODEL_DIR}:/model" \
  python:3.12-slim \
  bash -c 'pip install -q huggingface_hub && python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id=\"'"${REPO_ID}"'\",
    local_dir=\"/model\",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8,
    token=os.environ.get(\"HF_TOKEN\") or None,
)
open(\"/model/.download_complete\", \"w\").write(\"ok\\n\")
"'

echo "Download finished. Verify MTP heads:"
echo "  ls -l ${MODEL_DIR}/out-mtp-*"
