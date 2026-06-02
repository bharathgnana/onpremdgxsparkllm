#!/usr/bin/env bash
# ============================================================================
# onpremdgxsparkllm - vLLM lifecycle helper (DGX Spark / Blackwell GB10)
# ============================================================================
# Manages the local vLLM + Ollama stack served from this repo:
#   - download : pull model weights from Hugging Face into ./models/
#   - start    : docker compose up -d (vllm + ollama)
#   - stop     : docker compose stop
#   - validate : fire a quick chat-completion timing
#   - status   : show what's downloaded / running / loaded
#
# Backend-side switching (LLM_PROVIDER=vllm|ollama|groq) is now owned by
# the client repo (e.g. jh_ed_document_intelligence). This script does not
# touch any client repo's .env or containers.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
MODEL_DIR="${REPO_ROOT}/models/qwen3-coder-30b-a3b-instruct-fp8"
MODEL_ID="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
DOWNLOAD_CONTAINER="qwen3-coder-download"
SERVED_MODEL_NAME="qwen3-coder-30b"
VLLM_IMAGE_REF="nvcr.io/nvidia/vllm:25.12.post1-py3"
VLLM_CONTAINER="onprem-vllm"
OLLAMA_CONTAINER="onprem-ollama"

cmd="${1:-status}"

case "${cmd}" in
    download)
        echo "==> Downloading ${MODEL_ID} to ${MODEL_DIR}"
        mkdir -p "${MODEL_DIR}"
        if [[ -f "${MODEL_DIR}/.download_complete" ]]; then
            echo "  Already downloaded. Delete ${MODEL_DIR}/.download_complete to re-pull."
            exit 0
        fi
        if docker ps -a --format '{{.Names}}' | grep -q "^${DOWNLOAD_CONTAINER}$"; then
            echo "  Existing download container detected; following its logs..."
            docker logs -f "${DOWNLOAD_CONTAINER}" || true
            exit 0
        fi
        HF_TOKEN_VAL="$(grep -E '^HF_TOKEN=' "${ENV_FILE}" 2>/dev/null | cut -d= -f2- || true)"
        if [[ -z "${HF_TOKEN_VAL}" ]]; then
            echo "WARN: no HF_TOKEN in ${ENV_FILE}; download may hit Hugging Face anon rate limits."
        fi
        docker run --rm \
            --name "${DOWNLOAD_CONTAINER}" \
            -v "${MODEL_DIR}:/model" \
            -e "HF_TOKEN=${HF_TOKEN_VAL}" \
            python:3.12-slim \
            bash -c "pip install --quiet huggingface_hub && \
                     python -c 'from huggingface_hub import snapshot_download; \
                                snapshot_download(repo_id=\"${MODEL_ID}\", \
                                                  local_dir=\"/model\", \
                                                  max_workers=8, \
                                                  resume_download=True)' && \
                     echo DOWNLOAD_COMPLETE > /model/.download_complete && \
                     echo DOWNLOAD_COMPLETE"
        echo "==> Done. Model size:"
        du -sh "${MODEL_DIR}"
        ;;

    start)
        echo "==> Starting on-prem LLM stack (vllm + ollama)"
        if [[ ! -f "${MODEL_DIR}/.download_complete" ]]; then
            echo "ERROR: model not yet downloaded. Run: $0 download"
            exit 1
        fi
        cd "${REPO_ROOT}"
        docker compose up -d
        echo "==> Waiting for vLLM /health (this may take 1-3 minutes for model load)..."
        for i in $(seq 1 60); do
            if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
                echo "==> vLLM is healthy."
                break
            fi
            sleep 5
        done
        if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
            echo "WARN: vLLM did not become healthy in 5 minutes. Check logs:"
            echo "  docker logs ${VLLM_CONTAINER} --tail 80"
            exit 1
        fi
        echo "==> Waiting for Ollama..."
        for i in $(seq 1 30); do
            if curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
                echo "==> Ollama is healthy."
                exit 0
            fi
            sleep 2
        done
        echo "WARN: Ollama not responding on 11434. Check: docker logs ${OLLAMA_CONTAINER}"
        exit 1
        ;;

    stop)
        echo "==> Stopping on-prem LLM stack"
        cd "${REPO_ROOT}"
        docker compose stop
        ;;

    validate)
        echo "==> Hitting vLLM /v1/chat/completions for a quick timing"
        if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
            echo "ERROR: vLLM /health unreachable at 127.0.0.1:8000. Run: $0 start"
            exit 1
        fi
        python3 - <<PY
import json, time, urllib.request
payload = {
    "model": "${SERVED_MODEL_NAME}",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Write a Python binary search function with a one-line docstring."},
    ],
    "temperature": 0.2,
    "max_tokens": 256,
    "stream": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer not-needed"},
)
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    body = json.load(r)
dt = time.time() - t0
content = body["choices"][0]["message"]["content"]
usage = body.get("usage", {})
prompt_tok = usage.get("prompt_tokens", 0)
out_tok = usage.get("completion_tokens", 0)
print(f"  Prompt tokens:     {prompt_tok}")
print(f"  Completion tokens: {out_tok}")
print(f"  Wall time:         {dt:.2f}s")
if out_tok and dt > 0:
    print(f"  Output speed:      {out_tok/dt:.2f} tok/s")
print(f"\n--- Output (first 240 chars) ---\n{content[:240]}")
PY
        ;;

    status)
        echo "==> on-prem LLM stack status"
        echo ""
        echo "Active model (${MODEL_ID}):"
        if [[ -f "${MODEL_DIR}/.download_complete" ]]; then
            echo "  [OK] Downloaded -> ${MODEL_DIR} ($(du -sh "${MODEL_DIR}" 2>/dev/null | awk '{print $1}'))"
        else
            sz=$(du -sh "${MODEL_DIR}" 2>/dev/null | awk '{print $1}')
            echo "  [..] Pending (current size: ${sz:-0})"
        fi
        echo ""
        echo "Parked models:"
        parked_any=0
        for parked in \
            "${REPO_ROOT}/models/qwen3.5-35b-a3b-fp8.parked" \
            "${REPO_ROOT}/models/qwen3-30b-a3b-instruct-fp8.parked"; do
            if [[ -d "${parked}" ]]; then
                echo "  [PARKED] ${parked} ($(du -sh "${parked}" 2>/dev/null | awk '{print $1}'))"
                parked_any=1
            fi
        done
        [[ ${parked_any} -eq 0 ]] && echo "  (none)"
        echo ""
        echo "vLLM Docker images available:"
        if docker images "${VLLM_IMAGE_REF}" --format '{{.Size}}' | grep -q .; then
            echo "  [present] ${VLLM_IMAGE_REF} ($(docker images "${VLLM_IMAGE_REF}" --format '{{.Size}}'))  -- needs driver >= 590.44"
        fi
        if docker images vllm-patched:qwen3.5 --format '{{.Size}}' | grep -q .; then
            echo "  [present] vllm-patched:qwen3.5 ($(docker images vllm-patched:qwen3.5 --format '{{.Size}}'))   -- sm_121 patched fallback (driver 580.x compatible)"
        fi
        if ! docker images "${VLLM_IMAGE_REF}" --format '{{.Size}}' | grep -q . \
            && ! docker images vllm-patched:qwen3.5 --format '{{.Size}}' | grep -q .; then
            echo "  [..] No vLLM image found. Build the patched image (Dockerfile.vllm-patch) or pull ${VLLM_IMAGE_REF}"
        fi
        echo ""
        echo "vLLM container:"
        if docker ps --format '{{.Names}}' | grep -q "^${VLLM_CONTAINER}$"; then
            running_img=$(docker inspect -f '{{.Config.Image}}' "${VLLM_CONTAINER}" 2>/dev/null)
            echo "  [OK] Running (image: ${running_img})"
            curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null | head -20 || true
        else
            echo "  [..] Not running (start with: $0 start)"
        fi
        echo ""
        echo "Ollama container:"
        if docker ps --format '{{.Names}}' | grep -q "^${OLLAMA_CONTAINER}$"; then
            echo "  [OK] Running"
            docker exec "${OLLAMA_CONTAINER}" ollama list 2>/dev/null | sed 's/^/    /' || true
        else
            echo "  [..] Not running (start with: $0 start)"
        fi
        ;;

    *)
        echo "Unknown command: ${cmd}"
        echo "Usage: $0 {download|start|stop|validate|status}"
        exit 2
        ;;
esac
