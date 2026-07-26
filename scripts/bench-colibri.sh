#!/usr/bin/env bash
# A/B benchmark for the Colibrì GPU tuning work.
#
#   ./scripts/bench-colibri.sh baseline   # pre-tuning config (systemd drop-in)
#   ./scripts/bench-colibri.sh tuned      # launcher defaults
#
# Restarts the service into the requested profile, replays a fixed decode probe
# and a fixed prefill probe, and reports wall time, tok/s, the CPU/GPU split
# sampled during the run, and the engine's own [PROF] block.
#
# Fairness: both profiles start from the same .coli_usage snapshot (the learning
# cache would otherwise hand the second run a better pin profile) and with the
# persisted KV dropped (or the second run replays a cached prefix for free).
# The page cache cannot be dropped without root, so a cold-then-warm ordering
# still favours whichever profile runs second -- alternate runs if it is close.
set -uo pipefail

PROFILE="${1:-tuned}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
UNIT="colibri-glm52.service"
DROPIN_DIR="${HOME}/.config/systemd/user/${UNIT}.d"
DROPIN="${DROPIN_DIR}/zz-bench.conf"
RESULTS="${ROOT}/.bench"

[[ -f "${ENV_FILE}" ]] && { set -a; source "${ENV_FILE}"; set +a; }
PORT="${COLIBRI_PORT:-8001}"
BASE="http://127.0.0.1:${PORT}"
MODEL_DIR="${COLI_MODEL:-${ROOT}/models/glm52-colibri-int4}"
MODEL_ID="${COLI_MODEL_ID:-glm-5.2-colibri}"
KEY="${COLI_API_KEY:-}"

GEN_TOKENS="${BENCH_GEN:-64}"
PREFILL_TOKENS="${BENCH_PREFILL:-4000}"
HEALTH_WAIT="${BENCH_HEALTH_WAIT:-900}"
REQ_TIMEOUT="${BENCH_TIMEOUT:-1800}"

mkdir -p "${RESULTS}"

case "${PROFILE}" in
  baseline|tuned) ;;
  *) echo "usage: $0 [baseline|tuned]" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- profile ----
mkdir -p "${DROPIN_DIR}"
if [[ "${PROFILE}" == "baseline" ]]; then
  # Pin every knob the tuning added back to its pre-change value. PROF stays on
  # in both arms: it is a diagnostic, not a tuning lever.
  cat > "${DROPIN}" <<'EOF'
[Service]
Environment=CUDA_EXPERT_GB=60
Environment=RAM_GB=45
Environment=CUDA_RELEASE_HOST=0
Environment=CUDA_DENSE=0
Environment=COLI_CUDA_ATTN=0
Environment=COLI_CUDA_PIPE=1
Environment=COLI_CUDA_TC_W4A16=0
Environment=COLI_CUDA_SHARED_W4A16=0
Environment=COLI_CUDA_MTP=1
Environment=DRAFT=-1
Environment=PIN_FILL=0
Environment=PIN_GB=10
Environment=REPIN=0
Environment=URING=0
Environment=PROF=1
EOF
else
  rm -f "${DROPIN}"
fi

# ------------------------------------------------------------- fair start ----
SNAPSHOT="${RESULTS}/coli_usage.snapshot"
if [[ -f "${MODEL_DIR}/.coli_usage" && ! -f "${SNAPSHOT}" ]]; then
  cp "${MODEL_DIR}/.coli_usage" "${SNAPSHOT}"
  echo "==> saved .coli_usage snapshot for reproducible pin profiles"
fi
[[ -f "${SNAPSHOT}" ]] && cp "${SNAPSHOT}" "${MODEL_DIR}/.coli_usage"
rm -f "${MODEL_DIR}/.coli_kv"

echo "==> restarting ${UNIT} into '${PROFILE}'"
systemctl --user daemon-reload
systemctl --user restart "${UNIT}"

printf '==> waiting for health (up to %ss)' "${HEALTH_WAIT}"
deadline=$(( $(date +%s) + HEALTH_WAIT ))
until curl -sf --max-time 5 "${BASE}/health" >/dev/null 2>&1; do
  if (( $(date +%s) > deadline )); then
    echo " timeout"; systemctl --user status "${UNIT}" --no-pager | tail -20; exit 1
  fi
  if ! systemctl --user is-active --quiet "${UNIT}"; then
    echo " unit died"; journalctl --user -u "${UNIT}" -n 40 --no-pager; exit 1
  fi
  printf '.'; sleep 5
done
LOAD_SECS=$(( HEALTH_WAIT - (deadline - $(date +%s)) ))
echo " up in ~${LOAD_SECS}s"

# --------------------------------------------------------------- sampling ----
SAMPLE_FILE="$(mktemp)"
sample_start() {
  ( while :; do
      g="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
      p="$(pgrep -f '/colibri/c/colibri ' | head -1)"
      c="$(ps -p "${p:-0}" -o pcpu= --no-headers 2>/dev/null | tr -d ' ')"
      echo "${g:-0} ${c:-0}" >> "${SAMPLE_FILE}"
      sleep 2
    done ) &
  SAMPLER_PID=$!
}
sample_stop() {
  kill "${SAMPLER_PID}" 2>/dev/null; wait "${SAMPLER_PID}" 2>/dev/null
  awk '{g+=$1; c+=$2; n++} END{ if(n) printf "  gpu avg %.0f%% | engine cpu avg %.0f%% (%.1f cores) over %d samples\n", g/n, c/n, (c/n)/100, n }' "${SAMPLE_FILE}"
  : > "${SAMPLE_FILE}"
}

# Instantaneous %CPU needs a prior sample to be meaningful.
ps -p "$(pgrep -f '/colibri/c/colibri ' | head -1)" -o pcpu= >/dev/null 2>&1

probe() {
  local label="$1" prompt="$2" maxtok="$3"
  echo
  echo "-- ${label}"
  local body out code secs
  body="$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],"max_tokens":int(sys.argv[2]),"temperature":0,
                  "messages":[{"role":"user","content":sys.argv[3]}]}))' \
    "${MODEL_ID}" "${maxtok}" "${prompt}")"
  sample_start
  out="$(curl -s --max-time "${REQ_TIMEOUT}" -w '\n%{http_code} %{time_total}' \
          -X POST "${BASE}/v1/messages" \
          -H 'Content-Type: application/json' -H "x-api-key: ${KEY}" \
          -d "${body}" 2>&1)"
  sample_stop
  code="$(tail -1 <<<"${out}" | awk '{print $1}')"
  secs="$(tail -1 <<<"${out}" | awk '{print $2}')"
  printf '  http %s in %ss\n' "${code}" "${secs}"
  if [[ "${code}" != "200" ]]; then
    printf '  body: %s\n' "$(head -c 300 <<<"${out}")"
    return
  fi
  sed '$d' <<<"${out}" | python3 -c '
import json,sys
d=json.load(sys.stdin); u=d.get("usage",{})
secs=float(sys.argv[1]); i=u.get("input_tokens",0); o=u.get("output_tokens",0)
print(f"  tokens: in {i} / out {o}")
print(f"  decode: {o/secs:.2f} tok/s overall (wall includes prefill)")
txt="".join(b.get("text","") for b in d.get("content",[]) if b.get("type")=="text")
print(f"  reply : {txt[:90]!r}")
for m in ("<|user|>","<|assistant|>","<|observation|>"):
    if m in txt: print(f"  WARN  : role marker {m} leaked into the reply")
' "${secs}"
}

FILLER="$(python3 -c "print(('the quick brown fox jumps over the lazy dog. ' * ${PREFILL_TOKENS})[:${PREFILL_TOKENS}*4])")"

probe "decode probe (short prompt, ${GEN_TOKENS} tokens)" \
      "Reply with a single short sentence about MoE inference." "${GEN_TOKENS}"

probe "prefill probe (~${PREFILL_TOKENS} tokens in, 8 out)" \
      "${FILLER}

Answer in one word: what animal is mentioned above?" 8

rm -f "${SAMPLE_FILE}"

echo
echo "-- engine [PROF] (this run)"
START_TS="$(systemctl --user show "${UNIT}" -p ExecMainStartTimestamp --value 2>/dev/null)"
journalctl --user -u "${UNIT}" --since "${START_TS}" --no-pager 2>/dev/null \
  | grep -E '\[PROF\]|\[CUDA\] hot expert|\[RAM_GB' \
  | sed -E 's/^.*start-colibri\.sh\[[0-9]+\]: /  /' | tail -24

echo
echo "==> profile '${PROFILE}' done."
[[ "${PROFILE}" == "baseline" ]] && echo "    drop-in still active: ${DROPIN}"
[[ "${PROFILE}" == "baseline" ]] && echo "    run './scripts/bench-colibri.sh tuned' to remove it and compare."
exit 0
