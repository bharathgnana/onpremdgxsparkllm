#!/usr/bin/env bash
# One-shot health report for the GLM-5.2 / Colibrì stack on DGX Spark.
#
# Answers the questions that come up when Claude Code "stops responding":
# is the server up, is a request in flight, where is the time going, how much
# of the expert set is actually resident, and is the GPU doing any work.
#
# Read-only: never restarts the service and never cancels an in-flight turn.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"
UNIT="${COLIBRI_UNIT:-colibri-glm52.service}"
PORT="${COLIBRI_PORT:-8001}"
HOST="127.0.0.1"

[[ -f "${ENV_FILE}" ]] && { set -a; source "${ENV_FILE}"; set +a; }
PORT="${COLIBRI_PORT:-8001}"
BASE="http://${HOST}:${PORT}"

section() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
warn()    { printf '  \033[33m! %s\033[0m\n' "$1"; }
bad()     { printf '  \033[31mX %s\033[0m\n' "$1"; }
ok()      { printf '  \033[32mok\033[0m %s\n' "$1"; }

# Journal window: everything since the current process started, so a restart
# does not mix two configurations into one report.
START_TS="$(systemctl --user show "${UNIT}" -p ExecMainStartTimestamp --value 2>/dev/null)"
jlog() {
  if [[ -n "${START_TS}" ]]; then
    journalctl --user -u "${UNIT}" --since "${START_TS}" --no-pager 2>/dev/null
  else
    journalctl --user -u "${UNIT}" -n 4000 --no-pager 2>/dev/null
  fi
}

section "Service"
STATE="$(systemctl --user is-active "${UNIT}" 2>&1)"
printf '  unit        : %s (%s)\n' "${UNIT}" "${STATE}"
printf '  started     : %s\n' "${START_TS:-unknown}"
printf '  restarts    : %s\n' "$(systemctl --user show "${UNIT}" -p NRestarts --value 2>/dev/null)"
COLI_PID="$(pgrep -f '/colibri/c/colibri ' | head -1)"
SRV_PID="$(pgrep -f 'coli serve --model' | head -1)"
printf '  gateway pid : %s\n' "${SRV_PID:-none}"
printf '  engine pid  : %s\n' "${COLI_PID:-none}"
[[ "${STATE}" != "active" ]] && bad "unit is not active"

section "Endpoint"
HEALTH="$(curl -s --max-time 5 "${BASE}/health" 2>&1)"
if [[ "${HEALTH}" == *'"ok"'* ]]; then
  ok "${BASE}/health -> ${HEALTH}"
else
  bad "${BASE}/health -> ${HEALTH:-no response}"
fi
# /health answers from the gateway thread and stays green while the engine is
# wedged, so it proves reachability only -- never that inference is healthy.
KEY="${COLI_API_KEY:-}"
MODELS="$(curl -s --max-time 5 -H "x-api-key: ${KEY}" "${BASE}/v1/models" 2>&1 | head -c 200)"
printf '  models      : %s\n' "${MODELS:-<none>}"

section "Request state"
# A turn is in flight when the last KV-slot line has no matching completion.
LAST_ACCEPT="$(jlog | grep -E '\[API\] KV slot' | tail -1)"
LAST_PREFILL="$(jlog | grep -E '\[prefill\] layer' | tail -1)"
LAST_PROF="$(jlog | grep -E '\[PROF\] verdict' | tail -1)"
printf '  last accept : %s\n' "${LAST_ACCEPT:-<none>}"
printf '  last prefill: %s\n' "${LAST_PREFILL:-<none>}"
if [[ -n "${LAST_ACCEPT}" && -z "${LAST_PROF}" ]]; then
  warn "a turn was accepted but no PROF block was emitted yet -- still generating"
fi
printf '  http codes  :\n'
jlog | grep -oE '"(GET|POST|HEAD) [^"]+ HTTP/1.1" [0-9]{3}' \
     | awk '{n=NF; code=$n; $n=""; sub(/ +$/,""); key=$0" "code; c[key]++} END{for(k in c) printf "    %-52s %d\n", k, c[k]}' \
     | sort -k2 -r
if jlog | grep -q '" 429'; then
  warn "429s present: requests arrived while the single KV slot was busy"
fi
if jlog | grep -q '" 400'; then
  warn "400s present: request translation rejected input (see gateway errors)"
fi

section "Placement (this run)"
jlog | grep -E '\[CUDA\]|\[PIN\]|\[RAM_GB|\[MTP\]|\[USAGE\]|\[stop\]|\[KV\]' \
     | sed -E 's/^.*start-colibri\.sh\[[0-9]+\]: /  /' | tail -20

section "Residency and time split (last turn)"
if [[ -n "${LAST_PROF}" ]]; then
  jlog | grep -E '\[PROF\]' | tail -12 | sed -E 's/^.*start-colibri\.sh\[[0-9]+\]: /  /'
else
  warn "no [PROF] block yet -- PROF=1 emits one per completed turn"
  warn "complete one short turn, then re-run this script"
fi

section "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>&1 | sed 's/^/  /'
  if [[ -n "${COLI_PID}" ]] && nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q "^${COLI_PID}$"; then
    ok "engine holds a CUDA context"
  else
    bad "engine is NOT on the GPU -- check COLI_CUDA=1 and that the build used CUDA=1"
  fi
  printf '  util samples (gpu%%, power):\n'
  for _ in 1 2 3 4 5; do
    printf '    %s\n' "$(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader 2>/dev/null)"
    sleep 1
  done
else
  bad "nvidia-smi not found"
fi

section "CPU / memory"
if [[ -n "${COLI_PID}" ]]; then
  printf '  engine: %s\n' "$(ps -p "${COLI_PID}" -o pcpu=,pmem=,rss=,etime= --no-headers 2>/dev/null)"
  printf '          (%%cpu %%mem rss-kb elapsed) -- %s cores online\n' "$(nproc)"
fi
free -g | sed 's/^/  /'
printf '  loadavg: %s\n' "$(cut -d' ' -f1-3 /proc/loadavg)"

section "Effective engine env"
# Read from the live process, not this shell: systemd, .env and the launcher
# defaults all layer, and only /proc shows what actually took effect.
if [[ -n "${COLI_PID}" && -r "/proc/${COLI_PID}/environ" ]]; then
  tr '\0' '\n' < "/proc/${COLI_PID}/environ" \
    | grep -E '^(COLI_|CUDA_|PIN|PIN_GB|PIN_FILL|REPIN|RAM_GB|CTX|DRAFT|MTP|SPEC_PIN|PIPE|DIRECT|URING|PILOT|PROF)' \
    | sed -E 's/(API_KEY=).*/\1***/' | sort | sed 's/^/  /'
else
  warn "cannot read engine environment (process gone?)"
fi

section "Client (Claude Code)"
printf '  claude      : %s\n' "$(claude --version 2>/dev/null || echo 'not on PATH')"
if [[ -n "${ANTHROPIC_API_KEY:-}" && -n "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  warn "both ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN are set in this shell"
  warn "launch via scripts/claude-glm52.sh, which unsets ANTHROPIC_API_KEY"
fi
SETTINGS="${HOME}/.claude/settings.json"
if [[ -f "${SETTINGS}" ]]; then
  if grep -q '"ANTHROPIC_API_KEY"' "${SETTINGS}"; then
    warn "${SETTINGS} still defines ANTHROPIC_API_KEY (use ANTHROPIC_AUTH_TOKEN only)"
  fi
  printf '  base url    : %s\n' "$(grep -o '"ANTHROPIC_BASE_URL"[^,]*' "${SETTINGS}" | head -1)"
  printf '  model       : %s\n' "$(grep -o '"ANTHROPIC_MODEL"[^,]*' "${SETTINGS}" | head -1)"
else
  warn "no ${SETTINGS}"
fi

section "Storage"
printf '  model dir   : %s\n' "${COLI_MODEL:-${ROOT}/models/glm52-colibri-int4}"
df -h "${COLI_MODEL:-${ROOT}/models/glm52-colibri-int4}" 2>/dev/null | tail -1 | sed 's/^/  /'

printf '\nHint: a stuck-looking session is usually a >20k-token prefill on one KV slot.\n'
printf 'Clear persisted KV for a clean short turn:  rm -f "%s/.coli_kv"\n' "${COLI_MODEL:-${ROOT}/models/glm52-colibri-int4}"
printf 'Watch it live with:  journalctl --user -u %s -f\n' "${UNIT}"
