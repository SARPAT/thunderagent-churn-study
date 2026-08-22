#!/usr/bin/env bash
# A/B sweep: 3 seeds x 2 betas = 6 runs, unattended.
#
# beta=0 short-circuits to upstream behaviour (control).
# beta=1 applies the fatigue penalty (treatment).
# Same seed for both arms of a pair => identical workload.
#
# Requires vLLM already serving on :8000. Manages ThunderAgent itself.
# Usage:  bash run_sweep.sh
#         SEEDS="31 32" bash run_sweep.sh      # custom seeds

set -uo pipefail
REPO="${REPO:-$HOME/thunderagent-churn-study}"
TA_PORT="${TA_PORT:-9000}"
SEEDS="${SEEDS:-21 22 23}"
BETAS="${BETAS:-0 1}"

log(){ printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

stop_ta(){ pkill -f thunderagent 2>/dev/null || true; sleep 3; }

start_ta(){
  local beta=$1 tag=$2
  CHURN_BETA="$beta" nohup thunderagent --backend-type vllm \
      --backends http://localhost:8000 --port "$TA_PORT" --metrics --profile 2>&1 \
    | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done \
    > "$REPO/results/scheduler_${tag}.log" 2>&1 &
  disown
}

wait_ready(){
  for i in $(seq 1 60); do
    curl -sf "http://localhost:${TA_PORT}/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  echo "  ERROR: ThunderAgent did not become ready"; return 1
}

run_client(){
  local seed=$1 tag=$2
  cd "$REPO" && python3 client/synthetic_agent.py \
    --n-programs 40 --n-steps 8 --max-tokens 64 \
    --tool-time-min 4.0 --tool-time-max 12.0 \
    --stagger-s 0.2 --max-inflight 40 --seed "$seed" \
    --out "results/${tag}.json"
}

curl -sf http://localhost:8000/health >/dev/null 2>&1 \
  || { echo "vLLM not reachable on :8000 -- start it first"; exit 1; }

TOTAL=0; for s in $SEEDS; do for b in $BETAS; do TOTAL=$((TOTAL+1)); done; done
i=0
for seed in $SEEDS; do
  for beta in $BETAS; do
    i=$((i+1)); tag="s${seed}_b${beta}"
    log "[$i/$TOTAL] seed=$seed beta=$beta -> $tag"
    stop_ta
    start_ta "$beta" "$tag"
    wait_ready || { echo "skipping $tag"; continue; }
    run_client "$seed" "$tag"
  done
done
stop_ta
log "sweep complete -- results/s*_b*.json and results/scheduler_s*_b*.log"
