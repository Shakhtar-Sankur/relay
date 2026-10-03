#!/usr/bin/env bash
# Starts a local relay cluster: worker processes plus the Swift API server in front.
#
#   scripts/cluster.sh MODEL_DIR colocated N            N workers with role both
#   scripts/cluster.sh MODEL_DIR disaggregated P D      P prefill workers, D decode workers
#
# Environment: RELAY_BACKEND=cpu|cuda (default cpu), RELAY_PORT (API port, default 8000),
# RELAY_BLOCKS (KV blocks per worker, default 512), RELAY_GPUS ("0 1 2 3": the GPU of each
# worker in start order, default all on GPU 0), RELAY_WORKER, RELAY_SERVER (binaries).
# Stops everything on Ctrl-C or when the script exits.
set -euo pipefail
model=$1 mode=$2
root=$(cd "$(dirname "$0")/.." && pwd)
worker_bin=${RELAY_WORKER:-$root/build/relay-worker}
server_bin=${RELAY_SERVER:-$root/.build/release/relay-server}
backend=${RELAY_BACKEND:-cpu}
blocks=${RELAY_BLOCKS:-512}
read -r -a gpus <<< "${RELAY_GPUS:-}"
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Workers are started from this shell, not from $(...): a worker stops when its parent
# process exits, and a command substitution's subshell exits at once.
specs=()
start_worker() {  # role
  local role=$1 log gpu=${gpus[${#pids[@]}]:-0}
  log=$(mktemp)
  "$worker_bin" --model "$model" --role "$role" --backend "$backend" --device "$gpu" --blocks "$blocks" > "$log" 2>&1 &
  pids+=($!)
  for _ in $(seq 1 600); do
    if grep -q "control=" "$log"; then
      specs+=("$role=127.0.0.1:$(sed -n 's/.*control=\([0-9]*\).*/\1/p' "$log")")
      return
    fi
    sleep 0.1
  done
  echo "worker failed to start:" >&2
  cat "$log" >&2
  exit 1
}

if [ "$mode" = colocated ]; then
  for _ in $(seq 1 "$3"); do start_worker both; done
else
  for _ in $(seq 1 "$3"); do start_worker prefill; done
  for _ in $(seq 1 "$4"); do start_worker decode; done
fi
workers=$(IFS=,; echo "${specs[*]}")
echo "workers: $workers"
"$server_bin" --model "$model" --workers "$workers" --port "${RELAY_PORT:-8000}" &
pids+=($!)
wait
