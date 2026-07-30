#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${RLVR_RUNS_DIR:-/mnt/shared-storage-user/liujinyi/lj/rlvr_runs}"
JOBS="${WANDB_SYNC_JOBS:-4}"
INTERVAL="${WANDB_SYNC_INTERVAL:-30}"
WATCH=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: sync_wandb_runs.sh [options]

Options:
  --root DIR       Root directory to scan for wandb offline runs.
  -j, --jobs N     Number of concurrent wandb sync processes. Default: 4
  -i, --interval S Scan interval in watch mode. Default: 30
  --watch          Keep rescanning for new offline runs.
  --dry-run        Print the wandb sync commands without executing them.
  -h, --help       Show this help message.

Environment:
  RLVR_RUNS_DIR       Default scan root if --root is not set.
  WANDB_SYNC_JOBS     Default concurrency if --jobs is not set.
  WANDB_SYNC_INTERVAL Default watch interval if --interval is not set.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT_DIR="${2:?missing value for --root}"
      shift 2
      ;;
    -j|--jobs)
      JOBS="${2:?missing value for --jobs}"
      shift 2
      ;;
    -i|--interval)
      INTERVAL="${2:?missing value for --interval}"
      shift 2
      ;;
    --watch)
      WATCH=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Root directory does not exist: $ROOT_DIR" >&2
  exit 1
fi

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || (( JOBS < 1 )); then
  echo "Invalid concurrency: $JOBS" >&2
  exit 1
fi

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || (( INTERVAL < 1 )); then
  echo "Invalid watch interval: $INTERVAL" >&2
  exit 1
fi

if (( DRY_RUN == 0 )) && ! command -v wandb >/dev/null 2>&1; then
  echo "wandb CLI not found in PATH. Activate the environment that provides wandb before running sync." >&2
  exit 1
fi

discover_run_dirs() {
  find "$ROOT_DIR" -type f \( -name '*.wandb' -o -name '*.wnb' \) -print0 \
    | while IFS= read -r -d '' file; do
        printf '%s\0' "$(dirname "$file")"
      done \
    | sort -zu
}

sync_run_dir() {
  local run_dir="$1"
  if (( DRY_RUN )); then
    printf '[dry-run] wandb sync %q\n' "$run_dir"
    return 0
  fi

  printf '[sync] %s\n' "$run_dir"
  wandb sync "$run_dir"
}

sync_batch() {
  local -a run_dirs=("$@")
  local -a pids=()
  local failures=0
  local run_dir pid

  for run_dir in "${run_dirs[@]}"; do
    sync_run_dir "$run_dir" &
    pids+=("$!")

    if (( ${#pids[@]} >= JOBS )); then
      pid="${pids[0]}"
      if ! wait "$pid"; then
        failures=1
      fi
      pids=("${pids[@]:1}")
    fi
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=1
    fi
  done

  return "$failures"
}

declare -A seen_runs=()

process_once() {
  local -a discovered=()
  local -a pending=()
  local run_dir

  mapfile -d '' -t discovered < <(discover_run_dirs || true)

  for run_dir in "${discovered[@]}"; do
    if [[ -z "${seen_runs[$run_dir]+x}" ]]; then
      seen_runs["$run_dir"]=1
      pending+=("$run_dir")
    fi
  done

  if (( ${#pending[@]} == 0 )); then
    echo "No new wandb offline runs found under $ROOT_DIR"
    return 0
  fi

  echo "Found ${#pending[@]} wandb offline run(s) under $ROOT_DIR"
  sync_batch "${pending[@]}"
}

if (( WATCH )); then
  while true; do
    process_once
    sleep "$INTERVAL"
  done
else
  process_once
fi
