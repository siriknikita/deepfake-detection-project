#!/usr/bin/env bash
#
# continue.sh — run a command with auto-restart on crash.
#
# Designed for the cluster + outage-prone power grid combination. Wraps any
# command in an `until ... do ... done` loop so power blips, OOM kills,
# preemption signals, and other transient failures restart automatically.
# Pairs with the resumable training / feature-extraction / pivot-study
# pieces so each restart picks up exactly where the previous attempt left
# off — never from zero.
#
# Usage:
#   ./scripts/continue.sh [OPTIONS] -- COMMAND [ARGS...]
#
# Options:
#   --max-restarts N    Stop after N restarts (default: unlimited).
#   --cooldown SECONDS  Sleep between restarts (default: 60).
#   --log-dir PATH      Where to write timestamped logs (default: logs/).
#   --tag NAME          Log-file prefix (default: command basename).
#   --quiet             Suppress wrapper's own status messages.
#   -h, --help          Show this help.
#
# Exit codes:
#   0    The wrapped command eventually succeeded.
#   130  User pressed Ctrl-C — wrapper stopped, did not restart.
#   143  Wrapper received SIGTERM — wrapper stopped, did not restart.
#   any  Other rc — max-restarts hit; last child rc returned.
#
# Examples:
#
#   # Canonical pivot study, restart on every crash:
#   ./scripts/continue.sh -- python scripts/pivot_study.py \
#       --data-root /scratch/$USER/data/FaceForensics++ \
#       --max-frames-per-video 30 --image-size 256 \
#       --epochs 30 --batch-size 32 --device cuda \
#       --runs-dir runs/pivot_full \
#       --output runs/pivot_full/report.json
#
#   # Just train the trust-map CNN, with auto-restart pointing at an
#   # existing run dir so resume picks up from the last checkpoint:
#   ./scripts/continue.sh --tag train -- forge-detect train \
#       --data-root /scratch/$USER/data/FaceForensics++ \
#       --resume runs/pivot_full/trust_map_run \
#       --device cuda --epochs 30
#
#   # Cap restarts so a genuinely broken command does not loop forever:
#   ./scripts/continue.sh --max-restarts 5 --cooldown 120 -- python my_script.py
#
# Recommended cluster invocation (lets you log out without losing the run):
#
#   tmux new -s forge
#   ./scripts/continue.sh -- python scripts/pivot_study.py ...
#   # Ctrl-b d to detach. ssh out. Come back: tmux attach -t forge.
#
# On SLURM: just run continue.sh from the sbatch script. SLURM's --requeue
# handles cluster-level restarts; continue.sh handles in-process crashes.

set -uo pipefail

MAX_RESTARTS=""
COOLDOWN=60
LOG_DIR="logs"
TAG=""
QUIET=0

usage() {
  sed -n '2,55p' "$0"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-restarts)
      MAX_RESTARTS="$2"
      shift 2
      ;;
    --cooldown)
      COOLDOWN="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -h | --help) usage 0 ;;
    -*)
      echo "unknown flag: $1" >&2
      usage 2
      ;;
    *)
      # Treat anything else as the start of the command (no `--` required).
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "no command supplied — pass the command after --" >&2
  usage 2
fi

mkdir -p "$LOG_DIR"

if [[ -z "$TAG" ]]; then
  TAG="$(basename "$1" | tr -c 'a-zA-Z0-9._-' '_')"
fi
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_DIR}/${TAG}-${STAMP}.log"
RESTART_LOG="${LOG_DIR}/restart-${TAG}.log"

log() {
  if [[ $QUIET -eq 0 ]]; then
    printf '[continue.sh %s] %s\n' "$(date +%H:%M:%S)" "$*"
  fi
}

# Soft warning if the user is not in a persistent session — outage protection
# only helps if the wrapper itself survives the SSH disconnect.
if [[ -z "${TMUX:-}" ]] && [[ -z "${STY:-}" ]] && [[ -z "${SLURM_JOB_ID:-}" ]]; then
  log "warning: not in tmux / screen / SLURM. An SSH disconnect kills the wrapper too."
  log "         Recommended: \`tmux new -s forge\` first, then re-run this script."
fi

if [[ $QUIET -eq 0 ]]; then
  cat <<EOF
[continue.sh]
  command:   $*
  log file:  $LOG_FILE
  restarts:  ${MAX_RESTARTS:-unlimited}
  cooldown:  ${COOLDOWN}s
  pid:       $$
  started:   $(date)
EOF
fi

# A single trap sets a flag that breaks the loop; we do NOT exit immediately
# because we want the in-flight child to clean up first.
INTERRUPTED=0
trap 'INTERRUPTED=1' INT TERM

restart_count=0
last_rc=0

while [[ $INTERRUPTED -eq 0 ]]; do
  log "starting attempt $((restart_count + 1))"
  # Run the command, copy output to both terminal and log file. PIPESTATUS[0]
  # is the rc of the wrapped command, not of tee.
  "$@" 2>&1 | tee -a "$LOG_FILE"
  last_rc=${PIPESTATUS[0]}

  # Successful exit — done.
  if [[ $last_rc -eq 0 ]]; then
    log "command exited cleanly (rc=0)"
    printf '[%s] success after %d restarts\n' "$(date)" "$restart_count" >>"$RESTART_LOG"
    exit 0
  fi

  # Interrupted by user / supervisor — stop the loop instead of restarting.
  if [[ $INTERRUPTED -eq 1 ]] || [[ $last_rc -eq 130 ]] || [[ $last_rc -eq 143 ]]; then
    log "interrupted (rc=$last_rc) — stopping the auto-restart loop"
    printf '[%s] interrupted (rc=%d) after %d restarts\n' \
      "$(date)" "$last_rc" "$restart_count" >>"$RESTART_LOG"
    exit "$last_rc"
  fi

  restart_count=$((restart_count + 1))
  printf '[%s] crash rc=%d — restart #%d\n' "$(date)" "$last_rc" "$restart_count" \
    >>"$RESTART_LOG"
  log "crash rc=$last_rc — restart #$restart_count after ${COOLDOWN}s"

  if [[ -n "$MAX_RESTARTS" ]] && [[ $restart_count -ge $MAX_RESTARTS ]]; then
    log "max restarts ($MAX_RESTARTS) reached — giving up"
    printf '[%s] gave up after %d restarts (last rc=%d)\n' \
      "$(date)" "$restart_count" "$last_rc" >>"$RESTART_LOG"
    exit "$last_rc"
  fi

  # Sleep, but allow a SIGINT/SIGTERM during the cooldown to break us out.
  sleep "$COOLDOWN" || true
done

log "interrupted before any successful run (rc=$last_rc)"
exit "$last_rc"
