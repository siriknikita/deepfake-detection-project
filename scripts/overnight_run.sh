#!/usr/bin/env bash
# Overnight FF++ pipeline runner.
#
# Runs the four end-to-end stages back-to-back, each to its own log file,
# stops at the first failure, and prints final test AUROCs at the end so
# you can read off the headline numbers in the morning:
#
#   01  extract_faces.py        face crops from existing frames/     (~30 min)
#   02  pivot_study.py          baseline_3ch trained on face crops   (~25 min)
#   03  cache_physics_maps.py   physics maps from face crops         (~50 min)
#   04  train_physics_cnn.py    physics_6ch_heuristic on face crops  (~25 min)
#
# Face extraction runs first so a failure there (MTCNN won't load, GPU
# OOM, no source frames on disk) aborts before any destructive cleanup
# of the existing physics caches. Cleanup of the obsolete full-frame
# physics_<variant>/ dirs only happens *after* face extraction
# succeeds.
#
# Frame extraction from raw videos (extract_frames.py) is NOT part of
# this script. It assumes <DATA_ROOT>/.../<comp>/frames/<vid>/<f>.png
# already exist — that's what your prior physics-cache run was reading.
#
# Total wall-clock: ~2 hours on RTX 3080 Ti / i7-11700KF.
#
# Usage (defaults are sane):
#
#   bash scripts/overnight_run.sh
#
# Override via env vars:
#
#   DATA_ROOT          default $HOME/data/FaceForensics++
#   RUNS_DIR           default runs
#   DEVICE             default cuda
#   EPOCHS             default 20
#   LR                 default 2e-4
#   FRAMES_PER_VIDEO   default 10  (matches what train scripts read)
#   IMAGE_SIZE         default 256
#   MARGIN             default 0.3 (face-crop padding around MTCNN bbox)
#   NUM_WORKERS        default 4   (cache_physics_maps thread pool)
#   SKIP_CLEANUP=1     keep old full-frame physics_heuristic caches
#                      (default removes them to free ~16 GB disk)
#   SKIP_SYNC=1        skip the `uv sync` step at start
#   KEEP_RUNS=1        don't wipe runs/baseline_3ch_faces and
#                      runs/physics_6ch_faces_heuristic before training
#                      (default starts each training from scratch)
#
# Logs land in runs/overnight_<timestamp>/:
#   00-uv-sync.log            (if not skipped)
#   01-extract-faces.log
#   02-baseline-3ch.log
#   03-cache-physics.log
#   04-physics-6ch.log
#   run.log                   (master timeline + final summary)

set -euo pipefail

# ── Config (override via env) ────────────────────────────────────────────────

DATA_ROOT="${DATA_ROOT:-$HOME/data/FaceForensics++}"
RUNS_DIR="${RUNS_DIR:-runs}"
DEVICE="${DEVICE:-cuda}"
FRAMES_PER_VIDEO="${FRAMES_PER_VIDEO:-10}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-2e-4}"
MARGIN="${MARGIN:-0.3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"
SKIP_SYNC="${SKIP_SYNC:-0}"
KEEP_RUNS="${KEEP_RUNS:-0}"

TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$RUNS_DIR/overnight_$TS"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.log"

BL_DIR="$RUNS_DIR/baseline_3ch_faces"
PH_DIR="$RUNS_DIR/physics_6ch_faces_heuristic"

# ── Helpers ──────────────────────────────────────────────────────────────────

HR='========================================================================'

log() {
    # Timestamped, mirrored to RUN_LOG
    printf '[%(%H:%M:%S)T] %s\n' -1 "$*" | tee -a "$RUN_LOG"
}

hr() {
    printf '%s\n' "$HR" | tee -a "$RUN_LOG"
}

run_step() {
    # Run a command, redirecting its output to a per-step log, with
    # human-friendly start/finish lines on stdout. Aborts the whole
    # script on failure (set -e), but prints the tail of the failed
    # log first so the cause is visible without grepping files.
    local name="$1"; shift
    local step_log="$LOG_DIR/${name}.log"
    hr
    log ">>> START  $name"
    log "    log: $step_log"
    log "    cmd: $*"
    local t0=$SECONDS
    if "$@" > "$step_log" 2>&1 ; then
        log ">>> OK     $name ($((SECONDS - t0))s)"
    else
        local rc=$?
        log ">>> FAIL   $name (exit=$rc) after $((SECONDS - t0))s"
        log "--- last 30 lines of $step_log ---"
        tail -30 "$step_log" | sed 's/^/    /' | tee -a "$RUN_LOG"
        exit "$rc"
    fi
}

# ── Pre-flight ───────────────────────────────────────────────────────────────

hr
log "Overnight FF++ pipeline run"
log "  DATA_ROOT=$DATA_ROOT"
log "  RUNS_DIR=$RUNS_DIR  LOG_DIR=$LOG_DIR"
log "  DEVICE=$DEVICE  EPOCHS=$EPOCHS  LR=$LR"
log "  FRAMES_PER_VIDEO=$FRAMES_PER_VIDEO  IMAGE_SIZE=$IMAGE_SIZE  MARGIN=$MARGIN"

if [ ! -d "$DATA_ROOT" ]; then
    log "ERROR: DATA_ROOT $DATA_ROOT does not exist"
    exit 1
fi

if [ ! -f "$DATA_ROOT/splits/train.json" ]; then
    log "ERROR: $DATA_ROOT/splits/train.json missing — required for --use-ff-splits."
    log "Download via:"
    log "  mkdir -p $DATA_ROOT/splits"
    log "  curl -L -o $DATA_ROOT/splits/train.json https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json"
    log "  curl -L -o $DATA_ROOT/splits/val.json   https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json"
    log "  curl -L -o $DATA_ROOT/splits/test.json  https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json"
    exit 1
fi

if ! command -v uv > /dev/null 2>&1 ; then
    log "ERROR: uv is not on PATH"
    exit 1
fi

# Informational: starting disk state
log "starting disk state on $DATA_ROOT:"
df -h "$DATA_ROOT" 2>&1 | sed 's/^/    /' | tee -a "$RUN_LOG" || true

# ── Step 0: uv sync (required for facenet-pytorch import in step 1) ──────────

if [ "$SKIP_SYNC" != "1" ]; then
    run_step "00-uv-sync" uv sync
fi

# ── Step 1: face extraction ──────────────────────────────────────────────────
#
# Runs FIRST so any failure (MTCNN/torch import, GPU OOM, missing
# frames/ source) aborts before we touch the existing physics caches.

run_step "01-extract-faces" \
    uv run python scripts/extract_faces.py \
        --data-root "$DATA_ROOT" \
        --dataset face-forensics --compression c23 \
        --output-size "$IMAGE_SIZE" --margin "$MARGIN" \
        --max-frames-per-video "$FRAMES_PER_VIDEO" \
        --device "$DEVICE"

# ── Step 1b: cleanup old full-frame physics caches ──────────────────────────
#
# Now that face crops exist, the full-frame physics_<variant>/ caches
# from earlier runs are not used by anything downstream. Remove them to
# free ~16 GB before the face-crop physics caching pass writes its own
# ~10-15 GB.

if [ "$SKIP_CLEANUP" != "1" ]; then
    hr
    log ">>> CLEANUP  removing obsolete full-frame physics_<variant>/ caches"
    OLD_CACHES=(
        "$DATA_ROOT/original_sequences/youtube/c23/physics_heuristic"
        "$DATA_ROOT/original_sequences/youtube/c23/physics_gtmask"
    )
    for method in Deepfakes Face2Face FaceSwap NeuralTextures ; do
        for variant in heuristic gtmask ; do
            OLD_CACHES+=("$DATA_ROOT/manipulated_sequences/$method/c23/physics_$variant")
        done
    done
    freed=0
    for d in "${OLD_CACHES[@]}"; do
        if [ -d "$d" ]; then
            sz=$(du -s "$d" | cut -f1)
            freed=$((freed + sz))
            log "    rm $d ($(du -sh "$d" | cut -f1))"
            rm -rf "$d"
        fi
    done
    log ">>> CLEANUP  done (freed approximately $((freed / 1024 / 1024)) GB)"
    log "post-cleanup disk state:"
    df -h "$DATA_ROOT" 2>&1 | sed 's/^/    /' | tee -a "$RUN_LOG" || true
fi

# ── Step 2: baseline_3ch on face crops ──────────────────────────────────────

if [ "$KEEP_RUNS" != "1" ] && [ -d "$BL_DIR" ]; then
    log "    wiping previous $BL_DIR (KEEP_RUNS=1 to preserve)"
    rm -rf "$BL_DIR"
fi
run_step "02-baseline-3ch" \
    uv run python scripts/pivot_study.py \
        --data-root "$DATA_ROOT" \
        --baselines pure-cnn \
        --frames-per-video-train "$FRAMES_PER_VIDEO" \
        --frames-per-video-eval "$FRAMES_PER_VIDEO" \
        --epochs "$EPOCHS" \
        --runs-dir "$BL_DIR" \
        --device "$DEVICE" \
        --use-ff-splits --use-face-crops --lr "$LR"

# ── Step 3: physics map cache from face crops ──────────────────────────────

run_step "03-cache-physics" \
    uv run python scripts/cache_physics_maps.py \
        --data-root "$DATA_ROOT" \
        --variant heuristic \
        --frames-subdir frames_faces \
        --frames-per-video "$FRAMES_PER_VIDEO" \
        --num-workers "$NUM_WORKERS"

# ── Step 4: physics_6ch on face crops ────────────────────────────────────────

if [ "$KEEP_RUNS" != "1" ] && [ -d "$PH_DIR" ]; then
    log "    wiping previous $PH_DIR (KEEP_RUNS=1 to preserve)"
    rm -rf "$PH_DIR"
fi
run_step "04-physics-6ch" \
    uv run python scripts/train_physics_cnn.py \
        --data-root "$DATA_ROOT" \
        --variant heuristic \
        --frames-per-video-train "$FRAMES_PER_VIDEO" \
        --frames-per-video-eval "$FRAMES_PER_VIDEO" \
        --epochs "$EPOCHS" \
        --runs-dir "$PH_DIR" \
        --device "$DEVICE" \
        --use-ff-splits --use-face-crops --lr "$LR"

# ── Final summary ────────────────────────────────────────────────────────────

hr
log "ALL STAGES DONE. Total wall-clock: ${SECONDS}s ($((SECONDS / 60))m $((SECONDS % 60))s)."
log ""
log "Headline numbers (FF++ test set):"

uv run python - "$BL_DIR" "$PH_DIR" << 'PYEOF' | tee -a "$RUN_LOG"
import json
import sys
from pathlib import Path

def fmt(x):
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x) if x is not None else "n/a"

def load(*candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
    return None

bl_dir = Path(sys.argv[1])
ph_dir = Path(sys.argv[2])

bl = load(bl_dir / "report.partial.json", bl_dir / "report.json")
ph = load(ph_dir / "report.json")

print()
print("baseline_3ch_faces (RGB only)")
if bl and "baselines" in bl and "pure_cnn" in bl["baselines"]:
    m = bl["baselines"]["pure_cnn"]
    print(f"  test frame AUROC :  {fmt(m.get('auroc'))}")
    print(f"  test video AUROC :  {fmt(m.get('video_auroc_mean'))}  (mean-pool)")
    print(f"  test video AUROC :  {fmt(m.get('video_auroc_max'))}  (max-pool)")
else:
    print("  (no report; check 02-baseline-3ch.log)")

print()
print("physics_6ch_faces_heuristic (RGB + W_cnn + z* + R)")
if ph and "ff_test" in ph:
    m = ph["ff_test"]
    print(f"  test frame AUROC :  {fmt(m.get('auroc'))}")
    print(f"  test video AUROC :  {fmt(m.get('video_auroc_mean'))}  (mean-pool)")
    print(f"  test video AUROC :  {fmt(m.get('video_auroc_max'))}  (max-pool)")
else:
    print("  (no report; check 04-physics-6ch.log)")

# Diff line
def gauroc(d):
    if not d:
        return None
    if "ff_test" in d:
        return d["ff_test"].get("auroc")
    if "baselines" in d and "pure_cnn" in d["baselines"]:
        return d["baselines"]["pure_cnn"].get("auroc")
    return None

bl_a, ph_a = gauroc(bl), gauroc(ph)
if isinstance(bl_a, (int, float)) and isinstance(ph_a, (int, float)):
    delta = ph_a - bl_a
    sign = "+" if delta >= 0 else ""
    print()
    print(f"physics - baseline frame AUROC delta:  {sign}{delta:.4f}")
PYEOF

hr
log "All logs in $LOG_DIR/"
log "  00-uv-sync.log, 01-extract-faces.log, 02-baseline-3ch.log,"
log "  03-cache-physics.log, 04-physics-6ch.log, run.log"
