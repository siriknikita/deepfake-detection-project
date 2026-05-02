#!/usr/bin/env bash
#
# prepare_data.sh — orchestrate the FaceForensics++ + Celeb-DF download + frame extraction.
#
# Both datasets require accepting a EULA before download. This script does
# not bypass that — it walks you through the canonical access process and
# automates only the parts that don't require personal credentials.
#
# Usage:
#   ./scripts/prepare_data.sh --root /scratch/data
#       [--ff-downloader /path/to/download-FaceForensics++.py]
#       [--celeb-archive /path/to/Celeb-DF-v2.zip]
#       [--compression c23]
#       [--fps 5]
#       [--no-extract]
#       [--methods Deepfakes,Face2Face,FaceSwap,NeuralTextures]
#
# Steps:
#   1. Validates that the user has the FF++ official downloader script, which
#      they get only after signing the EULA at
#      https://github.com/ondyari/FaceForensics. Reminds them if missing.
#   2. Calls the FF++ downloader to fetch the requested compression and
#      methods into <root>/FaceForensics++.
#   3. If a Celeb-DF archive path is given, extracts it into <root>/Celeb-DF-v2.
#      (Celeb-DF is downloaded manually after submitting the Google Form at
#      https://github.com/yuezunli/celeb-deepfakeforensics.)
#   4. Runs scripts/extract_frames.py over the FF++ videos to populate the
#      frame-folder layout the dataset adapter expects.
#
# Idempotent — running again only re-downloads missing items.

set -euo pipefail

ROOT=""
FF_DOWNLOADER=""
CELEB_ARCHIVE=""
COMPRESSION="c23"
FPS=5
DO_EXTRACT=1
METHODS="Deepfakes,Face2Face,FaceSwap,NeuralTextures"

usage() {
    sed -n '2,30p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)         ROOT="$2"; shift 2 ;;
        --ff-downloader) FF_DOWNLOADER="$2"; shift 2 ;;
        --celeb-archive) CELEB_ARCHIVE="$2"; shift 2 ;;
        --compression)  COMPRESSION="$2"; shift 2 ;;
        --fps)          FPS="$2"; shift 2 ;;
        --methods)      METHODS="$2"; shift 2 ;;
        --no-extract)   DO_EXTRACT=0; shift ;;
        -h|--help)      usage 0 ;;
        *)              echo "unknown flag: $1" >&2; usage 2 ;;
    esac
done

if [[ -z "$ROOT" ]]; then
    echo "--root is required" >&2
    usage 2
fi

mkdir -p "$ROOT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ------------ FaceForensics++ download ------------

FF_ROOT="$ROOT/FaceForensics++"
if [[ -d "$FF_ROOT/original_sequences" ]]; then
    log "FaceForensics++ already present at $FF_ROOT — skipping download"
else
    if [[ -z "$FF_DOWNLOADER" ]]; then
        cat <<'EOF' >&2
FaceForensics++ requires accepting the EULA before download. Steps:
  1. Open https://github.com/ondyari/FaceForensics
  2. Fill the Google Form linked in the README. They email you the
     `download-FaceForensics++.py` script (account-keyed).
  3. Re-run this script with `--ff-downloader /path/to/download-FaceForensics++.py`.

Without the keyed script we cannot fetch the videos automatically.
EOF
        exit 3
    fi
    if [[ ! -f "$FF_DOWNLOADER" ]]; then
        echo "FF++ downloader script not found at $FF_DOWNLOADER" >&2
        exit 3
    fi
    log "downloading FaceForensics++ ($COMPRESSION) into $FF_ROOT"
    mkdir -p "$FF_ROOT"
    # Real videos.
    python "$FF_DOWNLOADER" "$FF_ROOT" -d original_videos -c "$COMPRESSION" -t videos
    # Manipulated videos for each requested method.
    IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"
    for method in "${METHOD_ARRAY[@]}"; do
        log "  fetching $method ($COMPRESSION)"
        python "$FF_DOWNLOADER" "$FF_ROOT" -d "$method" -c "$COMPRESSION" -t videos
    done
    # Pixel-level masks for the manipulated videos (used by the CNN's pixel-level loss).
    log "  fetching binary masks (pixel-level supervision)"
    python "$FF_DOWNLOADER" "$FF_ROOT" -d masks -c "$COMPRESSION" -t videos || true
fi

# ------------ Celeb-DF v2 (manual, post-Google-Form) ------------

CELEB_ROOT="$ROOT/Celeb-DF-v2"
if [[ -d "$CELEB_ROOT/Celeb-real" ]]; then
    log "Celeb-DF v2 already present at $CELEB_ROOT — skipping"
elif [[ -n "$CELEB_ARCHIVE" ]]; then
    if [[ ! -f "$CELEB_ARCHIVE" ]]; then
        echo "Celeb-DF archive not found at $CELEB_ARCHIVE" >&2
        exit 3
    fi
    log "extracting Celeb-DF archive $CELEB_ARCHIVE -> $CELEB_ROOT"
    mkdir -p "$CELEB_ROOT"
    case "$CELEB_ARCHIVE" in
        *.zip)            unzip -q "$CELEB_ARCHIVE" -d "$CELEB_ROOT" ;;
        *.tar.gz|*.tgz)   tar -xzf "$CELEB_ARCHIVE" -C "$CELEB_ROOT" ;;
        *.tar)            tar -xf  "$CELEB_ARCHIVE" -C "$CELEB_ROOT" ;;
        *)                echo "unsupported archive format: $CELEB_ARCHIVE" >&2; exit 3 ;;
    esac
else
    cat <<'EOF'
Celeb-DF v2 requires submitting the Google Form linked at
https://github.com/yuezunli/celeb-deepfakeforensics. After approval the
authors share Google Drive links. Download the zip / tar manually, then
re-run this script with --celeb-archive /path/to/Celeb-DF-v2.zip.
(Skipping for now — FaceForensics++ alone is enough to get started.)
EOF
fi

# ------------ Frame extraction ------------

if [[ $DO_EXTRACT -eq 1 ]] && [[ -d "$FF_ROOT/original_sequences" ]]; then
    log "extracting frames from FF++ videos at $FPS fps"
    if [[ ! -x "$(command -v python)" ]]; then
        echo "python not on PATH" >&2; exit 4
    fi
    python scripts/extract_frames.py "$FF_ROOT" \
        --compression "$COMPRESSION" \
        --fps "$FPS"
fi

log "done."
log "  FaceForensics++ root: $FF_ROOT"
[[ -d "$CELEB_ROOT" ]] && log "  Celeb-DF root:        $CELEB_ROOT"
