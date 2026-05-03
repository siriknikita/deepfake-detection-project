#!/usr/bin/env bash
#
# cluster_diagnose.sh — print actionable diagnostics for the training VM.
#
# The cluster you train on is a Proxmox VM, which means several of the
# numbers you'd take for granted on bare metal — GPU performance, RAM
# headroom, "8 cores", disk capacity — are the *VM's view* and may
# differ from the host's. This script gathers what matters before you
# kick off a long job:
#
#   1. Virtualization detection           — what hypervisor are we on
#   2. CPU layout                         — vCPU count and pinning hints
#   3. RAM and ballooning                 — usable memory under host pressure
#   4. Disk capacity                      — including /scratch (the real budget)
#   5. GPU passthrough check              — driver, compute mode, memory
#   6. NVLink / multi-GPU layout          — for 2× 3080 sanity
#   7. Filesystem performance smell-test  — random-write throughput
#
# Run once before each long job:
#   ./scripts/cluster_diagnose.sh
#   ./scripts/cluster_diagnose.sh --json   # machine-readable, for logs
#
# Exit code is always 0 — diagnostics are informational, not pass/fail.

set -uo pipefail

JSON=0
if [[ "${1:-}" == "--json" ]]; then
  JSON=1
fi

# Plain key=value list rather than `declare -A` so this runs on bash 3.2
# (the macBook M4 dev shell) as well as bash 4+ (the cluster VM).
REPORT_KEYS=""
REPORT_VALUES=""

emit_human() {
  printf '\n=== %s ===\n' "$1"
}

set_kv() {
  # Newline-separated buffers; keys and values stay in lock-step. Values
  # are pre-escaped: replace internal " and \ with their JSON escapes,
  # since the JSON dump is naive concatenation.
  local v="$2"
  v="${v//\\/\\\\}"
  v="${v//\"/\\\"}"
  v="${v//$'\n'/\\n}"
  REPORT_KEYS="${REPORT_KEYS}${1}"$'\n'
  REPORT_VALUES="${REPORT_VALUES}${v}"$'\n'
}

# ---------- 1. Virtualization detection ----------
[[ $JSON -eq 0 ]] && emit_human "Virtualization"
VIRT_KIND="unknown"
if command -v systemd-detect-virt >/dev/null 2>&1; then
  VIRT_KIND="$(systemd-detect-virt 2>/dev/null || echo none)"
elif [[ -r /sys/class/dmi/id/product_name ]]; then
  VIRT_KIND="$(cat /sys/class/dmi/id/product_name)"
fi
set_kv "virt_kind" "$VIRT_KIND"
[[ $JSON -eq 0 ]] && printf 'hypervisor: %s\n' "$VIRT_KIND"
if [[ "$VIRT_KIND" == "kvm" || "$VIRT_KIND" == "qemu" ]]; then
  [[ $JSON -eq 0 ]] && printf 'note:       running under KVM/QEMU (Proxmox uses KVM).\n'
fi

# ---------- 2. CPU layout ----------
[[ $JSON -eq 0 ]] && emit_human "CPU"
NPROC=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN || echo "?")
set_kv "vcpu_count" "$NPROC"
CPU_MODEL="unknown"
if [[ -r /proc/cpuinfo ]]; then
  CPU_MODEL=$(awk -F: '/^model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ *//')
fi
set_kv "cpu_model" "$CPU_MODEL"
[[ $JSON -eq 0 ]] && printf 'vCPUs:     %s\nmodel:     %s\n' "$NPROC" "$CPU_MODEL"

# Detect oversubscription hint: if `lscpu` lists the hypervisor's CPU
# steal time, neighbors are competing for cycles.
if command -v vmstat >/dev/null 2>&1; then
  STEAL=$(vmstat 1 2 2>/dev/null | tail -1 | awk '{print $NF}')
  if [[ -n "$STEAL" && "$STEAL" =~ ^[0-9]+$ && "$STEAL" -gt 0 ]]; then
    set_kv "cpu_steal_pct" "$STEAL"
    [[ $JSON -eq 0 ]] && printf 'cpu steal: %s%% (host neighbors are taking cycles — expect jitter)\n' "$STEAL"
  fi
fi

# ---------- 3. RAM ----------
[[ $JSON -eq 0 ]] && emit_human "RAM"
MEM_TOTAL_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
MEM_AVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
MEM_TOTAL_GB=$((MEM_TOTAL_KB / 1024 / 1024))
MEM_AVAIL_GB=$((MEM_AVAIL_KB / 1024 / 1024))
set_kv "ram_total_gb" "$MEM_TOTAL_GB"
set_kv "ram_available_gb" "$MEM_AVAIL_GB"
[[ $JSON -eq 0 ]] && printf 'total:     %s GB\navailable: %s GB\n' "$MEM_TOTAL_GB" "$MEM_AVAIL_GB"

# Ballooning indicator: virtio_balloon module present means Proxmox can
# reclaim memory from this VM under host pressure. The actual current
# balloon size lives in /sys/devices/virtual/misc/balloon* on some
# guests but is not always exposed; the module presence is a reliable
# "yes, this VM is balloonable" signal.
if [[ -d /sys/module/virtio_balloon ]]; then
  set_kv "balloon_module" "loaded"
  [[ $JSON -eq 0 ]] && printf 'balloon:   virtio_balloon loaded — host can reclaim RAM under pressure.\n'
  [[ $JSON -eq 0 ]] && printf '           Watch dmesg for OOMs; lower --batch-size if you see them.\n'
else
  set_kv "balloon_module" "not-loaded"
  [[ $JSON -eq 0 ]] && printf 'balloon:   virtio_balloon not loaded.\n'
fi

# ---------- 4. Disk capacity ----------
[[ $JSON -eq 0 ]] && emit_human "Disk"
ROOT_AVAIL=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9G')
set_kv "root_avail" "${ROOT_AVAIL:-?}"
[[ $JSON -eq 0 ]] && printf 'root /:    %s available\n' "${ROOT_AVAIL:-?}"

# /scratch is where you should be staging data on most clusters.
SCRATCH_PATHS=("/scratch/$USER" "/scratch" "/data" "/mnt/scratch")
SCRATCH_FOUND=""
for sp in "${SCRATCH_PATHS[@]}"; do
  if [[ -d "$sp" && -w "$sp" ]]; then
    SCRATCH_FOUND="$sp"
    break
  fi
done

if [[ -n "$SCRATCH_FOUND" ]]; then
  SCRATCH_AVAIL=$(df -BG --output=avail "$SCRATCH_FOUND" 2>/dev/null | tail -1 | tr -dc '0-9G')
  set_kv "scratch_path" "$SCRATCH_FOUND"
  set_kv "scratch_avail" "${SCRATCH_AVAIL:-?}"
  [[ $JSON -eq 0 ]] && printf 'scratch:   %s, %s available  <-- stage data here\n' \
    "$SCRATCH_FOUND" "${SCRATCH_AVAIL:-?}"
else
  set_kv "scratch_path" "not-found"
  [[ $JSON -eq 0 ]] && printf 'scratch:   not found at usual mounts. Ask the admin where to stage data.\n'
fi

# Per-dataset disk-budget guidance.
if [[ $JSON -eq 0 ]]; then
  printf '\nrough disk budget (frames @ PNG 256 px, ~10 fps stride):\n'
  printf '  FaceForensics++ (videos c23):   ~30 GB\n'
  printf '  FF++ frames extracted:          ~15 GB\n'
  printf '  Celeb-DF v2 (videos):           ~16 GB\n'
  printf '  Celeb-DF v2 frames:             ~5  GB\n'
  printf '  Celeb-DF v1 (videos + frames):  ~12 GB\n'
  printf '  Total:                          ~80 GB\n'
fi

# ---------- 5. GPU passthrough ----------
[[ $JSON -eq 0 ]] && emit_human "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo 0)
  set_kv "gpu_count" "$GPU_COUNT"
  if [[ $JSON -eq 0 ]]; then
    printf 'nvidia-smi: ok\n'
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_mode \
      --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  fi
  # PCIe topology — passthrough'd cards typically show as direct PCIe
  # devices, not vGPU slices.
  if command -v lspci >/dev/null 2>&1; then
    NVIDIA_PCI_LINES=$(lspci 2>/dev/null | grep -ic 'nvidia' || true)
    set_kv "nvidia_pci_devices" "$NVIDIA_PCI_LINES"
    [[ $JSON -eq 0 ]] && printf 'lspci nvidia entries: %s\n' "$NVIDIA_PCI_LINES"
  fi
else
  set_kv "gpu_count" "0"
  [[ $JSON -eq 0 ]] && printf 'nvidia-smi NOT FOUND. Either GPUs are not passed through or driver missing.\n'
  [[ $JSON -eq 0 ]] && printf 'Ask the cluster admin: "Are the 2x 3080s PCIe-passthrough or vGPU?"\n'
fi

# CUDA-from-PyTorch sanity (the actually-load-bearing check for training).
if command -v python >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1; then
  PY=$(command -v python || command -v python3)
  TORCH_CUDA=$("$PY" - <<'PYEOF' 2>/dev/null || echo "no-torch"
try:
    import torch
    print(f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
except Exception as e:
    print(f"error={e!r}")
PYEOF
  )
  set_kv "torch_cuda" "$TORCH_CUDA"
  [[ $JSON -eq 0 ]] && printf 'torch.cuda: %s\n' "$TORCH_CUDA"
fi

# ---------- 6. Multi-GPU topology ----------
if command -v nvidia-smi >/dev/null 2>&1; then
  [[ $JSON -eq 0 ]] && emit_human "GPU topology"
  TOPO=$(nvidia-smi topo -m 2>/dev/null | head -20 || true)
  if [[ -n "$TOPO" ]]; then
    set_kv "nvidia_topo" "available"
    [[ $JSON -eq 0 ]] && printf '%s\n' "$TOPO"
  fi
fi

# ---------- 7. Filesystem write smoke test ----------
[[ $JSON -eq 0 ]] && emit_human "Filesystem write smoke test"
TMP_TARGET="${SCRATCH_FOUND:-/tmp}"
TMP_FILE="$TMP_TARGET/.cluster_diagnose_$$"
DD_OUT=$(dd if=/dev/zero of="$TMP_FILE" bs=1M count=128 conv=fsync 2>&1 || true)
DD_RATE=$(echo "$DD_OUT" | grep -oE '[0-9.]+ [MG]B/s' | tail -1 || echo "?")
set_kv "fs_write_128mb" "$DD_RATE"
rm -f "$TMP_FILE"
[[ $JSON -eq 0 ]] && printf 'sequential write at %s: %s\n' "$TMP_TARGET" "$DD_RATE"
[[ $JSON -eq 0 ]] && printf '(sub-50 MB/s suggests heavily-shared NFS — set num_workers=2 to avoid I/O thrash)\n'

# ---------- summary / json output ----------
if [[ $JSON -eq 1 ]]; then
  printf '{\n'
  # Iterate the parallel newline-separated buffers in lock-step.
  paste <(printf '%s' "$REPORT_KEYS") <(printf '%s' "$REPORT_VALUES") \
    | awk -F'\t' 'BEGIN{first=1} NF>=2 {
        if (!first) printf(",\n");
        printf("  \"%s\": \"%s\"", $1, $2);
        first=0;
      } END { printf("\n") }'
  printf '}\n'
else
  emit_human "Recommendations"
  printf 'Default cluster invocation pattern that survives this environment:\n\n'
  printf '  tmux new -s forge\n'
  printf '  ./scripts/continue.sh --tag quick -- python scripts/quick_classifier.py \\\n'
  printf '      --data-root %s/FaceForensics++ \\\n' "${SCRATCH_FOUND:-/scratch/\$USER}"
  printf '      --runs-dir %s/runs/quick_phase1 \\\n' "${SCRATCH_FOUND:-/scratch/\$USER}"
  printf '      --device cuda --batch-size 16 --num-workers 2\n\n'
  printf 'If RAM is balloonable (above), keep --batch-size <= 16 and watch dmesg.\n'
  printf 'If torch.cuda is not available, GPU passthrough is broken — ask the admin.\n'
fi

exit 0
