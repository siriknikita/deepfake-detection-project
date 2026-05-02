# Hyperplane-Forge

Deepfake detection via _physical-manifold settlement_. The full mathematical specification is in [`paper/main.typ`](paper/main.typ); this README is a quickstart for running the code.

## What it does

A real face is a sample of a smooth physical surface; a deepfake's synthetic injection is not. The pipeline recovers an initial depth manifold `z_forged` from the image (Phase 1-3), settles it under the Euler-Lagrange PDE of a physically motivated energy functional `E(z)` (Phase 4-5), and detects forged regions as _flow breaks_ (`R = z* − z_ideal`) and _geometric cracks_ (`L = Δz*`) in the settled manifold (Phase 6).

The math kernels are implemented in **Rust** (PyO3, `ndarray`, `rayon`) for the CPU path and exposed to **Python** through a single-function PyO3 entry point. A **PyTorch CUDA backend** mirrors the same operators on GPU and is verified numerically equivalent to the Rust core by 6 backend-equivalence tests. A trainable **ChromaticEfficientNet** trust-map predictor and a **Gradient Boosting** binary classifier sit on top of the math kernels.

## Requirements

- Rust toolchain (`cargo` ≥ 1.75) — the project is built with `rustc` 1.91
  but any recent stable toolchain should work.
- Python ≥ 3.10. The dev host runs 3.14 against the stable ABI.
- [`uv`](https://github.com/astral-sh/uv) for Python venv & dependency
  management.
- `typst` for compiling the paper (optional).

## Build & run

```bash
# One-shot dev install: creates .venv, installs deps, builds the
# Rust extension into the venv with full optimizations.
make dev-install

# End-to-end on a single image (uses heuristic trust map):
uv run --python .venv/bin/python forge-detect detect path/to/image.jpg

# Save the diagnostic 6-panel figure:
uv run --python .venv/bin/python forge-detect detect path/to/image.jpg \
    --visualize panel.png

# Print every feature:
uv run --python .venv/bin/python forge-detect detect path/to/image.jpg \
    --print-features
```

## Datasets — where to get the inputs

The pipeline only needs **face images** (real and forged) plus, optionally, **pixel-level fake masks** for stronger CNN supervision. Everything else (`z_forged`, `z_ideal`, `K`, the energy weights) is derived internally.

### Recommended datasets

| Dataset | Size | Provides | Access |
|---|---|---|---|
| **FaceForensics++** | ~1.5 TB (raw) / ~10 GB (c23) | 1k real videos + 4k fake (4 methods) + per-pixel masks | Sign EULA at https://github.com/ondyari/FaceForensics |
| **Celeb-DF (v2)** | ~50 GB | 590 real + 5639 fake videos, high quality | Sign EULA at https://github.com/yuezunli/celeb-deepfakeforensics |
| **DFDC** | ~470 GB | 100k+ videos, less curated, larger scale | Free at https://www.kaggle.com/c/deepfake-detection-challenge/data |
| **WildDeepfake** | ~5 GB | In-the-wild collection | https://github.com/deepfakeinthewild/deepfake-in-the-wild |
| **FFHQ** (real-only) | ~90 GB | 70k 1024² aligned faces | https://github.com/NVlabs/ffhq-dataset |
| **CelebA-HQ** (real-only) | ~30 GB | 30k aligned celebrity faces | https://github.com/tkarras/progressive_growing_of_gans |

**Recommended starting point:** FaceForensics++ at compression `c23`. It is the de facto benchmark, has both real and four families of fake, and ships the per-pixel masks that train the CNN with the strongest signal. The c23 (visually lossless H.264) variant is what most published leaderboards report against.

### Workflow: video files → trained classifier

```bash
# 1. Sign the FF++ EULA, run their downloader to fetch the videos.
#    (See https://github.com/ondyari/FaceForensics — their script
#    pulls original_sequences/ + manipulated_sequences/.)

# 2. Extract frames (5 fps is plenty for our purposes):
python scripts/extract_frames.py /path/to/FaceForensics++ \
    --compression c23 --fps 5

# 3. Train ChromaticEfficientNet on the resulting frames:
forge-detect train \
    --data-root /path/to/FaceForensics++ \
    --dataset face-forensics --compression c23 \
    --max-frames-per-video 30 --image-size 256 \
    --epochs 30 --batch-size 32 --device cuda \
    --checkpoint-dir runs/

# 4. Train + evaluate the binary classifier on the impact-map features:
forge-detect eval \
    --data-root /path/to/FaceForensics++ \
    --dataset face-forensics --compression c23 \
    --max-frames-per-video 30 --image-size 256 \
    --device cuda \
    --cache features.csv --output classifier.pkl

# 5. Now use the trained classifier in single-image inference:
forge-detect detect suspect_image.jpg --device cuda --visualize panel.png
```

### What if I only have a folder of real and a folder of fake images?

Use the generic `image-folder` adapter:

```bash
forge-detect eval --dataset image-folder \
    --data-root /path/to/real_frames \
    --fake-dir /path/to/fake_frames \
    --image-size 256 --device cuda
```

This works for **any** real/fake source — synthetic faces from FFHQ vs StyleGAN3, in-the-wild scrapes, your own data collection. The pipeline is dataset-agnostic; it only cares about pixels.

### Heuristic trust map (no CNN, no training)

The pipeline is **runnable today without any training**. The default trust map is a deterministic chromatic-residual heuristic (see `python/forge_detect/trust_map.py`). Run `forge-detect detect <image>` on your MacBook with no GPU, no dataset, no CNN weights — you get a working impact map and feature vector. The trained CNN is an upgrade for stronger separation, not a prerequisite.

## Running on the GPU cluster

The cluster workflow has four stages. Each stage is a single shell command and the previous stage's outputs are inputs to the next.

### 1. Bootstrap a fresh node

After `ssh`-ing into a cluster node and `git clone`-ing the repo:

```bash
./scripts/cluster_bootstrap.sh
```

This installs `rustup`, `uv`, builds the Rust extension into `.venv`, probes `torch.cuda.is_available()`, and runs both test suites. **Idempotent** — safe to re-run on partial failures. If `ffmpeg` is missing the script flags it but does not abort; `module load ffmpeg` or `apt install ffmpeg` covers it on most clusters.

### 2. Download datasets

```bash
./scripts/prepare_data.sh \
    --root /scratch/$USER/data \
    --ff-downloader ~/download-FaceForensics++.py \
    --celeb-archive ~/Celeb-DF-v2.zip \
    --compression c23 --fps 5
```

`download-FaceForensics++.py` is the script the FF++ team emails you after EULA acceptance — it is account-keyed, we cannot bundle it. The script wraps it with sensible defaults (real videos, all four manipulation methods, pixel-level masks) and then automatically calls `scripts/extract_frames.py` to convert videos → per-video frame folders. Celeb-DF requires a Google-Form approval; once you have the archive, point `--celeb-archive` at it.

### 3. Forensic check on a small slice (always do this first)

Before committing to a multi-day training run, verify the whole stack on a tiny slice — 1 frame per video, 128² resolution, 3 epochs:

```bash
python scripts/pivot_study.py \
    --data-root /scratch/$USER/data/FaceForensics++ \
    --max-frames-per-video 1 --image-size 128 \
    --epochs 3 --batch-size 16 --device cuda \
    --runs-dir runs/pivot_smoke
```

This trains every component (pure-CNN, trust-map CNN, classifier) on ~5000 frames in roughly 15 minutes on a single 4090. It exists to catch *plumbing* failures (out-of-memory, dataset path wrong, CUDA OOM under the chosen batch size) — **not to produce defendable numbers**. The AUROCs from this run are noisy because of the small dataset; that is expected.

### 4. Full training and the pivot study

Once the smoke run finishes cleanly, run the real comparison:

```bash
python scripts/pivot_study.py \
    --data-root /scratch/$USER/data/FaceForensics++ \
    --max-frames-per-video 30 --image-size 256 \
    --epochs 30 --batch-size 32 --device cuda \
    --runs-dir runs/pivot_full \
    --output runs/pivot_full/report.json
```

This trains:

1. **Pure CNN** — torchvision EfficientNet-B0 directly on `(image, label)` for binary classification.
2. **Pipeline + heuristic W_cnn** — math kernels with the deterministic chromatic-residual trust map → 24-D features → Gradient Boosting Classifier.
3. **Pipeline + trained CNN W_cnn** — same as 2, but `W_cnn` comes from a trained `ChromaticEfficientNet`.

All three use the same train / val / test stratified split and the same optimizer / scheduler / loss family. The script writes a JSON report with AUROC, accuracy, top feature importances, and approximate training time per baseline.

### Pivot decision rubric

| Outcome | What it means | What to do |
|---|---|---|
| Pure CNN > Full pipeline by **≥ 0.03 AUROC** | A learned classifier alone separates real / fake at least as well as the physics framework | Pivot. Keep the trained CNN, drop the math pipeline, defend the engineering work as a learned-classifier project |
| Full pipeline > Pure CNN by **≥ 0.02 AUROC** | The physics signal is real and the framework adds explanatory power | Defend the framework as designed; the math is the contribution |
| Heuristic ≈ Trained CNN, both ≈ Pure CNN | The pipeline carries useful signal but the trained `W_cnn` does not improve over the deterministic heuristic | Drop the CNN training step from the deliverable, keep the deterministic pipeline as a faster, simpler alternative |
| All three within ≈ 0.01 AUROC of each other | No method has signal on this split | Revisit assumptions: dataset balance, image resolution, pipeline parameters; or accept that this dataset is too easy / hard to discriminate methods |

Concrete rough timings on a single RTX 4090 with FF++ c23 at 256² and 30 frames per video (~120k frames):

| Step | Time |
|---|---|
| Frame extraction (all methods + masks at fps=5) | ~6 h |
| Pure-CNN training, 30 epochs | ~3 h |
| Trust-map CNN training, 30 epochs | ~4 h |
| Pipeline feature extraction over the dataset, both runs | ~12 h |
| Total pivot study | ~25 h on one GPU |

This is sequential; on a multi-GPU cluster the two CNN trainings run in parallel and the feature-extraction step parallelizes trivially across GPUs (run with `CUDA_VISIBLE_DEVICES=0,1,...` per node).

### 5. Surviving outages and not having to babysit the script

The Ukrainian power-grid context makes mid-training crashes a real concern. Every long-running step is **crash-resumable by design** — the same command works as both the first run and every restart.

**What is saved when:**

| Step | Saved artifact | Resume granularity |
|---|---|---|
| Pure-CNN training | `runs_dir/baseline_run/checkpoint.pt` | One epoch |
| Trust-map CNN training | `runs_dir/trust_map_run/checkpoint.pt` | One epoch |
| Pipeline feature extraction | `runs_dir/features-*.csv` | `save_every=100` images |
| Pivot-study stage progression | `runs_dir/report.partial.json` | One baseline |

`checkpoint.pt` contains the full optimizer + LR scheduler + AMP scaler state, so resume is **bit-identical** to a never-crashed run — not just "load the weights and start a fresh optimizer", which would lose Adam's momentum and the cosine schedule's position.

**Recommended way to run on the cluster (no babysitting):**

The repo ships [`scripts/continue.sh`](scripts/continue.sh) — an auto-restart wrapper that handles the loop, log redirection, and Ctrl-C semantics for you. Run any command through it; on a crash it sleeps for the cooldown and re-invokes the same command. Combined with `tmux` (keeps the process alive across SSH disconnects):

```bash
tmux new -s forge

# Inside the tmux session:
./scripts/continue.sh -- python scripts/pivot_study.py \
    --data-root /scratch/$USER/data/FaceForensics++ \
    --max-frames-per-video 30 --image-size 256 \
    --epochs 30 --batch-size 32 --device cuda \
    --runs-dir runs/pivot_full \
    --output runs/pivot_full/report.json

# Detach with Ctrl-b d. SSH out. Come back hours later:
tmux attach -t forge
```

`continue.sh` re-runs the same command every time it crashes. Because the pivot study is idempotent, the second invocation skips already-finished stages and resumes the in-flight one from its last checkpoint. The cooldown (60 s by default) gives the GPU time to recover from a transient driver hang or power blip. Useful flags:

- `--cooldown SECONDS`  — adjust the inter-restart sleep.
- `--max-restarts N`    — stop after N restarts so a permanently-broken command doesn't loop forever.
- `--tag NAME`          — log-file prefix; defaults to the command basename.
- `--log-dir PATH`      — where to write `<tag>-<timestamp>.log` and `restart-<tag>.log`.

Ctrl-C (or SIGTERM from a job scheduler) **stops the loop** rather than restarting; rc 130 / 143 are treated as deliberate interruptions, not crashes. So if you want to abort the run, two presses of Ctrl-C in tmux do it cleanly.

**Monitoring without attaching:**

```bash
# From any other shell on the same node:
tail -f logs/pivot_*.log

# Watch checkpoint timestamps:
watch -n 30 'ls -lt runs/pivot_full/*/last.pt runs/pivot_full/*.csv 2>/dev/null'

# Read the partial report after each completed baseline:
cat runs/pivot_full/report.partial.json | jq '.baselines'
```

**SLURM clusters:**

If your cluster runs SLURM, the same `until ... do ... done` pattern is unnecessary — submit the job with `--requeue` and SLURM relaunches the same script after preemption. Example sbatch:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=forge-pivot
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@60
#SBATCH --output=logs/pivot_%j.log

source $HOME/.cargo/env
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

python scripts/pivot_study.py \
    --data-root /scratch/$USER/data/FaceForensics++ \
    --max-frames-per-video 30 --image-size 256 \
    --epochs 30 --batch-size 32 --device cuda \
    --runs-dir runs/pivot_full \
    --output runs/pivot_full/report.json
```

Submit with `sbatch script.sh`. After preemption SLURM relaunches the script automatically and resume kicks in on its own.

### 6. Export and reuse the trained model

The trust-map weights live at `runs/pivot_full/trust_map/<timestamp>/best.pt`. Use them anywhere:

```bash
forge-detect detect new_image.jpg \
    --cnn-checkpoint runs/pivot_full/trust_map/<timestamp>/best.pt \
    --device cuda --visualize panel.png

forge-detect eval --data-root /any/other/dataset \
    --cnn-checkpoint runs/pivot_full/trust_map/<timestamp>/best.pt \
    --device cuda
```

The classifier pickle (`runs/pivot_full/features-*.csv` → `train_classifier()`) is similarly portable. Both files are self-contained — copy them off the cluster, ship them with your application, and the inference path on a CPU-only laptop works the same as on the cluster.

## Tests

```bash
make test          # cargo test --release && pytest
make test-rust     # 88 Rust unit tests
make test-python   # 17 Python integration tests
```

## Linting, formatting, and types

```bash
make fmt           # cargo fmt + ruff format
make fmt-check
make lint          # cargo clippy + ruff check
make typecheck     # mypy strict
make pre-commit    # all of the above through pre-commit hooks
```

## Compile the paper

```bash
make paper          # typst compile paper/main.typ paper/main.pdf
make paper-watch    # live-reload while editing
```

The compiled PDF is gitignored; run the command above to regenerate.

## Project layout

```
deepfake-detection-v2/
├── paper/                  # Typst diploma paper, one section per phase
├── src/                    # Rust math core (one module per phase)
│   ├── lib.rs              # crate root
│   ├── luminance.rs        # Phase 1
│   ├── scharr.rs           # Phase 2 — gradients
│   ├── tensor.rs           # Phase 2 — structural tensor + classifier
│   ├── stencils.rs         # Discrete Laplacian, biharmonic, divergence
│   ├── hyperplane.rs       # Phase 3 — namesake stage
│   ├── energy.rs           # Phase 4 — E(z) functional
│   ├── pde.rs              # Phase 5 — Jacobi solver with line search
│   ├── impact.rs           # Pipeline orchestrator
│   ├── py_bindings.rs      # PyO3 entry (gated behind python-extension)
│   └── util.rs             # mirror() helper
├── python/forge_detect/    # Python orchestrator
│   ├── config.py           # PipelineParams, PdeParams
│   ├── types.py            # SolveResult dataclass
│   ├── backends/           # CpuBackend (Rust), CudaBackend (stub)
│   ├── trust_map.py        # Heuristic chromatic-residual W_cnn
│   ├── cnn.py              # ChromaticEfficientNet scaffold
│   ├── features.py         # 24-D feature vector
│   ├── pipeline.py         # detect() entry point
│   ├── viz.py              # 6-panel diagnostic figure
│   └── cli.py              # forge-detect CLI
├── tests/                  # pytest suite
├── examples/               # demo scripts
├── Cargo.toml              # Rust crate manifest
├── pyproject.toml          # maturin + ruff + mypy + pytest config
├── Makefile                # `make help` for all targets
└── .pre-commit-config.yaml # rustfmt, clippy, ruff, ruff-format, mypy
```

## Status

| Component                                    | Status                              |
| -------------------------------------------- | ----------------------------------- |
| Rust math core (Phases 1-5 + impact)         | Complete, 88 unit tests             |
| CPU backend (PyO3 wrapper)                   | Complete                            |
| CUDA backend (PyTorch reimplementation)      | Complete, 6 backend-equivalence tests |
| Heuristic trust map                          | Complete                            |
| ChromaticEfficientNet (architecture + training) | Complete, AdamW + cosine LR + AMP |
| Dataset adapters (FF++, ImageFolder)         | Complete, 12 tests                  |
| Frame-extraction script (ffmpeg)             | Complete                            |
| Binary classifier (sklearn GBC)              | Complete, 8 tests                   |
| End-to-end evaluation (AUROC + features)     | Complete                            |
| 6-panel visualization                        | Complete                            |
| CLI: detect / train / eval / bench           | Complete, 5 tests                   |
| Typst paper (Sections 1-9)                   | Complete, ~25 pages                 |
| **Pretrained CNN weights**                   | **Not bundled — train on your dataset** |
| **Empirical AUROC on FF++ / Celeb-DF / DFDC** | **Pending — that is the diploma's empirical chapter** |

Total test count: **88 Rust + 55 Python = 143 tests**, all green.

## License

MIT. See [`LICENSE`](LICENSE).
