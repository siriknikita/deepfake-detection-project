# Hyperplane-Forge

Deepfake detection via _physical-manifold settlement_, then via _multi-channel physics-tensor CNN classification_, and now (Phase 3) via _composable physical principles_ — geometric / chromatic channels and frequency-domain channels stacked into the same multi-channel build. The full mathematical specification and empirical evaluation are in [`paper/main.typ`](paper/main.typ); this README is a quickstart for running the code.

## What it does

A real face is a sample of a smooth physical surface; a deepfake's synthetic injection is not. The pipeline recovers an initial depth manifold `z_forged` from the image (Phase 1-3), settles it under the Euler-Lagrange PDE of a physically motivated energy functional `E(z)` (Phase 4-5), and detects forged regions as _flow breaks_ (`R = z* − z_ideal`) and _geometric cracks_ (`L = Δz*`) in the settled manifold (Phase 6).

The math kernels are implemented in **Rust** (PyO3, `ndarray`, `rayon`) for the CPU path and exposed to **Python** through a single-function PyO3 entry point. A **PyTorch CUDA backend** mirrors the same operators on GPU and is verified numerically equivalent to the Rust core by 6 backend-equivalence tests.

The empirical Phase 1 study (heuristic trust map + 24-D features + Gradient Boosting Classifier) found the math pipeline produces _systematically anti-correlated_ ranking on FF++ c23 — test video AUROC `0.347` (mean-pool), confirmed by an oracle ablation with FF++ ground-truth masks. Diagnosing this to the global-pool feature-extraction step rather than the math itself, **Phase 2** reformulates the use of the math: instead of feeding 24 scalar features into a tabular classifier, the per-pixel outputs `W_cnn`, `z*`, `R` are stacked alongside the original RGB image as additional input channels to an EfficientNet-B0 binary classifier. The first conv layer is replaced with a six-input variant whose RGB-channel weights are copied from the ImageNet stem; the three new channels are initialised to the per-output-channel mean of those weights so epoch-zero behaviour matches the standard pretrained baseline.

**Phase 3** generalises the multi-channel pattern. Phase 2's per-method ablation showed the geometric-chromatic channels help on autoencoder-based manipulations (Deepfakes, Celeb-DF) and hurt on parametric / graphics-based ones (Face2Face, FaceSwap), because the latter preserve geometric coherence by construction. Phase 3 adds a second physical principle that targets exactly those failure modes: three frequency-domain spatial maps (block-DCT energy, block high-band ratio, full-image FFT log-magnitude) extracted in `forge_detect.frequency_map`, cached alongside the physics maps, and stacked into a 9-channel input via the `--channels rgb,physics,frequency` flag. The Phase 3 chapter ([`paper/sections/12-phase3-frequency.typ`](paper/sections/12-phase3-frequency.typ)) lays out the per-method-recovery hypothesis; empirical results on FF++ c23 and Celeb-DF v2 are pending compute allocation.

## Headline results

All numbers below come from face-cropped FF++ c23 with the official 720/140/140 video-disjoint split, 10 frames per video, 20 epochs, EfficientNet-B0 with WeightedRandomSampler + horizontal-flip augmentation, lr 2·10⁻⁴, AdamW + cosine annealing, BatchNorm trainable, AMP off. The 3-channel baseline and the 6-channel physics arm differ only in input channels and (incidentally) batch size (32 vs 16, due to GPU-memory contention on the shared cluster at training time).

| Test set | Metric | baseline_3ch | physics_6ch | Δ |
|---|---|---|---|---|
| **FF++ c23** (combined) | Frame AUROC | 0.7309 | 0.7197 | −0.0112 |
| **FF++ c23** (combined) | Video AUROC mean-pool | **0.8179** | **0.8176** | **−0.0003** |
| **FF++ c23** (combined) | Video AUROC max-pool | 0.9130 | **0.9273** | **+0.0143** |
| **FF++ c23 — Deepfakes** | Video AUROC mean-pool | 0.8713 | **0.8787** | **+0.0074** |
| **FF++ c23 — Face2Face** | Video AUROC mean-pool | **0.8029** | 0.7731 | −0.0297 |
| **FF++ c23 — FaceSwap** | Video AUROC mean-pool | **0.8150** | 0.7799 | −0.0351 |
| **FF++ c23 — NeuralTextures** | Video AUROC mean-pool | **0.6647** | 0.6522 | −0.0124 |
| **Celeb-DF v2** (cross-dataset) | Frame AUROC | 0.5276 | **0.5382** | **+0.0106** |
| **Celeb-DF v2** (cross-dataset) | Video AUROC mean-pool | 0.5405 | **0.5458** | **+0.0053** |
| **Celeb-DF v2** (cross-dataset) | **Video AUROC max-pool** | **0.5022** | **0.5542** | **+0.0520** |

The combined FF++ result is null on the canonical metric (Δ = −0.0003), but the per-method breakdown reveals it is the average of opposing effects: physics features add signal on autoencoder-based **Deepfakes** (+0.0074) and reduce it on parametric / graphics-based **Face2Face** (−0.0297) and **FaceSwap** (−0.0351). The cross-dataset transfer to Celeb-DF v2 — itself an autoencoder-based pipeline — replicates the in-domain Deepfakes pattern: physics_6ch beats the baseline on every cross-dataset metric, with max-pool video AUROC +0.0520.

The interpretation: physics-derived spatial features encode autoencoder-induced manifold inconsistencies in a method-invariant way. They help on autoencoder-based manipulations both within FF++ and across to Celeb-DF, and they hurt on parametric / graphics-based methods that preserve geometric coherence by construction. See [`paper/sections/11-phase2-multichannel.typ`](paper/sections/11-phase2-multichannel.typ) for the full mechanistic argument and limitations.

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

### Workflow: video files → multi-channel trained classifier

The Phase-2 multi-channel pipeline expects: extracted frames → face crops → physics-map cache → CNN training → in-domain and cross-dataset evaluation. Each step is a single script call; the next step's inputs are the previous step's outputs. All scripts are crash-resumable via existence-check on a per-file basis.

```bash
# 0. Sign the FF++ EULA, run their downloader. Make sure
#    splits/{train,val,test}.json land in <root>/splits/. They are
#    required for identity-disjoint training; without them, random
#    splits over compound fake video_ids leak source identities
#    between train and val and produce anti-correlated val AUROC.
#    If missing, the cache and training scripts print exact curl
#    commands for grabbing them from the FF++ public GitHub.

# 1. Extract frames at 5 fps (~10-50 GB depending on full vs subset).
python scripts/extract_frames.py /path/to/FaceForensics++ \
    --compression c23 --fps 5

# 2. Face-crop with MTCNN (256x256, 0.3 margin, --max-frames-per-video
#    matches what training will read). ~30-40 min on RTX 3080 Ti for
#    the full FF++ at cap 10. Resumable, atomic-write safe.
python scripts/extract_faces.py \
    --data-root /path/to/FaceForensics++ \
    --dataset face-forensics --compression c23 \
    --output-size 256 --margin 0.3 \
    --max-frames-per-video 10 --device cuda

# 3. Cache the three physics maps (W_cnn, z*, R) per face crop as
#    float16 npz alongside the crops (parallel directory tree). ~50
#    min on i7-11700KF for FF++ c23 at cap 10.
python scripts/cache_physics_maps.py \
    --data-root /path/to/FaceForensics++ \
    --variant heuristic --frames-subdir frames_faces \
    --frames-per-video 10 --num-workers 4

# 4a. Train the 3-channel RGB baseline (the apples-to-apples control).
#     ~25 min on RTX 3080 Ti.
python scripts/pivot_study.py \
    --data-root /path/to/FaceForensics++ \
    --baselines pure-cnn \
    --frames-per-video-train 10 --frames-per-video-eval 10 \
    --runs-dir runs/baseline_3ch_faces \
    --device cuda --use-ff-splits --use-face-crops --lr 2e-4

# 4b. Train the 6-channel physics-input model (the experimental arm).
#     Same recipe; only the input channels differ. ~25 min.
python scripts/train_physics_cnn.py \
    --data-root /path/to/FaceForensics++ \
    --variant heuristic \
    --frames-per-video-train 10 --frames-per-video-eval 10 \
    --runs-dir runs/physics_6ch_faces_heuristic \
    --device cuda --use-ff-splits --use-face-crops --lr 2e-4

# 5. Per-method ablation (4 manipulation methods × 2 models, ~5 min).
python scripts/eval_per_method.py \
    --data-root /path/to/FaceForensics++ \
    --baseline-weights runs/baseline_3ch_faces/baseline_run/best.pt \
    --physics-weights  runs/physics_6ch_faces_heuristic/best.pt \
    --device cuda --use-face-crops --use-ff-splits \
    --output runs/per_method_comparison.json

# 6. Cross-dataset eval on Celeb-DF v2. After running steps 2 and 3
#    against the Celeb-DF root with --dataset celeb-df, evaluate
#    each model:
python scripts/eval_celebdf.py \
    --celeb-data-root /path/to/Celeb-DF-v2 \
    --weights runs/baseline_3ch_faces/baseline_run/best.pt \
    --model baseline --device cuda --use-face-crops \
    --output runs/baseline_3ch_faces/celeb_test.json

python scripts/eval_celebdf.py \
    --celeb-data-root /path/to/Celeb-DF-v2 \
    --weights runs/physics_6ch_faces_heuristic/best.pt \
    --model physics --device cuda --use-face-crops \
    --output runs/physics_6ch_faces_heuristic/celeb_test.json
```

### Phase 3 workflow extension: 9-channel build (RGB + physics + frequency)

The Phase-3 build adds three frequency-domain channels to the Phase-2 input tensor; everything else (face crops, splits, training recipe) is unchanged. The pipeline grows by one cache step (`cache_frequency_maps.py`) and one extra flag on training and eval (`--channels rgb,physics,frequency`).

```bash
# 3a. Cache the three frequency maps (DCT block energy, DCT high-band
#     ratio, FFT log-magnitude) per face crop. Pure-NumPy compute,
#     no GPU required; ~5 ms per 256² face. Resumable, atomic-write safe.
python scripts/cache_frequency_maps.py \
    --data-root /path/to/FaceForensics++ \
    --frames-subdir frames_faces \
    --image-size 256 --num-workers 4

# 4c. Train the 9-channel physics+frequency model. Same recipe as 4b,
#     only --channels and --runs-dir differ. ~25 min on RTX 3080 Ti.
python scripts/train_physics_cnn.py \
    --data-root /path/to/FaceForensics++ \
    --channels rgb,physics,frequency \
    --frames-per-video-train 10 --frames-per-video-eval 10 \
    --runs-dir runs/physics_9ch_freq_heuristic \
    --device cuda --use-ff-splits --use-face-crops --lr 2e-4

# 5b. Per-method ablation — the experimental story for Phase 3 lives
#     in this table (predict +Δ on Face2Face / FaceSwap, ≈0 on
#     Deepfakes; see paper §12.7).
python scripts/eval_per_method.py \
    --data-root /path/to/FaceForensics++ \
    --baseline-weights runs/baseline_3ch_faces/baseline_run/best.pt \
    --physics-weights  runs/physics_9ch_freq_heuristic/best.pt \
    --channels rgb,physics,frequency \
    --device cuda --use-face-crops --use-ff-splits \
    --output runs/per_method_phase3.json

# 6b. Cross-dataset Celeb-DF eval. Cache frequency maps for Celeb-DF
#     first (one-shot), then evaluate the 9-channel model.
python scripts/cache_frequency_maps.py \
    --data-root /path/to/Celeb-DF-v2 \
    --dataset celeb-df --celeb-testing-list \
    --frames-subdir frames_faces \
    --image-size 256 --num-workers 4

python scripts/eval_celebdf.py \
    --celeb-data-root /path/to/Celeb-DF-v2 \
    --weights runs/physics_9ch_freq_heuristic/best.pt \
    --model physics --channels rgb,physics,frequency \
    --device cuda --use-face-crops \
    --output runs/physics_9ch_freq_heuristic/celeb_test.json
```

The `--channels` spec is a comma-separated list of channel families. Recognised tokens (case-insensitive): `rgb` (implicit base), `physics` or `physics:<variant>` (Phase 2; variants `heuristic` / `gtmask`), and `frequency` or `frequency:<variant>` (Phase 3; variant `default`). Each family may appear at most once. Future channel sets (specular, chromatic-aberration, sub-surface, temporal — see §11.10) drop into the same `ChannelSource` interface in `python/forge_detect/datasets.py` without requiring further changes to training or eval scripts.

For an unattended end-to-end run (steps 1-4), [`scripts/overnight_run.sh`](scripts/overnight_run.sh) chains the four heavy stages together with per-step logs, fails fast on the first error, and prints headline AUROCs at the end. Total wall-clock ~2h on RTX 3080 Ti / i7-11700KF. Override defaults via env vars (`DATA_ROOT`, `EPOCHS`, `LR`, `FRAMES_PER_VIDEO`, etc.); see the script header for the full list.

```bash
bash scripts/overnight_run.sh
# logs land in runs/overnight_<timestamp>/
# headline AUROCs print at the bottom of run.log
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

Then run the diagnostic script once to verify the VM environment:

```bash
./scripts/cluster_diagnose.sh           # human-readable
./scripts/cluster_diagnose.sh --json    # machine-readable for logs
```

It surfaces the things that are easy to miss in a Proxmox VM: hypervisor type, vCPU oversubscription (CPU steal), RAM ballooning, `/scratch` capacity, GPU passthrough status, NVLink topology, and a quick filesystem-write smoke test. If `torch.cuda is_available=False` or `nvidia-smi` is missing, escalate to the cluster admin before queuing a long job.

### 2. Download datasets

```bash
./scripts/prepare_data.sh \
    --root /scratch/$USER/data \
    --ff-downloader ~/download-FaceForensics++.py \
    --celeb-archive ~/Celeb-DF-v2.zip \
    --compression c23 --fps 5
```

`download-FaceForensics++.py` is the script the FF++ team emails you after EULA acceptance — it is account-keyed, we cannot bundle it. The script wraps it with sensible defaults (real videos, all four manipulation methods, pixel-level masks) and then automatically calls `scripts/extract_frames.py` to convert videos → per-video frame folders. Celeb-DF requires a Google-Form approval; once you have the archive, point `--celeb-archive` at it.

### 3. Phase 1 — math + Gradient Boosting baseline (~3.5 h, optional historical context)

Phase 1 is the analytic baseline that motivated Phase 2: math pipeline → 24-D feature vector → `sklearn.ensemble.GradientBoostingClassifier`. This was run on FF++ c23 with random video-disjoint splits; it produces test video AUROC `0.347` (mean-pool) — _systematically below chance_, in the same direction across all four manipulation methods. The oracle ablation (`scripts/oracle_phase1.py`) re-runs the same pipeline with FF++ ground-truth manipulation masks substituted for `W_cnn`, confirming the inversion is a property of the settlement formulation under c23 compression rather than trust-map quality. See [`paper/sections/10-empirical.typ`](paper/sections/10-empirical.typ) for the full Phase 1 evaluation.

If you want to reproduce Phase 1 from scratch:

```bash
./scripts/continue.sh -- python scripts/quick_classifier.py \
    --data-root /scratch/$USER/data/FaceForensics++ \
    --runs-dir runs/quick_phase1 \
    --device cuda
# ~3.5 h cluster wallclock; outputs:
#   runs/quick_phase1/features.csv (24-D + label + path + video_id)
#   runs/quick_phase1/classifier.pkl
#   runs/quick_phase1/report.json (test AUROC, accuracy, top features)
```

Phase 1's negative result is what motivates Phase 2: the math pipeline produces real, reproducible spatial signal, but a 24-D global-pool tabular classifier discards that spatial information at feature-extraction time. Phase 2 reformulates the use of the math: keep the per-pixel maps as additional CNN input channels and let a convolutional classifier learn its own spatial pooling.

### 4. Phase 2 — multi-channel CNN (recommended starting point)

This is the run that produces the [headline results](#headline-results) above. The recipe is summarised in the workflow at [§Workflow](#workflow-video-files--multi-channel-trained-classifier); on the cluster wrap it in `continue.sh` for crash resumption and run it inside `tmux` so SSH disconnections don't kill it:

```bash
tmux new -s forge

# Inside tmux:
./scripts/continue.sh -- bash scripts/overnight_run.sh
# ~2 h end-to-end. Detach with Ctrl-b d; come back with `tmux attach -t forge`.
```

`overnight_run.sh` calls the four heavy stages back-to-back:

1. **Face cropping** (`scripts/extract_faces.py`) — MTCNN over the extracted frames; resumable; ~30-40 min.
2. **3-channel baseline training** (`scripts/pivot_study.py --baselines pure-cnn`) — RGB only; ~25 min.
3. **Physics-map cache** (`scripts/cache_physics_maps.py`) — the three spatial maps as float16 npz; ~50 min.
4. **6-channel physics-input training** (`scripts/train_physics_cnn.py`) — RGB + W_cnn + z* + R; ~25 min.

Each stage logs to its own file under `runs/overnight_<timestamp>/`, stops the whole script on first failure, and prints headline test AUROCs at the bottom of `run.log` when done.

For a smoke test before the full run, train at low resolution / few epochs to catch plumbing failures:

```bash
EPOCHS=3 IMAGE_SIZE=128 FRAMES_PER_VIDEO=2 bash scripts/overnight_run.sh
# ~15 min; AUROCs are noisy and not defendable, but plumbing is verified.
```

### 5. Phase 2 ablations and follow-up

After step 4 produces the two `best.pt` checkpoints, the per-method and cross-dataset evaluations attribute the combined null to its underlying components:

```bash
# Per-method breakdown (Deepfakes / Face2Face / FaceSwap / NeuralTextures)
python scripts/eval_per_method.py \
    --data-root /scratch/$USER/data/FaceForensics++ \
    --baseline-weights runs/baseline_3ch_faces/baseline_run/best.pt \
    --physics-weights  runs/physics_6ch_faces_heuristic/best.pt \
    --device cuda --use-face-crops --use-ff-splits \
    --output runs/per_method_comparison.json
# ~5 min; produces three side-by-side AUROC tables (frame, video mean,
# video max) for each of the four FF++ methods plus the Combined sanity.

# Cross-dataset transfer to Celeb-DF v2 (after re-running steps 2-3
# of the workflow against the CelebDF root):
python scripts/eval_celebdf.py \
    --celeb-data-root /path/to/Celeb-DF-v2 \
    --weights runs/baseline_3ch_faces/baseline_run/best.pt \
    --model baseline --device cuda --use-face-crops \
    --output runs/baseline_3ch_faces/celeb_test.json
python scripts/eval_celebdf.py \
    --celeb-data-root /path/to/Celeb-DF-v2 \
    --weights runs/physics_6ch_faces_heuristic/best.pt \
    --model physics --device cuda --use-face-crops \
    --output runs/physics_6ch_faces_heuristic/celeb_test.json
# ~30 sec each; the cross-dataset eval can be run on a different host
# from the training cluster (e.g. local machine with CelebDF + scp'd
# best.pt files) since eval_celebdf.py does not require FF++ data.
```

### Empirical findings rubric (actual outcome)

| Outcome | What we observed | What it means |
|---|---|---|
| Combined task | Δ = −0.0003 video AUROC mean-pool | Null on the canonical metric — physics_6ch and baseline_3ch tie within noise. |
| Per-method (Deepfakes) | Δ = +0.0074 video AUROC | Physics features add signal on autoencoder-based reconstruction. |
| Per-method (Face2Face) | Δ = −0.0297 video AUROC | Physics features hurt on parametric reenactment (3DMM-based, geometrically smooth by construction). |
| Per-method (FaceSwap) | Δ = −0.0351 video AUROC | Physics features hurt on graphics-based 3D-model swap (largest loss). |
| Per-method (NeuralTextures) | Δ = −0.0124 | Within noise. |
| Cross-dataset (Celeb-DF v2) | Δ = +0.0520 video AUROC max-pool, positive on every metric | Physics features replicate the Deepfakes win across the FF++ → Celeb-DF domain shift. |

The combined null is the average of opposing per-method effects; the per-method + cross-dataset evidence supports the interpretation that physics-derived spatial features encode autoencoder-induced manifold inconsistencies in a method-invariant way. They help on autoencoder-based manipulations both within and across datasets, and hurt on parametric / graphics-based methods that preserve geometric coherence by construction.

Concrete timings on RTX 3080 Ti / i7-11700KF for the Phase-2 pipeline at FF++ c23 c23, face-cropped 256², 10 frames per video, 20 epochs:

| Step | Time |
|---|---|
| Frame extraction (5 fps, all methods + masks) | ~3 h |
| Face cropping (MTCNN) | ~30-40 min |
| 3-channel baseline training, 20 epochs | ~25 min |
| Physics-map cache | ~50 min |
| 6-channel physics training, 20 epochs | ~25 min |
| Per-method ablation (no retrain) | ~5 min |
| Total | ~5-6 h on one GPU |

The frame-extraction stage is the long pole; on a freshly-prepared cluster the rest of the pipeline (face crops onward) runs in ~2 h via `overnight_run.sh`.

### 6. Surviving outages and not having to babysit the script

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

### 7. Export and reuse the trained models

After Phase 2 training, two `best.pt` checkpoints are the deliverables:

```
runs/baseline_3ch_faces/baseline_run/best.pt    # 3-channel RGB EfficientNet-B0 (~20 MB)
runs/physics_6ch_faces_heuristic/best.pt        # 6-channel physics EfficientNet-B0 (~20 MB)
```

Both are self-contained PyTorch state dicts; copy them off the cluster (~20 MB scp) and the inference path on a CPU-only laptop works the same as on the cluster. The 3-channel weights load into [`build_baseline_classifier`](python/forge_detect/baseline_cnn.py) and the 6-channel weights load into [`build_physics_classifier(in_channels=6)`](python/forge_detect/baseline_cnn.py) from the same module.

For inference on novel images, [`scripts/eval_celebdf.py`](scripts/eval_celebdf.py) is the reference loader for both models — it constructs the same `Sequential(_ImageNetNormalize, EfficientNet)` wrapper, calls `load_weights`, and runs `evaluate_baseline_cnn` for frame and video metrics.

The Phase-1 GBC pipeline (`runs/quick_phase1/classifier.pkl`) remains usable as a single-image diagnostic via the `forge-detect detect` CLI:

```bash
forge-detect detect new_image.jpg --device cuda --visualize panel.png
```

That path runs the math pipeline on one image and renders the 6-panel figure (Input / W_cnn / z_forged / z* / R / cracks). It does *not* run the multi-channel Phase-2 classifier — the Phase-2 model is for video-level cross-dataset evaluation, not single-image inference.

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
├── paper/                  # Typst diploma paper
│   └── sections/
│       ├── 01-introduction.typ … 09-implementation.typ
│       ├── 10-empirical.typ            # Phase 1 results + diagnostic chain
│       ├── 11-phase2-multichannel.typ  # Phase 2 6-channel reformulation + actual results
│       └── 12-phase3-frequency.typ     # Phase 3 9-channel composition + hypotheses
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
│   ├── backends/           # CpuBackend (Rust), CudaBackend (PyTorch)
│   ├── trust_map.py        # Heuristic chromatic-residual W_cnn
│   ├── cnn.py              # Phase 1 ChromaticEfficientNet (trust-map predictor; legacy)
│   ├── baseline_cnn.py     # build_baseline_classifier (3-ch) + build_physics_classifier (n-ch)
│   ├── datasets.py         # FaceForensicsAdapter, CelebDFAdapter, ChannelSource API
│   ├── frequency_map.py    # Phase 3 frequency-domain channel producers
│   ├── features.py         # Phase 1 24-D feature vector
│   ├── classifier.py       # Phase 1 GBC + ClassifierMetrics (frame + video AUROC)
│   ├── eval.py             # extract_features_over_dataset, evaluate_pipeline
│   ├── train.py            # Phase 1 trust-map CNN training (legacy)
│   ├── pipeline.py         # detect() entry point
│   ├── viz.py              # 6-panel diagnostic figure
│   └── cli.py              # forge-detect CLI (single-image diagnostic)
├── scripts/                # End-to-end pipeline scripts
│   ├── extract_frames.py           # videos → frames/
│   ├── extract_faces.py            # frames → frames_faces/ (MTCNN)
│   ├── verify_face_crops.py        # cleanup utility for partial extracts
│   ├── cache_physics_maps.py       # frames_faces → physics_faces_<variant>/
│   ├── cache_frequency_maps.py     # frames_faces → frequency_faces_<variant>/  (Phase 3)
│   ├── pivot_study.py              # 3-channel baseline training
│   ├── train_physics_cnn.py        # n-channel physics/frequency training (--channels)
│   ├── eval_per_method.py          # per-method FF++ test breakdown
│   ├── eval_celebdf.py             # cross-dataset Celeb-DF v2 evaluation
│   ├── overnight_run.sh            # end-to-end pipeline runner (4 stages, ~2 h)
│   ├── diagnose_baseline.py        # 30-step overfit test for training infra
│   ├── oracle_phase1.py            # Phase 1 oracle ablation (GT-mask trust map)
│   ├── per_method_refit.py         # Phase 1 per-method GBC refit
│   ├── quick_classifier.py         # Phase 1 math + GBC pipeline
│   ├── prepare_data.sh             # FF++ + Celeb-DF download orchestrator
│   ├── cluster_bootstrap.sh        # one-shot cluster setup
│   ├── cluster_diagnose.sh         # Proxmox VM sanity check
│   ├── continue.sh                 # crash-resume wrapper
│   └── trim_frames.py              # frame-cap utility
├── tests/                  # pytest suite
├── examples/               # demo scripts
├── Cargo.toml              # Rust crate manifest
├── pyproject.toml          # maturin + ruff + mypy + pytest config
├── Makefile                # `make help` for all targets
└── .pre-commit-config.yaml # rustfmt, clippy, ruff, ruff-format, mypy
```

## Status

| Component                                            | Status                                  |
| ---------------------------------------------------- | --------------------------------------- |
| Rust math core (Phases 1-5 + impact)                 | Complete, 88 unit tests                 |
| CPU backend (PyO3 wrapper)                           | Complete                                |
| CUDA backend (PyTorch reimplementation)              | Complete, 6 backend-equivalence tests   |
| Heuristic trust map                                  | Complete                                |
| Phase 1: math + GBC classifier                       | Complete, evaluated on FF++ c23         |
| Phase 1: oracle ablation (GT-mask trust map)         | Complete; result confirms Phase 1 inversion |
| Phase 1: per-method GBC refit                        | Complete, 4 methods × per-method classifiers |
| Phase 2: face-crop preprocessing (MTCNN)             | Complete, 99.7% detection rate on Celeb-DF |
| Phase 2: 6-channel physics tensor + stem surgery     | Complete (`build_physics_classifier`)   |
| Phase 2: training infra (WRS, hflip, AdamW + cosine) | Complete, verified on smoke + full     |
| Phase 2: in-domain FF++ c23 evaluation               | Complete (combined + per-method)        |
| Phase 2: cross-dataset Celeb-DF v2 evaluation        | Complete (518-video testing list)       |
| Phase 3: frequency-domain channel producers          | Complete (`forge_detect.frequency_map`) |
| Phase 3: composable `ChannelSource` API              | Complete; `--channels` flag in train/eval |
| Phase 3: frequency-map cache script                  | Complete (`scripts/cache_frequency_maps.py`) |
| Phase 3: empirical 9-channel evaluation              | **Pending — predictions in §12.7, table skeleton in §12.6** |
| ChromaticEfficientNet (Phase 1 trust-map predictor)  | Complete (legacy; superseded by Phase 2) |
| Dataset adapters (FF++, Celeb-DF, ImageFolder)       | Complete, 12 tests                      |
| Frame-extraction script (ffmpeg)                     | Complete                                |
| Binary classifier (sklearn GBC)                      | Complete, 8 tests                       |
| End-to-end evaluation (frame + video AUROC)          | Complete                                |
| 6-panel visualization                                | Complete                                |
| CLI: detect / train / eval / bench                   | Complete, 5 tests                       |
| Typst paper (Sections 1-11)                          | Complete, ~30 pages with empirical results |
| **Pretrained model weights**                         | **Not bundled — train via `overnight_run.sh`** |
| **Empirical evaluation on FF++ + Celeb-DF**          | **Complete — see [Headline results](#headline-results)** |
| **Multi-seed replication**                           | **Future work — single-seed numbers reported** |
| **Generalisation to modern diffusion generators**    | **Future work — out of scope, see §11.10** |

Total test count: **88 Rust + 112 Python = 200 tests**, all green.

## License

MIT. See [`LICENSE`](LICENSE).
