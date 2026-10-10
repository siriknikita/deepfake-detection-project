# MacBook runs: corrected solver × backbone matrix (2026-10-10)

Branch `feat/exact-energy-gradient-mobilenet`. All numbers in this file come from
`runs_mac/mac_fixedsolver_matrix.json` and the per-run `report.json` / `analysis/` files.
Hardware: Apple M4 Pro (14-core CPU, 20-core GPU via MPS), 24 GB. Single seed per run;
differences below ~0.02 AUROC should be read as noise.

## Question and design

Two things changed relative to the diploma's Phase-3 model (`runs_cluster/physics_9ch_freq`,
EfficientNet-B0, 9 channels): the settlement solver now descends along the exact gradient of
the discrete energy (commit `bbe225d`; the previous operator carried the consistency term with
the wrong sign and stopped at a non-stationary `z*`), and a MobileNet backbone was added. A
2×2-style matrix separates the two factors:

| run | backbone | solver | channels | role |
|---|---|---|---|---|
| reference | EfficientNet-B0 | old (sign defect) | RGB + physics + frequency | diploma Phase-3 model (cluster, CUDA) |
| A | EfficientNet-B0 | corrected | RGB + physics + frequency | solver effect at fixed backbone |
| B | MobileNet-V3-Large | corrected | RGB + physics + frequency | backbone effect at fixed solver |
| C | MobileNet-V3-Large | — | RGB only | channel effect at fixed backbone |

Common protocol: FF++ c23 face crops (MTCNN, 256 px, margin 0.3; `scripts/stream_datasets.py`
streamed the archives from Drive and kept only crops), 10 frames per video, official
identity-disjoint splits (36 000 / 7 000 / 7 000 frames), 20 epochs, batch 16, AdamW lr 2e-4,
cosine schedule, balanced sampling, horizontal flip, gradient clip 1.0, no AMP. Physics maps
cached once with the Rust CPU core (`n_scales 3`, `max_iter 200`, heuristic trust, uint8
storage); frequency maps computed on the fly. Cross-dataset: Celeb-DF v2 published testing
list (518 videos × 10 frames), never used for training or selection.

Three prespecified checkpoint rules per run, all evaluated, none chosen after seeing test
numbers: `best_loss` (min validation loss), `best_auc` (max validation AUROC), `last`.
Run B predates the `best_auc` rule in the trainer; its AUROC-best epoch (8, val 0.675) was not
preserved and no later epoch exceeded it, so run B has no `best_auc` entry.

## Headline results

FF++ test = frame AUROC / video AUROC (mean-pool) / video AUROC (max-pool). Celeb-DF = same
three on the 518-video list.

| run | checkpoint | FF++ test | Celeb-DF |
|---|---|---|---|
| reference | last | 0.707 / 0.797 / 0.926 | 0.561 / 0.610 / 0.563 |
| **A** EffNet, corrected | best_loss (ep 5) | 0.701 / 0.763 / 0.917 | 0.594 / 0.629 / 0.575 |
| **A** | best_auc (ep 10) | **0.736 / 0.832** / 0.917 | 0.597 / 0.638 / 0.587 |
| **A** | last | 0.721 / 0.808 / 0.900 | 0.582 / 0.630 / 0.592 |
| **B** MobileNet, corrected | best_loss (ep 4) | 0.647 / 0.711 / 0.845 | 0.475 / 0.473 / 0.437 |
| **B** | last | 0.674 / 0.758 / 0.710 | 0.535 / 0.574 / 0.523 |
| **C** MobileNet, RGB | best_loss (ep 9) | 0.703 / 0.778 / 0.892 | 0.602 / 0.647 / 0.616 |
| **C** | best_auc (ep 12) | 0.719 / 0.757 / **0.928** | 0.613 / 0.666 / 0.625 |
| **C** | last | 0.722 / 0.782 / 0.913 | **0.622 / 0.674 / 0.660** |

### Per manipulation method (FF++ test, frame AUROC / video mean-pool AUROC)

| run, checkpoint | Deepfakes | Face2Face | FaceSwap | NeuralTextures | combined |
|---|---|---|---|---|---|
| reference, last | 0.770 / 0.858 | 0.716 / 0.828 | 0.719 / 0.815 | 0.606 / 0.669 | 0.703 / 0.785 |
| A, last | 0.795 / 0.876 | 0.721 / 0.815 | 0.736 / 0.818 | 0.632 / 0.691 | 0.721 / 0.808 |
| A, best_auc | 0.827 / 0.885 | 0.739 / 0.818 | 0.740 / 0.812 | 0.641 / 0.693 | 0.736 / 0.832 |
| B, last | 0.758 / 0.854 | 0.649 / 0.748 | 0.699 / 0.798 | 0.592 / 0.646 | 0.674 / 0.758 |
| C, last | 0.803 / 0.876 | 0.730 / 0.802 | 0.737 / 0.799 | 0.617 / 0.656 | 0.722 / 0.782 |
| C, best_auc | 0.805 / 0.860 | 0.725 / 0.774 | 0.734 / 0.789 | 0.612 / 0.646 | 0.719 / 0.757 |

## Reading

1. **Solver fix (A vs reference, same backbone and recipe).** Neutral to positive: +0.014 frame
   / +0.011 video-mean in-domain, +0.021 / +0.020 on Celeb-DF, and ahead on all four
   manipulation methods (largest gains on the autoencoder-based Deepfakes and
   NeuralTextures). The consistent sign across methods and checkpoints makes a regression
   very unlikely; the magnitude is at the edge of single-seed noise.
2. **Backbone (B vs A).** MobileNet-V3-Large with the 9-channel input is 0.05–0.07 behind
   EfficientNet-B0 in-domain and ~0.05 behind cross-dataset. The deficit concentrates on
   Face2Face (−0.07 frame vs the reference), the graphics-based reenactment on which the
   geometric channels were already weakest in the diploma.
3. **Channels at fixed backbone (C vs B).** For MobileNet the physics/frequency channels
   *hurt*: RGB-only is better by 0.05 in-domain and 0.10 on Celeb-DF. The channel benefit the
   diploma reported for EfficientNet (neutral in-domain, positive cross-dataset) does not
   transfer to this backbone under this recipe (stem surgery with mean-initialised extra
   channels, same learning rate).
4. **Best cross-dataset model in the matrix is RGB-only MobileNet** (Celeb-DF video-mean 0.674
   vs 0.638 for A's best checkpoint), while trailing A on in-domain video-mean. This is the
   result that most needs a second seed before being leaned on.
5. **Checkpoint rule.** Loss-best picked under-trained epochs (4, 5, 9) and was the worst rule
   for every run; AUROC-best or last was consistently better. Max-pool video AUROC is fragile
   (run B: 0.845 vs 0.710 between its two checkpoints).

## Training dynamics (same shape in all three runs)

| | A | B | C |
|---|---|---|---|
| s / epoch | 395 | 222 | 254 |
| val AUROC peak (epoch) | 0.732 (10) | 0.675 (8) | 0.716 (12) |
| val AUROC at epoch 20 | 0.723 | 0.652 | 0.714 |
| val loss epoch 5 → 20 | 0.55 → 0.97 | 0.62 → 1.86 | 0.76 → 1.16 |
| train AUROC at 20 | 0.976 | 0.985 | 0.976 |
| raw (pre-clip) grad norm, epoch 1 → 20 | 3.0 → 25.7 | 2.5 → 66.9 (offline probe) | 1.4 → 33.6 |

Validation AUROC plateaus by epoch 8–12 in every run while training AUROC continues to 0.98
and validation loss climbs: the recipe memorises frames after the plateau. The reference run
shows the same curve (val 0.71 plateau from epoch 6, val loss 1.06 at epoch 20). Raw gradient
norms grow monotonically and become heavy-tailed per batch (maxima > 100 by epoch 15) without
any non-finite values or loss spikes; with AdamW and the clip this is a symptom of
memorisation, not an optimisation instability. The 9-channel models start with ~2× the
gradient norm of the RGB-only model and 99% of steps clipped from epoch 1 (57% for RGB-only),
pointing at the mean-initialised stem channels as a candidate for a smaller learning rate.

**Calibration vs ranking (run B, validation).** Between epochs 4 and 20 the mean validation
logit went from +1.1 to +9.5, the confidently-wrong fraction from 0.6% to 19.7%, and 92% of the
final validation loss came from confidently-wrong frames, while AUROC moved only 0.629 → 0.656.
The bias is one-sided (reals classified correctly 31–34%, fakes 81–84% at threshold 0): the
rising validation loss is a confidence problem on top of a mediocre ranking, not ranking decay.
Per-method validation ordering (Deepfakes > FaceSwap > Face2Face > NeuralTextures) matches the
test ordering of every other run.

## Caveats

- One seed per run; FF++ frame-level AUROC varies by ~±0.01–0.02 across seeds.
- Reference trained on CUDA with batch 16 (physics) / 32 (baseline); Mac runs used batch 16
  throughout on MPS. Float rounding differs; the Rust solver is not bit-deterministic across
  threads (≤3e-4 in `z*`).
- Physics maps were stored as uint8 codes (loader dequantises; < 0.01 input change after the
  per-image normalisation) instead of the cluster's float16.
- Run B's AUROC-best weights were not preserved (trainer predates the rule).
- Celeb-DF list = 518 videos; its intervals are wide (a few hundred videos per class).

## Files

- `runs_mac/mac_fixedsolver_matrix.json` — every number above plus per-epoch curves.
- `runs_mac/physics_9ch_effb0_fixedsolver/`, `runs_mac/physics_9ch_mnv3l_fixedsolver/`,
  `runs_mac/baseline_3ch_mnv3l/baseline_run/` — `report.json`, `history.json`, `config.json`,
  `best_auc.json`, checkpoints (`best.pt`, `best_auc.pt`, `last.pt`), `analysis/` (per-method,
  Celeb-DF, validation logits).
- Data preparation: `scripts/stream_datasets.py` (crops from Drive archives),
  `scripts/cache_physics_maps.py --store-dtype uint8`, `--channels rgb,physics,frequency:compute`.
