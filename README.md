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
