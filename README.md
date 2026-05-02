# Hyperplane-Forge

Deepfake detection via *physical-manifold settlement*. The full mathematical
specification is in [`paper/main.typ`](paper/main.typ); this README is a
quickstart for running the code.

## What it does

A real face is a sample of a smooth physical surface; a deepfake's
synthetic injection is not. The pipeline recovers an initial depth manifold
`z_forged` from the image (Phase 1-3), settles it under the Euler-Lagrange
PDE of a physically motivated energy functional `E(z)` (Phase 4-5), and
detects forged regions as *flow breaks* (`R = z* − z_ideal`) and *geometric
cracks* (`L = Δz*`) in the settled manifold (Phase 6).

The math kernels are implemented in **Rust** (PyO3, `ndarray`, `rayon`) for
the CPU path and exposed to **Python** through a single-function PyO3 entry
point. A CUDA path that mirrors the operators in PyTorch is scaffolded for
GPU-cluster runs (current status: stub).

## Requirements

* Rust toolchain (`cargo` ≥ 1.75) — the project is built with `rustc` 1.91
  but any recent stable toolchain should work.
* Python ≥ 3.10. The dev host runs 3.14 against the stable ABI.
* [`uv`](https://github.com/astral-sh/uv) for Python venv & dependency
  management.
* `typst` for compiling the paper (optional).

## Build & run

```bash
# One-shot dev install: creates .venv, installs deps, builds the
# Rust extension into the venv with full optimizations.
make dev-install

# End-to-end on a single image:
uv run --python .venv/bin/python forge-detect detect path/to/image.jpg

# Save the diagnostic 6-panel figure:
uv run --python .venv/bin/python forge-detect detect path/to/image.jpg \
    --visualize panel.png

# Print every feature:
uv run --python .venv/bin/python forge-detect detect path/to/image.jpg \
    --print-features
```

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

## Status & future work

| Component | Status |
|---|---|
| Rust math core (Phases 1-5 + impact) | Complete, 88 unit tests |
| CPU backend (PyO3 wrapper) | Complete, 17 integration tests |
| Heuristic trust map | Complete |
| End-to-end pipeline + CLI | Complete |
| 6-panel visualization | Complete |
| ChromaticEfficientNet (architecture) | Scaffolded |
| ChromaticEfficientNet (training) | Future work |
| CUDA backend (PyTorch reimplementation) | Stub |
| Backend-equivalence tests (CPU = CUDA) | Pending CUDA backend |
| Typst paper (Sections 1-9) | Complete, ~25 pages |

## License

MIT. See [`LICENSE`](LICENSE).
