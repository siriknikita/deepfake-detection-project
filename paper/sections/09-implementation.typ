= Implementation

This chapter — the only one in the paper that discusses code or
deployment — describes how the algorithm of Sections 2 through 8 is
realized as a working system. It covers the Rust math core, the Python
orchestrator, the PyO3 bridge between them, the build and tooling
choices, and the test strategy that keeps the implementation honest
with respect to the mathematical specification.

== Architecture

The system is a hybrid Rust/Python project laid out as a single
maturin-style cdylib + rlib crate at the repository root with a
sibling Python package under `python/forge_detect/`:

#table(
  columns: 2,
  align: (left, left),
  stroke: 0.5pt,
  table.header([*Path*], [*Role*]),
  [`src/`],
    [Rust math core. One module per algorithm phase plus a
     thin PyO3 wrapper.],
  [`python/forge_detect/`],
    [Python orchestrator: configuration, backend dispatch, pipeline,
     CNN scaffold, visualization, CLI.],
  [`tests/`],
    [Pytest suite for the Python layer; Rust unit tests live inside
     each module's `#[cfg(test)] mod tests` block.],
  [`paper/`],
    [The Typst sources of this document.],
  [`Cargo.toml`, `pyproject.toml`, `.cargo/config.toml`, `Makefile`],
    [Build configuration, dependency pinning, developer commands.],
)

The high-level data flow is straight-line: a caller (the CLI, a
notebook, or a training loop) hands the image and a trust map to a
`Backend`; the backend runs Phases 1 → 5 of the paper and returns a
`SolveResult` with the impact map and the energy decomposition; the
orchestrator turns that into a feature vector and (optionally) into a
deepfake probability. The :class:`Backend` protocol is satisfied by
two concrete implementations: a CPU backend that calls the Rust
extension, and a CUDA backend (currently a stub) that mirrors the
operators in PyTorch.

== Rust math core

The eight algorithm modules under `src/` correspond one-to-one with
the paper's phases. Each module is small, self-contained, and accompanied
by a `#[cfg(test)]` block of unit tests.

#table(
  columns: 2,
  align: (left, left),
  stroke: 0.5pt,
  table.header([*Module*], [*Responsibility*]),
  [`luminance.rs`],
    [Phase 1 — `weighted_luminance`, `gaussian_blur` (separable), `dog_band`,
     `dog_pyramid`. 16 unit tests.],
  [`scharr.rs`],
    [Phase 2 — `scharr_gradients`, `gradient_magnitude`. 11 unit tests.],
  [`tensor.rs`],
    [Phase 2 — `structural_tensor`, `eigenvalues_2x2`, `classify`,
     `adaptive_thresholds`, `keypoints`. 14 unit tests.],
  [`stencils.rs`],
    [`laplacian`, `biharmonic` (iterated 5-point), `divergence`. 12 tests.],
  [`hyperplane.rs`],
    [Phase 3 — `LocalHyperplane`, `local_albedo` (median), `build_hyperplanes`
     (Lambertian transfer), `forge_per_scale` (Min step), `compose_max_over_scales`,
     `fill_uncovered`. 16 tests.],
  [`energy.rs`],
    [Phase 4 — `EnergyTerms`, `energy(...)`. 8 tests.],
  [`pde.rs`],
    [Phase 5 — `PdeParams`, `SolveResult`, `jacobi_solve`. 7 tests.],
  [`impact.rs`],
    [Pipeline orchestrator (`run_pipeline`, `PipelineParams`,
     `ImpactResult`). 3 integration tests.],
)

A small private `util.rs` module hosts the `mirror(i, n)` helper that
implements Neumann index reflection used uniformly by every operator
that touches a pixel grid boundary. The PyO3 entry point lives in
`py_bindings.rs` and is gated behind a `python-extension` feature flag
so `cargo test` does not link against `libpython` (a precondition for
test binaries to load on macOS without a Python interpreter).

=== Parallelism and allocation

Every kernel that operates on an `H × W` field follows the same
template:

1. Allocate a contiguous owned `Array2<f32>` for the output.
2. Borrow it as a row-stride slice via `as_slice_mut().expect("contiguous")`.
3. Iterate `par_chunks_mut(W).enumerate()` so each chunk is one row,
   the closure runs on a `rayon` worker, and there is no inter-thread
   synchronization on the inner loop.

This pattern is used in `weighted_luminance`, both Gaussian
convolution passes, the Scharr inner loop, gradient-magnitude, the
structural-tensor box average, the eigenvalue closed-form, the 5-point
Laplacian, the divergence, the median-window albedo, and the energy
sums. Allocations are concentrated at module entry (no scratch
buffers persist between calls) and the math runs over `f32` end-to-end
because `f64` doubles bandwidth for no perceptual gain on 8-bit image
inputs.

The Hyperplane Forge's Min-over-neighborhood is the one operator that
does *not* fit the row-parallel template: each keypoint stamps a
neighborhood that may overlap with other keypoints' stamps, so a naive
parallel reduction would race on the shared output buffer. The
implementation iterates keypoints serially and updates the output
in-place; performance is acceptable because the keypoint count is
sparse (a few thousand on a 1024² image with default thresholds) and
this loop is dominated by the per-keypoint $(2r+1)^2$ stamp anyway.

=== Step-size selection in the Jacobi solver

The Jacobi update `z^(n+1) = z^n − τ R(z^n)` is stable when
`τ < 2/L` where `L` is the spectral radius of the discrete linear
operator `λ I + α Δ² − β div(W² K² ∇·)`. We bound `L` analytically by
the diagonal sum

$ D = lambda + 20 alpha + 16 beta dot.c max(W^2 K^2), $

with `20 α` the analytic biharmonic diagonal (5-point center weight is
$-4$, squared is $16$, plus four $+1$ neighbor contributions from
$Delta(Delta z)$) and `16 β · max(W²K²)` an empirically generous bound
on the Scharr-then-central-difference divergence's spectral radius.

The starting step is `τ = 0.5 / D`. On stiff inputs at large `β` the
analytic bound is sometimes still optimistic — the Scharr gradient
followed by the central-difference divergence couples slightly more
than the immediate 5-point neighborhood, and small grids (≤ 16²) admit
boundary modes the bound does not see. To guarantee monotone descent
on `E(z)` regardless of the bound's tightness, every step runs a
*backtracking line search*: try `τ`, halve up to six times if the
energy increased, commit the smallest tried step. This adds at most a
small constant factor of overhead per iteration on degenerate inputs
and is invisible on well-conditioned ones.

=== PyO3 bridge

The Rust core is exposed to Python through a single function,
`solve_and_extract`, defined in `src/py_bindings.rs` and gated behind
the `python-extension` Cargo feature.

Inputs use the `numpy` crate's zero-copy view types
(`PyReadonlyArray3<f32>` for the RGB image, `PyReadonlyArray2<f32>`
for the trust map). Outputs are allocated by the Rust core, then
converted to `PyArray2<f32>` / `PyArray1<f32>` via `to_pyarray_bound`
(this is a copy, but only on the way out, after the math is done).
The function takes 21 keyword arguments matching `PipelineParams` /
`PdeParams`, with the same defaults as the Rust-side `Default` impls;
the GIL is released through `py.allow_threads` while the Rust pipeline
runs, so other Python threads can make progress and `rayon`
parallelism is unaffected.

The extension links against the stable Python ABI (`abi3-py310`) so a
single compiled artifact runs against any CPython 3.10+ — including
Python 3.14, which the project's host machine uses (PyO3 0.22 does not
otherwise advertise support for that version).

== Python orchestrator

The Python package mirrors the Rust core's module boundary at a
higher abstraction level.

#table(
  columns: 2,
  align: (left, left),
  stroke: 0.5pt,
  table.header([*Module*], [*Responsibility*]),
  [`config.py`],
    [`PdeParams`, `PipelineParams` dataclasses with the same defaults as
     the Rust side; `as_core_kwargs()` flattens them into the keyword
     arguments the PyO3 entry point accepts.],
  [`types.py`],
    [`SolveResult` dataclass + `from_core_dict` that converts the dict
     the Rust extension returns into a typed object.],
  [`backends/cpu.py`],
    [`CpuBackend` — thin wrapper around `forge_detect._core.solve_and_extract`.
     Validates input shapes and dtypes; lazy-imports the extension so
     ruff / mypy can run on hosts without a build.],
  [`backends/cuda.py`],
    [`CudaBackend` — stub raising `NotImplementedError`. The shape of
     the PyTorch reimplementation is described above; the actual
     implementation is future work.],
  [`trust_map.py`],
    [`heuristic_trust_map(rgb)` — chromatic-residual fallback used
     when no trained CNN is available. Pure NumPy, no SciPy.],
  [`cnn.py`],
    [`ChromaticEfficientNet` — torchvision EfficientNet-B0 + chromatic
     6-channel adapter + UNet decoder, produced by `.build()`.
     Importable without PyTorch present; `__call__` requires it.],
  [`features.py`],
    [`extract_features(SolveResult)` — produces the 24-dimensional
     feature vector tabulated in Phase 6, including spectral entropy
     and convergence features.],
  [`pipeline.py`],
    [`detect(image, *, device, params, classifier, trust_map)` — the
     end-to-end entry point. Loads the image, computes the trust map,
     routes to the chosen backend, extracts features, and (when a
     classifier is supplied) returns a deepfake probability.],
  [`viz.py`],
    [`crack_overlay`, `panel`, `save_panel` — matplotlib figures for
     the diploma defense.],
  [`cli.py`],
    [argparse-based `forge-detect detect <image>` with `--device`,
     `--n-scales`, `--max-iter`, `--visualize`, `--print-features`.],
)

The Backend protocol is the formal seam: a `Backend` is anything with
a `name: str` attribute and a `solve(rgb, w_cnn, params)` method
returning a `SolveResult`. Because the protocol is what callers depend
on, swapping the Rust core for the PyTorch reimplementation is a
matter of changing `device="cpu"` to `device="cuda"` at the call site.

== Build, packaging, and tooling

#table(
  columns: 2,
  align: (left, left),
  stroke: 0.5pt,
  table.header([*Concern*], [*Tool*]),
  [Rust dependencies & profile], [`Cargo.toml`, `.cargo/config.toml`],
  [Python dependencies, ABI, CLI script], [`pyproject.toml` (PEP 621)],
  [Rust ↔ Python build], [`maturin` (PyO3-aware build backend)],
  [Python virtualenv & dep installer], [`uv` (Astral)],
  [Rust formatter], [`rustfmt` (config in `rustfmt.toml`)],
  [Rust linter], [`clippy` (`pedantic` + `nursery` with documented
                  numerical-code allowances)],
  [Python formatter & linter], [`ruff` (`format` and `check`)],
  [Python type checker], [`mypy` (strict)],
  [Pre-commit hooks], [`.pre-commit-config.yaml`],
  [Developer commands], [`Makefile` (one-liner targets)],
  [Paper], [`typst` (`make paper` and `make paper-watch`)],
)

`maturin develop --release` is the canonical build command — it
compiles the Rust extension with full optimizations, links it against
the stable Python ABI, and installs it editably into the current
virtualenv. The `Makefile` wraps both steps:

```
make dev-install   # uv sync + maturin develop --release
make test          # cargo test --release && pytest
make fmt-check
make lint
make typecheck
```

== Verification path

The implementation is verified at three layers:

+ *Rust unit tests* — 88 tests across the 9 modules, executed by
  `cargo test --release`. Each module's tests exercise its public API
  on inputs whose ground-truth outputs are derivable from the
  algorithm specification: $Delta(x^2) = 2$, the structural tensor of a
  pure-x gradient is $"diag"(1, 0)$, the energy of a constant field
  with zero gradients is zero, the Min-Max composition recovers a
  known plane on coplanar keypoints, and so on.

+ *Python integration tests* — 17 tests under `tests/`, executed by
  `pytest`. These run against the *installed* extension, so they
  validate the PyO3 contract end-to-end: a constant input produces
  a near-constant `z*`, the feature vector has the documented length
  and names, the pipeline accepts paths and in-memory arrays, and
  the heuristic trust map satisfies its range / shape invariants.

+ *End-to-end smoke* — `forge-detect detect <image>` runs the full
  pipeline on a real image and prints the impact-map summary. This
  is the user-facing acceptance test; if it produces a finite
  `z*`, a feature vector with no NaNs, and a sensibly-shaped energy
  decomposition, the system is healthy.

A future addition, listed in the project's open work, is a
*backend-equivalence test* that asserts `(z_forged, z*, R, L, energy)`
agree to ≤ 1e-4 between `CpuBackend` and `CudaBackend` on a fixed
seed. Until the PyTorch backend lands, this test is replaced by the
mathematical-identity tests in the Rust unit suite.

#pagebreak()
