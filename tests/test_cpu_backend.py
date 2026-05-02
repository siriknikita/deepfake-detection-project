"""Smoke tests for the CPU backend (the Rust extension wrapper).

These run against the installed extension at ``forge_detect._core``; the
test session must therefore have been preceded by ``maturin develop``
(usually done by ``uv pip install -e .[dev]``).
"""

from __future__ import annotations

import numpy as np
import pytest

from forge_detect.backends.cpu import CpuBackend
from forge_detect.config import PdeParams, PipelineParams
from forge_detect.types import SolveResult


def _rgb_disk(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    cy, cx = h / 2.0, w / 2.0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    bright = (r2 < (min(h, w) / 3.0) ** 2).astype(np.float32) * 0.6 + 0.2
    return np.stack([bright, bright, bright], axis=-1)


def _params(**overrides: object) -> PipelineParams:
    p = PipelineParams(
        n_scales=2,
        pde=PdeParams(max_iter=30, log_every=5),
    )
    for k, v in overrides.items():
        setattr(p, k, v)
    return p


def test_cpu_backend_runs_end_to_end() -> None:
    rgb = _rgb_disk((24, 24))
    w_cnn = np.full((24, 24), 1.0, dtype=np.float32)
    backend = CpuBackend()
    result = backend.solve(rgb, w_cnn, _params())
    assert isinstance(result, SolveResult)
    assert result.z_forged.shape == (24, 24)
    assert result.z_star.shape == (24, 24)
    assert result.residual.shape == (24, 24)
    assert result.laplacian.shape == (24, 24)
    assert np.isfinite(result.z_star).all()
    assert np.isfinite(result.residual).all()
    assert np.isfinite(result.laplacian).all()


def test_cpu_backend_energy_components_non_negative() -> None:
    rgb = _rgb_disk((24, 24))
    w_cnn = np.full((24, 24), 1.0, dtype=np.float32)
    result = CpuBackend().solve(rgb, w_cnn, _params())
    # Each term is a sum of squares with a non-negative weight ⇒ ≥ 0.
    assert result.energy_data >= 0.0
    assert result.energy_smoothness >= 0.0
    assert result.energy_consistency >= 0.0
    assert result.energy_total >= 0.0


def test_cpu_backend_energy_trace_recorded() -> None:
    rgb = _rgb_disk((16, 16))
    w_cnn = np.full((16, 16), 1.0, dtype=np.float32)
    # log_every = 1 ensures the trace records every iteration so even fast
    # convergence (≤ 5 steps) leaves observable samples.
    p = _params(pde=PdeParams(max_iter=200, log_every=1))
    result = CpuBackend().solve(rgb, w_cnn, p)
    # The trace always contains at least the initial energy.
    assert len(result.energy_trace) >= 1
    # Final energy must not exceed initial — the canonical convergence
    # indicator that the line-search-damped Jacobi guarantees by
    # construction.
    initial = result.energy_trace[0]
    final = result.energy_trace[-1]
    assert final <= initial * 1.01, f"final energy {final} must be <= initial {initial}"


def test_cpu_backend_rejects_wrong_shape() -> None:
    rgb_bad = np.zeros((10, 10, 4), dtype=np.float32)
    w_cnn = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="rgb must be"):
        CpuBackend().solve(rgb_bad, w_cnn, _params())


def test_cpu_backend_rejects_mismatched_w_cnn() -> None:
    rgb = _rgb_disk((12, 12))
    w_cnn = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        CpuBackend().solve(rgb, w_cnn, _params())


def test_cpu_backend_constant_image_yields_uniform_z_star() -> None:
    rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
    w_cnn = np.full((16, 16), 1.0, dtype=np.float32)
    result = CpuBackend().solve(rgb, w_cnn, _params())
    # No gradients ⇒ z* is essentially constant (data term anchors it).
    assert result.z_star.std() < 1e-2


def test_cuda_backend_runs_on_torch_cpu() -> None:
    """CudaBackend in CPU-fallback mode produces a valid SolveResult."""
    pytest.importorskip("torch")
    from forge_detect.backends.cuda import CudaBackend

    rgb = _rgb_disk((16, 16))
    w_cnn = np.full((16, 16), 1.0, dtype=np.float32)
    result = CudaBackend(device="cpu").solve(rgb, w_cnn, _params())
    assert result.z_star.shape == (16, 16)
    assert np.isfinite(result.z_star).all()
