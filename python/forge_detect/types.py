"""Shared dataclasses used by both backends and consumers of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SolveResult:
    """Output of :meth:`forge_detect.backends.Backend.solve`.

    The same schema is produced by both the CPU (Rust) and CUDA (PyTorch)
    backends so callers can swap between them transparently.

    Attributes:
        z_forged: Phase 3 initial manifold, shape ``(H, W)`` float32.
        z_star: Phase 5 settled manifold, shape ``(H, W)`` float32.
        residual: ``z* − z_ideal``, the "flow break" map, shape ``(H, W)``.
        laplacian: ``Δz*``, the "geometric cracks" map, shape ``(H, W)``.
        energy_total: ``E(z*)`` at the minimum.
        energy_data: ``λ · Σ (z* − z_forged)²``.
        energy_smoothness: ``α · Σ (Δz*)²``.
        energy_consistency: ``β · Σ ‖W ⊙ (∇I − K∇z*)‖²``.
        energy_trace: 1D array of ``E(z^(n))`` totals sampled by the solver.
        iterations: number of Jacobi iterations performed.
        converged: whether the relative-L² stopping criterion fired.
    """

    z_forged: np.ndarray
    z_star: np.ndarray
    residual: np.ndarray
    laplacian: np.ndarray
    energy_total: float
    energy_data: float
    energy_smoothness: float
    energy_consistency: float
    energy_trace: np.ndarray
    iterations: int
    converged: bool

    @classmethod
    def from_core_dict(cls, d: dict[str, object]) -> SolveResult:
        """Construct a :class:`SolveResult` from the dict returned by the Rust core."""
        return cls(
            z_forged=np.asarray(d["z_forged"], dtype=np.float32),
            z_star=np.asarray(d["z_star"], dtype=np.float32),
            residual=np.asarray(d["R"], dtype=np.float32),
            laplacian=np.asarray(d["L"], dtype=np.float32),
            energy_total=float(d["energy_total"]),  # type: ignore[arg-type]
            energy_data=float(d["energy_data"]),  # type: ignore[arg-type]
            energy_smoothness=float(d["energy_smoothness"]),  # type: ignore[arg-type]
            energy_consistency=float(d["energy_consistency"]),  # type: ignore[arg-type]
            energy_trace=np.asarray(d["energy_trace"], dtype=np.float32),
            iterations=int(d["iterations"]),  # type: ignore[arg-type]
            converged=bool(d["converged"]),
        )
