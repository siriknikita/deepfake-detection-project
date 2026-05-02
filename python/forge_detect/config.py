"""Pipeline configuration dataclasses.

Mirrors the Rust ``PipelineParams`` / ``PdeParams`` so a single Python config
object drives both backends. The defaults match the values the paper
recommends (see paper/sections/05-phase4-energy.typ).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PdeParams:
    """Parameters of the Jacobi PDE solver (Phase 5)."""

    lambda_: float = 1.0
    alpha: float = 0.5
    beta: float = 5.0
    max_iter: int = 500
    tol: float = 1.0e-5
    log_every: int = 10


@dataclass
class PipelineParams:
    """End-to-end pipeline parameters covering Phases 1, 2, 3, 5, and 6."""

    # Phase 1: chromatic luminance + DoG pyramid.
    w_r: float = 0.299
    w_g: float = 0.587
    w_b: float = 0.114
    sigma_base: float = 1.0
    k_ratio: float = math.sqrt(2.0)
    n_scales: int = 4

    # Phase 2: structural tensor + classifier.
    tensor_window_radius: int = 2
    p_flat: float = 0.30
    p_edge: float = 0.70

    # Phase 3: hyperplane forge.
    neigh_radius: int = 6
    albedo_window_radius: int = 2
    epsilon: float = 1.0e-3
    c_z: float = 1.0

    # Phase 6: impact map reference manifold smoothing.
    sigma_ref: float = 4.0

    # Phase 5: PDE solver (nested).
    pde: PdeParams = field(default_factory=PdeParams)

    def as_core_kwargs(self) -> dict[str, float | int]:
        """Flatten to keyword arguments for :func:`forge_detect._core.solve_and_extract`."""
        return {
            "w_r": self.w_r,
            "w_g": self.w_g,
            "w_b": self.w_b,
            "sigma_base": self.sigma_base,
            "k_ratio": self.k_ratio,
            "n_scales": self.n_scales,
            "tensor_window_radius": self.tensor_window_radius,
            "p_flat": self.p_flat,
            "p_edge": self.p_edge,
            "neigh_radius": self.neigh_radius,
            "albedo_window_radius": self.albedo_window_radius,
            "epsilon": self.epsilon,
            "c_z": self.c_z,
            "sigma_ref": self.sigma_ref,
            "pde_lambda": self.pde.lambda_,
            "pde_alpha": self.pde.alpha,
            "pde_beta": self.pde.beta,
            "pde_max_iter": self.pde.max_iter,
            "pde_tol": self.pde.tol,
            "pde_log_every": self.pde.log_every,
        }
