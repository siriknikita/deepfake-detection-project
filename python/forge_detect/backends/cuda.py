"""CUDA backend — pure-PyTorch reimplementation of the Rust math core.

This backend is functionally equivalent to :class:`forge_detect.backends.cpu.CpuBackend`
but executes on a CUDA device through PyTorch. It is the path used for
training and large-scale evaluation runs on the NVIDIA GPU cluster.

Implementation note: the bulk of the work mirrors the Rust core's per-phase
operators using ``torch.nn.functional`` convolutions for Scharr, Laplacian,
and biharmonic; tensor-native min/max for the Hyperplane Forge composition;
and a vectorized Jacobi loop. The PyTorch ops dispatch automatically to
CUDA kernels when tensors live on a CUDA device.

The CUDA backend is currently a *stub*: it raises ``NotImplementedError``.
Wiring up the full PyTorch path is tracked as future work — the intended
shape of the implementation is described in the implementation chapter of
the paper.
"""

from __future__ import annotations

import numpy as np

from forge_detect.config import PipelineParams
from forge_detect.types import SolveResult


class CudaBackend:
    """PyTorch-backed pipeline runner targeting CUDA devices."""

    name: str = "cuda"

    def __init__(self) -> None:
        # We probe for torch lazily so the package remains importable on
        # hosts without a PyTorch install.
        try:
            import torch  # noqa: F401  (probe only)
        except ImportError as e:
            msg = "CudaBackend requires the optional PyTorch dependency"
            raise ImportError(msg) from e

    def solve(
        self,
        rgb: np.ndarray,
        w_cnn: np.ndarray,
        params: PipelineParams,
    ) -> SolveResult:
        del rgb, w_cnn, params  # unused — see module docstring
        msg = (
            "CudaBackend is not yet implemented. Use the CPU backend "
            "(make_backend('cpu')) until the PyTorch reimplementation lands."
        )
        raise NotImplementedError(msg)
