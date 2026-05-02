"""Backend protocol and factory for swapping CPU (Rust) and CUDA (PyTorch) execution."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from forge_detect.config import PipelineParams
from forge_detect.types import SolveResult


class Backend(Protocol):
    """Common interface for the CPU and CUDA pipeline backends.

    Both implementations run the full Phase 1 → Phase 5 chain and return a
    :class:`SolveResult` with the identical schema. ``z_forged`` is computed
    *internally* — callers do not (and cannot) pre-compute it.
    """

    name: str

    def solve(
        self,
        rgb: np.ndarray,
        w_cnn: np.ndarray,
        params: PipelineParams,
    ) -> SolveResult:
        """Run the pipeline.

        Args:
            rgb: ``(H, W, 3)`` float32 image in ``[0, 1]``.
            w_cnn: ``(H, W)`` float32 trust map in ``[0, 1]``.
            params: pipeline configuration.

        Returns:
            A :class:`SolveResult` with the impact map and energy decomposition.
        """
        ...


def make_backend(device: str) -> Backend:
    """Construct the backend matching the requested device.

    Args:
        device: Either ``"cpu"`` (Rust core) or ``"cuda"`` (PyTorch on GPU).

    Raises:
        ValueError: On any other ``device`` string.
    """
    if device == "cpu":
        from forge_detect.backends.cpu import CpuBackend

        return CpuBackend()
    if device == "cuda":
        from forge_detect.backends.cuda import CudaBackend

        return CudaBackend()
    msg = f"unknown device {device!r}; expected 'cpu' or 'cuda'"
    raise ValueError(msg)


__all__ = ["Backend", "make_backend"]
