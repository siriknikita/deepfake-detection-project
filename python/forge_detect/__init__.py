"""Hyperplane-Forge: deepfake detection via physical-manifold settlement.

Public API exports the high-level entry points; the Rust math core lives at
``forge_detect._core`` (built via ``maturin develop``) and is normally
accessed through the :class:`forge_detect.backends.cpu.CpuBackend` wrapper.
"""

from __future__ import annotations

__version__ = "0.1.0"

from forge_detect.config import PdeParams, PipelineParams
from forge_detect.types import SolveResult

__all__ = [
    "PdeParams",
    "PipelineParams",
    "SolveResult",
    "__version__",
]
