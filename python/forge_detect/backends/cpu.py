"""CPU backend — thin wrapper around the Rust extension at ``forge_detect._core``."""

from __future__ import annotations

import numpy as np

from forge_detect.config import PipelineParams
from forge_detect.types import SolveResult


class CpuBackend:
    """Pipeline backend that runs the Rust math core on the host CPU.

    The Rust core exposes a single function ``solve_and_extract`` that takes
    the RGB image and trust map plus the flattened pipeline parameters as
    keyword arguments, and returns a Python ``dict`` matching the
    :class:`SolveResult` schema. This wrapper only handles the array
    contiguity / dtype contract, the kwarg flattening, and the dict-to-
    dataclass conversion.
    """

    name: str = "cpu"

    def solve(
        self,
        rgb: np.ndarray,
        w_cnn: np.ndarray,
        params: PipelineParams,
    ) -> SolveResult:
        rgb_f32 = np.ascontiguousarray(rgb, dtype=np.float32)
        w_cnn_f32 = np.ascontiguousarray(w_cnn, dtype=np.float32)
        if rgb_f32.ndim != 3 or rgb_f32.shape[2] != 3:
            msg = f"rgb must be (H, W, 3), got {rgb_f32.shape}"
            raise ValueError(msg)
        if w_cnn_f32.shape != rgb_f32.shape[:2]:
            msg = f"w_cnn shape {w_cnn_f32.shape} does not match image H × W {rgb_f32.shape[:2]}"
            raise ValueError(msg)

        # Imported lazily so the rest of the package is usable on hosts where
        # the Rust extension has not been built yet (e.g. when only running
        # type-checks or unit tests that mock the backend).
        from forge_detect import _core

        out = _core.solve_and_extract(rgb_f32, w_cnn_f32, **params.as_core_kwargs())
        return SolveResult.from_core_dict(out)
