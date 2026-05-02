"""Phase 6 — feature vector for the binary classifier.

Builds the 24-dimensional feature vector described in
``paper/sections/07-phase6-impact.typ`` from a :class:`SolveResult`. The
features cover:

- residual statistics (4): mean, std, p95, p99 of ``R = z* − z_ideal``;
- absolute-Laplacian statistics (4): same on ``|L|``;
- Laplacian morphology (2): edge density above the p95 threshold,
  spectral entropy of |L|;
- energy decomposition (4): total + three terms;
- energy ratios (2): smoothness/total, consistency/total;
- convergence features (3): iterations to convergence, energy decrease
  in the first 10 % of iterations, energy slope at the trace tail;
- reserved (5): zero-padded, held for downstream extensions.

Total length is 24, named in :data:`FEATURE_NAMES`.
"""

from __future__ import annotations

import numpy as np

from forge_detect.types import SolveResult

FEATURE_NAMES: tuple[str, ...] = (
    "R_mean",
    "R_std",
    "R_p95",
    "R_p99",
    "absL_mean",
    "absL_std",
    "absL_p95",
    "absL_p99",
    "L_edge_density",
    "L_spectral_entropy",
    "E_total",
    "E_data",
    "E_smoothness",
    "E_consistency",
    "E_smoothness_ratio",
    "E_consistency_ratio",
    "iterations",
    "E_decrease_first_decile",
    "E_tail_slope",
    "_reserved_0",
    "_reserved_1",
    "_reserved_2",
    "_reserved_3",
    "_reserved_4",
)


def extract_features(result: SolveResult) -> np.ndarray:
    """Build the fixed-length feature vector from a :class:`SolveResult`."""
    r = result.residual.astype(np.float64, copy=False)
    abs_l = np.abs(result.laplacian.astype(np.float64, copy=False))
    e_total = float(result.energy_total) or 1e-12

    feats: list[float] = [
        float(r.mean()),
        float(r.std()),
        float(np.percentile(r, 95)),
        float(np.percentile(r, 99)),
        float(abs_l.mean()),
        float(abs_l.std()),
        float(np.percentile(abs_l, 95)),
        float(np.percentile(abs_l, 99)),
        _edge_density(abs_l),
        _spectral_entropy(abs_l),
        float(result.energy_total),
        float(result.energy_data),
        float(result.energy_smoothness),
        float(result.energy_consistency),
        float(result.energy_smoothness) / e_total,
        float(result.energy_consistency) / e_total,
        float(result.iterations),
        _energy_decrease_first_decile(result.energy_trace),
        _energy_tail_slope(result.energy_trace),
        # 5 reserved slots, zero-padded for forward compatibility.
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert len(feats) == len(FEATURE_NAMES)
    return np.asarray(feats, dtype=np.float64)


def _edge_density(abs_l: np.ndarray) -> float:
    """Fraction of pixels with |L| above its own p95 — the "crack density"."""
    threshold = float(np.percentile(abs_l, 95))
    if threshold == 0.0:
        return 0.0
    return float((abs_l > threshold).mean())


def _spectral_entropy(abs_l: np.ndarray) -> float:
    """Shannon entropy of the (normalized) 2D power spectrum of |L|.

    A real face's |L| concentrates power in a narrow set of low frequencies
    (smooth surface); a deepfake's |L| has heavier high-frequency content
    and therefore higher spectral entropy.
    """
    spec = np.abs(np.fft.fft2(abs_l - abs_l.mean())) ** 2
    s = spec.sum()
    if s <= 0:
        return 0.0
    p = (spec / s).ravel()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _energy_decrease_first_decile(trace: np.ndarray) -> float:
    """``(E[0] − E[k]) / E[0]`` where ``k = ceil(0.1 · len(trace))``.

    Non-zero only when the trace has at least two samples. Real faces drop
    energy faster in the first 10 % of iterations than deepfakes do.
    """
    if trace.size < 2:
        return 0.0
    k = max(1, int(np.ceil(0.1 * trace.size)))
    e0 = float(trace[0])
    if e0 == 0.0:
        return 0.0
    return float((e0 - trace[min(k, trace.size - 1)]) / e0)


def _energy_tail_slope(trace: np.ndarray) -> float:
    """Slope of the last 25 % of the energy trace (least-squares fit).

    Plateaued (real-face) traces have a small magnitude slope; deepfake
    traces often have steeper residual slopes because the consistency term
    cannot be driven to zero.
    """
    if trace.size < 4:
        return 0.0
    k = max(2, int(np.ceil(0.25 * trace.size)))
    tail = trace[-k:].astype(np.float64)
    xs = np.arange(tail.size, dtype=np.float64)
    a, _ = np.polyfit(xs, tail, 1)
    return float(a)


__all__ = ["FEATURE_NAMES", "extract_features"]
