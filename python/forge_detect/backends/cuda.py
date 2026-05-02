"""CUDA backend — PyTorch reimplementation of the Rust math core.

Functionally equivalent to :class:`forge_detect.backends.cpu.CpuBackend`
but executes on a configurable PyTorch device (CUDA when available,
falling back to MPS or CPU). This is the path used for training and
large-scale evaluation runs on the NVIDIA GPU cluster.

Numerical equivalence with the Rust path is exercised by
``tests/test_backend_equivalence.py``: `(z_forged, z_star, R, L,
energies)` must agree to ≤ 1e-3 relative error on a fixed-seed input.

Implementation strategy:

- **Convolutions** (Scharr, Laplacian, biharmonic, divergence): a
  single fixed kernel is built once and applied with
  :func:`torch.nn.functional.conv2d`. Boundaries use
  ``F.pad(..., mode='reflect')`` to match the Rust ``mirror`` helper.
- **Hyperplane Forge Min step** (the only non-grid operator): we stack
  every keypoint's hyperplane evaluation over its
  ``(2r+1)²`` neighborhood into one big tensor and reduce with
  :func:`torch.Tensor.scatter_reduce_` ``'amin'`` into a flat ``H·W``
  buffer. This is the same algorithm as the CPU path but parallel
  across keypoints.
- **Jacobi solver**: vectorized fixed-point iteration with the same
  backtracking line search as the Rust implementation, ported one-to-
  one. The whole loop runs inside ``torch.no_grad()`` because we are
  not training the solver — only the trust-map CNN takes gradients.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from forge_detect.config import PdeParams, PipelineParams
from forge_detect.types import SolveResult

if TYPE_CHECKING:
    pass


def _select_device(prefer: str = "auto") -> str:
    """Resolve the requested torch device string."""
    import torch

    if prefer == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return prefer


# --------------- internal building blocks ----------------


def _conv2d(x: Any, weight: Any) -> Any:
    """2D convolution with reflect-padding, no bias, stride 1."""
    import torch.nn.functional as F

    pad = weight.shape[-1] // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_padded, weight)


def _scharr_kernels(device: str, dtype: Any) -> tuple[Any, Any]:
    import torch

    kx = (
        torch.tensor(
            [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
            device=device,
            dtype=dtype,
        )
        / 32.0
    )
    ky = (
        torch.tensor(
            [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
            device=device,
            dtype=dtype,
        )
        / 32.0
    )
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def _laplacian_kernel(device: str, dtype: Any) -> Any:
    import torch

    return torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)


def _gaussian_kernel_1d(sigma: float, device: str, dtype: Any) -> Any:
    import torch

    radius = int(math.ceil(3.0 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def _gaussian_blur(x: Any, sigma: float) -> Any:
    """Separable 2D Gaussian via two 1D conv2d passes with reflect padding."""
    import torch.nn.functional as F

    if sigma <= 0:
        return x
    k = _gaussian_kernel_1d(sigma, str(x.device), x.dtype)
    pad = (k.shape[0] - 1) // 2
    kh = k.view(1, 1, 1, -1)
    kv = k.view(1, 1, -1, 1)
    x_pad_h = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    horiz = F.conv2d(x_pad_h, kh)
    x_pad_v = F.pad(horiz, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(x_pad_v, kv)


def _scharr_gradients(x: Any) -> tuple[Any, Any]:
    kx, ky = _scharr_kernels(str(x.device), x.dtype)
    return _conv2d(x, kx), _conv2d(x, ky)


def _laplacian(x: Any) -> Any:
    return _conv2d(x, _laplacian_kernel(str(x.device), x.dtype))


def _box_average(x: Any, radius: int) -> Any:
    import torch.nn.functional as F

    if radius == 0:
        return x
    k = 2 * radius + 1
    x_pad = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    return F.avg_pool2d(x_pad, kernel_size=k, stride=1)


def _local_median(x: Any, radius: int) -> Any:
    """Sliding-window median via unfold + kthvalue.

    Memory cost is ``O(H · W · (2r+1)²)`` — fine for the small
    ``r ∈ [1, 3]`` we use in practice.
    """
    import torch.nn.functional as F

    if radius == 0:
        return x
    k = 2 * radius + 1
    x_pad = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)  # (N, 1, H, W, k, k)
    patches = patches.reshape(*patches.shape[:4], -1)
    return patches.median(dim=-1).values


def _eigenvalues_2x2(jxx: Any, jxy: Any, jyy: Any) -> tuple[Any, Any]:
    import torch

    half_t = 0.5 * (jxx + jyy)
    half_diff = 0.5 * (jxx - jyy)
    disc = (half_diff * half_diff + jxy * jxy).clamp(min=0.0)
    s = torch.sqrt(disc)
    l1 = half_t + s
    l2 = (half_t - s).clamp(min=0.0)
    return l1, l2


def _classify(l1: Any, l2: Any, tau_flat: float, tau_edge: float) -> Any:
    """Per-pixel ``0=flat / 1=edge / 2=corner`` codes.

    Same rule as ``crate::tensor::classify``:
    flat if l1 < tau_flat, corner if l2 >= tau_edge, edge otherwise.
    """
    import torch

    flat = l1 < tau_flat
    corner = l2 >= tau_edge
    out = torch.ones_like(l1, dtype=torch.uint8)  # default: edge
    out = out.masked_fill(flat, 0)
    return out.masked_fill(corner, 2)


def _adaptive_thresholds(l1: Any, l2: Any, p_flat: float, p_edge: float) -> tuple[float, float]:
    import torch

    return (
        float(torch.quantile(l1.flatten(), p_flat)),
        float(torch.quantile(l2.flatten(), p_edge)),
    )


def _build_hyperplane_params(
    i_w: Any,
    ix_k: Any,
    iy_k: Any,
    rho: Any,
    yx: Any,  # (N, 2) keypoint coordinates
    epsilon: float,
    c_z: float,
) -> Any:
    """Stack hyperplane parameters at every keypoint.

    Returns an ``(N, 5)`` tensor whose columns are
    ``(yi, xi, zi, dz_dx, dz_dy)``.
    """
    import torch

    yi = yx[:, 0]
    xi = yx[:, 1]
    rho_kp = rho[yi, xi]
    ix_kp = ix_k[yi, xi]
    iy_kp = iy_k[yi, xi]
    iw_kp = i_w[yi, xi]
    denom = rho_kp + epsilon
    dzdx = -ix_kp / denom
    dzdy = -iy_kp / denom
    zi = c_z * iw_kp
    return torch.stack([yi.to(zi.dtype), xi.to(zi.dtype), zi, dzdx, dzdy], dim=1)


def _forge_per_scale(
    shape: tuple[int, int],
    hp_params: Any,  # (N, 5)
    neigh_radius: int,
) -> Any:
    """Stamp every hyperplane onto its (2r+1)² neighborhood and reduce by min."""
    import torch

    h, w = shape
    device = hp_params.device
    dtype = hp_params.dtype
    if hp_params.shape[0] == 0:
        return torch.full((h, w), float("inf"), device=device, dtype=dtype)

    yi = hp_params[:, 0]
    xi = hp_params[:, 1]
    zi = hp_params[:, 2]
    dzdx = hp_params[:, 3]
    dzdy = hp_params[:, 4]

    r = neigh_radius
    dy = torch.arange(-r, r + 1, device=device, dtype=dtype)
    dx = torch.arange(-r, r + 1, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(dy, dx, indexing="ij")  # (k, k)

    # Hyperplane evaluations at every (keypoint, dy, dx).
    values = (
        zi[:, None, None] + dzdx[:, None, None] * grid_x[None] + dzdy[:, None, None] * grid_y[None]
    )

    # Pixel coordinates clamped to the grid bounds.
    yi_long = yi.long()[:, None, None]
    xi_long = xi.long()[:, None, None]
    y_idx = (yi_long + grid_y.long()).clamp(0, h - 1)
    x_idx = (xi_long + grid_x.long()).clamp(0, w - 1)
    flat_idx = y_idx * w + x_idx

    out_flat = torch.full((h * w,), float("inf"), device=device, dtype=dtype)
    out_flat.scatter_reduce_(0, flat_idx.flatten(), values.flatten(), reduce="amin")
    return out_flat.view(h, w)


def _compose_max_over_scales(per_scale: list[Any]) -> Any:
    import torch

    return torch.stack(per_scale, dim=0).amax(dim=0)


def _fill_uncovered(z: Any, fallback: Any) -> Any:
    import torch

    return torch.where(torch.isfinite(z), z, fallback)


def _energy_terms(
    z: Any,
    z_forged: Any,
    ix: Any,
    iy: Any,
    w_cnn: Any,
    k_alb: Any,
    pde: PdeParams,
) -> dict[str, float]:

    data_sum = ((z - z_forged) ** 2).sum().item()
    lap = _laplacian(z.unsqueeze(0).unsqueeze(0)).squeeze()
    smooth_sum = (lap**2).sum().item()
    zx, zy = _scharr_gradients(z.unsqueeze(0).unsqueeze(0))
    rx = w_cnn * (ix - k_alb * zx.squeeze())
    ry = w_cnn * (iy - k_alb * zy.squeeze())
    cons_sum = ((rx**2) + (ry**2)).sum().item()
    e_data = pde.lambda_ * data_sum
    e_smooth = pde.alpha * smooth_sum
    e_cons = pde.beta * cons_sum
    return {
        "data": float(e_data),
        "smoothness": float(e_smooth),
        "consistency": float(e_cons),
        "total": float(e_data + e_smooth + e_cons),
    }


def _jacobi_solve(
    z_forged: Any,
    ix: Any,
    iy: Any,
    w_cnn: Any,
    k_alb: Any,
    pde: PdeParams,
) -> tuple[Any, list[float], int, bool]:
    """Vectorized Jacobi iteration with backtracking line search."""
    import torch

    z = z_forged.clone()
    max_wk2 = float((w_cnn * k_alb).pow(2).max().item())
    diag = pde.lambda_ + 20.0 * pde.alpha + 16.0 * pde.beta * max_wk2
    tau = 0.5 / diag if diag > 0 else 0.5
    backoff_tries = 6

    energy_trace: list[float] = []
    if pde.log_every > 0:
        energy_trace.append(_energy_terms(z, z_forged, ix, iy, w_cnn, k_alb, pde)["total"])
    e_prev = energy_trace[-1] if energy_trace else float("inf")

    iterations = 0
    converged = False
    for n in range(1, pde.max_iter + 1):
        z_prev_norm = float(torch.linalg.vector_norm(z).item()) or 1e-12

        lap = _laplacian(z.unsqueeze(0).unsqueeze(0)).squeeze()
        bilap = _laplacian(lap.unsqueeze(0).unsqueeze(0)).squeeze()
        zx, zy = _scharr_gradients(z.unsqueeze(0).unsqueeze(0))
        zx = zx.squeeze()
        zy = zy.squeeze()
        fx = w_cnn * w_cnn * k_alb * (ix - k_alb * zx)
        fy = w_cnn * w_cnn * k_alb * (iy - k_alb * zy)
        # Central-difference divergence.
        import torch.nn.functional as F

        kxd = torch.tensor(
            [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], device=z.device, dtype=z.dtype
        ).view(1, 1, 3, 3)
        kyd = torch.tensor(
            [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], device=z.device, dtype=z.dtype
        ).view(1, 1, 3, 3)
        fx_pad = F.pad(fx.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
        fy_pad = F.pad(fy.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
        div_f = (F.conv2d(fx_pad, kxd) + F.conv2d(fy_pad, kyd)).squeeze()

        residual = pde.lambda_ * (z - z_forged) + pde.alpha * bilap - pde.beta * div_f

        # Line search.
        accepted_z = z
        accepted_e = e_prev
        delta_norm_sq = 0.0
        cur_tau = tau
        for _ in range(backoff_tries):
            z_trial = z - cur_tau * residual
            e_trial = _energy_terms(z_trial, z_forged, ix, iy, w_cnn, k_alb, pde)["total"]
            delta_norm_sq = float((z_trial - z).pow(2).sum().item())
            if e_trial <= e_prev * (1.0 + 1.0e-6):
                accepted_z = z_trial
                accepted_e = e_trial
                break
            cur_tau *= 0.5
            accepted_z = z_trial
            accepted_e = e_trial
        z = accepted_z
        e_prev = accepted_e
        tau = cur_tau
        iterations = n

        delta_norm = math.sqrt(delta_norm_sq)
        if delta_norm / z_prev_norm < pde.tol:
            converged = True

        if pde.log_every > 0 and n % pde.log_every == 0:
            energy_trace.append(
                _energy_terms(z, z_forged, ix, iy, w_cnn, k_alb, pde)["total"],
            )
        if converged:
            break

    return z, energy_trace, iterations, converged


# --------------- public backend ----------------


class CudaBackend:
    """PyTorch-backed pipeline runner targeting CUDA / MPS / CPU.

    Args:
        device: ``"auto"`` (default — selects CUDA, then MPS, then CPU),
            ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """

    name: str = "cuda"

    def __init__(self, device: str = "auto") -> None:
        try:
            import torch  # noqa: F401  (probe)
        except ImportError as e:
            msg = "CudaBackend requires the optional PyTorch dependency"
            raise ImportError(msg) from e
        self.device = _select_device(device)

    def solve(
        self,
        rgb: np.ndarray,
        w_cnn: np.ndarray,
        params: PipelineParams,
    ) -> SolveResult:
        import torch

        rgb_np = np.ascontiguousarray(rgb, dtype=np.float32)
        w_cnn_np = np.ascontiguousarray(w_cnn, dtype=np.float32)
        if rgb_np.ndim != 3 or rgb_np.shape[2] != 3:
            msg = f"rgb must be (H, W, 3), got {rgb_np.shape}"
            raise ValueError(msg)
        if w_cnn_np.shape != rgb_np.shape[:2]:
            msg = f"w_cnn shape {w_cnn_np.shape} does not match image {rgb_np.shape[:2]}"
            raise ValueError(msg)

        with torch.no_grad():
            return self._solve(
                torch.from_numpy(rgb_np).to(self.device),
                torch.from_numpy(w_cnn_np).to(self.device),
                params,
            )

    def _solve(self, rgb: Any, w_cnn: Any, params: PipelineParams) -> SolveResult:
        import torch

        h, w, _ = rgb.shape

        # Phase 1: weighted luminance + DoG bands.
        weights = torch.tensor(
            [params.w_r, params.w_g, params.w_b], device=self.device, dtype=rgb.dtype
        )
        i_w = (rgb * weights[None, None, :]).sum(dim=-1)  # (H, W)
        # Robust albedo (median window) and all-scales gradient for the
        # consistency term.
        rho = _local_median(
            i_w.unsqueeze(0).unsqueeze(0),
            params.albedo_window_radius,
        ).squeeze()
        ix_full, iy_full = _scharr_gradients(i_w.unsqueeze(0).unsqueeze(0))
        ix_full = ix_full.squeeze()
        iy_full = iy_full.squeeze()

        per_scale: list[Any] = []
        for j in range(params.n_scales):
            sigma_j = params.sigma_base * (params.k_ratio**j)
            outer = _gaussian_blur(i_w.unsqueeze(0).unsqueeze(0), sigma_j * params.k_ratio)
            inner = _gaussian_blur(i_w.unsqueeze(0).unsqueeze(0), sigma_j)
            band = (outer - inner).squeeze()

            # Phase 2: Scharr per scale + structural tensor + classifier.
            ix_k, iy_k = _scharr_gradients(band.unsqueeze(0).unsqueeze(0))
            ix_k = ix_k.squeeze()
            iy_k = iy_k.squeeze()
            ixx = (ix_k * ix_k).unsqueeze(0).unsqueeze(0)
            ixy = (ix_k * iy_k).unsqueeze(0).unsqueeze(0)
            iyy = (iy_k * iy_k).unsqueeze(0).unsqueeze(0)
            jxx = _box_average(ixx, params.tensor_window_radius).squeeze()
            jxy = _box_average(ixy, params.tensor_window_radius).squeeze()
            jyy = _box_average(iyy, params.tensor_window_radius).squeeze()
            l1, l2 = _eigenvalues_2x2(jxx, jxy, jyy)
            tau_flat, tau_edge = _adaptive_thresholds(l1, l2, params.p_flat, params.p_edge)
            classes = _classify(l1, l2, tau_flat, tau_edge)
            yx = (classes >= 1).nonzero()  # (N, 2)

            # Phase 3.5: hyperplane construction.
            hp = _build_hyperplane_params(
                i_w,
                ix_k,
                iy_k,
                rho,
                yx,
                params.epsilon,
                params.c_z,
            )
            per_scale.append(_forge_per_scale((h, w), hp, params.neigh_radius))

        # Phase 3.6: Min-Max composition.
        z_forged = _compose_max_over_scales(per_scale)
        fallback = params.c_z * i_w
        z_forged = _fill_uncovered(z_forged, fallback)

        # Phase 5: Jacobi solve.
        z_star, e_trace, iters, converged = _jacobi_solve(
            z_forged,
            ix_full,
            iy_full,
            w_cnn,
            rho,
            params.pde,
        )

        # Phase 6: impact map.
        z_ideal = _gaussian_blur(z_forged.unsqueeze(0).unsqueeze(0), params.sigma_ref).squeeze()
        residual = z_star - z_ideal
        l_map = _laplacian(z_star.unsqueeze(0).unsqueeze(0)).squeeze()

        e_final = _energy_terms(z_star, z_forged, ix_full, iy_full, w_cnn, rho, params.pde)

        return SolveResult(
            z_forged=z_forged.detach().cpu().numpy().astype(np.float32),
            z_star=z_star.detach().cpu().numpy().astype(np.float32),
            residual=residual.detach().cpu().numpy().astype(np.float32),
            laplacian=l_map.detach().cpu().numpy().astype(np.float32),
            energy_total=e_final["total"],
            energy_data=e_final["data"],
            energy_smoothness=e_final["smoothness"],
            energy_consistency=e_final["consistency"],
            energy_trace=np.asarray(e_trace, dtype=np.float32),
            iterations=iters,
            converged=converged,
        )


__all__ = ["CudaBackend"]
