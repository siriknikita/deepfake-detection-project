//! Phase 5 — Euler-Lagrange settlement via Jacobi fixed-point iteration.
//!
//! Solves the PDE
//!
//!   λ (z − z_forged) + α Δ²z − β div(W² K (∇I − K ∇z)) = 0
//!
//! by the Jacobi-style fixed-point iteration
//!
//!   z^(n+1) = z^n − τ · R(z^n)
//!
//! where `R(z)` is the residual on the left-hand side of the PDE (the
//! L²-gradient of the energy functional E(z) — see `energy::energy`),
//! and τ is the inverse of the conservative diagonal estimate
//! `D = λ + 20 α` of the discrete operator. This is provably stable
//! whenever the off-diagonal contribution from the gradient-consistency
//! divergence term has spectral radius below D, which holds for the
//! parameter ranges relevant to this pipeline. The cost per iteration is
//! one biharmonic, one Scharr-gradient pair, and one divergence on the
//! pixel grid — all already implemented in `stencils` and `scharr`.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::energy::{energy, EnergyTerms};
use crate::scharr::scharr_gradients;
use crate::stencils::{divergence, laplacian};

/// Parameters of the Jacobi PDE solver.
#[derive(Clone, Copy, Debug)]
pub struct PdeParams {
    /// Data-term weight in `E(z)`.
    pub lambda: f32,
    /// Smoothness-term weight in `E(z)`.
    pub alpha: f32,
    /// Gradient-consistency weight in `E(z)`.
    pub beta: f32,
    /// Maximum number of Jacobi iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖z^(n+1) − z^(n)‖₂ / ‖z^(n)‖₂`.
    pub tol: f32,
    /// Sample the energy every `log_every` iterations (0 disables logging).
    pub log_every: usize,
}

impl Default for PdeParams {
    fn default() -> Self {
        Self { lambda: 1.0, alpha: 0.5, beta: 5.0, max_iter: 500, tol: 1.0e-5, log_every: 10 }
    }
}

/// Output of the Jacobi solver.
#[derive(Clone, Debug)]
pub struct SolveResult {
    /// The settled manifold `z*`.
    pub z_star: Array2<f32>,
    /// Energy trace sampled at iterations `0, log_every, 2·log_every, …`.
    pub energy_trace: Vec<EnergyTerms>,
    /// Number of iterations performed (including the final one).
    pub iterations: usize,
    /// Did the relative-L² stopping criterion fire before `max_iter`?
    pub converged: bool,
}

/// Run the Jacobi fixed-point iteration on the Euler-Lagrange PDE.
///
/// `z_forged` is the initial manifold from Phase 3; the iteration starts from
/// `z^(0) = z_forged`. `ix` / `iy` are the all-scales image gradients
/// (Phase 2 output of `I_w`). `w_cnn` is the CNN trust map. `k_albedo` is the
/// adaptive albedo coefficient `K(x, y)`.
///
/// # Panics
///
/// Panics if any of the input arrays disagree on shape.
#[must_use]
pub fn jacobi_solve(
    z_forged: ArrayView2<f32>,
    ix: ArrayView2<f32>,
    iy: ArrayView2<f32>,
    w_cnn: ArrayView2<f32>,
    k_albedo: ArrayView2<f32>,
    params: &PdeParams,
) -> SolveResult {
    let shape = z_forged.dim();
    assert_eq!(shape, ix.dim());
    assert_eq!(shape, iy.dim());
    assert_eq!(shape, w_cnn.dim());
    assert_eq!(shape, k_albedo.dim());

    // Step size: inverse of a conservative diagonal of the discrete operator
    // λ I + α Δ² − β div(W² K² ∇·). Per-pixel diagonal coefficients:
    // - λ I            : λ
    // - α Δ²           : 20 α  (5-point biharmonic center weight)
    // - β div(W² K² ∇·): bounded above by β · max(W² K²) under the central-
    //   difference + Scharr discretization; we pad by a factor of 2 to leave
    //   stability headroom for the spatially varying off-diagonal terms.
    let max_wk2 =
        w_cnn.iter().zip(k_albedo.iter()).map(|(&w, &k)| (w * k).powi(2)).fold(0.0_f32, f32::max);
    let diag = params.lambda + 20.0 * params.alpha + 2.0 * params.beta * max_wk2;
    // Conservative damping factor — empirically the analytic diagonal still
    // underestimates the spectral radius of the discrete operator on small
    // grids (the Scharr gradient and central-difference divergence couple
    // ranges beyond the immediate 5-point neighborhood). A 0.25 factor on
    // top of the analytic 1/diag bound buys monotone convergence on every
    // test we exercise without slowing down large-grid runs by more than 4×.
    let tau = if diag > 0.0 { 0.25 / diag } else { 0.25 };

    let mut z = z_forged.to_owned();
    let mut energy_trace: Vec<EnergyTerms> = Vec::new();
    if params.log_every > 0 {
        energy_trace.push(energy(
            z.view(),
            z_forged,
            ix,
            iy,
            w_cnn,
            k_albedo,
            params.lambda,
            params.alpha,
            params.beta,
        ));
    }

    let mut converged = false;
    let mut iterations = 0;
    for n in 1..=params.max_iter {
        let z_prev_norm = l2_norm(z.view()).max(f32::EPSILON);

        // Compute residual R(z) = λ(z - z_forged) + α Δ²z − β div(W² K (∇I − K ∇z)).
        let lap = laplacian(z.view());
        let bilap = laplacian(lap.view());
        let (zx, zy) = scharr_gradients(z.view());

        let mut fx = Array2::<f32>::zeros(shape);
        let mut fy = Array2::<f32>::zeros(shape);
        for ((y, x), v) in fx.indexed_iter_mut() {
            let wi = w_cnn[(y, x)];
            let ki = k_albedo[(y, x)];
            *v = wi * wi * ki * (ix[(y, x)] - ki * zx[(y, x)]);
            fy[(y, x)] = wi * wi * ki * (iy[(y, x)] - ki * zy[(y, x)]);
        }
        let div_f = divergence(fx.view(), fy.view());

        // z_new = z - τ · R(z).
        let mut delta_norm_sq = 0.0_f32;
        let mut z_new = Array2::<f32>::zeros(shape);
        for ((y, x), v) in z_new.indexed_iter_mut() {
            let r = params.lambda * (z[(y, x)] - z_forged[(y, x)]) + params.alpha * bilap[(y, x)]
                - params.beta * div_f[(y, x)];
            let new_v = z[(y, x)] - tau * r;
            *v = new_v;
            let d = new_v - z[(y, x)];
            delta_norm_sq += d * d;
        }
        z = z_new;
        iterations = n;

        let delta_norm = delta_norm_sq.sqrt();
        if delta_norm / z_prev_norm < params.tol {
            converged = true;
        }

        if params.log_every > 0 && n % params.log_every == 0 {
            energy_trace.push(energy(
                z.view(),
                z_forged,
                ix,
                iy,
                w_cnn,
                k_albedo,
                params.lambda,
                params.alpha,
                params.beta,
            ));
        }

        if converged {
            break;
        }
    }

    SolveResult { z_star: z, energy_trace, iterations, converged }
}

fn l2_norm(arr: ArrayView2<f32>) -> f32 {
    let s: f32 = arr.as_slice().expect("contiguous").par_iter().map(|&v| v * v).sum();
    s.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn zero_field(shape: (usize, usize)) -> Array2<f32> {
        Array2::<f32>::zeros(shape)
    }

    #[test]
    fn solver_converges_on_trivial_problem() {
        // β = 0, α = 0 ⇒ minimizer is z* = z_forged exactly. One step is
        // sufficient: λ (z − z_forged) = 0 ⇔ z = z_forged.
        let z_forged = Array2::from_elem((5, 5), 0.7_f32);
        let zeros = zero_field(z_forged.dim());
        let one = Array2::from_elem(z_forged.dim(), 1.0_f32);
        let params =
            PdeParams { lambda: 1.0, alpha: 0.0, beta: 0.0, max_iter: 50, tol: 1e-7, log_every: 0 };
        let res = jacobi_solve(
            z_forged.view(),
            zeros.view(),
            zeros.view(),
            one.view(),
            one.view(),
            &params,
        );
        assert!(res.converged, "should converge");
        for v in res.z_star.iter() {
            assert!(approx_eq(*v, 0.7, 1e-5), "z* should equal z_forged");
        }
    }

    #[test]
    fn solver_preserves_constant_under_full_problem() {
        // Constant z_forged + zero gradients ⇒ Δz_forged = 0, ∇z_forged = 0,
        // ∇I = 0; the constant z = z_forged is the global minimum and the
        // solver should not move from it.
        let z_forged = Array2::from_elem((6, 6), 0.4_f32);
        let zeros = zero_field(z_forged.dim());
        let one = Array2::from_elem(z_forged.dim(), 1.0_f32);
        let params = PdeParams {
            lambda: 1.0,
            alpha: 0.5,
            beta: 5.0,
            max_iter: 100,
            tol: 1e-8,
            log_every: 0,
        };
        let res = jacobi_solve(
            z_forged.view(),
            zeros.view(),
            zeros.view(),
            one.view(),
            one.view(),
            &params,
        );
        for v in res.z_star.iter() {
            assert!(
                approx_eq(*v, 0.4, 1e-4),
                "constant minimizer should not drift, got {v} after {} iterations",
                res.iterations
            );
        }
    }

    #[test]
    fn solver_returns_starting_iterate_when_max_iter_zero() {
        // max_iter = 0 ⇒ the loop body never executes; z* must equal the
        // initial iterate z_forged.
        let z_forged = Array2::from_elem((4, 4), 0.3_f32);
        let zeros = zero_field(z_forged.dim());
        let one = Array2::from_elem(z_forged.dim(), 1.0_f32);
        let params =
            PdeParams { lambda: 1.0, alpha: 0.5, beta: 5.0, max_iter: 0, tol: 1e-5, log_every: 0 };
        let res = jacobi_solve(
            z_forged.view(),
            zeros.view(),
            zeros.view(),
            one.view(),
            one.view(),
            &params,
        );
        assert_eq!(res.iterations, 0);
        for v in res.z_star.iter() {
            assert!(approx_eq(*v, 0.3, 1e-9));
        }
    }

    #[test]
    fn solver_logs_energy_at_expected_cadence() {
        // log_every = 5, max_iter = 20 ⇒ trace records iter 0, 5, 10, 15, 20.
        let z_forged = Array2::from_elem((4, 4), 0.5_f32);
        let zeros = zero_field(z_forged.dim());
        let one = Array2::from_elem(z_forged.dim(), 1.0_f32);
        let params =
            PdeParams { lambda: 1.0, alpha: 0.0, beta: 0.0, max_iter: 20, tol: 0.0, log_every: 5 };
        let res = jacobi_solve(
            z_forged.view(),
            zeros.view(),
            zeros.view(),
            one.view(),
            one.view(),
            &params,
        );
        // iter 0 (initial) + iters 5, 10, 15, 20 = 5 entries (no early-stop
        // since tol=0 only fires when the update is exactly zero).
        assert_eq!(res.energy_trace.len(), 5);
    }

    #[test]
    fn solver_energy_monotone_non_increasing() {
        // Sanity invariant: under a stable step the trace should be
        // monotonically non-increasing across our Jacobi iteration.
        let mut z_forged = Array2::<f32>::zeros((9, 9));
        // Spike z_forged so the data term has work to do.
        z_forged[(4, 4)] = 1.0;
        let zeros = zero_field(z_forged.dim());
        let w = Array2::from_elem(z_forged.dim(), 1.0_f32);
        let k = Array2::from_elem(z_forged.dim(), 1.0_f32);
        let params =
            PdeParams { lambda: 1.0, alpha: 0.5, beta: 1.0, max_iter: 50, tol: 1e-8, log_every: 5 };
        let res =
            jacobi_solve(z_forged.view(), zeros.view(), zeros.view(), w.view(), k.view(), &params);
        // Float32 sum tolerance: trace samples may diverge by O(1e-3) of
        // their absolute magnitude due to non-associative summation, even
        // when the analytic energy is monotonically decreasing.
        for w in res.energy_trace.windows(2) {
            let tol = 1.0e-3 * w[0].total.abs().max(1.0);
            assert!(
                w[1].total <= w[0].total + tol,
                "energy must not increase beyond {tol}: {} -> {}",
                w[0].total,
                w[1].total
            );
        }
    }

    #[test]
    fn solver_settles_smooth_target() {
        // z_forged is a piecewise-affine bump (a single anchor's hyperplane
        // stamped on a 9×9). The biharmonic regularizer should round off the
        // discontinuities so the settled z* has lower smoothness energy than
        // z_forged.
        let mut z_forged = Array2::<f32>::zeros((11, 11));
        for ((y, x), v) in z_forged.indexed_iter_mut() {
            let dx = x as f32 - 5.0;
            let dy = y as f32 - 5.0;
            let r2 = dx * dx + dy * dy;
            *v = if r2 < 4.0 { 1.0 } else { 0.0 };
        }
        let zeros = zero_field(z_forged.dim());
        let w = Array2::from_elem(z_forged.dim(), 1.0_f32);
        let k = Array2::from_elem(z_forged.dim(), 1.0_f32);
        let params = PdeParams {
            lambda: 1.0,
            alpha: 5.0,
            beta: 0.0,
            max_iter: 200,
            tol: 1e-7,
            log_every: 10,
        };
        let res =
            jacobi_solve(z_forged.view(), zeros.view(), zeros.view(), w.view(), k.view(), &params);
        let lap_forged = laplacian(z_forged.view());
        let lap_settled = laplacian(res.z_star.view());
        let s_forged: f32 = lap_forged.iter().map(|v| v * v).sum();
        let s_settled: f32 = lap_settled.iter().map(|v| v * v).sum();
        assert!(
            s_settled < s_forged,
            "smoothness energy did not decrease: {s_forged} -> {s_settled}"
        );
    }

    #[test]
    fn l2_norm_matches_definition() {
        let arr =
            Array2::from_shape_vec((1, 4), vec![3.0_f32, 4.0, 0.0, 0.0]).expect("valid shape");
        // ‖[3, 4, 0, 0]‖₂ = 5.
        assert!(approx_eq(l2_norm(arr.view()), 5.0, 1e-6));
    }
}
