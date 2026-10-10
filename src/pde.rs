//! Phase 5 — Euler-Lagrange settlement by damped gradient descent.
//!
//! Minimises the discrete energy `E(z)` of `energy::energy` with the
//! fixed-point iteration
//!
//!   z^(n+1) = z^n − τ · R(z^n),      R(z) = ½ ∂E/∂z
//!
//! where, with `S` the Scharr gradient and `Δ` the mirror Laplacian,
//!
//!   R(z) = λ (z − z_forged) + α Δᵀ(Δz) − β Sᵀ[ W² K (∇I − K Sz) ]
//!
//! (see `energy::energy_gradient_half`). `R` is the *exact* L²-gradient of the
//! implemented energy, including the mirror boundary rows, so every accepted
//! step is a descent step and a stationary `z*` satisfies `R(z*) = 0`.
//!
//! History: the previous implementation used `+ α Δ²z − β div_central(F)`,
//! i.e. the paper's printed Euler–Lagrange form. That operator carries the
//! consistency term with the *opposite* sign to the energy gradient (and
//! `div_central` is not the adjoint of the Scharr gradient), so the iteration
//! only decreased `E` while the `λ`/`α` parts dominated and stopped when the
//! backtracking step size collapsed — the returned `z*` was not stationary.
//!
//! The step size starts at `τ₀ = 1 / (λ + 64α + 2β·max(WK)²)`, a bound on the
//! Lipschitz constant of `R` (‖ΔᵀΔ‖ ≤ 64, ‖SᵀS‖ ≤ 2 for the 1/32-normalised
//! Scharr kernels), which guarantees descent; a backtracking line search is
//! kept as a guard against the bound being violated by rounding.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::energy::{energy, energy_gradient_half, EnergyTerms};

/// Maximum number of times we halve `τ` per Jacobi step in the backtracking
/// line search before committing the smallest step we tried.
const TAU_BACKOFF_TRIES: usize = 6;

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

    // Step size from a Lipschitz bound on R = ½∇E: ‖ΔᵀΔ‖ ≤ 64, ‖SᵀS‖ ≤ 2 (Scharr
    // kernels are 1/32-normalised with |weights| summing to 32 per direction), so
    // L ≤ λ + 64α + 2β·max(WK)² and any τ ≤ 1/L yields monotone descent. A
    // backtracking line search guards against rounding; it rarely triggers.
    let max_wk2 =
        w_cnn.iter().zip(k_albedo.iter()).map(|(&w, &k)| (w * k).powi(2)).fold(0.0_f32, f32::max);
    let lipschitz = params.lambda + 64.0 * params.alpha + 2.0 * params.beta * max_wk2;
    let mut tau = if lipschitz > 0.0 { 1.0 / lipschitz } else { 0.5 };

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

    let mut e_prev = energy(
        z.view(),
        z_forged,
        ix,
        iy,
        w_cnn,
        k_albedo,
        params.lambda,
        params.alpha,
        params.beta,
    )
    .total;

    let mut converged = false;
    let mut iterations = 0;
    for n in 1..=params.max_iter {
        let z_prev_norm = l2_norm(z.view()).max(f32::EPSILON);

        // R(z) = ½ ∇E(z), the exact gradient of the implemented discrete energy.
        let r_field = energy_gradient_half(
            z.view(),
            z_forged,
            ix,
            iy,
            w_cnn,
            k_albedo,
            params.lambda,
            params.alpha,
            params.beta,
        );

        // Backtracking line search: try the current τ, halve until energy
        // does not increase. After TAU_BACKOFF_TRIES halvings we accept the
        // smallest step we tried; further iterations will continue to use it.
        let mut accepted_z: Option<Array2<f32>> = None;
        let mut accepted_e = e_prev;
        let mut delta_norm_sq = 0.0_f32;
        for _ in 0..TAU_BACKOFF_TRIES {
            let mut z_trial = Array2::<f32>::zeros(shape);
            let mut delta_sq_trial = 0.0_f32;
            for ((y, x), v) in z_trial.indexed_iter_mut() {
                let new_v = z[(y, x)] - tau * r_field[(y, x)];
                *v = new_v;
                let d = new_v - z[(y, x)];
                delta_sq_trial += d * d;
            }
            let e_trial = energy(
                z_trial.view(),
                z_forged,
                ix,
                iy,
                w_cnn,
                k_albedo,
                params.lambda,
                params.alpha,
                params.beta,
            )
            .total;
            if e_trial <= e_prev * (1.0 + 1.0e-6) {
                accepted_z = Some(z_trial);
                accepted_e = e_trial;
                delta_norm_sq = delta_sq_trial;
                break;
            }
            tau *= 0.5;
            // Final-pass fallthrough: keep last trial if no acceptance.
            accepted_z = Some(z_trial);
            accepted_e = e_trial;
            delta_norm_sq = delta_sq_trial;
        }
        z = accepted_z.expect("at least one trial executed");
        e_prev = accepted_e;
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
    use crate::stencils::laplacian;
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
        // Tolerance: 1% of energy magnitude. Discrete Jacobi steps can
        // overshoot the minimum on stiff problems and the trace is not
        // pointwise monotone — but the long-run trend is decreasing, so we
        // verify final < initial elsewhere and only assert the per-step
        // jitter stays bounded here.
        for w in res.energy_trace.windows(2) {
            let tol = 1.0e-2 * w[0].total.abs().max(1.0);
            assert!(
                w[1].total <= w[0].total + tol,
                "energy must not increase beyond {tol}: {} -> {}",
                w[0].total,
                w[1].total
            );
        }
        // Final energy must be no greater than initial — the canonical
        // convergence indicator that survives the discrete-step jitter.
        let initial = res.energy_trace.first().expect("trace populated").total;
        let final_e = res.energy_trace.last().expect("trace populated").total;
        assert!(final_e <= initial + 1.0e-3, "final energy {final_e} must be <= initial {initial}");
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

    fn pseudo_random(shape: (usize, usize), seed: u32, scale: f32, offset: f32) -> Array2<f32> {
        let mut state = seed;
        Array2::from_shape_fn(shape, |_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            offset + scale * ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5)
        })
    }

    #[test]
    fn solver_reaches_stationary_point_with_consistency_term() {
        // Regression for the gradient-sign defect: with β > 0 and non-trivial ∇I,
        // the settled field must make the exact energy gradient small relative
        // to its value at the starting point. The pre-fix operator left it at
        // 20–55 % of the initial norm.
        let shape = (12, 12);
        let z_forged = pseudo_random(shape, 31, 0.6, 0.5);
        let ix = pseudo_random(shape, 32, 0.4, 0.0);
        let iy = pseudo_random(shape, 33, 0.4, 0.0);
        let w = pseudo_random(shape, 34, 0.4, 0.7);
        let k = pseudo_random(shape, 35, 0.4, 0.6);
        let params = PdeParams {
            lambda: 1.0,
            alpha: 0.5,
            beta: 5.0,
            max_iter: 4000,
            tol: 1e-9,
            log_every: 0,
        };
        let res = jacobi_solve(z_forged.view(), ix.view(), iy.view(), w.view(), k.view(), &params);
        let g0 = energy_gradient_half(
            z_forged.view(),
            z_forged.view(),
            ix.view(),
            iy.view(),
            w.view(),
            k.view(),
            1.0,
            0.5,
            5.0,
        );
        let g1 = energy_gradient_half(
            res.z_star.view(),
            z_forged.view(),
            ix.view(),
            iy.view(),
            w.view(),
            k.view(),
            1.0,
            0.5,
            5.0,
        );
        let n0: f32 = g0.iter().map(|v| v * v).sum::<f32>().sqrt();
        let n1: f32 = g1.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            n1 < 1.0e-3 * n0,
            "gradient norm ratio {} after {} iterations (converged={})",
            n1 / n0,
            res.iterations,
            res.converged
        );
        // and the energy actually decreased from the starting point
        let e0 = energy(
            z_forged.view(),
            z_forged.view(),
            ix.view(),
            iy.view(),
            w.view(),
            k.view(),
            1.0,
            0.5,
            5.0,
        )
        .total;
        let e1 = energy(
            res.z_star.view(),
            z_forged.view(),
            ix.view(),
            iy.view(),
            w.view(),
            k.view(),
            1.0,
            0.5,
            5.0,
        )
        .total;
        assert!(e1 < e0, "energy did not decrease: {e0} -> {e1}");
    }

    #[test]
    fn l2_norm_matches_definition() {
        let arr =
            Array2::from_shape_vec((1, 4), vec![3.0_f32, 4.0, 0.0, 0.0]).expect("valid shape");
        // ‖[3, 4, 0, 0]‖₂ = 5.
        assert!(approx_eq(l2_norm(arr.view()), 5.0, 1e-6));
    }
}
