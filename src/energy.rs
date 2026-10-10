//! Phase 4 — Global energy functional.
//!
//! Evaluates the discrete Riemann sum approximation of
//!
//!   E(z) = ∬ [ λ (z − z_forged)²
//!            + α (Δz)²
//!            + β ‖W_cnn ⊙ (∇I − K ∇z)‖² ] dA
//!
//! and returns each term separately so the per-term contribution can be
//! inspected in the impact map's feature vector and used as a convergence
//! diagnostic by the Jacobi solver. All sums are over the full pixel grid
//! with unit cell area (h = 1); see Phase 4 of the paper for the
//! continuous formulation.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::scharr::{scharr_adjoint, scharr_gradients};
use crate::stencils::{laplacian, laplacian_adjoint};

/// Per-term decomposition of the energy functional plus its total.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnergyTerms {
    /// `λ · Σ (z − z_forged)²`.
    pub data: f32,
    /// `α · Σ (Δz)²`.
    pub smoothness: f32,
    /// `β · Σ ‖W ⊙ (∇I − K ∇z)‖²`.
    pub consistency: f32,
    /// Sum of the three terms above.
    pub total: f32,
}

/// Evaluate the discrete energy functional `E(z)` and return its per-term
/// breakdown. All array shapes must agree.
///
/// # Panics
///
/// Panics if any pair of input arrays disagree on shape.
#[must_use]
pub fn energy(
    z: ArrayView2<f32>,
    z_forged: ArrayView2<f32>,
    ix: ArrayView2<f32>,
    iy: ArrayView2<f32>,
    w_cnn: ArrayView2<f32>,
    k_albedo: ArrayView2<f32>,
    lambda: f32,
    alpha: f32,
    beta: f32,
) -> EnergyTerms {
    let shape = z.dim();
    assert_eq!(shape, z_forged.dim());
    assert_eq!(shape, ix.dim());
    assert_eq!(shape, iy.dim());
    assert_eq!(shape, w_cnn.dim());
    assert_eq!(shape, k_albedo.dim());

    // Data term: λ · Σ (z − z_forged)².
    let data_sum: f32 = z
        .as_slice()
        .expect("contiguous")
        .par_iter()
        .zip(z_forged.as_slice().expect("contiguous").par_iter())
        .map(|(&zi, &zf)| {
            let d = zi - zf;
            d * d
        })
        .sum();
    let data = lambda * data_sum;

    // Smoothness term: α · Σ (Δz)².
    let lap = laplacian(z);
    let smooth_sum: f32 = lap.as_slice().expect("contiguous").par_iter().map(|&v| v * v).sum();
    let smoothness = alpha * smooth_sum;

    // Consistency term: β · Σ ‖W ⊙ (∇I − K ∇z)‖².
    let (zx, zy) = scharr_gradients(z);
    let (h, w) = shape;
    let n = h * w;
    let cons_sum: f32 = (0..n)
        .into_par_iter()
        .map(|idx| {
            let y = idx / w;
            let x = idx % w;
            let wi = w_cnn[(y, x)];
            let ki = k_albedo[(y, x)];
            let rx = wi * (ix[(y, x)] - ki * zx[(y, x)]);
            let ry = wi * (iy[(y, x)] - ki * zy[(y, x)]);
            rx * rx + ry * ry
        })
        .sum();
    let consistency = beta * cons_sum;

    EnergyTerms { data, smoothness, consistency, total: data + smoothness + consistency }
}

/// Exact L²-gradient of the discrete energy, halved: `½ ∂E/∂z`.
///
/// With `S` the Scharr gradient and `Δ` the mirror Laplacian as implemented,
///
/// ```text
/// ½ ∇E(z) = λ (z − z_forged) + α Δᵀ(Δz) − β Sᵀ[ W² K (∇I − K Sz) ]
/// ```
///
/// The adjoints `Δᵀ`, `Sᵀ` are the exact transposes under the mirror boundary
/// (see `stencils::laplacian_adjoint`, `scharr::scharr_adjoint`), so this is the
/// true gradient of [`energy`] on the full grid. In the continuum `−Sᵀ ≈ div`, i.e.
/// the consistency term enters the Euler–Lagrange equation as `+β div(...)`.
///
/// # Panics
///
/// Panics if any pair of input arrays disagree on shape.
#[must_use]
pub fn energy_gradient_half(
    z: ArrayView2<f32>,
    z_forged: ArrayView2<f32>,
    ix: ArrayView2<f32>,
    iy: ArrayView2<f32>,
    w_cnn: ArrayView2<f32>,
    k_albedo: ArrayView2<f32>,
    lambda: f32,
    alpha: f32,
    beta: f32,
) -> Array2<f32> {
    let shape = z.dim();
    assert_eq!(shape, z_forged.dim());
    assert_eq!(shape, ix.dim());
    assert_eq!(shape, iy.dim());
    assert_eq!(shape, w_cnn.dim());
    assert_eq!(shape, k_albedo.dim());

    let lap = laplacian(z);
    let bilap_t = laplacian_adjoint(lap.view());
    let (zx, zy) = scharr_gradients(z);
    let mut fx = Array2::<f32>::zeros(shape);
    let mut fy = Array2::<f32>::zeros(shape);
    for ((y, x), v) in fx.indexed_iter_mut() {
        let wi = w_cnn[(y, x)];
        let ki = k_albedo[(y, x)];
        *v = wi * wi * ki * (ix[(y, x)] - ki * zx[(y, x)]);
        fy[(y, x)] = wi * wi * ki * (iy[(y, x)] - ki * zy[(y, x)]);
    }
    let s_t = scharr_adjoint(fx.view(), fy.view());
    let mut out = Array2::<f32>::zeros(shape);
    for ((y, x), v) in out.indexed_iter_mut() {
        *v = lambda * (z[(y, x)] - z_forged[(y, x)]) + alpha * bilap_t[(y, x)] - beta * s_t[(y, x)];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn pseudo_random(shape: (usize, usize), seed: u32, scale: f32, offset: f32) -> Array2<f32> {
        let mut state = seed;
        Array2::from_shape_fn(shape, |_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            offset + scale * ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5)
        })
    }

    #[test]
    fn gradient_matches_finite_difference_directional_derivative() {
        // d/dε E(z + ε v) |₀ = ⟨∇E(z), v⟩ = 2 ⟨½∇E(z), v⟩, with a central difference
        // in f64 built from f32 energy evaluations on a random direction.
        let shape = (9, 11);
        let z = pseudo_random(shape, 1, 1.0, 0.0);
        let zf = pseudo_random(shape, 2, 1.0, 0.0);
        let ix = pseudo_random(shape, 3, 0.5, 0.0);
        let iy = pseudo_random(shape, 4, 0.5, 0.0);
        let w = pseudo_random(shape, 5, 0.6, 0.6);
        let k = pseudo_random(shape, 6, 0.6, 0.6);
        let v = pseudo_random(shape, 7, 1.0, 0.0);
        let (lam, al, be) = (1.0_f32, 0.5_f32, 5.0_f32);
        let g = energy_gradient_half(
            z.view(),
            zf.view(),
            ix.view(),
            iy.view(),
            w.view(),
            k.view(),
            lam,
            al,
            be,
        );
        let analytic: f64 =
            2.0 * g.iter().zip(v.iter()).map(|(a, b)| f64::from(a * b)).sum::<f64>();
        let eps = 1.0e-2_f32;
        let zp = &z + &v.mapv(|t| eps * t);
        let zm = &z - &v.mapv(|t| eps * t);
        let ep =
            energy(zp.view(), zf.view(), ix.view(), iy.view(), w.view(), k.view(), lam, al, be)
                .total;
        let em =
            energy(zm.view(), zf.view(), ix.view(), iy.view(), w.view(), k.view(), lam, al, be)
                .total;
        let fd = (f64::from(ep) - f64::from(em)) / (2.0 * f64::from(eps));
        let rel = (fd - analytic).abs() / analytic.abs().max(1.0);
        assert!(rel < 2.0e-2, "finite difference {fd} vs analytic {analytic} (rel err {rel})");
    }

    #[test]
    fn gradient_consistency_term_has_opposite_sign_to_naive_divergence_form() {
        // The pre-fix operator used `−β div_central(F)`; the exact gradient uses
        // `−β Sᵀ F = +β div_S F` on the interior. Check the sign on a field where
        // only the consistency term is active (λ = α = 0).
        let shape = (11, 11);
        let z = pseudo_random(shape, 8, 1.0, 0.0);
        let zeros = Array2::<f32>::zeros(shape);
        let ix = pseudo_random(shape, 9, 0.5, 0.0);
        let iy = pseudo_random(shape, 10, 0.5, 0.0);
        let one = Array2::from_elem(shape, 1.0_f32);
        let g = energy_gradient_half(
            z.view(),
            zeros.view(),
            ix.view(),
            iy.view(),
            one.view(),
            one.view(),
            0.0,
            0.0,
            1.0,
        );
        let (zx, zy) = scharr_gradients(z.view());
        let fx = &ix - &zx;
        let fy = &iy - &zy;
        let (fxx, _) = scharr_gradients(fx.view());
        let (_, fyy) = scharr_gradients(fy.view());
        let mut dot_plus = 0.0_f32;
        let mut norm = 0.0_f32;
        for y in 2..9 {
            for x in 2..9 {
                let div_s = fxx[(y, x)] + fyy[(y, x)];
                dot_plus += g[(y, x)] * div_s;
                norm += div_s * div_s;
            }
        }
        // interior: g = +div_S F exactly ⇒ ⟨g, div_S F⟩ = ‖div_S F‖².
        assert!(
            approx_eq(dot_plus, norm, 1e-3 * norm.max(1.0)),
            "⟨g, div_S F⟩={dot_plus} vs ‖div_S F‖²={norm}"
        );
    }

    fn zero(shape: (usize, usize)) -> Array2<f32> {
        Array2::<f32>::zeros(shape)
    }

    #[test]
    fn energy_of_z_equals_zforged_zero_data_term() {
        let z = Array2::from_elem((6, 6), 0.5_f32);
        let zf = z.clone();
        let zero = zero(z.dim());
        let one = Array2::from_elem(z.dim(), 1.0_f32);
        let e = energy(
            z.view(),
            zf.view(),
            zero.view(),
            zero.view(),
            one.view(),
            one.view(),
            1.0,
            1.0,
            1.0,
        );
        // Constant z ⇒ Δz = 0 ⇒ smoothness = 0.
        // ∇I = 0, ∇z = 0 ⇒ consistency = 0.
        // z = z_forged ⇒ data = 0.
        assert!(approx_eq(e.data, 0.0, 1e-5));
        assert!(approx_eq(e.smoothness, 0.0, 1e-5));
        assert!(approx_eq(e.consistency, 0.0, 1e-5));
        assert!(approx_eq(e.total, 0.0, 1e-5));
    }

    #[test]
    fn energy_terms_are_non_negative() {
        // Each term is a sum of squares (with non-negative weights) ⇒ ≥ 0.
        let mut z = Array2::<f32>::zeros((8, 8));
        let mut zf = Array2::<f32>::zeros((8, 8));
        for ((y, x), v) in z.indexed_iter_mut() {
            *v = (y as f32) * 0.1 + (x as f32) * 0.2;
            zf[(y, x)] = (x as f32) * 0.3;
        }
        let one = Array2::from_elem(z.dim(), 1.0_f32);
        let e = energy(
            z.view(),
            zf.view(),
            one.view(),
            one.view(),
            one.view(),
            one.view(),
            1.0,
            1.0,
            1.0,
        );
        assert!(e.data >= 0.0);
        assert!(e.smoothness >= 0.0);
        assert!(e.consistency >= 0.0);
        assert!(e.total >= 0.0);
    }

    #[test]
    fn data_term_is_lambda_squared_deviation() {
        // z − z_forged = 0.5 everywhere, λ = 2 ⇒ data = 2 · N · 0.25.
        let z = Array2::from_elem((4, 4), 0.5_f32);
        let zf = Array2::from_elem((4, 4), 0.0_f32);
        let zero = zero(z.dim());
        let one = Array2::from_elem(z.dim(), 1.0_f32);
        let e = energy(
            z.view(),
            zf.view(),
            zero.view(),
            zero.view(),
            one.view(),
            one.view(),
            2.0,
            0.0,
            0.0,
        );
        assert!(approx_eq(e.data, 2.0 * 16.0 * 0.25, 1e-4));
        assert!(approx_eq(e.smoothness, 0.0, 1e-5));
        assert!(approx_eq(e.consistency, 0.0, 1e-5));
    }

    #[test]
    fn smoothness_zero_on_constant_z() {
        // Constant z ⇒ Δz = 0 everywhere, even at mirror-reflected
        // boundaries (constants are eigenfunctions of the Laplacian with
        // eigenvalue 0 under any boundary).
        let z = Array2::from_elem((9, 9), 4.2_f32);
        let zf = z.clone();
        let zeros = zero(z.dim());
        let one = Array2::from_elem(z.dim(), 1.0_f32);
        let e = energy(
            z.view(),
            zf.view(),
            zeros.view(),
            zeros.view(),
            one.view(),
            one.view(),
            0.0,
            1.0,
            0.0,
        );
        assert!(approx_eq(e.smoothness, 0.0, 1e-5));
    }

    #[test]
    fn consistency_zero_when_gradients_vanish() {
        // ∇I = 0 and z constant ⇒ ∇z = 0 ⇒ residual = 0 everywhere,
        // including the boundary; consistency term is exactly 0.
        let z = Array2::from_elem((9, 9), 1.5_f32);
        let zf = z.clone();
        let zeros = zero(z.dim());
        let w_cnn = Array2::from_elem(z.dim(), 1.0_f32);
        let k = Array2::from_elem(z.dim(), 1.0_f32);
        let e = energy(
            z.view(),
            zf.view(),
            zeros.view(),
            zeros.view(),
            w_cnn.view(),
            k.view(),
            0.0,
            0.0,
            1.0,
        );
        assert!(approx_eq(e.consistency, 0.0, 1e-6));
    }

    #[test]
    fn weight_zero_disables_term() {
        // Setting one of (λ, α, β) to 0 must zero out exactly that term.
        let mut z = Array2::<f32>::zeros((6, 6));
        for ((y, x), v) in z.indexed_iter_mut() {
            *v = (x + y) as f32 * 0.1;
        }
        let zf = Array2::<f32>::zeros(z.dim());
        let one = Array2::from_elem(z.dim(), 1.0_f32);
        let e_no_data = energy(
            z.view(),
            zf.view(),
            one.view(),
            one.view(),
            one.view(),
            one.view(),
            0.0,
            1.0,
            1.0,
        );
        assert!(approx_eq(e_no_data.data, 0.0, 1e-9));
        let e_no_smooth = energy(
            z.view(),
            zf.view(),
            one.view(),
            one.view(),
            one.view(),
            one.view(),
            1.0,
            0.0,
            1.0,
        );
        assert!(approx_eq(e_no_smooth.smoothness, 0.0, 1e-9));
        let e_no_cons = energy(
            z.view(),
            zf.view(),
            one.view(),
            one.view(),
            one.view(),
            one.view(),
            1.0,
            1.0,
            0.0,
        );
        assert!(approx_eq(e_no_cons.consistency, 0.0, 1e-9));
    }

    #[test]
    fn total_equals_sum_of_terms() {
        let mut z = Array2::<f32>::zeros((5, 5));
        for ((y, x), v) in z.indexed_iter_mut() {
            *v = (y * 5 + x) as f32 * 0.05;
        }
        let zf = Array2::<f32>::zeros(z.dim());
        let one = Array2::from_elem(z.dim(), 1.0_f32);
        let e = energy(
            z.view(),
            zf.view(),
            one.view(),
            one.view(),
            one.view(),
            one.view(),
            1.0,
            1.0,
            1.0,
        );
        let summed = e.data + e.smoothness + e.consistency;
        assert!(approx_eq(e.total, summed, 1e-4));
    }

    #[test]
    fn w_cnn_zero_eliminates_consistency() {
        // W_cnn = 0 everywhere ⇒ the consistency term is 0 regardless of ∇I, K, ∇z.
        let mut z = Array2::<f32>::zeros((6, 6));
        for ((y, x), v) in z.indexed_iter_mut() {
            *v = (y * 7 + x * 3) as f32 * 0.1;
        }
        let zf = Array2::<f32>::zeros(z.dim());
        let arbitrary = Array2::from_elem(z.dim(), 5.0_f32);
        let zero_w = Array2::<f32>::zeros(z.dim());
        let e = energy(
            z.view(),
            zf.view(),
            arbitrary.view(),
            arbitrary.view(),
            zero_w.view(),
            arbitrary.view(),
            0.0,
            0.0,
            1.0,
        );
        assert!(approx_eq(e.consistency, 0.0, 1e-6));
    }
}
