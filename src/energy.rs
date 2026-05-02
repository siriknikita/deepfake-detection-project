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

use ndarray::ArrayView2;
use rayon::prelude::*;

use crate::scharr::scharr_gradients;
use crate::stencils::laplacian;

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
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
