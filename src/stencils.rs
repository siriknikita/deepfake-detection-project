//! Discrete differential operators on a unit-spaced pixel grid.
//!
//! Implements:
//! - 5-point Laplacian:
//!   `(Δz)_(i,j) = z_(i+1,j) + z_(i-1,j) + z_(i,j+1) + z_(i,j-1) − 4·z_(i,j)`.
//! - Biharmonic operator as the iterated Laplacian: `Δ²z = Δ(Δz)`.
//! - Divergence of a 2D vector field by central differences:
//!   `(div F)_(i,j) = (Fx_(i,j+1) − Fx_(i,j-1))/2 + (Fy_(i+1,j) − Fy_(i-1,j))/2`.
//!
//! All operators use Neumann (mirror) boundary handling so they are
//! well-defined on the entire domain. The free-edge condition for the
//! biharmonic — `∂(Δz)/∂n = 0` — is satisfied automatically because the
//! mirroring is applied to the *intermediate* `Δz` field as well.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::util::mirror;

/// 5-point Laplacian with Neumann (mirror) boundary handling.
#[must_use]
pub fn laplacian(z: ArrayView2<f32>) -> Array2<f32> {
    let (h, w) = z.dim();
    let h_i = h as i32;
    let w_i = w as i32;

    let mut out = Array2::<f32>::zeros((h, w));
    let row_len = w;
    let out_slice = out.as_slice_mut().expect("contiguous");
    out_slice.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        let y_m = mirror(y as i32 - 1, h_i);
        let y_p = mirror(y as i32 + 1, h_i);
        for (x, dst) in row.iter_mut().enumerate() {
            let x_m = mirror(x as i32 - 1, w_i);
            let x_p = mirror(x as i32 + 1, w_i);
            *dst = z[(y_p, x)] + z[(y_m, x)] + z[(y, x_p)] + z[(y, x_m)] - 4.0 * z[(y, x)];
        }
    });
    out
}

/// Adjoint (transpose) of [`laplacian`] under the mirror boundary convention,
/// so that `⟨Δu, v⟩ = ⟨u, Δᵀv⟩` holds exactly on the whole grid.
///
/// Away from the boundary the 5-point stencil is symmetric and `Δᵀ = Δ`; the
/// mirror reflection breaks that symmetry on the outermost ring (a boundary
/// pixel reads its interior neighbour twice). The adjoint is the exact
/// transpose of the gather: every tap `out[p] += w · z[mirror(p + d)]` becomes
/// the scatter `out[mirror(p + d)] += w · v[p]`. Needed for the exact gradient
/// of the smoothness energy `α Σ (Δz)²`, whose L²-gradient is `2α ΔᵀΔz`.
#[must_use]
pub fn laplacian_adjoint(v: ArrayView2<f32>) -> Array2<f32> {
    let (h, w) = v.dim();
    let h_i = h as i32;
    let w_i = w as i32;
    let mut out = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        let y_m = mirror(y as i32 - 1, h_i);
        let y_p = mirror(y as i32 + 1, h_i);
        for x in 0..w {
            let x_m = mirror(x as i32 - 1, w_i);
            let x_p = mirror(x as i32 + 1, w_i);
            let val = v[(y, x)];
            out[(y_p, x)] += val;
            out[(y_m, x)] += val;
            out[(y, x_p)] += val;
            out[(y, x_m)] += val;
            out[(y, x)] -= 4.0 * val;
        }
    }
    out
}

/// Biharmonic operator computed as the iterated Laplacian: `Δ²z = Δ(Δz)`.
///
/// Applied with Neumann boundary handling on both passes, so the natural
/// free-edge condition `∂(Δz)/∂n = 0` is enforced automatically.
#[must_use]
pub fn biharmonic(z: ArrayView2<f32>) -> Array2<f32> {
    let lap = laplacian(z);
    laplacian(lap.view())
}

/// Central-difference divergence of a 2D vector field `F = (Fx, Fy)`.
///
/// # Panics
///
/// Panics if `fx` and `fy` have differing shapes.
#[must_use]
pub fn divergence(fx: ArrayView2<f32>, fy: ArrayView2<f32>) -> Array2<f32> {
    assert_eq!(fx.dim(), fy.dim(), "Fx and Fy must share shape");
    let (h, w) = fx.dim();
    let h_i = h as i32;
    let w_i = w as i32;

    let mut out = Array2::<f32>::zeros((h, w));
    let row_len = w;
    let out_slice = out.as_slice_mut().expect("contiguous");
    out_slice.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        let y_m = mirror(y as i32 - 1, h_i);
        let y_p = mirror(y as i32 + 1, h_i);
        for (x, dst) in row.iter_mut().enumerate() {
            let x_m = mirror(x as i32 - 1, w_i);
            let x_p = mirror(x as i32 + 1, w_i);
            let dfx = (fx[(y, x_p)] - fx[(y, x_m)]) * 0.5;
            let dfy = (fy[(y_p, x)] - fy[(y_m, x)]) * 0.5;
            *dst = dfx + dfy;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Iterate over interior pixels — the outer ring is reached via mirror
    /// boundaries which trade exactness for symmetry, so analytic identities
    /// are checked away from the edge.
    fn interior_iter(arr: &Array2<f32>) -> impl Iterator<Item = ((usize, usize), f32)> + '_ {
        let (h, w) = arr.dim();
        arr.indexed_iter()
            .filter(move |((y, x), _)| *y > 0 && *y + 1 < h && *x > 0 && *x + 1 < w)
            .map(|((y, x), v)| ((y, x), *v))
    }

    #[test]
    fn laplacian_of_constant_is_zero() {
        let arr = Array2::from_elem((8, 10), 0.7_f32);
        let lap = laplacian(arr.view());
        for v in lap.iter() {
            assert!(approx_eq(*v, 0.0, 1e-6));
        }
    }

    #[test]
    fn laplacian_of_linear_ramp_is_zero() {
        // Δ(a·x + b·y + c) = 0 everywhere on the interior.
        let mut arr = Array2::<f32>::zeros((9, 9));
        for ((y, x), v) in arr.indexed_iter_mut() {
            *v = 2.0 * (x as f32) - 3.0 * (y as f32) + 7.0;
        }
        let lap = laplacian(arr.view());
        for ((_y, _x), v) in interior_iter(&lap) {
            assert!(approx_eq(v, 0.0, 1e-5));
        }
    }

    #[test]
    fn laplacian_of_x_squared() {
        // Δ(x²) = 2; for a pure x² field, Δ(x²) = 2 by the 5-point stencil
        // exactly: z_(i, j±1) − 2·z_(i, j) = ((x+1)² + (x−1)² − 2x²) = 2.
        let mut arr = Array2::<f32>::zeros((9, 9));
        for ((_y, x), v) in arr.indexed_iter_mut() {
            let xf = x as f32;
            *v = xf * xf;
        }
        let lap = laplacian(arr.view());
        for ((_y, _x), v) in interior_iter(&lap) {
            assert!(approx_eq(v, 2.0, 1e-4));
        }
    }

    #[test]
    fn laplacian_of_y_squared() {
        let mut arr = Array2::<f32>::zeros((9, 9));
        for ((y, _x), v) in arr.indexed_iter_mut() {
            let yf = y as f32;
            *v = yf * yf;
        }
        let lap = laplacian(arr.view());
        for ((_y, _x), v) in interior_iter(&lap) {
            assert!(approx_eq(v, 2.0, 1e-4));
        }
    }

    #[test]
    fn laplacian_of_xy_is_zero() {
        // Δ(x·y) = 0.
        let mut arr = Array2::<f32>::zeros((9, 9));
        for ((y, x), v) in arr.indexed_iter_mut() {
            *v = (x as f32) * (y as f32);
        }
        let lap = laplacian(arr.view());
        for ((_y, _x), v) in interior_iter(&lap) {
            assert!(approx_eq(v, 0.0, 1e-5));
        }
    }

    #[test]
    fn biharmonic_of_quadratic_is_zero() {
        // Δ²(x² + y²) = Δ(2 + 2) = 0.
        let mut arr = Array2::<f32>::zeros((11, 11));
        for ((y, x), v) in arr.indexed_iter_mut() {
            *v = (x as f32).powi(2) + (y as f32).powi(2);
        }
        let bilap = biharmonic(arr.view());
        // Iterated Laplacian needs a 2-pixel margin for the analytic identity.
        let (h, w) = bilap.dim();
        for ((y, x), &v) in bilap.indexed_iter() {
            if y > 1 && y + 2 < h && x > 1 && x + 2 < w {
                assert!(approx_eq(v, 0.0, 1e-3), "Δ²(x²+y²) at ({y}, {x}) = {v}");
            }
        }
    }

    #[test]
    fn biharmonic_of_linear_ramp_is_zero() {
        let mut arr = Array2::<f32>::zeros((11, 11));
        for ((y, x), v) in arr.indexed_iter_mut() {
            *v = 3.0 * (x as f32) - 2.0 * (y as f32) + 1.0;
        }
        let bilap = biharmonic(arr.view());
        let (h, w) = bilap.dim();
        for ((y, x), &v) in bilap.indexed_iter() {
            if y > 1 && y + 2 < h && x > 1 && x + 2 < w {
                assert!(approx_eq(v, 0.0, 1e-4));
            }
        }
    }

    #[test]
    fn biharmonic_of_quartic() {
        // Δ²(x⁴) = Δ(12 x²) = 24 on the interior.
        let mut arr = Array2::<f32>::zeros((13, 13));
        for ((_y, x), v) in arr.indexed_iter_mut() {
            let xf = x as f32;
            *v = xf.powi(4);
        }
        let bilap = biharmonic(arr.view());
        let (h, w) = bilap.dim();
        for ((y, x), &v) in bilap.indexed_iter() {
            if y > 1 && y + 2 < h && x > 1 && x + 2 < w {
                // Discrete biharmonic of x⁴ matches the continuous result up
                // to a small finite-difference truncation error: discrete
                // Δ²(x⁴) at integer pixels equals 24 exactly because
                // x⁴ has no terms above degree 4 and the iterated 5-point
                // stencil is exact on polynomials of degree ≤ 4 in x alone.
                assert!(approx_eq(v, 24.0, 1e-2), "Δ²(x⁴) at ({y}, {x}) = {v}");
            }
        }
    }

    /// Deterministic pseudo-random field (LCG) for adjointness checks.
    fn pseudo_random(shape: (usize, usize), seed: u32) -> Array2<f32> {
        let mut state = seed;
        Array2::from_shape_fn(shape, |_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
    }

    #[test]
    fn laplacian_adjoint_satisfies_inner_product_identity() {
        // ⟨Δu, v⟩ = ⟨u, Δᵀv⟩ must hold on the full grid, boundary included.
        let u = pseudo_random((9, 7), 1);
        let v = pseudo_random((9, 7), 2);
        let lhs: f32 = laplacian(u.view()).iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let rhs: f32 = u.iter().zip(laplacian_adjoint(v.view()).iter()).map(|(a, b)| a * b).sum();
        assert!(approx_eq(lhs, rhs, 1e-4 * lhs.abs().max(1.0)), "⟨Δu,v⟩={lhs} vs ⟨u,Δᵀv⟩={rhs}");
    }

    #[test]
    fn laplacian_adjoint_equals_laplacian_on_interior() {
        // Interior stencil is symmetric, so Δᵀv = Δv two pixels away from the edge.
        let v = pseudo_random((11, 11), 3);
        let a = laplacian(v.view());
        let b = laplacian_adjoint(v.view());
        for y in 2..9 {
            for x in 2..9 {
                assert!(approx_eq(a[(y, x)], b[(y, x)], 1e-5));
            }
        }
    }

    #[test]
    fn divergence_of_constant_field_is_zero() {
        let fx = Array2::from_elem((6, 6), 0.5_f32);
        let fy = Array2::from_elem((6, 6), -0.3_f32);
        let div = divergence(fx.view(), fy.view());
        for ((_y, _x), v) in interior_iter(&div) {
            assert!(approx_eq(v, 0.0, 1e-6));
        }
    }

    #[test]
    fn divergence_of_x_id() {
        // F = (x, 0) ⇒ div F = ∂x/∂x = 1.
        let mut fx = Array2::<f32>::zeros((9, 9));
        for ((_y, x), v) in fx.indexed_iter_mut() {
            *v = x as f32;
        }
        let fy = Array2::<f32>::zeros((9, 9));
        let div = divergence(fx.view(), fy.view());
        for ((_y, _x), v) in interior_iter(&div) {
            assert!(approx_eq(v, 1.0, 1e-5));
        }
    }

    #[test]
    fn divergence_of_xy_id() {
        // F = (x, y) ⇒ div F = 1 + 1 = 2.
        let mut fx = Array2::<f32>::zeros((9, 9));
        let mut fy = Array2::<f32>::zeros((9, 9));
        for ((y, x), v) in fx.indexed_iter_mut() {
            *v = x as f32;
            fy[(y, x)] = y as f32;
        }
        let div = divergence(fx.view(), fy.view());
        for ((_y, _x), v) in interior_iter(&div) {
            assert!(approx_eq(v, 2.0, 1e-5));
        }
    }

    #[test]
    fn divergence_of_rotational_field_is_zero() {
        // F = (-y, x) ⇒ div F = ∂(-y)/∂x + ∂x/∂y = 0.
        let mut fx = Array2::<f32>::zeros((9, 9));
        let mut fy = Array2::<f32>::zeros((9, 9));
        for ((y, x), v) in fx.indexed_iter_mut() {
            *v = -(y as f32);
            fy[(y, x)] = x as f32;
        }
        let div = divergence(fx.view(), fy.view());
        for ((_y, _x), v) in interior_iter(&div) {
            assert!(approx_eq(v, 0.0, 1e-5));
        }
    }
}
