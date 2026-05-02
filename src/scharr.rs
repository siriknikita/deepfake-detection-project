//! Phase 2 — Scharr gradient operators.
//!
//! Implements the rotationally symmetric 3×3 first-derivative kernels
//!
//!   K_x = (1/32) [[-3, 0, 3],
//!                 [-10, 0, 10],
//!                 [-3, 0, 3]]
//!
//!   K_y = (1/32) [[-3, -10, -3],
//!                 [ 0,   0,  0],
//!                 [ 3,  10,  3]]
//!
//! and exposes them through `scharr_gradients(I) -> (Ix, Iy)`. The 1/32
//! prefactor is the calibrated normalization that makes a unit-slope ramp
//! produce a unit gradient, equivalent to the general form
//! `K(a, b) = 1/(4a + 2b)` evaluated at the Scharr coefficients (a, b) = (3, 10).
//! Boundary handling is Neumann (mirror), consistent with the rest of the
//! pipeline.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::util::mirror;

/// Common normalization factor for both kernels: 1/(4a + 2b) at (a, b) = (3, 10).
const SCHARR_NORM: f32 = 1.0 / 32.0;
/// Outer-row coefficient.
const A: f32 = 3.0;
/// Center-row coefficient.
const B: f32 = 10.0;

/// Compute the Scharr gradient of a 2D field, returning `(Ix, Iy)`.
///
/// The two output arrays have the same shape as the input. Mirror boundary
/// conditions are applied so the result is well-defined on the entire domain.
#[must_use]
pub fn scharr_gradients(input: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
    let (h, w) = input.dim();
    let mut ix = Array2::<f32>::zeros((h, w));
    let mut iy = Array2::<f32>::zeros((h, w));

    let h_i = h as i32;
    let w_i = w as i32;

    let row_len = w;
    let ix_slice = ix.as_slice_mut().expect("contiguous output");
    let iy_slice = iy.as_slice_mut().expect("contiguous output");

    ix_slice.par_chunks_mut(row_len).zip(iy_slice.par_chunks_mut(row_len)).enumerate().for_each(
        |(y, (ix_row, iy_row))| {
            let y_m = mirror(y as i32 - 1, h_i);
            let y_p = mirror(y as i32 + 1, h_i);

            for x in 0..w {
                let x_m = mirror(x as i32 - 1, w_i);
                let x_p = mirror(x as i32 + 1, w_i);

                // Read 3×3 neighborhood.
                let p_mm = input[(y_m, x_m)];
                let p_mc = input[(y_m, x)];
                let p_mp = input[(y_m, x_p)];
                let p_cm = input[(y, x_m)];
                // p_cc is the center pixel; not needed for Scharr (zero column / row).
                let p_cp = input[(y, x_p)];
                let p_pm = input[(y_p, x_m)];
                let p_pc = input[(y_p, x)];
                let p_pp = input[(y_p, x_p)];

                let gx = A * (p_mp - p_mm) + B * (p_cp - p_cm) + A * (p_pp - p_pm);
                let gy = A * (p_pm - p_mm) + B * (p_pc - p_mc) + A * (p_pp - p_mp);

                ix_row[x] = SCHARR_NORM * gx;
                iy_row[x] = SCHARR_NORM * gy;
            }
        },
    );

    (ix, iy)
}

/// Pixel-wise gradient magnitude `|∇I| = √(I_x² + I_y²)`.
#[must_use]
pub fn gradient_magnitude(ix: ArrayView2<f32>, iy: ArrayView2<f32>) -> Array2<f32> {
    assert_eq!(ix.dim(), iy.dim(), "Ix and Iy must have matching shape");
    let mut mag = Array2::<f32>::zeros(ix.dim());
    let mag_slice = mag.as_slice_mut().expect("contiguous output");
    let row_len = ix.dim().1;
    mag_slice.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        for (x, dst) in row.iter_mut().enumerate() {
            *dst = ix[(y, x)].hypot(iy[(y, x)]);
        }
    });
    mag
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Crop the 1-pixel boundary of an array view; gradient kernels reach into
    /// the boundary via mirroring, which trades exactness for symmetry — most
    /// analytic checks below ignore the outermost ring of pixels.
    fn interior_iter(ix: &Array2<f32>) -> impl Iterator<Item = ((usize, usize), f32)> + '_ {
        let (h, w) = ix.dim();
        ix.indexed_iter()
            .filter(move |((y, x), _)| *y > 0 && *y + 1 < h && *x > 0 && *x + 1 < w)
            .map(|((y, x), v)| ((y, x), *v))
    }

    #[test]
    fn gradient_of_constant_is_zero() {
        let arr = Array2::from_elem((10, 12), 0.7_f32);
        let (ix, iy) = scharr_gradients(arr.view());
        for v in ix.iter() {
            assert!(approx_eq(*v, 0.0, 1e-6), "Ix should be 0, got {v}");
        }
        for v in iy.iter() {
            assert!(approx_eq(*v, 0.0, 1e-6), "Iy should be 0, got {v}");
        }
    }

    #[test]
    fn gradient_of_x_ramp() {
        // f(y, x) = x ⇒ ∂f/∂x = 1, ∂f/∂y = 0.
        let mut arr = Array2::<f32>::zeros((9, 9));
        for ((_, x), v) in arr.indexed_iter_mut() {
            *v = x as f32;
        }
        let (ix, iy) = scharr_gradients(arr.view());
        for ((_y, _x), v) in interior_iter(&ix) {
            assert!(approx_eq(v, 1.0, 1e-5), "Ix should be 1 on a unit x-ramp, got {v}");
        }
        for ((_y, _x), v) in interior_iter(&iy) {
            assert!(approx_eq(v, 0.0, 1e-5), "Iy should be 0 on a unit x-ramp, got {v}");
        }
    }

    #[test]
    fn gradient_of_y_ramp() {
        // f(y, x) = y ⇒ ∂f/∂x = 0, ∂f/∂y = 1.
        let mut arr = Array2::<f32>::zeros((9, 9));
        for ((y, _), v) in arr.indexed_iter_mut() {
            *v = y as f32;
        }
        let (ix, iy) = scharr_gradients(arr.view());
        for ((_y, _x), v) in interior_iter(&ix) {
            assert!(approx_eq(v, 0.0, 1e-5), "Ix should be 0 on a unit y-ramp, got {v}");
        }
        for ((_y, _x), v) in interior_iter(&iy) {
            assert!(approx_eq(v, 1.0, 1e-5), "Iy should be 1 on a unit y-ramp, got {v}");
        }
    }

    #[test]
    fn gradient_of_diagonal_ramp() {
        // f(y, x) = x + y ⇒ Ix = Iy = 1.
        let mut arr = Array2::<f32>::zeros((9, 9));
        for ((y, x), v) in arr.indexed_iter_mut() {
            *v = (x + y) as f32;
        }
        let (ix, iy) = scharr_gradients(arr.view());
        for ((_y, _x), v) in interior_iter(&ix) {
            assert!(approx_eq(v, 1.0, 1e-5), "Ix should be 1 on diag ramp, got {v}");
        }
        for ((_y, _x), v) in interior_iter(&iy) {
            assert!(approx_eq(v, 1.0, 1e-5), "Iy should be 1 on diag ramp, got {v}");
        }
    }

    #[test]
    fn gradient_of_scaled_ramp() {
        // f(y, x) = 3.5 · x ⇒ Ix = 3.5.
        let mut arr = Array2::<f32>::zeros((9, 9));
        for ((_, x), v) in arr.indexed_iter_mut() {
            *v = 3.5 * x as f32;
        }
        let (ix, _iy) = scharr_gradients(arr.view());
        for ((_y, _x), v) in interior_iter(&ix) {
            assert!(approx_eq(v, 3.5, 1e-5), "Ix should be 3.5, got {v}");
        }
    }

    #[test]
    fn kernel_normalization() {
        // The full-row L¹ norm of K_x (positive + negative weights of equal
        // magnitude) is 2·(3 + 10 + 3) = 32; the 1/32 prefactor makes a
        // unit-slope ramp produce a unit gradient.
        let half_row_l1 = A + B + A;
        let full_row_l1 = 2.0 * half_row_l1;
        assert!(approx_eq(SCHARR_NORM * full_row_l1, 1.0, 1e-9));
    }

    #[test]
    fn x_kernel_antisymmetric_gives_zero_on_constant() {
        // The Kx kernel sums to zero (DC-blind property).
        let kx_sum = -A + 0.0 + A + (-B) + 0.0 + B + (-A) + 0.0 + A;
        assert!(approx_eq(kx_sum, 0.0, 1e-9));
    }

    #[test]
    fn y_kernel_antisymmetric_gives_zero_on_constant() {
        let ky_sum = -A + (-B) + (-A) + 0.0 + 0.0 + 0.0 + A + B + A;
        assert!(approx_eq(ky_sum, 0.0, 1e-9));
    }

    #[test]
    fn gradient_magnitude_pythagorean() {
        let ix = Array2::from_elem((4, 4), 3.0_f32);
        let iy = Array2::from_elem((4, 4), 4.0_f32);
        let mag = gradient_magnitude(ix.view(), iy.view());
        for v in mag.iter() {
            assert!(approx_eq(*v, 5.0, 1e-6), "|(3, 4)| should be 5, got {v}");
        }
    }

    #[test]
    fn gradient_at_isotropic_angles() {
        // Rotational-symmetry sanity: gradient magnitude is roughly invariant
        // under rotation of the input by 45°. Build a unit-slope ramp aligned
        // with the +x axis and another aligned with the +x+y diagonal; the
        // magnitudes on the interior should agree to high precision.
        let mut axis = Array2::<f32>::zeros((11, 11));
        let mut diag = Array2::<f32>::zeros((11, 11));
        for ((y, x), v) in axis.indexed_iter_mut() {
            *v = x as f32;
            let _ = y;
        }
        for ((y, x), v) in diag.indexed_iter_mut() {
            *v = (x + y) as f32 / std::f32::consts::SQRT_2;
        }
        let (ax_x, ax_y) = scharr_gradients(axis.view());
        let (dx, dy) = scharr_gradients(diag.view());
        let mag_axis = gradient_magnitude(ax_x.view(), ax_y.view());
        let mag_diag = gradient_magnitude(dx.view(), dy.view());
        // Compare interior pixels only.
        for y in 1..10 {
            for x in 1..10 {
                let ma = mag_axis[(y, x)];
                let md = mag_diag[(y, x)];
                assert!(
                    approx_eq(ma, md, 5.0e-3),
                    "axis vs diagonal ramp mag: {ma} vs {md} at ({y}, {x})"
                );
            }
        }
    }
}
