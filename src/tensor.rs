//! Phase 2 (continued) — Structural tensor, eigenvalue decomposition, and the
//! edge / corner / flat classifier.
//!
//! Given the Phase 2 gradient field (Ix, Iy), this module forms the
//! per-pixel structural tensor
//!
//!   J = [[<Ix²>, <Ix·Iy>],
//!        [<Ix·Iy>, <Iy²>]]
//!
//! averaged over a box window of radius `r_J`, computes the closed-form
//! 2×2 eigenvalues λ1 ≥ λ2 ≥ 0, and classifies every pixel as flat,
//! edge, or corner using the two thresholds (τ_flat, τ_edge).

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::util::mirror;

/// Pixel classification label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PixelClass {
    Flat = 0,
    Edge = 1,
    Corner = 2,
}

impl PixelClass {
    /// Decode the `u8` value used in the classification array.
    #[must_use]
    pub const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Edge,
            2 => Self::Corner,
            _ => Self::Flat,
        }
    }
}

/// Compute the three independent entries of the structural tensor at every pixel.
///
/// Each output entry is the box-window average of the corresponding gradient
/// outer-product term:
///   `Jxx = <Ix²>`, `Jxy = <Ix·Iy>`, `Jyy = <Iy²>`.
/// Mirror boundary handling is used so the averages are well-defined on the
/// entire domain.
///
/// # Panics
///
/// Panics if `ix` and `iy` have differing shapes.
#[must_use]
pub fn structural_tensor(
    ix: ArrayView2<f32>,
    iy: ArrayView2<f32>,
    window_radius: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    assert_eq!(ix.dim(), iy.dim(), "Ix and Iy must have matching shapes");
    let (h, w) = ix.dim();

    let mut ixx = Array2::<f32>::zeros((h, w));
    let mut ixy = Array2::<f32>::zeros((h, w));
    let mut iyy = Array2::<f32>::zeros((h, w));
    for ((y, x), v) in ixx.indexed_iter_mut() {
        let gx = ix[(y, x)];
        let gy = iy[(y, x)];
        *v = gx * gx;
        ixy[(y, x)] = gx * gy;
        iyy[(y, x)] = gy * gy;
    }

    let jxx = box_average(ixx.view(), window_radius);
    let jxy = box_average(ixy.view(), window_radius);
    let jyy = box_average(iyy.view(), window_radius);
    (jxx, jxy, jyy)
}

/// Closed-form 2×2 eigenvalues of the symmetric matrix at every pixel.
///
/// Returns `(lambda1, lambda2)` with `lambda1 ≥ lambda2 ≥ 0`. Small negative
/// values from floating-point cancellation are clamped to zero since `J` is
/// positive semidefinite by construction.
///
/// # Panics
///
/// Panics if the three input shapes do not all match.
#[must_use]
pub fn eigenvalues_2x2(
    jxx: ArrayView2<f32>,
    jxy: ArrayView2<f32>,
    jyy: ArrayView2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    assert_eq!(jxx.dim(), jxy.dim());
    assert_eq!(jxx.dim(), jyy.dim());
    let (h, w) = jxx.dim();

    let mut l1 = Array2::<f32>::zeros((h, w));
    let mut l2 = Array2::<f32>::zeros((h, w));

    let row_len = w;
    let l1_slice = l1.as_slice_mut().expect("contiguous");
    let l2_slice = l2.as_slice_mut().expect("contiguous");

    l1_slice.par_chunks_mut(row_len).zip(l2_slice.par_chunks_mut(row_len)).enumerate().for_each(
        |(y, (r1, r2))| {
            for x in 0..w {
                let a = jxx[(y, x)];
                let b = jxy[(y, x)];
                let d = jyy[(y, x)];
                let trace = a + d;
                // Discriminant of the 2x2 characteristic polynomial,
                // factored to avoid the small a·d − b·b cancellation
                // (clippy::suspicious_arithmetic_impl flags that form).
                let half_diff = (a - d) * 0.5;
                let disc = half_diff.mul_add(half_diff, b * b).max(0.0);
                let s = disc.sqrt();
                let half_t = trace * 0.5;
                r1[x] = half_t + s;
                r2[x] = (half_t - s).max(0.0); // J is PSD; clamp FP noise.
            }
        },
    );

    (l1, l2)
}

/// Classify every pixel as Flat / Edge / Corner from the eigenvalue pair
/// `(lambda1, lambda2)` and the two thresholds.
///
/// The labels are:
/// - `Flat`   if `lambda1 < tau_flat`
/// - `Edge`   if `lambda1 ≥ tau_flat` and `lambda2 < tau_edge`
/// - `Corner` if `lambda2 ≥ tau_edge`
///
/// Returns an `H×W` array of `u8` codes (`0=Flat, 1=Edge, 2=Corner`).
#[must_use]
pub fn classify(
    lambda1: ArrayView2<f32>,
    lambda2: ArrayView2<f32>,
    tau_flat: f32,
    tau_edge: f32,
) -> Array2<u8> {
    assert_eq!(lambda1.dim(), lambda2.dim());
    let (h, w) = lambda1.dim();
    let mut out = Array2::<u8>::zeros((h, w));
    for ((y, x), v) in out.indexed_iter_mut() {
        let l1 = lambda1[(y, x)];
        let l2 = lambda2[(y, x)];
        *v = if l2 >= tau_edge {
            PixelClass::Corner as u8
        } else if l1 >= tau_flat {
            PixelClass::Edge as u8
        } else {
            PixelClass::Flat as u8
        };
    }
    out
}

/// Pick adaptive thresholds from empirical eigenvalue distributions.
///
/// Returns `(tau_flat, tau_edge) = (percentile(λ1, p_flat),
/// percentile(λ2, p_edge))`. Defaults `p_flat = 0.30`, `p_edge = 0.70`
/// follow the recommendation in the paper.
///
/// # Panics
///
/// Panics if either percentile is outside `(0, 1)` or if either eigenvalue
/// array is empty.
#[must_use]
pub fn adaptive_thresholds(
    lambda1: ArrayView2<f32>,
    lambda2: ArrayView2<f32>,
    p_flat: f32,
    p_edge: f32,
) -> (f32, f32) {
    assert!((0.0..=1.0).contains(&p_flat), "p_flat must be in [0, 1], got {p_flat}");
    assert!((0.0..=1.0).contains(&p_edge), "p_edge must be in [0, 1], got {p_edge}");
    let tau_flat = percentile(lambda1, p_flat);
    let tau_edge = percentile(lambda2, p_edge);
    (tau_flat, tau_edge)
}

/// Extract keypoint coordinates `(y, x)` from the classification array — the
/// pixels classified as `Edge` or `Corner`. Pixels classified as `Flat` are
/// excluded.
#[must_use]
pub fn keypoints(classes: ArrayView2<u8>) -> Vec<(usize, usize)> {
    classes
        .indexed_iter()
        .filter_map(|((y, x), &c)| if c >= PixelClass::Edge as u8 { Some((y, x)) } else { None })
        .collect()
}

// ---------------- internal helpers ----------------

fn box_average(input: ArrayView2<f32>, radius: usize) -> Array2<f32> {
    let (h, w) = input.dim();
    if radius == 0 {
        return input.to_owned();
    }
    let area = ((2 * radius + 1) * (2 * radius + 1)) as f32;
    let h_i = h as i32;
    let w_i = w as i32;
    let r_i = radius as i32;

    let mut out = Array2::<f32>::zeros((h, w));
    let row_len = w;
    let out_slice = out.as_slice_mut().expect("contiguous");
    out_slice.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        for (x, dst) in row.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for dy in -r_i..=r_i {
                let yi = mirror(y as i32 + dy, h_i);
                for dx in -r_i..=r_i {
                    let xi = mirror(x as i32 + dx, w_i);
                    sum += input[(yi, xi)];
                }
            }
            *dst = sum / area;
        }
    });
    out
}

fn percentile(arr: ArrayView2<f32>, p: f32) -> f32 {
    assert!(!arr.is_empty(), "percentile of empty array is undefined");
    let mut buf: Vec<f32> = arr.iter().copied().collect();
    buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = buf.len();
    let idx = ((p * (n - 1) as f32).round() as usize).min(n - 1);
    buf[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn structural_tensor_zero_on_zero_gradient() {
        let zero = Array2::<f32>::zeros((6, 6));
        let (jxx, jxy, jyy) = structural_tensor(zero.view(), zero.view(), 1);
        for v in jxx.iter().chain(jxy.iter()).chain(jyy.iter()) {
            assert!(approx_eq(*v, 0.0, 1e-9));
        }
    }

    #[test]
    fn structural_tensor_pure_x_gradient() {
        // Constant Ix = 1, Iy = 0 ⇒ Jxx = 1, Jxy = 0, Jyy = 0.
        let ix = Array2::from_elem((8, 8), 1.0_f32);
        let iy = Array2::<f32>::zeros((8, 8));
        let (jxx, jxy, jyy) = structural_tensor(ix.view(), iy.view(), 1);
        for v in jxx.iter() {
            assert!(approx_eq(*v, 1.0, 1e-6));
        }
        for v in jxy.iter() {
            assert!(approx_eq(*v, 0.0, 1e-6));
        }
        for v in jyy.iter() {
            assert!(approx_eq(*v, 0.0, 1e-6));
        }
    }

    #[test]
    fn structural_tensor_diagonal_gradient() {
        // Ix = 1, Iy = 1 ⇒ Jxx = Jyy = 1, Jxy = 1.
        let ix = Array2::from_elem((6, 6), 1.0_f32);
        let iy = Array2::from_elem((6, 6), 1.0_f32);
        let (jxx, jxy, jyy) = structural_tensor(ix.view(), iy.view(), 0);
        for v in jxx.iter().chain(jxy.iter()).chain(jyy.iter()) {
            assert!(approx_eq(*v, 1.0, 1e-6));
        }
    }

    #[test]
    fn eigenvalues_of_zero_tensor_are_zero() {
        let zero = Array2::<f32>::zeros((4, 4));
        let (l1, l2) = eigenvalues_2x2(zero.view(), zero.view(), zero.view());
        for v in l1.iter().chain(l2.iter()) {
            assert!(approx_eq(*v, 0.0, 1e-9));
        }
    }

    #[test]
    fn eigenvalues_of_diag_tensor() {
        // J = diag(a, d) ⇒ eigenvalues are a and d.
        let jxx = Array2::from_elem((3, 3), 4.0_f32);
        let jxy = Array2::<f32>::zeros((3, 3));
        let jyy = Array2::from_elem((3, 3), 1.0_f32);
        let (l1, l2) = eigenvalues_2x2(jxx.view(), jxy.view(), jyy.view());
        for v in l1.iter() {
            assert!(approx_eq(*v, 4.0, 1e-6));
        }
        for v in l2.iter() {
            assert!(approx_eq(*v, 1.0, 1e-6));
        }
    }

    #[test]
    fn eigenvalues_of_isotropic_tensor() {
        // J = I (identity) ⇒ both eigenvalues are 1.
        let one = Array2::from_elem((3, 3), 1.0_f32);
        let zero = Array2::<f32>::zeros((3, 3));
        let (l1, l2) = eigenvalues_2x2(one.view(), zero.view(), one.view());
        for v in l1.iter().chain(l2.iter()) {
            assert!(approx_eq(*v, 1.0, 1e-6));
        }
    }

    #[test]
    fn eigenvalues_ordered() {
        // Random-ish PSD matrix per pixel; verify λ1 ≥ λ2 ≥ 0 throughout.
        let mut jxx = Array2::<f32>::zeros((5, 5));
        let mut jxy = Array2::<f32>::zeros((5, 5));
        let mut jyy = Array2::<f32>::zeros((5, 5));
        for ((y, x), v) in jxx.indexed_iter_mut() {
            let f = (y as f32) + 0.5 * (x as f32);
            *v = f * f;
            jxy[(y, x)] = 0.3 * f;
            jyy[(y, x)] = (y as f32 + 1.0) * 0.5;
        }
        let (l1, l2) = eigenvalues_2x2(jxx.view(), jxy.view(), jyy.view());
        for ((y, x), &a) in l1.indexed_iter() {
            let b = l2[(y, x)];
            assert!(a >= b, "λ1 ({a}) must be ≥ λ2 ({b}) at ({y}, {x})");
            assert!(b >= -1e-6, "λ2 must be non-negative, got {b}");
        }
    }

    #[test]
    fn eigenvalues_recover_off_diagonal() {
        // For J = [[0, 1], [1, 0]] the eigenvalues are ±1; J is not PSD here,
        // but the closed-form formula (trace/2 ± √(disc)) still applies and
        // tests our discriminant formula. PSD clamp is used downstream; here
        // we feed a non-PSD J on purpose and check the unclamped path with
        // the trace shifted to make eigenvalues positive: J' = J + 2 I has
        // eigenvalues 1, 3.
        let jxx = Array2::from_elem((2, 2), 2.0_f32);
        let jxy = Array2::from_elem((2, 2), 1.0_f32);
        let jyy = Array2::from_elem((2, 2), 2.0_f32);
        let (l1, l2) = eigenvalues_2x2(jxx.view(), jxy.view(), jyy.view());
        for v in l1.iter() {
            assert!(approx_eq(*v, 3.0, 1e-6));
        }
        for v in l2.iter() {
            assert!(approx_eq(*v, 1.0, 1e-6));
        }
    }

    #[test]
    fn classify_thresholds() {
        let l1 = Array2::from_shape_vec((2, 2), vec![0.0_f32, 0.5, 1.0, 2.0]).unwrap();
        let l2 = Array2::from_shape_vec((2, 2), vec![0.0_f32, 0.0, 0.2, 0.9]).unwrap();
        let classes = classify(l1.view(), l2.view(), 0.4, 0.5);
        // (l1, l2) = (0, 0)   -> flat
        // (l1, l2) = (0.5, 0) -> edge (l1 ≥ τ_flat=0.4, l2 < τ_edge=0.5)
        // (l1, l2) = (1, 0.2) -> edge
        // (l1, l2) = (2, 0.9) -> corner
        assert_eq!(classes[(0, 0)], PixelClass::Flat as u8);
        assert_eq!(classes[(0, 1)], PixelClass::Edge as u8);
        assert_eq!(classes[(1, 0)], PixelClass::Edge as u8);
        assert_eq!(classes[(1, 1)], PixelClass::Corner as u8);
    }

    #[test]
    fn keypoints_excludes_flat() {
        let classes = Array2::from_shape_vec((2, 3), vec![0u8, 1, 2, 1, 0, 2]).unwrap();
        let kps = keypoints(classes.view());
        // Keypoints should be at all (y, x) where the value is non-zero.
        let expected = vec![(0, 1), (0, 2), (1, 0), (1, 2)];
        assert_eq!(kps, expected);
    }

    #[test]
    fn box_average_preserves_constant() {
        let arr = Array2::from_elem((10, 10), 3.0_f32);
        let blurred = box_average(arr.view(), 2);
        for v in blurred.iter() {
            assert!(approx_eq(*v, 3.0, 1e-6));
        }
    }

    #[test]
    fn box_average_radius_zero_is_identity() {
        let mut arr = Array2::<f32>::zeros((4, 4));
        for ((y, x), v) in arr.indexed_iter_mut() {
            *v = (y * 4 + x) as f32;
        }
        let same = box_average(arr.view(), 0);
        for ((y, x), v) in arr.indexed_iter() {
            assert_eq!(*v, same[(y, x)]);
        }
    }

    #[test]
    fn percentile_basic() {
        let arr = Array2::from_shape_vec((1, 5), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
        // p=0 -> min, p=1 -> max, p=0.5 -> median(=3).
        assert!(approx_eq(percentile(arr.view(), 0.0), 1.0, 1e-9));
        assert!(approx_eq(percentile(arr.view(), 1.0), 5.0, 1e-9));
        assert!(approx_eq(percentile(arr.view(), 0.5), 3.0, 1e-9));
    }

    #[test]
    fn adaptive_thresholds_in_range() {
        // For a uniform distribution, p30 of λ1 ≤ p70 of λ2 only if the data
        // are non-degenerate; here we just check that the returned thresholds
        // are finite and within the empirical range.
        let mut l1 = Array2::<f32>::zeros((6, 6));
        let mut l2 = Array2::<f32>::zeros((6, 6));
        for ((y, x), v) in l1.indexed_iter_mut() {
            *v = (y * 6 + x) as f32 * 0.1;
            l2[(y, x)] = (y * 6 + x) as f32 * 0.05;
        }
        let (tf, te) = adaptive_thresholds(l1.view(), l2.view(), 0.30, 0.70);
        assert!(tf.is_finite() && te.is_finite());
        assert!((0.0..=3.5).contains(&tf));
        assert!((0.0..=1.75).contains(&te));
    }
}
