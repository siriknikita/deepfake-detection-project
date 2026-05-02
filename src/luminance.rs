//! Phase 1 — Signal decomposition.
//!
//! Implements the chromatic linear projection
//!   I_w = w_R · R + w_G · G + w_B · B
//! and the multi-scale difference-of-Gaussians (DoG) pyramid
//!   DoG_k = I_w * G(σ · k) − I_w * G(σ).
//!
//! Boundary handling is Neumann (mirror) throughout, consistent with the
//! free-edge boundary conditions derived in Phase 5 of the paper.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use rayon::prelude::*;

/// Tolerance for the chromatic-weight unit-sum constraint.
pub const WEIGHT_SUM_TOL: f32 = 1.0e-3;

/// Minimum standard deviation accepted by the Gaussian filter.
pub const SIGMA_MIN: f32 = 1.0e-3;

/// Project an RGB image to a scalar luminance via the weighted linear combination
/// `I_w = w_R · R + w_G · G + w_B · B` with `w_R + w_G + w_B = 1`.
///
/// # Panics
///
/// Panics if the weights do not sum to 1 within `WEIGHT_SUM_TOL`, or if the
/// input does not have exactly 3 channels on the last axis.
#[must_use]
pub fn weighted_luminance(rgb: ArrayView3<f32>, w_r: f32, w_g: f32, w_b: f32) -> Array2<f32> {
    let (h, w, c) = rgb.dim();
    assert_eq!(c, 3, "weighted_luminance expects an H x W x 3 RGB image, got {c} channels");
    let s = w_r + w_g + w_b;
    assert!(
        (s - 1.0).abs() < WEIGHT_SUM_TOL,
        "luminance weights must sum to 1 within {WEIGHT_SUM_TOL}, got {s}"
    );

    let mut out = Array2::<f32>::zeros((h, w));
    let row_len = w;
    let out_slice = out.as_slice_mut().expect("Array2 is contiguous in default layout");

    out_slice.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            *v = w_r * rgb[(y, x, 0)] + w_g * rgb[(y, x, 1)] + w_b * rgb[(y, x, 2)];
        }
    });
    out
}

/// Build a normalized 1D Gaussian kernel of radius `r = ceil(3·σ)`.
fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let len = 2 * radius + 1;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut k = Vec::with_capacity(len);
    let mut sum = 0.0_f32;
    for i in 0..len {
        let x = i as f32 - radius as f32;
        let v = (-x * x / two_sigma_sq).exp();
        k.push(v);
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Mirror an out-of-range index back into `[0, n)` for Neumann boundary handling.
#[inline]
fn mirror(i: i32, n: i32) -> usize {
    debug_assert!(n > 0);
    let mut i = if i < 0 { -i } else { i };
    if i >= n {
        i = 2 * n - i - 2;
    }
    i.clamp(0, n - 1) as usize
}

/// Apply a separable 2D Gaussian blur of standard deviation `sigma` with mirror
/// boundary conditions. Performs a horizontal pass then a vertical pass, each
/// parallelized over rows / columns by `rayon`.
///
/// # Panics
///
/// Panics if `sigma < SIGMA_MIN`.
#[must_use]
pub fn gaussian_blur(input: ArrayView2<f32>, sigma: f32) -> Array2<f32> {
    assert!(sigma >= SIGMA_MIN, "sigma must be at least {SIGMA_MIN}, got {sigma}");
    let kernel = gaussian_kernel_1d(sigma);
    let radius = (kernel.len() - 1) / 2;
    let (h, w) = input.dim();

    let mut tmp = Array2::<f32>::zeros((h, w));
    convolve_horizontal(input, &kernel, radius, &mut tmp);
    let mut out = Array2::<f32>::zeros((h, w));
    convolve_vertical(tmp.view(), &kernel, radius, &mut out);
    out
}

fn convolve_horizontal(
    input: ArrayView2<f32>,
    kernel: &[f32],
    radius: usize,
    out: &mut Array2<f32>,
) {
    let (_, w) = input.dim();
    let w_i = w as i32;
    let r_i = radius as i32;

    let row_len = w;
    let out_slice = out.as_slice_mut().expect("contiguous output");

    out_slice.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        for (x, dst) in row.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let xi = mirror(x as i32 + ki as i32 - r_i, w_i);
                sum += kv * input[(y, xi)];
            }
            *dst = sum;
        }
    });
}

fn convolve_vertical(input: ArrayView2<f32>, kernel: &[f32], radius: usize, out: &mut Array2<f32>) {
    let (h, w) = input.dim();
    let h_i = h as i32;
    let r_i = radius as i32;

    let row_len = w;
    let out_slice = out.as_slice_mut().expect("contiguous output");

    out_slice.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        for (x, dst) in row.iter_mut().enumerate() {
            let mut sum = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let yi = mirror(y as i32 + ki as i32 - r_i, h_i);
                sum += kv * input[(yi, x)];
            }
            *dst = sum;
        }
    });
}

/// Difference-of-Gaussians band: `DoG_k = input * G(k·σ) − input * G(σ)`.
#[must_use]
pub fn dog_band(input: ArrayView2<f32>, sigma: f32, k_ratio: f32) -> Array2<f32> {
    assert!(k_ratio > 1.0, "k_ratio must be > 1 for a valid band-pass, got {k_ratio}");
    let outer = gaussian_blur(input, sigma * k_ratio);
    let inner = gaussian_blur(input, sigma);
    outer - inner
}

/// Stack of `n_scales` DoG bands at scales `σ · k_ratio^j`, j = 0, …, n_scales-1.
/// Returns an `H × W × n_scales` array; the trailing axis is the scale index.
///
/// # Panics
///
/// Panics if `n_scales == 0`.
#[must_use]
pub fn dog_pyramid(
    input: ArrayView2<f32>,
    sigma_base: f32,
    k_ratio: f32,
    n_scales: usize,
) -> Array3<f32> {
    assert!(n_scales > 0, "n_scales must be at least 1");
    let (h, w) = input.dim();
    let mut out = Array3::<f32>::zeros((h, w, n_scales));
    for j in 0..n_scales {
        let sigma_j = sigma_base * k_ratio.powi(j as i32);
        let band = dog_band(input, sigma_j, k_ratio);
        out.index_axis_mut(Axis(2), j).assign(&band);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr3, Array2, Array3};

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn weighted_luminance_constant_image() {
        // R=G=B=v with any unit-sum weights returns v at every pixel.
        let rgb = arr3(&[[[0.42_f32, 0.42, 0.42], [0.42, 0.42, 0.42]]]);
        let out = weighted_luminance(rgb.view(), 0.299, 0.587, 0.114);
        for v in out.iter() {
            assert!(approx_eq(*v, 0.42, 1e-6), "expected 0.42, got {v}");
        }
    }

    #[test]
    fn weighted_luminance_bt601_white() {
        // White pixel under BT.601 luma weights: 1.
        let rgb = arr3(&[[[1.0_f32, 1.0, 1.0]]]);
        let out = weighted_luminance(rgb.view(), 0.299, 0.587, 0.114);
        assert!(approx_eq(out[(0, 0)], 1.0, 1e-6));
    }

    #[test]
    fn weighted_luminance_per_channel_response() {
        // Pure-red pixel responds with exactly w_R.
        let rgb_r = arr3(&[[[1.0_f32, 0.0, 0.0]]]);
        let rgb_g = arr3(&[[[0.0_f32, 1.0, 0.0]]]);
        let rgb_b = arr3(&[[[0.0_f32, 0.0, 1.0]]]);
        let out_r = weighted_luminance(rgb_r.view(), 0.299, 0.587, 0.114);
        let out_g = weighted_luminance(rgb_g.view(), 0.299, 0.587, 0.114);
        let out_b = weighted_luminance(rgb_b.view(), 0.299, 0.587, 0.114);
        assert!(approx_eq(out_r[(0, 0)], 0.299, 1e-6));
        assert!(approx_eq(out_g[(0, 0)], 0.587, 1e-6));
        assert!(approx_eq(out_b[(0, 0)], 0.114, 1e-6));
    }

    #[test]
    #[should_panic(expected = "luminance weights must sum to 1")]
    fn weighted_luminance_rejects_non_unit_weights() {
        let rgb = arr3(&[[[0.5_f32, 0.5, 0.5]]]);
        let _ = weighted_luminance(rgb.view(), 0.5, 0.5, 0.5);
    }

    #[test]
    #[should_panic(expected = "expects an H x W x 3 RGB image")]
    fn weighted_luminance_rejects_non_rgb() {
        let bad = Array3::<f32>::zeros((2, 2, 4));
        let _ = weighted_luminance(bad.view(), 0.299, 0.587, 0.114);
    }

    #[test]
    fn gaussian_kernel_normalized() {
        for &sigma in &[0.5_f32, 1.0, 1.5, 3.0] {
            let k = gaussian_kernel_1d(sigma);
            let s: f32 = k.iter().sum();
            assert!(approx_eq(s, 1.0, 1e-5), "kernel(σ={sigma}) sum = {s}, want 1");
        }
    }

    #[test]
    fn gaussian_kernel_symmetric() {
        let k = gaussian_kernel_1d(1.5);
        let n = k.len();
        for i in 0..n / 2 {
            assert!(
                approx_eq(k[i], k[n - 1 - i], 1e-7),
                "kernel must be symmetric, k[{i}] = {} vs k[{}] = {}",
                k[i],
                n - 1 - i,
                k[n - 1 - i]
            );
        }
    }

    #[test]
    fn mirror_basic() {
        // n = 5, valid indices [0, 4].
        assert_eq!(mirror(-1, 5), 1);
        assert_eq!(mirror(-2, 5), 2);
        assert_eq!(mirror(0, 5), 0);
        assert_eq!(mirror(4, 5), 4);
        assert_eq!(mirror(5, 5), 3);
        assert_eq!(mirror(6, 5), 2);
    }

    #[test]
    fn gaussian_blur_preserves_constant() {
        // A constant field is the eigenfunction of the Gaussian operator with
        // eigenvalue 1; the blurred result must equal the input pixel-for-pixel.
        let arr = Array2::from_elem((16, 24), 0.7_f32);
        let blurred = gaussian_blur(arr.view(), 1.5);
        for v in blurred.iter() {
            assert!(approx_eq(*v, 0.7, 1e-4), "constant not preserved: got {v}");
        }
    }

    #[test]
    fn gaussian_blur_preserves_zero() {
        let arr = Array2::<f32>::zeros((8, 8));
        let blurred = gaussian_blur(arr.view(), 1.0);
        for v in blurred.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn gaussian_blur_conserves_total_mass() {
        // Mass conservation: Σ G·I = Σ I when the kernel is normalized and
        // boundary handling is mirror (sums of mirror reflections equal the
        // sums of interior values they reflect).
        let mut arr = Array2::<f32>::zeros((15, 15));
        arr[(7, 7)] = 1.0; // Kronecker delta at the center
        let blurred = gaussian_blur(arr.view(), 1.2);
        let total: f32 = blurred.iter().sum();
        assert!(approx_eq(total, 1.0, 1e-4), "mass not conserved: total = {total}");
    }

    #[test]
    fn gaussian_blur_reduces_impulse_peak() {
        let mut arr = Array2::<f32>::zeros((15, 15));
        arr[(7, 7)] = 1.0;
        let blurred = gaussian_blur(arr.view(), 1.5);
        assert!(blurred[(7, 7)] < 1.0, "peak must drop after blur");
        assert!(blurred[(7, 7)] > 0.0, "peak must remain positive");
    }

    #[test]
    fn dog_band_zero_on_constant() {
        // DoG of a constant field is identically zero — the constant is
        // annihilated by both Gaussians equally and cancels in the difference.
        let arr = Array2::from_elem((16, 16), 0.5_f32);
        let dog = dog_band(arr.view(), 1.0, std::f32::consts::SQRT_2);
        for v in dog.iter() {
            assert!(approx_eq(*v, 0.0, 1e-4), "DoG of constant must be zero, got {v}");
        }
    }

    #[test]
    fn dog_pyramid_correct_shape() {
        let arr = Array2::<f32>::zeros((10, 12));
        let pyr = dog_pyramid(arr.view(), 1.0, std::f32::consts::SQRT_2, 4);
        assert_eq!(pyr.dim(), (10, 12, 4));
    }

    #[test]
    fn dog_pyramid_zero_dc_per_band() {
        // Every band has approximately zero spatial mean (zero-mean property).
        let mut arr = Array2::<f32>::zeros((32, 32));
        for ((y, x), v) in arr.indexed_iter_mut() {
            *v = ((y as f32) * 0.1 + (x as f32) * 0.05).sin();
        }
        let pyr = dog_pyramid(arr.view(), 1.0, std::f32::consts::SQRT_2, 3);
        for j in 0..3 {
            let band = pyr.index_axis(Axis(2), j);
            let mean: f32 = band.iter().sum::<f32>() / (band.len() as f32);
            assert!(mean.abs() < 1e-2, "band {j} DC = {mean}, want |DC| < 1e-2");
        }
    }

    #[test]
    #[should_panic(expected = "k_ratio must be > 1")]
    fn dog_band_rejects_unit_ratio() {
        let arr = Array2::<f32>::zeros((4, 4));
        let _ = dog_band(arr.view(), 1.0, 1.0);
    }
}
