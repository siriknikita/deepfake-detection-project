//! Phase 3 — The Hyperplane Forge.
//!
//! Builds per-keypoint local linear depth models from intensity gradients
//! under a Lambertian reflectance model, then fuses overlapping models
//! across spatial neighborhoods (Min-over-neighborhood) and across DoG
//! scales (Max-over-scales) to produce the initial manifold `z_forged`.
//!
//! The full Phase 1 → 3 chain is assembled in `impact.rs`; this module
//! exposes the four kernels needed by that chain:
//! - `local_albedo` — robust per-pixel albedo estimate from `I_w`.
//! - `build_hyperplanes` — Lambertian gradient transfer for one scale.
//! - `forge_per_scale` — stamp + spatial Min for one scale.
//! - `compose_max_over_scales` — elementwise Max across scale buffers.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::util::mirror;

/// A local linear depth model anchored at one keypoint at one DoG scale.
///
/// At pixel (x, y) the hyperplane evaluates to
/// `z_i + dz_dx · (x − x_i) + dz_dy · (y − y_i)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalHyperplane {
    pub x_i: f32,
    pub y_i: f32,
    pub z_i: f32,
    pub dz_dx: f32,
    pub dz_dy: f32,
}

impl LocalHyperplane {
    /// Evaluate the hyperplane at the integer pixel `(y, x)`.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, y: usize, x: usize) -> f32 {
        let dx = x as f32 - self.x_i;
        let dy = y as f32 - self.y_i;
        self.z_i + self.dz_dx * dx + self.dz_dy * dy
    }
}

/// Robust per-pixel albedo estimate: the median of `I_w` in a square window
/// of half-side `radius`. Mirror boundary handling.
///
/// Median is preferred over mean because single-pixel noise (sensor / JPEG
/// artefacts) would otherwise feed directly into the Lambertian denominator
/// and amplify itself in the gradient transfer.
#[must_use]
pub fn local_albedo(i_w: ArrayView2<f32>, radius: usize) -> Array2<f32> {
    let (h, w) = i_w.dim();
    if radius == 0 {
        return i_w.to_owned();
    }
    let h_i = h as i32;
    let w_i = w as i32;
    let r_i = radius as i32;
    let win_size = (2 * radius + 1) * (2 * radius + 1);
    let mid = win_size / 2;

    let mut out = Array2::<f32>::zeros((h, w));
    let row_len = w;
    let out_slice = out.as_slice_mut().expect("contiguous");
    out_slice.par_chunks_mut(row_len).enumerate().for_each(|(y, row)| {
        let mut buf: Vec<f32> = Vec::with_capacity(win_size);
        for (x, dst) in row.iter_mut().enumerate() {
            buf.clear();
            for dy in -r_i..=r_i {
                let yi = mirror(y as i32 + dy, h_i);
                for dx in -r_i..=r_i {
                    let xi = mirror(x as i32 + dx, w_i);
                    buf.push(i_w[(yi, xi)]);
                }
            }
            buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            *dst = buf[mid];
        }
    });
    out
}

/// Build the hyperplane at every keypoint of one scale.
///
/// For each `(y_i, x_i)` in `keypoints` the surface gradient is recovered
/// from the intensity gradient via the linearized Lambertian model
///   `dz_dx ≈ −Ix / (ρ + ε)`,  `dz_dy ≈ −Iy / (ρ + ε)`,
/// and the depth anchor is the depth-from-intensity proxy `z_i = c_z · I_w`.
///
/// # Panics
///
/// Panics if any of `i_w`, `ix`, `iy`, `rho` have differing shapes, or if
/// any keypoint coordinate is out of bounds.
#[must_use]
pub fn build_hyperplanes(
    i_w: ArrayView2<f32>,
    ix: ArrayView2<f32>,
    iy: ArrayView2<f32>,
    rho: ArrayView2<f32>,
    keypoints: &[(usize, usize)],
    epsilon: f32,
    c_z: f32,
) -> Vec<LocalHyperplane> {
    assert_eq!(i_w.dim(), ix.dim());
    assert_eq!(i_w.dim(), iy.dim());
    assert_eq!(i_w.dim(), rho.dim());
    keypoints
        .iter()
        .map(|&(y, x)| {
            let denom = rho[(y, x)] + epsilon;
            let gx = ix[(y, x)];
            let gy = iy[(y, x)];
            LocalHyperplane {
                x_i: x as f32,
                y_i: y as f32,
                z_i: c_z * i_w[(y, x)],
                dz_dx: -gx / denom,
                dz_dy: -gy / denom,
            }
        })
        .collect()
}

/// Phase 3.6 step 1 — per-scale Min-over-neighborhood.
///
/// For every keypoint, evaluate its hyperplane on every pixel within
/// `(2 r + 1)²` of the anchor and take the elementwise minimum into a
/// shared buffer. Pixels covered by no hyperplane retain `+∞`; the caller
/// must fill them via `fill_uncovered` (or supply a fallback).
#[must_use]
pub fn forge_per_scale(
    shape: (usize, usize),
    hyperplanes: &[LocalHyperplane],
    neigh_radius: usize,
) -> Array2<f32> {
    let (h, w) = shape;
    let mut out = Array2::from_elem((h, w), f32::INFINITY);
    let r_i = neigh_radius as i32;
    let h_i = h as i32;
    let w_i = w as i32;

    for hp in hyperplanes {
        let cy = hp.y_i.round() as i32;
        let cx = hp.x_i.round() as i32;
        for dy in -r_i..=r_i {
            let yi = (cy + dy).clamp(0, h_i - 1) as usize;
            for dx in -r_i..=r_i {
                let xi = (cx + dx).clamp(0, w_i - 1) as usize;
                let val = hp.evaluate(yi, xi);
                let cell = &mut out[(yi, xi)];
                if val < *cell {
                    *cell = val;
                }
            }
        }
    }
    out
}

/// Replace `+∞` cells in `z` with `fallback[(y, x)]`. Used to backfill the
/// pixels that lay outside every keypoint's neighborhood at every scale.
pub fn fill_uncovered(z: &mut Array2<f32>, fallback: ArrayView2<f32>) {
    assert_eq!(z.dim(), fallback.dim());
    for ((y, x), v) in z.indexed_iter_mut() {
        if !v.is_finite() {
            *v = fallback[(y, x)];
        }
    }
}

/// Phase 3.6 step 2 — Max across DoG scales.
///
/// Returns the elementwise maximum of the per-scale buffers. `+∞` cells
/// are propagated and should be filled via `fill_uncovered` afterwards.
///
/// # Panics
///
/// Panics if `per_scale` is empty or if the per-scale arrays disagree on shape.
#[must_use]
pub fn compose_max_over_scales(per_scale: &[Array2<f32>]) -> Array2<f32> {
    assert!(!per_scale.is_empty(), "compose_max_over_scales requires at least one scale");
    let shape = per_scale[0].dim();
    for arr in per_scale {
        assert_eq!(arr.dim(), shape, "all scale buffers must share shape");
    }
    let mut out = Array2::from_elem(shape, f32::NEG_INFINITY);
    for arr in per_scale {
        for ((y, x), &v) in arr.indexed_iter() {
            let cell = &mut out[(y, x)];
            if v > *cell {
                *cell = v;
            }
        }
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

    #[test]
    fn hyperplane_evaluates_at_anchor() {
        let hp = LocalHyperplane { x_i: 3.0, y_i: 5.0, z_i: 1.5, dz_dx: 2.0, dz_dy: -1.0 };
        assert!(approx_eq(hp.evaluate(5, 3), 1.5, 1e-6));
    }

    #[test]
    fn hyperplane_extrapolates_linearly() {
        let hp = LocalHyperplane { x_i: 2.0, y_i: 2.0, z_i: 0.0, dz_dx: 1.0, dz_dy: 0.0 };
        // Move +2 in x ⇒ z = 0 + 1·2 = 2.
        assert!(approx_eq(hp.evaluate(2, 4), 2.0, 1e-6));
        // Move -1 in y at fixed x ⇒ z = 0 (dz_dy = 0).
        assert!(approx_eq(hp.evaluate(1, 2), 0.0, 1e-6));
    }

    #[test]
    fn local_albedo_constant_image() {
        let arr = Array2::from_elem((5, 5), 0.4_f32);
        let alb = local_albedo(arr.view(), 1);
        for v in alb.iter() {
            assert!(approx_eq(*v, 0.4, 1e-6));
        }
    }

    #[test]
    fn local_albedo_radius_zero_is_identity() {
        let mut arr = Array2::<f32>::zeros((3, 3));
        for ((y, x), v) in arr.indexed_iter_mut() {
            *v = (y * 3 + x) as f32;
        }
        let same = local_albedo(arr.view(), 0);
        for ((y, x), v) in arr.indexed_iter() {
            assert_eq!(*v, same[(y, x)]);
        }
    }

    #[test]
    fn local_albedo_robust_to_single_outlier() {
        // 5x5 of 1.0 with one extreme outlier — the median in a 3x3 window
        // around the outlier should still be 1.0.
        let mut arr = Array2::from_elem((5, 5), 1.0_f32);
        arr[(2, 2)] = 100.0;
        let alb = local_albedo(arr.view(), 1);
        assert!(approx_eq(alb[(2, 2)], 1.0, 1e-6), "median should reject the outlier");
    }

    #[test]
    fn build_hyperplanes_zero_gradient() {
        let i_w = Array2::from_elem((4, 4), 0.5_f32);
        let ix = Array2::<f32>::zeros((4, 4));
        let iy = Array2::<f32>::zeros((4, 4));
        let rho = Array2::from_elem((4, 4), 0.5_f32);
        let kps = vec![(1, 1), (2, 2)];
        let hps = build_hyperplanes(i_w.view(), ix.view(), iy.view(), rho.view(), &kps, 1e-3, 1.0);
        assert_eq!(hps.len(), 2);
        for hp in &hps {
            assert!(approx_eq(hp.dz_dx, 0.0, 1e-6));
            assert!(approx_eq(hp.dz_dy, 0.0, 1e-6));
            assert!(approx_eq(hp.z_i, 0.5, 1e-6));
        }
    }

    #[test]
    fn build_hyperplanes_lambertian_transfer() {
        // Ix = 0.5, Iy = -0.25, ρ = 0.5, ε = 0
        // ⇒ dz_dx = -0.5 / 0.5 = -1, dz_dy = -(-0.25) / 0.5 = 0.5.
        let i_w = Array2::from_elem((3, 3), 0.5_f32);
        let ix = Array2::from_elem((3, 3), 0.5_f32);
        let iy = Array2::from_elem((3, 3), -0.25_f32);
        let rho = Array2::from_elem((3, 3), 0.5_f32);
        let kps = vec![(1, 1)];
        let hps = build_hyperplanes(i_w.view(), ix.view(), iy.view(), rho.view(), &kps, 0.0, 1.0);
        let hp = hps[0];
        assert!(approx_eq(hp.dz_dx, -1.0, 1e-6));
        assert!(approx_eq(hp.dz_dy, 0.5, 1e-6));
    }

    #[test]
    fn build_hyperplanes_epsilon_prevents_singular_dark_pixels() {
        // ρ = 0 plus ε > 0 must yield a finite, bounded slope.
        let i_w = Array2::from_elem((3, 3), 0.0_f32);
        let ix = Array2::from_elem((3, 3), 0.1_f32);
        let iy = Array2::<f32>::zeros((3, 3));
        let rho = Array2::<f32>::zeros((3, 3));
        let hps =
            build_hyperplanes(i_w.view(), ix.view(), iy.view(), rho.view(), &[(1, 1)], 0.05, 1.0);
        assert!(hps[0].dz_dx.is_finite());
        // dz_dx = -0.1 / 0.05 = -2.
        assert!(approx_eq(hps[0].dz_dx, -2.0, 1e-6));
    }

    #[test]
    fn forge_per_scale_single_keypoint() {
        // Single hyperplane with anchor (5, 5), z_i = 0, dz_dx = 1, dz_dy = 0.
        // Inside its r=2 neighborhood we should see the linear ramp it stamps.
        let hp = LocalHyperplane { x_i: 5.0, y_i: 5.0, z_i: 0.0, dz_dx: 1.0, dz_dy: 0.0 };
        let z = forge_per_scale((11, 11), &[hp], 2);
        // At the anchor, z = 0.
        assert!(approx_eq(z[(5, 5)], 0.0, 1e-6));
        // 1 pixel right of anchor, z = 1.
        assert!(approx_eq(z[(5, 6)], 1.0, 1e-6));
        // 2 pixels left, z = -2.
        assert!(approx_eq(z[(5, 3)], -2.0, 1e-6));
        // Pixel outside the neighborhood: still +∞.
        assert!(z[(0, 0)].is_infinite());
    }

    #[test]
    fn forge_per_scale_min_picks_lower() {
        // Two coincident-anchor hyperplanes: one z=0, one z=-5. The Min should
        // take the z=-5 one everywhere they overlap.
        let lo = LocalHyperplane { x_i: 3.0, y_i: 3.0, z_i: -5.0, dz_dx: 0.0, dz_dy: 0.0 };
        let hi = LocalHyperplane { x_i: 3.0, y_i: 3.0, z_i: 0.0, dz_dx: 0.0, dz_dy: 0.0 };
        let z = forge_per_scale((7, 7), &[hi, lo], 2);
        for v in z.iter() {
            // Either covered (=-5) or uncovered (=+∞); never the +0 hyperplane.
            if v.is_finite() {
                assert!(approx_eq(*v, -5.0, 1e-6), "min step did not pick lower, got {v}");
            }
        }
    }

    #[test]
    fn forge_per_scale_no_keypoints_is_all_inf() {
        let z = forge_per_scale((4, 4), &[], 1);
        for v in z.iter() {
            assert!(v.is_infinite());
        }
    }

    #[test]
    fn fill_uncovered_replaces_infinities() {
        let mut z = Array2::from_elem((3, 3), f32::INFINITY);
        z[(1, 1)] = 0.7;
        let fallback = Array2::from_elem((3, 3), 0.3_f32);
        fill_uncovered(&mut z, fallback.view());
        assert!(approx_eq(z[(1, 1)], 0.7, 1e-6), "covered cells must be preserved");
        for &v in z.iter() {
            assert!(v.is_finite());
        }
        assert!(approx_eq(z[(0, 0)], 0.3, 1e-6));
    }

    #[test]
    fn max_over_scales_picks_higher() {
        let a = Array2::from_elem((3, 3), 1.0_f32);
        let b = Array2::from_elem((3, 3), 3.0_f32);
        let c = Array2::from_elem((3, 3), 2.0_f32);
        let out = compose_max_over_scales(&[a, b, c]);
        for v in out.iter() {
            assert!(approx_eq(*v, 3.0, 1e-6));
        }
    }

    #[test]
    fn max_over_scales_propagates_inf() {
        // A pixel with +∞ at every scale stays +∞ after max.
        let inf = Array2::from_elem((2, 2), f32::INFINITY);
        let out = compose_max_over_scales(&[inf]);
        for v in out.iter() {
            assert!(v.is_infinite());
        }
    }

    #[test]
    #[should_panic(expected = "at least one scale")]
    fn max_over_scales_rejects_empty() {
        let _ = compose_max_over_scales(&[]);
    }

    #[test]
    fn full_min_max_pipeline_recovers_planar_input() {
        // Three keypoints on a 9x9 grid, all defining the *same* hyperplane
        // (slope 1 in x, 0 in y, anchored differently). The Min-Max should
        // recover the underlying plane on every covered pixel.
        let mk = |x: f32, y: f32, z: f32| LocalHyperplane {
            x_i: x,
            y_i: y,
            z_i: z,
            dz_dx: 1.0,
            dz_dy: 0.0,
        };
        // The plane is z = x (so at (xi, yi), z_i = xi).
        let kps = vec![mk(2.0, 2.0, 2.0), mk(5.0, 5.0, 5.0), mk(7.0, 7.0, 7.0)];
        let per_scale = forge_per_scale((9, 9), &kps, 3);
        let composed = compose_max_over_scales(&[per_scale]);
        // Every covered pixel should equal x.
        for ((_y, x), &v) in composed.indexed_iter() {
            if v.is_finite() {
                assert!(approx_eq(v, x as f32, 1e-5), "expected x={x}, got {v}");
            }
        }
    }
}
