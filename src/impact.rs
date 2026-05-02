//! Pipeline orchestrator and impact-map assembly.
//!
//! Threads Phases 1 → 5 of the paper into a single function `run_pipeline`,
//! then assembles the impact map (`R = z* − z_ideal`, `L = Δz*`) and the
//! energy decomposition into a single `ImpactResult` consumed by the
//! Python orchestrator (via the PyO3 wrapper in `lib.rs`).

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};

use crate::energy::EnergyTerms;
use crate::hyperplane::{
    build_hyperplanes, compose_max_over_scales, fill_uncovered, forge_per_scale, local_albedo,
    LocalHyperplane,
};
use crate::luminance::{dog_band, gaussian_blur, weighted_luminance};
use crate::pde::{jacobi_solve, PdeParams};
use crate::scharr::scharr_gradients;
use crate::stencils::laplacian;
use crate::tensor::{adaptive_thresholds, classify, eigenvalues_2x2, keypoints, structural_tensor};

/// End-to-end pipeline parameters.
#[derive(Clone, Copy, Debug)]
pub struct PipelineParams {
    // ---- Phase 1: luminance & DoG ----
    pub w_r: f32,
    pub w_g: f32,
    pub w_b: f32,
    pub sigma_base: f32,
    pub k_ratio: f32,
    pub n_scales: usize,

    // ---- Phase 2: structural tensor ----
    pub tensor_window_radius: usize,
    pub p_flat: f32,
    pub p_edge: f32,

    // ---- Phase 3: hyperplane forge ----
    pub neigh_radius: usize,
    pub albedo_window_radius: usize,
    pub epsilon: f32,
    pub c_z: f32,

    // ---- Phase 5: PDE solver ----
    pub pde: PdeParams,

    // ---- Phase 6: impact map ----
    pub sigma_ref: f32,
}

impl Default for PipelineParams {
    fn default() -> Self {
        Self {
            w_r: 0.299,
            w_g: 0.587,
            w_b: 0.114,
            sigma_base: 1.0,
            k_ratio: std::f32::consts::SQRT_2,
            n_scales: 4,
            tensor_window_radius: 2,
            p_flat: 0.30,
            p_edge: 0.70,
            neigh_radius: 6,
            albedo_window_radius: 2,
            epsilon: 1.0e-3,
            c_z: 1.0,
            pde: PdeParams::default(),
            sigma_ref: 4.0,
        }
    }
}

/// Output of the full pipeline.
#[derive(Clone, Debug)]
pub struct ImpactResult {
    pub z_forged: Array2<f32>,
    pub z_star: Array2<f32>,
    pub r: Array2<f32>,
    pub l: Array2<f32>,
    pub energy_final: EnergyTerms,
    pub energy_trace: Vec<f32>,
    pub iterations: usize,
    pub converged: bool,
}

/// Run the full Phase 1 → Phase 5 pipeline, then assemble the impact map.
///
/// The image must be `H × W × 3` float32 in `[0, 1]`. `w_cnn` is the
/// per-pixel trust map, also `H × W` float32 in `[0, 1]`.
///
/// # Panics
///
/// Panics if `rgb.dim().2 != 3` or `w_cnn.dim() != (rgb.dim().0, rgb.dim().1)`,
/// or if any chromatic-weight or DoG parameter is invalid (see the
/// individual modules' contracts).
#[must_use]
pub fn run_pipeline(
    rgb: ArrayView3<f32>,
    w_cnn: ArrayView2<f32>,
    params: &PipelineParams,
) -> ImpactResult {
    let (h, w, _) = rgb.dim();
    assert_eq!(w_cnn.dim(), (h, w), "trust map and image must have matching H × W");

    // Phase 1: weighted luminance + DoG bands.
    let i_w = weighted_luminance(rgb, params.w_r, params.w_g, params.w_b);
    let pyramid = build_dog_pyramid(i_w.view(), params.sigma_base, params.k_ratio, params.n_scales);

    // Phase 2 + 3 (per scale): gradients, classification, hyperplanes.
    let albedo = local_albedo(i_w.view(), params.albedo_window_radius);
    let mut per_scale_z = Vec::with_capacity(params.n_scales);
    for j in 0..params.n_scales {
        let band = pyramid.index_axis(Axis(2), j);
        let (ix_k, iy_k) = scharr_gradients(band);
        let (jxx, jxy, jyy) =
            structural_tensor(ix_k.view(), iy_k.view(), params.tensor_window_radius);
        let (l1, l2) = eigenvalues_2x2(jxx.view(), jxy.view(), jyy.view());
        let (tau_flat, tau_edge) =
            adaptive_thresholds(l1.view(), l2.view(), params.p_flat, params.p_edge);
        let classes = classify(l1.view(), l2.view(), tau_flat, tau_edge);
        let kps = keypoints(classes.view());
        let hps = build_hyperplanes_for_scale(
            i_w.view(),
            ix_k.view(),
            iy_k.view(),
            albedo.view(),
            &kps,
            params.epsilon,
            params.c_z,
        );
        per_scale_z.push(forge_per_scale((h, w), &hps, params.neigh_radius));
    }

    // Phase 3.6: Min-Max composition.
    let mut z_forged = compose_max_over_scales(&per_scale_z);
    let fallback = i_w.mapv(|v| params.c_z * v);
    fill_uncovered(&mut z_forged, fallback.view());

    // Phase 5: solve the Euler-Lagrange PDE.
    let (ix, iy) = scharr_gradients(i_w.view());
    let res =
        jacobi_solve(z_forged.view(), ix.view(), iy.view(), w_cnn, albedo.view(), &params.pde);

    // Phase 6: impact map.
    let z_ideal = gaussian_blur(z_forged.view(), params.sigma_ref);
    let r = &res.z_star - &z_ideal;
    let l = laplacian(res.z_star.view());

    let energy_final = res.energy_trace.last().copied().unwrap_or_default();
    let energy_trace_total: Vec<f32> = res.energy_trace.iter().map(|e| e.total).collect();

    ImpactResult {
        z_forged,
        z_star: res.z_star,
        r,
        l,
        energy_final,
        energy_trace: energy_trace_total,
        iterations: res.iterations,
        converged: res.converged,
    }
}

fn build_dog_pyramid(
    i_w: ArrayView2<f32>,
    sigma_base: f32,
    k_ratio: f32,
    n_scales: usize,
) -> Array3<f32> {
    let (h, w) = i_w.dim();
    let mut pyr = Array3::<f32>::zeros((h, w, n_scales));
    for j in 0..n_scales {
        let sigma_j = sigma_base * k_ratio.powi(j as i32);
        let band = dog_band(i_w, sigma_j, k_ratio);
        pyr.index_axis_mut(Axis(2), j).assign(&band);
    }
    pyr
}

fn build_hyperplanes_for_scale(
    i_w: ArrayView2<f32>,
    ix_k: ArrayView2<f32>,
    iy_k: ArrayView2<f32>,
    rho: ArrayView2<f32>,
    keypoints: &[(usize, usize)],
    epsilon: f32,
    c_z: f32,
) -> Vec<LocalHyperplane> {
    build_hyperplanes(i_w, ix_k, iy_k, rho, keypoints, epsilon, c_z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn synthetic_face(shape: (usize, usize)) -> Array3<f32> {
        // Simple rgb: a bright disk on dark background. R=G=B for simplicity.
        let (h, w) = shape;
        let mut rgb = Array3::<f32>::zeros((h, w, 3));
        let cy = h as f32 / 2.0;
        let cx = w as f32 / 2.0;
        let r2_max = (h.min(w) as f32 / 3.0).powi(2);
        for ((y, x, c), v) in rgb.indexed_iter_mut() {
            let dy = y as f32 - cy;
            let dx = x as f32 - cx;
            let intensity = if dx * dx + dy * dy < r2_max { 0.8 } else { 0.2 };
            let _ = c;
            *v = intensity;
        }
        rgb
    }

    #[test]
    fn pipeline_runs_end_to_end_on_synthetic_image() {
        let rgb = synthetic_face((24, 24));
        let w_cnn = Array2::from_elem((24, 24), 1.0_f32);
        let mut params = PipelineParams::default();
        // Reduce iteration count for the test — convergence shape is what we
        // care about, not absolute precision.
        params.pde.max_iter = 50;
        params.pde.log_every = 10;
        params.n_scales = 2;
        let result = run_pipeline(rgb.view(), w_cnn.view(), &params);

        assert_eq!(result.z_forged.dim(), (24, 24));
        assert_eq!(result.z_star.dim(), (24, 24));
        assert_eq!(result.r.dim(), (24, 24));
        assert_eq!(result.l.dim(), (24, 24));
        // Outputs are finite.
        for v in result.z_forged.iter() {
            assert!(v.is_finite(), "z_forged contains non-finite value");
        }
        for v in result.z_star.iter() {
            assert!(v.is_finite(), "z_star contains non-finite value");
        }
        // Energy trace was logged.
        assert!(!result.energy_trace.is_empty());
        // Each energy component is non-negative (sum-of-squares invariant).
        assert!(result.energy_final.data >= 0.0);
        assert!(result.energy_final.smoothness >= 0.0);
        assert!(result.energy_final.consistency >= 0.0);
    }

    #[test]
    fn pipeline_constant_image_produces_uniform_output() {
        // A constant RGB image should yield a near-constant z_star (no
        // gradients, no keypoints, fallback z_forged = c_z · I_w everywhere).
        let rgb = Array3::from_elem((16, 16, 3), 0.5_f32);
        let w_cnn = Array2::from_elem((16, 16), 1.0_f32);
        let mut params = PipelineParams::default();
        params.pde.max_iter = 50;
        params.pde.log_every = 0;
        params.n_scales = 2;
        let result = run_pipeline(rgb.view(), w_cnn.view(), &params);

        let mean: f32 = result.z_star.iter().sum::<f32>() / (16 * 16) as f32;
        let max_dev = result.z_star.iter().map(|v| (v - mean).abs()).fold(0.0_f32, f32::max);
        assert!(max_dev < 1e-3, "z_star should be near-constant, max deviation = {max_dev}");

        // R = z_star − z_ideal should be near zero for a constant input.
        let r_max = result.r.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        assert!(r_max < 1e-3, "residual should be near zero, max |R| = {r_max}");
    }

    #[test]
    fn pipeline_default_params_construct() {
        let p = PipelineParams::default();
        assert!(p.w_r + p.w_g + p.w_b > 0.999 && p.w_r + p.w_g + p.w_b < 1.001);
        assert!(p.sigma_base > 0.0);
        assert!(p.n_scales > 0);
        assert!(p.epsilon > 0.0);
    }
}
