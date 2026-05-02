//! PyO3 bindings — the Python entry point for the Rust math core.
//!
//! Compiled only when the `python-extension` cargo feature is enabled
//! (set by maturin via pyproject.toml). The default `cargo test` build
//! omits this module so the test binary does not need to resolve
//! libpython symbols at load time.

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Bound;

use crate::impact::{run_pipeline, PipelineParams};
use crate::pde::PdeParams;

/// PyO3 module entry point. Exposes `solve_and_extract`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_and_extract, m)?)?;
    Ok(())
}

/// Run the full Hyperplane-Forge pipeline on an RGB image and trust map and
/// return a dictionary with the impact map, energy decomposition, and
/// settled manifold.
#[pyfunction]
#[pyo3(signature = (
    rgb,
    w_cnn,
    *,
    w_r = 0.299,
    w_g = 0.587,
    w_b = 0.114,
    sigma_base = 1.0,
    k_ratio = std::f32::consts::SQRT_2,
    n_scales = 4,
    tensor_window_radius = 2,
    p_flat = 0.30,
    p_edge = 0.70,
    neigh_radius = 6,
    albedo_window_radius = 2,
    epsilon = 1.0e-3,
    c_z = 1.0,
    sigma_ref = 4.0,
    pde_lambda = 1.0,
    pde_alpha = 0.5,
    pde_beta = 5.0,
    pde_max_iter = 500,
    pde_tol = 1.0e-5,
    pde_log_every = 10,
))]
fn solve_and_extract<'py>(
    py: Python<'py>,
    rgb: PyReadonlyArray3<'py, f32>,
    w_cnn: PyReadonlyArray2<'py, f32>,
    w_r: f32,
    w_g: f32,
    w_b: f32,
    sigma_base: f32,
    k_ratio: f32,
    n_scales: usize,
    tensor_window_radius: usize,
    p_flat: f32,
    p_edge: f32,
    neigh_radius: usize,
    albedo_window_radius: usize,
    epsilon: f32,
    c_z: f32,
    sigma_ref: f32,
    pde_lambda: f32,
    pde_alpha: f32,
    pde_beta: f32,
    pde_max_iter: usize,
    pde_tol: f32,
    pde_log_every: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let rgb_owned: ndarray::Array3<f32> = rgb.as_array().to_owned();
    let w_cnn_owned: Array2<f32> = w_cnn.as_array().to_owned();

    let params = PipelineParams {
        w_r,
        w_g,
        w_b,
        sigma_base,
        k_ratio,
        n_scales,
        tensor_window_radius,
        p_flat,
        p_edge,
        neigh_radius,
        albedo_window_radius,
        epsilon,
        c_z,
        pde: PdeParams {
            lambda: pde_lambda,
            alpha: pde_alpha,
            beta: pde_beta,
            max_iter: pde_max_iter,
            tol: pde_tol,
            log_every: pde_log_every,
        },
        sigma_ref,
    };

    let result = py.allow_threads(|| run_pipeline(rgb_owned.view(), w_cnn_owned.view(), &params));

    let dict = PyDict::new_bound(py);
    dict.set_item("z_forged", to_py_2d(py, &result.z_forged))?;
    dict.set_item("z_star", to_py_2d(py, &result.z_star))?;
    dict.set_item("R", to_py_2d(py, &result.r))?;
    dict.set_item("L", to_py_2d(py, &result.l))?;
    dict.set_item("energy_total", result.energy_final.total)?;
    dict.set_item("energy_data", result.energy_final.data)?;
    dict.set_item("energy_smoothness", result.energy_final.smoothness)?;
    dict.set_item("energy_consistency", result.energy_final.consistency)?;
    dict.set_item("energy_trace", PyArray1::from_slice_bound(py, &result.energy_trace))?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict)
}

fn to_py_2d<'py>(py: Python<'py>, arr: &Array2<f32>) -> Bound<'py, PyArray2<f32>> {
    arr.to_pyarray_bound(py)
}
