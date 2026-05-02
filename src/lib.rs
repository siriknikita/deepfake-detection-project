//! Hyperplane-Forge math core.
//!
//! Each module corresponds to one phase of the algorithm specified in
//! `paper/main.typ`. The PyO3 wrapper at the bottom of this file (gated
//! behind the `python-extension` feature) exposes a single function
//! `solve_and_extract` that runs the full pipeline; in-Rust consumers call
//! `impact::run_pipeline` directly.

pub mod energy;
pub mod hyperplane;
pub mod impact;
pub mod luminance;
pub mod pde;
pub mod scharr;
pub mod stencils;
pub mod tensor;

mod util;

#[cfg(feature = "python-extension")]
mod py_bindings;
