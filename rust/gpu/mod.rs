//! GPU acceleration layer using wgpu v29.
//!
//! # Design rationale
//!
//! | Operation                    | Strategy          | Reason                                            |
//! |------------------------------|-------------------|----------------------------------------------------|
//! | Palette mapping              | **GPU compute**   | Embarrassingly parallel; 256-color NN search/pixel |
//! | Dominant-color downscale     | **GPU compute**   | Per-output-pixel scan; trivially parallel          |
//! | Mode (freq) downscale        | **GPU compute**   | Independent per output pixel                       |
//! | Wu quantizer histogram build | **GPU compute**   | u32 atomic adds; 1 write/pixel                     |
//! | Wu box-cutting / palette gen | **CPU**           | Sequential tree recursion, O(K) not worth GPU trip |
//! | Content-adaptive (EM-C)      | **CPU rayon**     | Iterative global reduction steps between kernels   |
//! | Scale detection              | **CPU rayon**     | Very fast, I/O bound not compute bound             |
//! | Background flood-fill        | **CPU**           | Inherently sequential BFS                          |
//!
//! # wgpu v29 features used
//! * `immediate_size` / `set_immediates` – renamed from push constants (v28).
//! * `var<immediate>` address space in WGSL (v28).
//! * `request_adapter` returns `Result` (v25).
//! * Optional `entry_point: None` in pipeline descriptors (v23).
//! * `map_buffer_on_submit` for deferred readback (v27+).

use std::sync::{Arc, OnceLock};

#[cfg(feature = "gpu")]
pub use context::GpuContext;

#[cfg(feature = "gpu")]
pub mod context;

#[cfg(feature = "gpu")]
pub mod palette;

#[cfg(feature = "gpu")]
pub mod downscale;

#[cfg(feature = "gpu")]
pub mod histogram;

/// Returns the lazily-initialized shared GPU context, or None if no GPU is available.
#[cfg(feature = "gpu")]
pub fn gpu_context() -> Option<Arc<GpuContext>> {
    static INSTANCE: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| pollster::block_on(GpuContext::new()).ok().map(Arc::new))
        .clone()
}
