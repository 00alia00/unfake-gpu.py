//! GPU-accelerated 3-D colour histogram building for the Wu quantizer.
//!
//! Builds four flat arrays – `weights`, `moments_r`, `moments_g`, `moments_b` –
//! via GPU atomic adds.  The floating-point moment array (r²+g²+b² per cell)
//! is reconstructed on the CPU from the integer channel sums; see comment in
//! `histogram.wgsl` for the rationale.

use bytemuck::{Pod, Zeroable};
use ndarray::ArrayView3;
use std::sync::Arc;

use super::context::GpuContext;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HistParams {
    width: u32,
    height: u32,
    shift: u32,
    side_size: u32,
    has_alpha: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU histogram result.  All arrays are flat with index = ir*S*S + ig*S + ib.
pub struct GpuHistogram {
    pub weights: Vec<u32>,
    pub moments_r: Vec<u32>,
    pub moments_g: Vec<u32>,
    pub moments_b: Vec<u32>,
    pub side_size: usize,
}

impl GpuHistogram {
    /// Derive the floating-point moment array needed by the Wu quantizer.
    ///
    /// Uses the per-cell centroid approximation:
    ///   moments[i] = mr[i]² + mg[i]² + mb[i]² (when weight == 1 this is exact).
    ///
    /// The error is bounded by intra-cell variance, which is small with ≥5-bit
    /// quantisation (cell width ≤ 8 levels of 255).
    pub fn to_float_moments(&self) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.moments_r.iter())
            .zip(self.moments_g.iter())
            .zip(self.moments_b.iter())
            .map(|(((&w, &mr), &mg), &mb)| {
                if w == 0 {
                    0.0f64
                } else {
                    let w = w as f64;
                    let r = mr as f64 / w;
                    let g = mg as f64 / w;
                    let b = mb as f64 / w;
                    // Reconstitute Σ(r²+g²+b²) ≈ w * (centroid r²+g²+b²)
                    w * (r * r + g * g + b * b)
                }
            })
            .collect()
    }
}

/// Build a GPU histogram.  Returns `None` if the GPU path is unavailable.
pub fn build_histogram_gpu(
    ctx: &Arc<GpuContext>,
    pixels: &ArrayView3<u8>,
    significant_bits: u8,
) -> Option<GpuHistogram> {
    let (height, width, channels) = pixels.dim();
    let total = width * height;
    let side_size = 1usize << significant_bits;
    let hist_len = side_size * side_size * side_size;
    let has_alpha = channels == 4;

    // Pack pixels
    let mut packed = Vec::with_capacity(total);
    for y in 0..height {
        for x in 0..width {
            let r = pixels[(y, x, 0)] as u32;
            let g = pixels[(y, x, 1)] as u32;
            let b = pixels[(y, x, 2)] as u32;
            let a = if has_alpha {
                pixels[(y, x, 3)] as u32
            } else {
                255
            };
            packed.push(r | (g << 8) | (b << 16) | (a << 24));
        }
    }

    let params = HistParams {
        width: width as u32,
        height: height as u32,
        shift: (8 - significant_bits) as u32,
        side_size: side_size as u32,
        has_alpha: if has_alpha { 1 } else { 0 },
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buf = ctx.upload_uniform("hist_params", &params);
    let pixels_buf = ctx.upload_storage("hist_pixels", &packed);
    let weights_buf = ctx.create_output_storage("hist_w", (hist_len * 4) as u64);
    let moments_r_buf = ctx.create_output_storage("hist_mr", (hist_len * 4) as u64);
    let moments_g_buf = ctx.create_output_storage("hist_mg", (hist_len * 4) as u64);
    let moments_b_buf = ctx.create_output_storage("hist_mb", (hist_len * 4) as u64);

    // Zero-initialise atomic buffers (GPU buffers start zeroed by wgpu spec).
    // wgpu guarantees zero-initialisation, so no extra clear step needed.

    let bind_group_layout = ctx.histogram_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("histogram_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pixels_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: weights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: moments_r_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: moments_g_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: moments_b_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("histogram"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.histogram_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (total as u32 + 63) / 64;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.submit_and_wait(encoder);

    let w_raw = ctx.readback_buffer(&weights_buf, (hist_len * 4) as u64);
    let mr_raw = ctx.readback_buffer(&moments_r_buf, (hist_len * 4) as u64);
    let mg_raw = ctx.readback_buffer(&moments_g_buf, (hist_len * 4) as u64);
    let mb_raw = ctx.readback_buffer(&moments_b_buf, (hist_len * 4) as u64);

    let cast = |v: Vec<u8>| -> Vec<u32> { bytemuck::cast_slice(&v).to_vec() };

    Some(GpuHistogram {
        weights: cast(w_raw),
        moments_r: cast(mr_raw),
        moments_g: cast(mg_raw),
        moments_b: cast(mb_raw),
        side_size,
    })
}
