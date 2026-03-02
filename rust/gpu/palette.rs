//! GPU-accelerated palette mapping.
//!
//! Maps every pixel in an RGBA / RGB image to the nearest colour in a palette
//! (Euclidean distance in RGB space).

use bytemuck::{Pod, Zeroable};
use ndarray::{Array3, ArrayView3};
use std::sync::Arc;

use super::context::GpuContext;

// ── Shader parameters layout (must match palette_map.wgsl `Params` struct) ───

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PaletteMapParams {
    width: u32,
    height: u32,
    num_colors: u32,
    has_alpha: u32,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Map every pixel to the nearest palette entry using a GPU compute shader.
///
/// Returns `None` if a bind-group or dispatch call fails unexpectedly.
/// The caller falls back to the CPU implementation in that case.
pub fn map_pixels_gpu(
    ctx: &Arc<GpuContext>,
    pixels: &ArrayView3<u8>,
    palette: &[(u8, u8, u8)],
) -> Option<Array3<u8>> {
    let (height, width, channels) = pixels.dim();
    let has_alpha = channels == 4;
    let total = width * height;
    if palette.is_empty() || total == 0 {
        return None;
    }

    // ── Pack input pixels as u32 (RGBA / RGBX) ───────────────────────────────
    let mut packed_in = Vec::with_capacity(total);
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
            packed_in.push(r | (g << 8) | (b << 16) | (a << 24));
        }
    }

    // ── Pack palette as u32 ───────────────────────────────────────────────────
    let mut pal_u32: Vec<u32> = palette
        .iter()
        .map(|&(r, g, b)| r as u32 | ((g as u32) << 8) | ((b as u32) << 16))
        .collect();
    // Pad palette buffer to a multiple of 4 bytes (bytemuck requires alignment)
    while pal_u32.len() % 4 != 0 {
        pal_u32.push(0);
    }

    // ── GPU buffers ─────────────────────────────────────────────────────────
    let params = PaletteMapParams {
        width: width as u32,
        height: height as u32,
        num_colors: palette.len() as u32,
        has_alpha: if has_alpha { 1 } else { 0 },
    };

    let params_buf = ctx.upload_uniform("palette_map_params", &params);
    let pixels_buf = ctx.upload_storage("palette_map_pixels", &packed_in);
    let palette_buf = ctx.upload_storage("palette_map_palette", &pal_u32);
    let output_buf = ctx.create_output_storage("palette_map_output", (total * 4) as u64);

    // ── Bind group (auto-layout from shader reflection) ───────────────────
    let bind_group_layout = ctx.palette_map_pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("palette_map_bg"),
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
                resource: palette_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    // ── Dispatch ───────────────────────────────────────────────────────────
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("palette_map"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("palette_map_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&ctx.palette_map_pipeline);
        // v23+: set_bind_group accepts `impl Into<Option<&BindGroup>>`
        cpass.set_bind_group(0, &bind_group, &[]);
        let workgroups = (total as u32 + 63) / 64;
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.submit_and_wait(encoder);

    // ── Readback ───────────────────────────────────────────────────────────
    let raw = ctx.readback_buffer(&output_buf, (total * 4) as u64);

    // ── Unpack into Array3<u8> ─────────────────────────────────────────────
    let mut result = Array3::<u8>::zeros((height, width, channels));
    let words: &[u32] = bytemuck::cast_slice(&raw);
    for y in 0..height {
        for x in 0..width {
            let w = words[y * width + x];
            result[(y, x, 0)] = (w & 0xff) as u8;
            result[(y, x, 1)] = ((w >> 8) & 0xff) as u8;
            result[(y, x, 2)] = ((w >> 16) & 0xff) as u8;
            if has_alpha {
                result[(y, x, 3)] = ((w >> 24) & 0xff) as u8;
            }
        }
    }

    Some(result)
}
