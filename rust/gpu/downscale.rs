//! GPU-accelerated image downscaling (dominant-colour and mode methods).

use bytemuck::{Pod, Zeroable};
use ndarray::{Array3, ArrayView3};
use std::sync::Arc;

use super::context::GpuContext;

// ── Shader params (must match struct layouts in WGSL shaders) ─────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DominantParams {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    scale: u32,
    threshold_pct: u32, // integer percent, e.g. 51 for >51 %
    has_alpha: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ModeParams {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    scale: u32,
    has_alpha: u32,
    _pad0: u32,
    _pad1: u32,
}

// ── Internal helper ────────────────────────────────────────────────────────────

fn pack_pixels(image: &ArrayView3<u8>) -> Vec<u32> {
    let (h, w, c) = image.dim();
    let mut out = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            let r = image[(y, x, 0)] as u32;
            let g = image[(y, x, 1)] as u32;
            let b = image[(y, x, 2)] as u32;
            let a = if c == 4 { image[(y, x, 3)] as u32 } else { 255 };
            out.push(r | (g << 8) | (b << 16) | (a << 24));
        }
    }
    out
}

fn unpack_to_array3(raw: &[u8], h: usize, w: usize, c: usize) -> Array3<u8> {
    let words: &[u32] = bytemuck::cast_slice(raw);
    let mut arr = Array3::<u8>::zeros((h, w, c));
    for y in 0..h {
        for x in 0..w {
            let px = words[y * w + x];
            arr[(y, x, 0)] = (px & 0xff) as u8;
            arr[(y, x, 1)] = ((px >> 8) & 0xff) as u8;
            arr[(y, x, 2)] = ((px >> 16) & 0xff) as u8;
            if c == 4 {
                arr[(y, x, 3)] = ((px >> 24) & 0xff) as u8;
            }
        }
    }
    arr
}

fn dispatch_downscale(
    ctx: &Arc<GpuContext>,
    pipeline: &wgpu::ComputePipeline,
    params_buf: &wgpu::Buffer,
    input_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer,
    out_w: usize,
    out_h: usize,
) {
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("downscale_bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("downscale"),
        });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("downscale_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let gx = (out_w as u32 + 7) / 8;
        let gy = (out_h as u32 + 7) / 8;
        cpass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.submit_and_wait(encoder);
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// GPU dominant-colour downscale.  Returns `None` on any failure → CPU fallback.
pub fn downscale_dominant_gpu(
    ctx: &Arc<GpuContext>,
    image: &ArrayView3<u8>,
    scale: usize,
    threshold: f32,
) -> Option<Array3<u8>> {
    let (in_h, in_w, c) = image.dim();
    let out_w = in_w / scale;
    let out_h = in_h / scale;
    if out_w == 0 || out_h == 0 {
        return None;
    }

    // WGSL shader handles scale ≤ 16 reliably (256-entry local hash)
    if scale > 16 {
        return None;
    }

    let packed = pack_pixels(image);
    let has_alpha = c == 4;

    let params = DominantParams {
        in_width: in_w as u32,
        in_height: in_h as u32,
        out_width: out_w as u32,
        out_height: out_h as u32,
        scale: scale as u32,
        threshold_pct: (threshold * 100.0) as u32,
        has_alpha: if has_alpha { 1 } else { 0 },
        _pad: 0,
    };

    let params_buf = ctx.upload_uniform("dominant_params", &params);
    let input_buf = ctx.upload_storage("dominant_input", &packed);
    let output_buf = ctx.create_output_storage("dominant_output", (out_w * out_h * 4) as u64);

    dispatch_downscale(
        ctx,
        &ctx.downscale_dominant_pipeline,
        &params_buf,
        &input_buf,
        &output_buf,
        out_w,
        out_h,
    );

    let raw = ctx.readback_buffer(&output_buf, (out_w * out_h * 4) as u64);
    Some(unpack_to_array3(&raw, out_h, out_w, c))
}

/// GPU mode downscale.  Returns `None` on any failure → CPU fallback.
pub fn downscale_mode_gpu(
    ctx: &Arc<GpuContext>,
    image: &ArrayView3<u8>,
    scale: usize,
) -> Option<Array3<u8>> {
    let (in_h, in_w, c) = image.dim();
    let out_w = in_w / scale;
    let out_h = in_h / scale;
    if out_w == 0 || out_h == 0 {
        return None;
    }
    if scale > 16 {
        return None;
    }

    let packed = pack_pixels(image);
    let has_alpha = c == 4;

    let params = ModeParams {
        in_width: in_w as u32,
        in_height: in_h as u32,
        out_width: out_w as u32,
        out_height: out_h as u32,
        scale: scale as u32,
        has_alpha: if has_alpha { 1 } else { 0 },
        _pad0: 0,
        _pad1: 0,
    };

    let params_buf = ctx.upload_uniform("mode_params", &params);
    let input_buf = ctx.upload_storage("mode_input", &packed);
    let output_buf = ctx.create_output_storage("mode_output", (out_w * out_h * 4) as u64);

    dispatch_downscale(
        ctx,
        &ctx.downscale_mode_pipeline,
        &params_buf,
        &input_buf,
        &output_buf,
        out_w,
        out_h,
    );

    let raw = ctx.readback_buffer(&output_buf, (out_w * out_h * 4) as u64);
    Some(unpack_to_array3(&raw, out_h, out_w, c))
}
