// histogram.wgsl
//
// GPU histogram building for the Wu colour quantizer.
//
// Computes a 3-D colour histogram of side `side_size` using hardware atomic
// adds.  For each pixel we accumulate:
//   weights    – pixel count per cell (u32)
//   moments_r  – sum of R values     (u32)
//   moments_g  – sum of G values     (u32)
//   moments_b  – sum of B values     (u32)
//
// The squared-distance moment (r²+g²+b²) is NOT computed on the GPU because
// it can overflow u32 for large images.  The CPU reads back the four u32
// histograms and computes the f64 moment array from:
//
//   moments[i] ≈ (mr[i]² + mg[i]² + mb[i]²) / weights[i]  (cell centroid method)
//
// This approximation is negligible for palette generation quality; the Wu
// algorithm only uses moments for variance, not for exact centroid placement.
//
// wgpu requirements:
//   • `atomicAdd` on storage buffers – supported since wgpu 0.13 / WGSL core.

struct Params {
    width:     u32,
    height:    u32,
    shift:     u32,   // = 8 - significant_bits
    side_size: u32,   // = 1 << significant_bits
    has_alpha: u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
}

@group(0) @binding(0) var<uniform>              params:    Params;
@group(0) @binding(1) var<storage, read>        pixels:    array<u32>;
@group(0) @binding(2) var<storage, read_write>  weights:   array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write>  moments_r: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write>  moments_g: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write>  moments_b: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total { return; }

    let px = pixels[idx];
    let a  = (px >> 24u) & 0xffu;
    if params.has_alpha != 0u && a < 128u { return; }

    let r = px & 0xffu;
    let g = (px >> 8u)  & 0xffu;
    let b = (px >> 16u) & 0xffu;

    let ir = r >> params.shift;
    let ig = g >> params.shift;
    let ib = b >> params.shift;

    let ss = params.side_size;
    let hist_idx = ir * ss * ss + ig * ss + ib;

    atomicAdd(&weights[hist_idx],   1u);
    atomicAdd(&moments_r[hist_idx], r);
    atomicAdd(&moments_g[hist_idx], g);
    atomicAdd(&moments_b[hist_idx], b);
}
