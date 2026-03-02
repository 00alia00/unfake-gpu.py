// palette_map.wgsl
//
// GPU-accelerated nearest-palette-colour mapping.
//
// Each thread processes one pixel, scanning the palette (up to 256 colours)
// and writing the closest RGB match.  Transparent pixels (alpha < 128) pass
// through as fully-transparent black.
//
// wgpu v29 note: params are in a uniform buffer rather than `var<immediate>`.
// For a future cut, replace the uniform buffer + binding with:
//
//   var<immediate> params: Params;   // wgpu v28+ "immediate data" (ex push-constants)
//
// and set `immediate_size` in `PipelineLayoutDescriptor` plus call
// `cpass.set_immediates(0, bytemuck::bytes_of(&params))`.

struct Params {
    width:      u32,
    height:     u32,
    num_colors: u32,
    has_alpha:  u32,   // 0 = RGB image, 1 = RGBA image
}

// Palette colours packed as u32: 0x00BBGGRR (LSB = R)
struct PaletteEntry {
    rgba: u32,
}

@group(0) @binding(0) var<uniform>              params:  Params;
@group(0) @binding(1) var<storage, read>        pixels:  array<u32>;
@group(0) @binding(2) var<storage, read>        palette: array<PaletteEntry>;
@group(0) @binding(3) var<storage, read_write>  output:  array<u32>;

fn unpack_rgb(packed: u32) -> vec3<i32> {
    return vec3<i32>(
        i32(packed & 0xffu),
        i32((packed >> 8u) & 0xffu),
        i32((packed >> 16u) & 0xffu),
    );
}

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if idx >= total { return; }

    let packed = pixels[idx];
    let a = (packed >> 24u) & 0xffu;

    // Transparent pixels -> black, alpha 0
    if params.has_alpha != 0u && a < 128u {
        output[idx] = 0u;
        return;
    }

    let pixel = unpack_rgb(packed);

    var min_dist: i32 = 0x7fffffff;
    var best_r: u32 = 0u;
    var best_g: u32 = 0u;
    var best_b: u32 = 0u;

    // Branchless linear scan – GPU can execute 256 iterations efficiently
    for (var i: u32 = 0u; i < params.num_colors; i++) {
        let pal = unpack_rgb(palette[i].rgba);
        let dr = pixel.x - pal.x;
        let dg = pixel.y - pal.y;
        let db = pixel.z - pal.z;
        let dist = dr * dr + dg * dg + db * db;

        if dist < min_dist {
            min_dist = dist;
            best_r = u32(pal.x);
            best_g = u32(pal.y);
            best_b = u32(pal.z);
        }
    }

    let out_a = select(0u, 255u, params.has_alpha == 0u || a >= 128u);
    output[idx] = pack_rgba(best_r, best_g, best_b, out_a);
}
