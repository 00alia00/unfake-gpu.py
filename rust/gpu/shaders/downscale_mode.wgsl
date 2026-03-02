// downscale_mode.wgsl
//
// GPU-accelerated mode (most-frequently-occurring colour) downscaling.
//
// Like dominant, but always emits the plurality colour regardless of
// threshold.  Falls back to the average when the block is all-transparent.

struct Params {
    in_width:  u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
    scale:     u32,
    has_alpha: u32,
    _pad0:     u32,
    _pad1:     u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       input:  array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if ox >= params.out_width || oy >= params.out_height { return; }

    let xs = ox * params.scale;
    let ys = oy * params.scale;
    let xe = min(xs + params.scale, params.in_width);
    let ye = min(ys + params.scale, params.in_height);

    // Local histogram: up to 16 distinct RGB colours
    var keys:    array<u32, 16>;
    var counts:  array<u32, 16>;
    var alphas:  array<u32, 16>;   // one alpha sample per key
    var num_keys = 0u;
    var opaque_count = 0u;

    for (var y = ys; y < ye; y++) {
        for (var x = xs; x < xe; x++) {
            let px = input[y * params.in_width + x];
            let a  = (px >> 24u) & 0xffu;
            if params.has_alpha != 0u && a < 128u { continue; }

            let key = px & 0x00ffffffu;
            opaque_count++;

            var found = false;
            for (var k = 0u; k < num_keys; k++) {
                if keys[k] == key {
                    counts[k]++;
                    found = true;
                    break;
                }
            }
            if !found && num_keys < 16u {
                keys[num_keys]   = key;
                counts[num_keys] = 1u;
                alphas[num_keys] = a;
                num_keys++;
            }
        }
    }

    let out_idx = oy * params.out_width + ox;

    if opaque_count == 0u {
        output[out_idx] = 0u;
        return;
    }

    // Mode: find the colour with the highest count
    var best_count = 0u;
    var best_key   = 0u;
    var best_alpha = 255u;
    for (var k = 0u; k < num_keys; k++) {
        if counts[k] > best_count {
            best_count = counts[k];
            best_key   = keys[k];
            best_alpha = alphas[k];
        }
    }

    output[out_idx] = best_key | (best_alpha << 24u);
}
