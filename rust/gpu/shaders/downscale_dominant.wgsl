// downscale_dominant.wgsl
//
// GPU-accelerated dominant-colour downscaling.
//
// Each compute thread owns one output texel.  It scans the corresponding
// scale×scale input block, finds the colour that appears most often (the
// "dominant" colour), and emits it if its frequency exceeds `threshold`
// (stored as a u32 fraction numerator out of 256).  Otherwise it falls back
// to the average of the block.
//
// Constraint: scale ≤ 16 (max block = 256 pixels, fits a function-scope array).

struct Params {
    in_width:   u32,
    in_height:  u32,
    out_width:  u32,
    out_height: u32,
    scale:      u32,
    // threshold as integer percent 0-100 (dominant count / total > threshold/100)
    threshold_pct: u32,
    has_alpha:  u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       input:  array<u32>;  // packed RGBA
@group(0) @binding(2) var<storage, read_write> output: array<u32>;  // packed RGBA

fn unpack(px: u32) -> vec4<u32> {
    return vec4<u32>(
        px & 0xffu,
        (px >> 8u) & 0xffu,
        (px >> 16u) & 0xffu,
        (px >> 24u) & 0xffu,
    );
}

fn pack(v: vec4<u32>) -> u32 {
    return v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
}

// Count how many pixels in [xs..xe, ys..ye] match `tgt`.
fn count_matches(xs: u32, xe: u32, ys: u32, ye: u32, tgt: u32) -> u32 {
    var c = 0u;
    for (var y = ys; y < ye; y++) {
        for (var x = xs; x < xe; x++) {
            let idx = y * params.in_width + x;
            let px = input[idx];
            // Compare only RGB, ignoring alpha for dominant detection
            if (px & 0x00ffffffu) == (tgt & 0x00ffffffu) { c++; }
        }
    }
    return c;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if ox >= params.out_width || oy >= params.out_height { return; }

    let xs = ox * params.scale;
    let ys = oy * params.scale;
    let xe = min(xs + params.scale, params.in_width);
    let ye = min(ys + params.scale, params.in_height);

    let total = (xe - xs) * (ye - ys);
    if total == 0u {
        output[oy * params.out_width + ox] = 0u;
        return;
    }

    // Accumulate sum for average fallback
    var sum_r = 0u;
    var sum_g = 0u;
    var sum_b = 0u;
    var sum_a = 0u;
    var opaque_count = 0u;

    // For dominant detection: store up to 16 distinct RGB keys + their counts
    // (function-scope arrays live in registers on GPU)
    var keys:   array<u32, 16>;
    var counts: array<u32, 16>;
    var num_keys = 0u;

    for (var y = ys; y < ye; y++) {
        for (var x = xs; x < xe; x++) {
            let px  = input[y * params.in_width + x];
            let a   = (px >> 24u) & 0xffu;

            if params.has_alpha != 0u && a < 128u { continue; }

            let key = px & 0x00ffffffu;   // RGB only
            sum_r += key & 0xffu;
            sum_g += (key >> 8u) & 0xffu;
            sum_b += (key >> 16u) & 0xffu;
            sum_a += a;
            opaque_count++;

            // Insert into local histogram (max 16 distinct colours)
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
                num_keys++;
            }
        }
    }

    if opaque_count == 0u {
        output[oy * params.out_width + ox] = 0u;
        return;
    }

    // Find dominant colour
    var best_count = 0u;
    var best_key   = 0u;
    for (var k = 0u; k < num_keys; k++) {
        if counts[k] > best_count {
            best_count = counts[k];
            best_key   = keys[k];
        }
    }

    let out_idx = oy * params.out_width + ox;
    let threshold_count = (opaque_count * params.threshold_pct + 99u) / 100u;

    if best_count >= threshold_count {
        // Emit dominant colour, full alpha
        output[out_idx] = best_key | (255u << 24u);
    } else {
        // Emit average colour
        let r = sum_r / opaque_count;
        let g = sum_g / opaque_count;
        let b = sum_b / opaque_count;
        let a = sum_a / opaque_count;
        output[out_idx] = r | (g << 8u) | (b << 16u) | (a << 24u);
    }
}
