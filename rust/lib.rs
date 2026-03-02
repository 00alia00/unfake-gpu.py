//! `unfake` Rust core library.
//!
//! Feature flags:
//!   `python` – enable pyo3 / numpy Python bindings (required for `.so`).
//!   `gpu`    – enable wgpu compute-shader acceleration (default on).

// ── Conditional pyo3 imports ─────────────────────────────────────────────────
#[cfg(feature = "python")]
use numpy::{PyArray3, PyReadonlyArray3};
#[cfg(feature = "python")]
use pyo3::prelude::*;

use rayon::prelude::*;
use std::sync::Mutex;

// ── Core modules ─────────────────────────────────────────────────────────────
pub mod quantizer;
pub mod utils;

// ── GPU acceleration module (disabled if wgpu not requested) ────────────────
#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "python")]
use quantizer::WuQuantizerRust;
#[cfg(feature = "python")]
use utils::content_adaptive_downscale_rust;
use utils::downscale_dominant;
use utils::downscale_mode;
#[cfg(feature = "python")]
use utils::make_background_transparent_rust;

/// Calculate GCD of two numbers
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

/// Calculate GCD of a vector of numbers
fn gcd_of_vec(numbers: &[u32]) -> u32 {
    if numbers.is_empty() {
        return 1;
    }

    let mut result = numbers[0];
    for &num in &numbers[1..] {
        result = gcd(result, num);
        if result == 1 {
            return 1;
        }
    }
    result
}

/// Runs-based scale detection (pure Rust API)
pub fn runs_based_detect_rs(image: ndarray::ArrayView3<u8>) -> u32 {
    let (height, width, channels) = image.dim();
    let horizontal_runs: Vec<u32> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let mut runs = Vec::new();
            let mut run_length = 1u32;
            for x in 1..width {
                let same = (0..channels).all(|c| image[(y, x, c)] == image[(y, x - 1, c)]);
                if same {
                    run_length += 1;
                } else {
                    if run_length > 1 {
                        runs.push(run_length);
                    }
                    run_length = 1;
                }
            }
            if run_length > 1 {
                runs.push(run_length);
            }
            runs
        })
        .collect();
    let vertical_runs: Vec<u32> = (0..width)
        .into_par_iter()
        .flat_map(|x| {
            let mut runs = Vec::new();
            let mut run_length = 1u32;
            for y in 1..height {
                let same = (0..channels).all(|c| image[(y, x, c)] == image[(y - 1, x, c)]);
                if same {
                    run_length += 1;
                } else {
                    if run_length > 1 {
                        runs.push(run_length);
                    }
                    run_length = 1;
                }
            }
            if run_length > 1 {
                runs.push(run_length);
            }
            runs
        })
        .collect();
    let mut all_runs = horizontal_runs;
    all_runs.extend(vertical_runs);
    if all_runs.len() < 10 {
        return 1;
    }
    gcd_of_vec(&all_runs).max(1)
}

/// Runs-based scale detection
#[cfg(feature = "python")]
#[pyfunction]
fn runs_based_detect(_py: Python<'_>, image: PyReadonlyArray3<u8>) -> PyResult<u32> {
    let img_array = image.as_array();
    let (height, width, channels) = img_array.dim();

    // Collect all run lengths in parallel
    let horizontal_runs: Vec<u32> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let mut runs = Vec::new();
            let mut run_length = 1u32;

            for x in 1..width {
                let mut same = true;
                for c in 0..channels {
                    if img_array[(y, x, c)] != img_array[(y, x - 1, c)] {
                        same = false;
                        break;
                    }
                }

                if same {
                    run_length += 1;
                } else {
                    if run_length > 1 {
                        runs.push(run_length);
                    }
                    run_length = 1;
                }
            }

            if run_length > 1 {
                runs.push(run_length);
            }
            runs
        })
        .collect();

    let vertical_runs: Vec<u32> = (0..width)
        .into_par_iter()
        .flat_map(|x| {
            let mut runs = Vec::new();
            let mut run_length = 1u32;

            for y in 1..height {
                let mut same = true;
                for c in 0..channels {
                    if img_array[(y, x, c)] != img_array[(y - 1, x, c)] {
                        same = false;
                        break;
                    }
                }

                if same {
                    run_length += 1;
                } else {
                    if run_length > 1 {
                        runs.push(run_length);
                    }
                    run_length = 1;
                }
            }

            if run_length > 1 {
                runs.push(run_length);
            }
            runs
        })
        .collect();

    // Combine all runs
    let mut all_runs = horizontal_runs;
    all_runs.extend(vertical_runs);

    if all_runs.len() < 10 {
        return Ok(1);
    }

    // Calculate GCD of all run lengths
    let scale = gcd_of_vec(&all_runs);
    Ok(scale.max(1))
}

/// Map pixels to nearest palette colours (pure Rust, GPU-accelerated when available).
pub fn map_pixels_to_palette_rs(
    pixels: ndarray::ArrayView3<u8>,
    palette: Vec<(u8, u8, u8)>,
) -> ndarray::Array3<u8> {
    // ─ GPU fast path ───────────────────────────────────────────────────
    #[cfg(feature = "gpu")]
    if let Some(ctx) = gpu::gpu_context() {
        if let Some(result) = gpu::palette::map_pixels_gpu(&ctx, &pixels, &palette) {
            return result;
        }
    }
    // ─ CPU fallback ─────────────────────────────────────────────────────
    let (height, width, channels) = pixels.dim();
    let has_alpha = channels == 4;
    let mut output = ndarray::Array3::<u8>::zeros((height, width, channels));
    let results = Mutex::new(Vec::with_capacity(height * width));
    let chunk_size = 64;
    (0..height)
        .into_par_iter()
        .step_by(chunk_size)
        .for_each(|y_start| {
            let y_end = (y_start + chunk_size).min(height);
            let mut chunk = Vec::with_capacity((y_end - y_start) * width);
            for y in y_start..y_end {
                for x in 0..width {
                    if has_alpha && pixels[(y, x, 3)] < 128 {
                        chunk.push((y, x, 0u8, 0u8, 0u8, 0u8));
                        continue;
                    }
                    let (pr, pg, pb) = (
                        pixels[(y, x, 0)] as i32,
                        pixels[(y, x, 1)] as i32,
                        pixels[(y, x, 2)] as i32,
                    );
                    let (&best_r, &best_g, &best_b) = palette
                        .iter()
                        .map(|c| {
                            let dr = pr - c.0 as i32;
                            let dg = pg - c.1 as i32;
                            let db = pb - c.2 as i32;
                            (dr * dr + dg * dg + db * db, c)
                        })
                        .min_by_key(|&(d, _)| d)
                        .map(|(_, c)| (&c.0, &c.1, &c.2))
                        .unwrap();
                    let a = if has_alpha { 255u8 } else { 0u8 };
                    chunk.push((y, x, best_r, best_g, best_b, a));
                }
            }
            results.lock().unwrap().extend(chunk);
        });
    for (y, x, r, g, b, a) in results.into_inner().unwrap() {
        output[(y, x, 0)] = r;
        output[(y, x, 1)] = g;
        output[(y, x, 2)] = b;
        if has_alpha {
            output[(y, x, 3)] = a;
        }
    }
    output
}

/// Map pixels to nearest palette colors (Python binding)
#[cfg(feature = "python")]
#[pyfunction]
fn map_pixels_to_palette(
    py: Python<'_>,
    pixels: PyReadonlyArray3<u8>,
    palette: Vec<(u8, u8, u8)>,
) -> PyResult<Py<PyArray3<u8>>> {
    let arr = pixels.as_array();
    let result = map_pixels_to_palette_rs(arr, palette);
    let (h, w, c) = result.dim();
    let out = unsafe { PyArray3::<u8>::new(py, [h, w, c], false) };
    for ((y, x, ch), &v) in result.indexed_iter() {
        unsafe {
            *out.uget_mut([y, x, ch]) = v;
        }
    }
    Ok(out.into())
}

/// Downscale image using dominant color method (pure Rust, GPU-accelerated)
pub fn downscale_dominant_color_rs(
    image: ndarray::ArrayView3<u8>,
    scale: usize,
    threshold: f32,
) -> ndarray::Array3<u8> {
    #[cfg(feature = "gpu")]
    if let Some(ctx) = gpu::gpu_context() {
        if let Some(r) = gpu::downscale::downscale_dominant_gpu(&ctx, &image, scale, threshold) {
            return r;
        }
    }
    downscale_dominant(&image, scale, threshold)
}

/// Downscale image using dominant color method (Python binding)
#[cfg(feature = "python")]
#[pyfunction]
fn downscale_dominant_color(
    _py: Python<'_>,
    image: PyReadonlyArray3<u8>,
    scale: usize,
    threshold: f32,
) -> PyResult<Py<PyArray3<u8>>> {
    let img_array = image.as_array();
    let result = downscale_dominant_color_rs(img_array, scale, threshold);

    // Convert ndarray::Array3 to PyArray3
    let (h, w, c) = result.dim();
    let py_array = unsafe { PyArray3::<u8>::new(_py, [h, w, c], false) };

    // Copy data
    for ((y, x, ch), &value) in result.indexed_iter() {
        unsafe {
            *py_array.uget_mut([y, x, ch]) = value;
        }
    }

    Ok(py_array.into())
}

/// Downscale image using mode (most frequent color) method - pure Rust API
pub fn downscale_mode_color_rs(
    image: ndarray::ArrayView3<u8>,
    scale: usize,
) -> ndarray::Array3<u8> {
    #[cfg(feature = "gpu")]
    if let Some(ctx) = gpu::gpu_context() {
        if let Some(r) = gpu::downscale::downscale_mode_gpu(&ctx, &image, scale) {
            return r;
        }
    }
    downscale_mode(&image, scale)
}

/// Downscale image using mode (most frequent color) method (Python binding)
#[cfg(feature = "python")]
#[pyfunction]
fn downscale_mode_method(
    _py: Python<'_>,
    image: PyReadonlyArray3<u8>,
    scale: usize,
) -> PyResult<Py<PyArray3<u8>>> {
    let img_array = image.as_array();
    let result = downscale_mode_color_rs(img_array, scale);

    // Convert ndarray::Array3 to PyArray3
    let (h, w, c) = result.dim();
    let py_array = unsafe { PyArray3::<u8>::new(_py, [h, w, c], false) };

    // Copy data
    for ((y, x, ch), &value) in result.indexed_iter() {
        unsafe {
            *py_array.uget_mut([y, x, ch]) = value;
        }
    }

    Ok(py_array.into())
}

/// Count unique opaque colors in an image
#[cfg(feature = "python")]
#[pyfunction]
fn count_unique_colors(_py: Python<'_>, image: PyReadonlyArray3<u8>) -> PyResult<usize> {
    let img_array = image.as_array();
    let (height, width, channels) = img_array.dim();
    let has_alpha = channels == 4;

    use std::collections::HashSet;
    let mut unique_colors = HashSet::new();

    for y in 0..height {
        for x in 0..width {
            // Skip transparent pixels
            if has_alpha && img_array[(y, x, 3)] < 128 {
                continue;
            }

            // Pack RGB into 24-bit integer
            let r = img_array[(y, x, 0)] as u32;
            let g = img_array[(y, x, 1)] as u32;
            let b = img_array[(y, x, 2)] as u32;
            let color_key = (r << 16) | (g << 8) | b;

            unique_colors.insert(color_key);
        }
    }

    Ok(unique_colors.len())
}

/// Finalize pixels by ensuring binary alpha and black transparent pixels
#[cfg(feature = "python")]
#[pyfunction]
fn finalize_pixels_rust(
    _py: Python<'_>,
    image: PyReadonlyArray3<u8>,
) -> PyResult<Py<PyArray3<u8>>> {
    let img_array = image.as_array();
    let (height, width, channels) = img_array.dim();

    if channels < 4 {
        // No alpha channel, return as-is
        let result = unsafe { PyArray3::<u8>::new(_py, [height, width, channels], false) };
        for ((y, x, c), &value) in img_array.indexed_iter() {
            unsafe {
                *result.uget_mut([y, x, c]) = value;
            }
        }
        return Ok(result.into());
    }

    // Has alpha channel - process it
    let result = unsafe { PyArray3::<u8>::new(_py, [height, width, channels], false) };

    for y in 0..height {
        for x in 0..width {
            let alpha = img_array[(y, x, 3)];

            if alpha < 128 {
                // Transparent - set to black with 0 alpha
                unsafe {
                    *result.uget_mut([y, x, 0]) = 0;
                    *result.uget_mut([y, x, 1]) = 0;
                    *result.uget_mut([y, x, 2]) = 0;
                    *result.uget_mut([y, x, 3]) = 0;
                }
            } else {
                // Opaque - copy color with 255 alpha
                unsafe {
                    *result.uget_mut([y, x, 0]) = img_array[(y, x, 0)];
                    *result.uget_mut([y, x, 1]) = img_array[(y, x, 1)];
                    *result.uget_mut([y, x, 2]) = img_array[(y, x, 2)];
                    *result.uget_mut([y, x, 3]) = 255;
                }
            }
        }
    }

    Ok(result.into())
}

/// Content-adaptive downscaling using EM-C algorithm
#[cfg(feature = "python")]
#[pyfunction]
fn content_adaptive_downscale(
    py: Python<'_>,
    lab_image: PyReadonlyArray3<f32>,
    target_w: usize,
    target_h: usize,
    num_iterations: Option<usize>,
) -> PyResult<Py<PyArray3<f32>>> {
    let img_array = lab_image.as_array();
    let iterations = num_iterations.unwrap_or(5);

    let result = content_adaptive_downscale_rust(&img_array, target_w, target_h, iterations);

    // Convert ndarray::Array3 to PyArray3
    let (h, w, c) = result.dim();
    let py_array = unsafe { PyArray3::<f32>::new(py, [h, w, c], false) };

    // Copy data
    for ((y, x, ch), &value) in result.indexed_iter() {
        unsafe {
            *py_array.uget_mut([y, x, ch]) = value;
        }
    }

    Ok(py_array.into())
}

/// Make background transparent by flood-filling from specified starting points
#[cfg(feature = "python")]
#[pyfunction]
fn make_background_transparent(
    py: Python<'_>,
    image: PyReadonlyArray3<u8>,
    tolerance: Option<i32>,
    mode: Option<&str>,
) -> PyResult<Py<PyArray3<u8>>> {
    let img_array = image.as_array();
    let tol = tolerance.unwrap_or(10);
    let bg_mode = mode.unwrap_or("edges");

    let result = make_background_transparent_rust(&img_array, tol, bg_mode);

    // Convert ndarray::Array3 to PyArray3
    let (h, w, c) = result.dim();
    let py_array = unsafe { PyArray3::<u8>::new(py, [h, w, c], false) };

    // Copy data
    for ((y, x, ch), &value) in result.indexed_iter() {
        unsafe {
            *py_array.uget_mut([y, x, ch]) = value;
        }
    }

    Ok(py_array.into())
}

/// Python module
#[cfg(feature = "python")]
#[pymodule]
fn unfake(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(runs_based_detect, m)?)?;
    m.add_function(wrap_pyfunction!(map_pixels_to_palette, m)?)?;
    m.add_function(wrap_pyfunction!(downscale_dominant_color, m)?)?;
    m.add_function(wrap_pyfunction!(downscale_mode_method, m)?)?;
    m.add_function(wrap_pyfunction!(count_unique_colors, m)?)?;
    m.add_function(wrap_pyfunction!(finalize_pixels_rust, m)?)?;
    m.add_function(wrap_pyfunction!(content_adaptive_downscale, m)?)?;
    m.add_function(wrap_pyfunction!(make_background_transparent, m)?)?;
    m.add_class::<WuQuantizerRust>()?;
    Ok(())
}
