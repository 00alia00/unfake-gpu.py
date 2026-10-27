use ndarray::Array3;
use std::{env, error::Error, io, time::Instant};
use unfake::{downscale_dominant_color_rs, quantizer::WuQuantizerRust, runs_based_detect_rs};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 {
        eprintln!("Usage: unfake-cli <input-image> <output-image>");
        std::process::exit(2);
    }

    let started = Instant::now();
    let image = image::open(&args[1])?.into_rgba8();
    let (width, height) = image.dimensions();
    let pixels = Array3::from_shape_vec((height as usize, width as usize, 4), image.into_raw())?;
    let detected_scale = runs_based_detect_rs(pixels.view());

    // Match the interactive demo's defaults: 2x, dominant, 16 colors.
    let downscaled = downscale_dominant_color_rs(pixels.view(), 2, 0.5);
    let (mapped, palette) = WuQuantizerRust::new(16, 5).quantize_rs(&downscaled.view());
    let (output_height, output_width, _) = mapped.dim();
    let output = image::RgbaImage::from_raw(
        output_width as u32,
        output_height as u32,
        mapped.into_raw_vec(),
    )
    .ok_or_else(|| io::Error::other("invalid output image dimensions"))?;
    output.save(&args[2])?;
    println!("Processed {width}x{height} -> {output_width}x{output_height}; {} colors, detected scale {detected_scale}x (used 2x); {:.2?}", palette.len(), started.elapsed());
    Ok(())
}
