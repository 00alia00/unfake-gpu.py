//! `unfake-demo` — interactive egui test bench for the unfake GPU library.
//!
//! Build and run:
//!   cargo run --features demo --bin unfake-demo
//!
//! Controls
//! --------
//! • "Open image…" file-picker  — load a PNG/JPEG/WEBP
//! • Color count slider         — palette size for Wu quantisation
//! • Scale slider               — downscale factor (1–16)
//! • Method radio               — Dominant / Mode / (Content-adaptive is CPU)
//! • Threshold slider           — dominant-color threshold %
//! • "Process"                  — run the full pipeline and display result
//! • GPU status badge           — green if wgpu device was acquired

use eframe::egui;

// ── Bring in the library's public Rust API ──────────────────────────────────

// When building with `demo` feature both `gpu` and the raw Rust API are
// available without needing pyo3.
use unfake::gpu;
use unfake::quantizer::WuQuantizerRust;
use unfake::{downscale_dominant_color_rs, downscale_mode_color_rs, runs_based_detect_rs};

// ── Image wrapper ────────────────────────────────────────────────────────────

/// RGBA pixel buffer shared between CPU processing and the egui texture.
#[derive(Clone)]
struct RgbaImage {
    width: usize,
    height: usize,
    data: Vec<u8>, // RGBA, row-major
}

impl RgbaImage {
    fn from_image_crate(img: image::DynamicImage) -> Self {
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        Self {
            width: w as usize,
            height: h as usize,
            data: rgba.into_raw(),
        }
    }

    fn to_ndarray(&self) -> ndarray::Array3<u8> {
        ndarray::Array3::from_shape_fn((self.height, self.width, 4), |(y, x, c)| {
            self.data[(y * self.width + x) * 4 + c]
        })
    }

    fn from_ndarray(arr: ndarray::Array3<u8>) -> Self {
        let (h, w, c) = arr.dim();
        let channels = c;
        let mut data = vec![255u8; h * w * 4];
        for y in 0..h {
            for x in 0..w {
                let base = (y * w + x) * 4;
                data[base] = arr[(y, x, 0)];
                data[base + 1] = arr[(y, x, 1)];
                data[base + 2] = arr[(y, x, 2)];
                data[base + 3] = if channels == 4 { arr[(y, x, 3)] } else { 255 };
            }
        }
        Self {
            width: w,
            height: h,
            data,
        }
    }

    fn to_egui_image(&self) -> egui::ColorImage {
        egui::ColorImage::from_rgba_unmultiplied([self.width, self.height], &self.data)
    }
}

// ── Error statistics ────────────────────────────────────────────────────────

/// Per-pixel error metrics between the downscaled image and the palette-mapped
/// result.  All values are in the 0-255 range (absolute pixel units).
#[derive(Clone, Default)]
struct ErrorStats {
    /// Root-mean-square error across all channels and pixels.
    rmse: f32,
    /// Peak signal-to-noise ratio (dB).  Higher is better; ≥40 dB is excellent.
    psnr: f32,
    /// Mean absolute error across all channels.
    mae: f32,
    /// Per-channel mean absolute errors.
    mae_r: f32,
    mae_g: f32,
    mae_b: f32,
    /// Largest single-channel difference found.
    max_err: f32,
}

impl ErrorStats {
    fn severity_color(e: f32) -> egui::Color32 {
        // Green < 5, Yellow < 15, Orange < 30, Red >= 30  (absolute 0-255 units)
        if e < 5.0 {
            egui::Color32::from_rgb(70, 200, 70)
        } else if e < 15.0 {
            egui::Color32::from_rgb(220, 200, 50)
        } else if e < 30.0 {
            egui::Color32::from_rgb(230, 130, 30)
        } else {
            egui::Color32::from_rgb(220, 60, 60)
        }
    }
}

/// Compute error metrics and produce an amplified difference heatmap.
/// Both arrays must have the same (H, W, C) shape.
fn compute_error(
    reference: &ndarray::ArrayView3<u8>,
    result: &ndarray::ArrayView3<u8>,
) -> (ErrorStats, RgbaImage) {
    let (h, w, c) = reference.dim();
    debug_assert_eq!((h, w, c), result.dim());
    let channels = c.min(3); // only R/G/B

    let total_pixels = h * w;
    let mut sum_sq = 0u64;
    let mut sum_abs = 0u64;
    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;
    let mut max_err = 0u8;

    let mut heatmap = vec![0u8; h * w * 4];

    for y in 0..h {
        for x in 0..w {
            let mut max_ch = 0u8;
            for ch in 0..channels {
                let a = reference[(y, x, ch)];
                let b = result[(y, x, ch)];
                let diff = a.abs_diff(b);
                sum_sq += (diff as u64) * (diff as u64);
                sum_abs += diff as u64;
                match ch {
                    0 => sum_r += diff as u64,
                    1 => sum_g += diff as u64,
                    2 => sum_b += diff as u64,
                    _ => {}
                }
                if diff > max_ch {
                    max_ch = diff;
                }
            }
            if max_ch > max_err {
                max_err = max_ch;
            }

            // Amplify the error 6× for visibility, clamp to 255
            let amp = (max_ch as u32 * 6).min(255) as u8;
            // Map magnitude through a heat palette: black→blue→cyan→green→yellow→red
            let (hr, hg, hb) = heat_color(amp);
            let base = (y * w + x) * 4;
            heatmap[base] = hr;
            heatmap[base + 1] = hg;
            heatmap[base + 2] = hb;
            heatmap[base + 3] = 255;
        }
    }

    let n = (total_pixels * channels) as f64;
    let rmse = ((sum_sq as f64 / n).sqrt()) as f32;
    let mae = (sum_abs as f64 / n) as f32;
    let mae_r = (sum_r as f64 / total_pixels as f64) as f32;
    let mae_g = (sum_g as f64 / total_pixels as f64) as f32;
    let mae_b = (sum_b as f64 / total_pixels as f64) as f32;
    let psnr = if rmse < 0.001 {
        f32::INFINITY
    } else {
        20.0 * (255.0_f32 / rmse).log10()
    };

    let stats = ErrorStats {
        rmse,
        psnr,
        mae,
        mae_r,
        mae_g,
        mae_b,
        max_err: max_err as f32,
    };
    let img = RgbaImage {
        width: w,
        height: h,
        data: heatmap,
    };
    (stats, img)
}

/// Map a byte value 0-255 through a classic "hot" heat palette.
fn heat_color(v: u8) -> (u8, u8, u8) {
    // Five-stop gradient: black → blue → cyan → green → yellow → red
    let t = v as f32 / 255.0;
    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, 0.0, s)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 1.0 - s)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Draw a texture fitted to `max_size`, then show a magnifier popup on hover.
///
/// The popup renders a 5× zoom of the area under the cursor using egui UV
/// cropping — no extra CPU work per frame.
fn show_zoomable_image(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    texture: &egui::TextureHandle,
    max_size: egui::Vec2,
    id: egui::Id,
    zoom: f32,
) {
    let tex_size = texture.size_vec2();
    if tex_size.x < 1.0 || tex_size.y < 1.0 {
        return;
    }

    // Fit into max_size preserving aspect ratio.
    let aspect = tex_size.y / tex_size.x;
    let w = max_size.x.min(max_size.y / aspect.max(f32::EPSILON));
    let h = (w * aspect).min(max_size.y);
    let display_size = egui::Vec2::new(w.max(1.0), h.max(1.0));

    let (response, painter) =
        ui.allocate_painter(display_size, egui::Sense::hover());
    let rect = response.rect;

    // Draw full image.
    painter.image(
        texture.id(),
        rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    if let Some(pos) = response.hover_pos() {
        // Subtle crosshair.
        let dim = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 60);
        painter.hline(rect.x_range(), pos.y, egui::Stroke::new(0.5, dim));
        painter.vline(pos.x, rect.y_range(), egui::Stroke::new(0.5, dim));

        // UV coordinates under the cursor.
        let uv_cx = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
        let uv_cy = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0);

        // half_r is the half-width of the UV window; smaller = more zoom.
        let half_r = (0.5 / zoom.max(1.0)).min(0.5);
        let u0 = (uv_cx - half_r).max(0.0);
        let u1 = (uv_cx + half_r).min(1.0);
        let v0 = (uv_cy - half_r).max(0.0);
        let v1 = (uv_cy + half_r).min(1.0);
        let zoom_uv =
            egui::Rect::from_min_max(egui::pos2(u0, v0), egui::pos2(u1, v1));
        let zoom_factor = 1.0 / ((u1 - u0).max(f32::EPSILON));

        let popup_size = egui::Vec2::splat(200.0);
        // Quadrant placement: pick the screen quadrant that keeps the popup
        // away from the cursor, so it never overlaps and causes hover flicker.
        let screen = ctx.screen_rect();
        let gap = 14.0_f32;
        // Prefer right side if there is room, otherwise left.
        let px = if pos.x + gap + popup_size.x <= screen.max.x - 4.0 {
            pos.x + gap
        } else {
            pos.x - gap - popup_size.x
        };
        // Prefer above cursor if there is room, otherwise below.
        let py = if pos.y - gap - popup_size.y >= screen.min.y + 4.0 {
            pos.y - gap - popup_size.y
        } else {
            pos.y + gap
        };
        // Final clamp to keep inside the screen.
        let px = px.clamp(screen.min.x + 4.0, screen.max.x - popup_size.x - 4.0);
        let py = py.clamp(screen.min.y + 4.0, screen.max.y - popup_size.y - 4.0);

        egui::Area::new(id.with("magnifier"))
            .fixed_pos(egui::pos2(px, py))
            .order(egui::Order::Tooltip)
            .interactable(false)
            .show(ctx, |ui| {
                let (r, p) = ui.allocate_painter(popup_size, egui::Sense::hover());
                p.rect_filled(r.rect.expand(2.0), 4.0, egui::Color32::from_gray(18));
                p.image(texture.id(), r.rect, zoom_uv, egui::Color32::WHITE);
                p.rect_stroke(
                    r.rect,
                    0.0,
                    egui::Stroke::new(1.5, egui::Color32::from_gray(160)),
                    egui::StrokeKind::Outside,
                );
                p.text(
                    r.rect.right_bottom() + egui::Vec2::new(-4.0, -4.0),
                    egui::Align2::RIGHT_BOTTOM,
                    format!("{:.0}×", zoom_factor),
                    egui::FontId::proportional(11.0),
                    egui::Color32::from_gray(200),
                );
            });
    }
}

// ── App state ────────────────────────────────────────────────────────────────

#[derive(PartialEq, Clone, Copy)]
enum DownscaleMethod {
    Dominant,
    Mode,
}

struct UnfakeDemo {
    // ── user settings ──
    color_count: usize,
    scale: usize,
    threshold: f32,
    sig_bits: u8,
    method: DownscaleMethod,

    // ── loaded images ──
    original: Option<RgbaImage>,
    processed: Option<RgbaImage>,
    heatmap_image: Option<RgbaImage>, // pre-computed diff heatmap (set in process())
    palette: Vec<(u8, u8, u8)>,
    scale_result: Option<u32>,

    // ── egui textures ──
    orig_texture: Option<egui::TextureHandle>,
    proc_texture: Option<egui::TextureHandle>,
    error_texture: Option<egui::TextureHandle>,

    // ── status / errors ──
    gpu_available: bool,
    status_msg: String,
    error_stats: Option<ErrorStats>,

    // ── view settings ──
    zoom_level: f32,
}

impl Default for UnfakeDemo {
    fn default() -> Self {
        let gpu_available = gpu::gpu_context().is_some();
        Self {
            color_count: 16,
            scale: 2,
            threshold: 0.5,
            sig_bits: 5,
            method: DownscaleMethod::Dominant,
            original: None,
            processed: None,
            heatmap_image: None,
            palette: Vec::new(),
            scale_result: None,
            orig_texture: None,
            proc_texture: None,
            error_texture: None,
            gpu_available,
            error_stats: None,
            zoom_level: 5.0,
            status_msg: if gpu_available {
                "GPU ready".to_string()
            } else {
                "No GPU — using CPU fallback".to_string()
            },
        }
    }
}

impl UnfakeDemo {
    fn load_image(&mut self, path: &std::path::Path) {
        match image::open(path) {
            Ok(img) => {
                self.original = Some(RgbaImage::from_image_crate(img));
                self.processed = None;
                self.heatmap_image = None;
                self.palette.clear();
                self.orig_texture = None;
                self.proc_texture = None;
                self.error_texture = None;
                self.error_stats = None;
                self.status_msg = format!(
                    "Loaded {} ({}×{})",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    self.original.as_ref().unwrap().width,
                    self.original.as_ref().unwrap().height,
                );
            }
            Err(e) => self.status_msg = format!("Load error: {e}"),
        }
    }

    fn process(&mut self) {
        let Some(orig) = &self.original else {
            self.status_msg = "No image loaded.".to_string();
            return;
        };

        let arr = orig.to_ndarray();
        let view = arr.view();
        self.status_msg = "Processing…".to_string();

        // 1. Scale detection
        let detected_scale = runs_based_detect_rs(view);
        self.scale_result = Some(detected_scale);
        let effective_scale = if self.scale > 1 {
            self.scale
        } else {
            detected_scale.max(1) as usize
        };

        // 2. Downscale
        let downscaled = match self.method {
            DownscaleMethod::Dominant => {
                downscale_dominant_color_rs(view, effective_scale, self.threshold)
            }
            DownscaleMethod::Mode => downscale_mode_color_rs(view, effective_scale),
        };

        // 3. Quantise
        let mut quantizer = WuQuantizerRust::new(self.color_count, self.sig_bits);
        let ds_view = downscaled.view();
        let (mapped, palette) = quantizer.quantize_rs(&ds_view);

        // 4. Error metrics: compare downscaled (pre-palette) vs palette-mapped
        let (stats, error_img) = compute_error(&ds_view, &mapped.view());

        self.palette = palette;
        self.processed = Some(RgbaImage::from_ndarray(mapped));
        self.heatmap_image = Some(error_img); // stored for texture build in update()
        self.proc_texture = None;
        self.error_texture = None; // will be rebuilt from heatmap_image at top of next update()
        self.error_stats = Some(stats);
        self.status_msg = format!(
            "Done — palette {} colours, detected scale {}×  |  RMSE {:.2}  PSNR {:.1} dB",
            self.palette.len(),
            detected_scale,
            self.error_stats.as_ref().map_or(0.0, |s| s.rmse),
            self.error_stats.as_ref().map_or(0.0, |s| s.psnr),
        );
    }
}

// ── egui App impl ──────────────────────────────────────────────────────────────

impl eframe::App for UnfakeDemo {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Build error heatmap texture once from the image stored by process().
        // Must happen before any panel renders so the texture is ready on the same frame.
        if self.error_texture.is_none() {
            if let Some(hm) = &self.heatmap_image {
                self.error_texture = Some(ctx.load_texture(
                    "error_heatmap",
                    hm.to_egui_image(),
                    egui::TextureOptions::NEAREST,
                ));
            }
        }

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // GPU badge
                let (badge_label, badge_color) = if self.gpu_available {
                    ("GPU", egui::Color32::from_rgb(60, 180, 60))
                } else {
                    ("CPU only", egui::Color32::from_rgb(200, 120, 30))
                };
                ui.label(egui::RichText::new(badge_label).color(badge_color).strong());
                ui.separator();

                if ui.button("Open image…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp"])
                        .pick_file()
                    {
                        self.load_image(&path);
                    }
                }
                ui.separator();
                ui.label(&self.status_msg);
            });
        });

        egui::SidePanel::left("controls")
            .min_width(220.0)
            .show(ctx, |ui| {
                ui.heading("Parameters");
                ui.separator();

                ui.label("Color count");
                ui.add(egui::Slider::new(&mut self.color_count, 2..=256).suffix(" colors"));

                ui.label("Significant bits (quantiser)");
                ui.add(egui::Slider::new(&mut self.sig_bits, 4..=7));

                ui.separator();
                ui.label("Downscale factor");
                ui.add(egui::Slider::new(&mut self.scale, 1..=16).suffix("×"));

                ui.label("Method");
                ui.radio_value(
                    &mut self.method,
                    DownscaleMethod::Dominant,
                    "Dominant color",
                );
                ui.radio_value(
                    &mut self.method,
                    DownscaleMethod::Mode,
                    "Mode (most frequent)",
                );

                if self.method == DownscaleMethod::Dominant {
                    ui.label("Dominant threshold");
                    ui.add(egui::Slider::new(&mut self.threshold, 0.0..=1.0).fixed_decimals(2));
                }

                ui.separator();
                ui.label("Magnifier zoom");
                ui.add(
                    egui::Slider::new(&mut self.zoom_level, 2.0..=20.0)
                        .suffix("×")
                        .fixed_decimals(0),
                );

                ui.separator();
                if let Some(scale) = self.scale_result {
                    ui.label(format!("Detected scale: {}×", scale));
                }

                ui.separator();
                let can_process = self.original.is_some();
                ui.add_enabled_ui(can_process, |ui| {
                    if ui.button("▶  Process").clicked() {
                        self.process();
                    }
                });

                // Palette swatches
                if !self.palette.is_empty() {
                    ui.separator();
                    ui.label(format!("Palette ({} colors)", self.palette.len()));
                    let swatch_size = 20.0;
                    let cols = 8;
                    egui::Grid::new("palette_grid")
                        .spacing([2.0, 2.0])
                        .show(ui, |ui| {
                            for (i, &(r, g, b)) in self.palette.iter().enumerate() {
                                let (response, painter) = ui.allocate_painter(
                                    egui::Vec2::splat(swatch_size),
                                    egui::Sense::hover(),
                                );
                                painter.rect_filled(
                                    response.rect,
                                    2.0,
                                    egui::Color32::from_rgb(r, g, b),
                                );
                                response.on_hover_text(format!("#{:02X}{:02X}{:02X}", r, g, b));
                                if (i + 1) % cols == 0 {
                                    ui.end_row();
                                }
                            }
                        });
                }
            });

        // ── Error visualisation panel (bottom) ─────────────────────────────────
        egui::TopBottomPanel::bottom("error_panel")
            .resizable(true)
            .min_height(160.0)
            .max_height(340.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.heading("Quantisation Error");
                ui.separator();

                if let Some(stats) = &self.error_stats.clone() {
                    ui.horizontal(|ui| {
                        // ── 1. Heatmap ───────────────────────────────────────
                        ui.vertical(|ui| {
                            ui.label("Error heatmap (6× amplified)");
                            // Build legend row
                            ui.horizontal(|ui| {
                                for (label, v) in [
                                    ("0", 0u8),
                                    ("low", 42),
                                    ("mid", 128),
                                    ("high", 210),
                                    ("max", 255),
                                ] {
                                    let (r, g, b) = heat_color(v);
                                    let (resp, painter) = ui.allocate_painter(
                                        egui::Vec2::new(28.0, 12.0),
                                        egui::Sense::hover(),
                                    );
                                    painter.rect_filled(
                                        resp.rect,
                                        0.0,
                                        egui::Color32::from_rgb(r, g, b),
                                    );
                                    ui.label(egui::RichText::new(label).small());
                                }
                            });
                            if let Some(err_img) = self.error_texture.as_ref() {
                                // Fixed height cap; available_height() is near-zero
                                // inside a nested horizontal→vertical layout.
                                show_zoomable_image(
                                    ui,
                                    ctx,
                                    err_img,
                                    egui::Vec2::new(260.0, 110.0),
                                    egui::Id::new("img_heatmap"),
                                    self.zoom_level,
                                );
                            } else {
                                ui.label("(no heatmap yet)");
                            }
                        });

                        ui.separator();

                        // ── 2. Numeric stats ─────────────────────────────────
                        ui.vertical(|ui| {
                            ui.label(egui::RichText::new("Metrics").strong());
                            ui.add_space(4.0);

                            let metric = |ui: &mut egui::Ui, name: &str, val: f32, unit: &str| {
                                let color = ErrorStats::severity_color(val);
                                ui.horizontal(|ui| {
                                    ui.label(format!("{name:<12}"));
                                    ui.label(
                                        egui::RichText::new(format!("{val:.2} {unit}"))
                                            .color(color)
                                            .strong(),
                                    );
                                });
                            };

                            metric(ui, "RMSE", stats.rmse, "px");
                            metric(ui, "MAE", stats.mae, "px");
                            metric(ui, "Max err", stats.max_err, "px");

                            ui.add_space(4.0);
                            ui.separator();
                            // PSNR — for PSNR higher is better, invert the colour logic
                            let psnr_color = if stats.psnr > 40.0 {
                                egui::Color32::from_rgb(70, 200, 70)
                            } else if stats.psnr > 30.0 {
                                egui::Color32::from_rgb(220, 200, 50)
                            } else if stats.psnr > 20.0 {
                                egui::Color32::from_rgb(230, 130, 30)
                            } else {
                                egui::Color32::from_rgb(220, 60, 60)
                            };
                            ui.horizontal(|ui| {
                                ui.label("PSNR        ");
                                ui.label(
                                    egui::RichText::new(if stats.psnr.is_infinite() {
                                        "∞ dB (lossless)".to_string()
                                    } else {
                                        format!("{:.1} dB", stats.psnr)
                                    })
                                    .color(psnr_color)
                                    .strong(),
                                );
                            });
                        });

                        ui.separator();

                        // ── 3. Per-channel bar chart ──────────────────────────
                        ui.vertical(|ui| {
                            ui.label(egui::RichText::new("Per-channel MAE").strong());
                            ui.add_space(6.0);

                            let bar_w = 40.0_f32;
                            let bar_h = 100.0_f32;
                            let gap = 12.0_f32;
                            let total_w = (bar_w + gap) * 3.0 + gap;
                            let (resp, painter) = ui.allocate_painter(
                                egui::Vec2::new(total_w, bar_h + 24.0),
                                egui::Sense::hover(),
                            );
                            let origin = resp.rect.left_top();

                            let max_val = stats.mae_r.max(stats.mae_g).max(stats.mae_b).max(1.0);
                            let channels = [
                                ("R", stats.mae_r, egui::Color32::from_rgb(220, 60, 60)),
                                ("G", stats.mae_g, egui::Color32::from_rgb(60, 200, 60)),
                                ("B", stats.mae_b, egui::Color32::from_rgb(60, 100, 220)),
                            ];

                            for (i, (label, val, color)) in channels.iter().enumerate() {
                                let x = origin.x + gap + i as f32 * (bar_w + gap);
                                let filled = (val / max_val * bar_h).max(1.0);
                                let top_y = origin.y + bar_h - filled;
                                let bot_y = origin.y + bar_h;

                                // Background trough
                                painter.rect_filled(
                                    egui::Rect::from_min_max(
                                        egui::pos2(x, origin.y),
                                        egui::pos2(x + bar_w, bot_y),
                                    ),
                                    3.0,
                                    egui::Color32::from_gray(45),
                                );
                                // Filled bar
                                painter.rect_filled(
                                    egui::Rect::from_min_max(
                                        egui::pos2(x, top_y),
                                        egui::pos2(x + bar_w, bot_y),
                                    ),
                                    3.0,
                                    *color,
                                );
                                // Value label above bar
                                painter.text(
                                    egui::pos2(x + bar_w / 2.0, top_y - 14.0),
                                    egui::Align2::CENTER_CENTER,
                                    format!("{:.1}", val),
                                    egui::FontId::proportional(11.0),
                                    egui::Color32::WHITE,
                                );
                                // Channel label below bar
                                painter.text(
                                    egui::pos2(x + bar_w / 2.0, bot_y + 10.0),
                                    egui::Align2::CENTER_CENTER,
                                    *label,
                                    egui::FontId::proportional(12.0),
                                    *color,
                                );
                            }

                            // Scale line + label at 50% mark
                            let y_mid = origin.y + bar_h / 2.0;
                            painter.hline(
                                origin.x..=(origin.x + total_w),
                                y_mid,
                                egui::Stroke::new(0.5, egui::Color32::from_gray(90)),
                            );
                            painter.text(
                                egui::pos2(origin.x + total_w - 2.0, y_mid - 6.0),
                                egui::Align2::RIGHT_CENTER,
                                format!("{:.1}", max_val * 0.5),
                                egui::FontId::proportional(9.0),
                                egui::Color32::from_gray(130),
                            );
                        });
                    });
                } else {
                    ui.vertical_centered(|ui| {
                        let space = (ui.available_height() / 3.0 - 10.0).max(0.0);
                        ui.add_space(space);
                        ui.label(
                            egui::RichText::new(
                                "Load an image and press ▶ Process to see error metrics.",
                            )
                            .color(egui::Color32::from_gray(140))
                            .italics(),
                        );
                    });
                }
            });

        // ── Main image area ─────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            let available = ui.available_size();
            // Guard against tiny or negative space (e.g. window very small)
            if available.x < 10.0 || available.y < 10.0 {
                return;
            }
            let half_w = (available.x / 2.0 - 8.0).max(1.0);
            let img_max_h = (available.y - 10.0).max(1.0);

            ui.horizontal(|ui| {
                // Original image
                ui.vertical(|ui| {
                    ui.label("Original");
                    if let Some(orig) = &self.original {
                        let texture = self.orig_texture.get_or_insert_with(|| {
                            ctx.load_texture(
                                "original",
                                orig.to_egui_image(),
                                egui::TextureOptions::LINEAR,
                            )
                        });
                        show_zoomable_image(
                            ui, ctx, texture,
                            egui::Vec2::new(half_w, img_max_h),
                            egui::Id::new("img_orig"),
                            self.zoom_level,
                        );
                    } else {
                        ui.label("(no image loaded)");
                    }
                });

                ui.separator();

                // Processed image
                ui.vertical(|ui| {
                    ui.label("Processed");
                    if let Some(proc) = &self.processed {
                        let texture = self.proc_texture.get_or_insert_with(|| {
                            ctx.load_texture(
                                "processed",
                                proc.to_egui_image(),
                                egui::TextureOptions::NEAREST,
                            )
                        });
                        show_zoomable_image(
                            ui, ctx, texture,
                            egui::Vec2::new(half_w, img_max_h),
                            egui::Id::new("img_proc"),
                            self.zoom_level,
                        );
                    } else {
                        ui.label("(press Process)");
                    }
                });
            });
        });
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("unfake demo")
            .with_inner_size([1200.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "unfake-demo",
        options,
        Box::new(|_cc| Ok(Box::new(UnfakeDemo::default()))),
    )
}
