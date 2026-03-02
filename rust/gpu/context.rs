//! Owns the wgpu `Device`, `Queue`, and cached `ComputePipeline`s.

use wgpu::util::DeviceExt;

/// Shared GPU state.  Created once (see `gpu::gpu_context()`).
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    // Cached pipelines (built lazily on first use)
    pub palette_map_pipeline: wgpu::ComputePipeline,
    pub downscale_dominant_pipeline: wgpu::ComputePipeline,
    pub downscale_mode_pipeline: wgpu::ComputePipeline,
    pub histogram_pipeline: wgpu::ComputePipeline,
}

impl GpuContext {
    /// Asynchronously create a GPU context.  Returns `Err` if no suitable adapter found.
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // wgpu v29: `create_instance` takes `&InstanceDescriptor`
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // wgpu v25+: `request_adapter` returns `Result`
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("No GPU adapter: {e}"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("unfake-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            })
            .await?;

        let palette_map_pipeline = build_palette_map_pipeline(&device);
        let downscale_dominant_pipeline = build_downscale_dominant_pipeline(&device);
        let downscale_mode_pipeline = build_downscale_mode_pipeline(&device);
        let histogram_pipeline = build_histogram_pipeline(&device);

        Ok(Self {
            device,
            queue,
            palette_map_pipeline,
            downscale_dominant_pipeline,
            downscale_mode_pipeline,
            histogram_pipeline,
        })
    }

    /// Convenience: write a slice to a newly-allocated `STORAGE | COPY_SRC` buffer.
    pub fn upload_storage<T: bytemuck::Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Convenience: create a writeable storage buffer of `byte_size` bytes.
    pub fn create_output_storage(&self, label: &str, byte_size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Convenience: create a staging (MAP_READ | COPY_DST) buffer for readback.
    pub fn create_readback(&self, label: &str, byte_size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// Upload a small plain-struct to a uniform buffer.
    pub fn upload_uniform<T: bytemuck::Pod>(&self, label: &str, data: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Submit a command encoder and block until idle (v25+: PollType).
    pub fn submit_and_wait(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }

    /// Read back a buffer from the GPU queue synchronously.
    /// Uses the v27 `map_buffer_on_submit` pattern via a blocking poll.
    pub fn readback_buffer(&self, src: &wgpu::Buffer, byte_size: u64) -> Vec<u8> {
        let staging = self.create_readback("readback", byte_size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, byte_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map synchronously (only suitable for offline / non-frame usage)
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().unwrap().expect("buffer map failed");

        let data = staging.slice(..).get_mapped_range().to_vec();
        staging.unmap();
        data
    }
}

// ── Pipeline constructors ────────────────────────────────────────────────────────

fn build_palette_map_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/palette_map.wgsl"));
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("palette_map"),
        // Auto-layout: wgpu derives the bind group layout from the shader reflection.
        layout: None,
        module: &shader,
        // v23+: entry_point is Optional; None = single entry point in shader
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn build_downscale_dominant_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader =
        device.create_shader_module(wgpu::include_wgsl!("shaders/downscale_dominant.wgsl"));
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("downscale_dominant"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn build_downscale_mode_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/downscale_mode.wgsl"));
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("downscale_mode"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn build_histogram_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/histogram.wgsl"));
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("histogram"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}
