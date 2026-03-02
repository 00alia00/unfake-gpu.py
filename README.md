# unfake-gpu

* Moving to the GPU gives us another 10x speed improvement from unfake-opt. 
* All ops sub 1 second for example image. I get about 233 ms total on my laptop.
* Added demo with error metric and zoom tool (Has odd warping near edges of zoom tool)
* Added benchmarking python script

```
unfake benchmark
  image      : orig_2025-09-08-005621__0.png  (1312×976 px, 1.28 Mpx)
  iterations : 10  (+ 2 warmup per stage)
  colors     : 16   sig_bits=5
  method     : both
  rust       : yes

─── Rust ───
  scale detect                  mean 7.33 ms  min 4.53 ms  max 10.83 ms  med 7.28 ms  σ 2.17 ms  p95 10.83 ms
  downscale dominant            mean 13.84 ms  min 11.87 ms  max 15.81 ms  med 13.94 ms  σ 1.20 ms  p95 15.81 ms
  downscale mode                mean 15.19 ms  min 12.46 ms  max 20.70 ms  med 14.49 ms  σ 2.90 ms  p95 20.70 ms
  wu quantize                   mean 167.95 ms  min 149.31 ms  max 188.94 ms  med 167.83 ms  σ 11.65 ms  p95 188.94 ms
  palette map                   mean 14.42 ms  min 11.84 ms  max 19.66 ms  med 13.43 ms  σ 2.76 ms  p95 19.66 ms
  full pipeline                 mean 233.17 ms  min 219.32 ms  max 254.20 ms  med 233.29 ms  σ 10.52 ms  p95 254.20 ms

─── Python ───
  scale detect                  mean 3.706 s  min 3.662 s  max 3.778 s  med 3.695 s  σ 39.23 ms  p95 3.778 s
  downscale dominant            mean 36.014 s  min 35.805 s  max 36.265 s  med 36.019 s  σ 139.23 ms  p95 36.265 s
  downscale mode                mean 23.97 ms  min 23.24 ms  max 25.41 ms  med 23.70 ms  σ 718.8 µs  p95 25.41 ms
  wu quantize                   mean 53.391 s  min 51.971 s  max 62.651 s  med 52.316 s  σ 3.268 s  p95 62.651 s
  palette map                   mean 449.64 ms  min 446.50 ms  max 458.49 ms  med 448.78 ms  σ 3.30 ms  p95 458.49 ms
  full pipeline: skipped (no pure-Python mode)

─── Speedup summary (Python ÷ Rust mean) ───
  Stage                             Rust      Python    Speedup
  --------------------------  ----------  ----------  ---------
  scale detect                   7.33 ms     3.706 s   505.6×
  downscale dominant            13.84 ms    36.014 s  2602.7×
  downscale mode                15.19 ms    23.97 ms     1.6×
  wu quantize                  167.95 ms    53.391 s   317.9×
  palette map                   14.42 ms   449.64 ms    31.2×
  --------------------------  ----------  ----------  ---------
  OVERALL (pipeline)           218.73 ms    93.585 s   427.9×
```

---

Improve AI-generated pixel art through scale detection, color quantization, and smart downscaling — now significantly faster and more accurate thanks to algorithmic and performance enhancements.  
This optimized fork features **10–40× faster content-adaptive downscaling**, improved dominant color selection using **KMeans**, a new **hybrid downscaling method**, and additional preprocessing/postprocessing options for sharper, cleaner pixel art.

Based on the excellent work by:
- **Eugeniy Smirnov** ([jenissimo/unfake.js](https://github.com/jenissimo/unfake.js)) – Original JavaScript implementation  
- **Igor Bezkrovnyi** ([ibezkrovnyi/image-quantization](https://github.com/ibezkrovnyi/image-quantization)) – Image quantization algorithms  
- **Benjamin Paine** ([painebenjamin/unfake.py](https://github.com/painebenjamin/unfake.py)) – Original Python/Rust port  

## Examples  

Original Generated Image
![Original Generated Image](https://raw.githubusercontent.com/2dameneko/unfake-opt.py/main/samples/orig_2025-09-08-005621__0.png)

Original Dominant color method
![Original Dominant color method](https://raw.githubusercontent.com/2dameneko/unfake-opt.py/main/samples/orig_dom_pixelart_2025-09-08-005621__0_8x.png)

Enhanced Dominant color method
![Enhanced Dominant color method](https://raw.githubusercontent.com/2dameneko/unfake-opt.py/main/samples/enh_dom_pixelart_2025-09-08-005621__0_8x.png)

---

## ✨ Key Improvements (vs. original port)

- **10–40× faster `content-adaptive` downscaling** via optimized Rust implementation  
- **Improved `dominant` method**: uses **KMeans clustering** for better color selection, especially on complex pixel-art backgrounds  
- **New `hybrid` downscaling method**: automatically combine the best from `dominant` and `content-adaptive` methods
- **Preprocessing**: optional light blur (`--pre-filter`) before quantization to reduce noise  
- **Edge preservation**: `--edge-preserve` enhances contour sharpness during downscaling  
- **Post-sharpening**: experimental `--post-sharpen` (currently under refinement, produce mostly unwanted results)  
- **Adaptive threshold tuning**: `--iterations N` allows iterative refinement of the dominant color threshold for `dominant` method  

---

## Features

- **Automatic Scale Detection**: Detects the inherent scale of pixel art using both runs-based and edge-aware methods  
- **Advanced Color Quantization**: Wu algorithm with Rust acceleration + KMeans-enhanced dominant color selection  
- **Smart Downscaling**: Multiple methods including `dominant`, `median`, `mode`, `content-adaptive`, and new `hybrid`  
- **Image Cleanup**: Alpha binarization, morphological operations, jaggy edge removal  
- **Grid Snapping**: Automatic alignment to pixel grid for clean results  
- **Flexible API**: Both synchronous and asynchronous interfaces  
- **Blazing Fast**: Process a 1-megapixel image in under a second (with Rust acceleration)

### Upcoming

- Refined post-sharpening algorithm  
- Vectorization support  

---

## Installation

### From Source (recommended for now)

```bash
# Clone the optimized fork
git clone https://github.com/2dameneko/unfake-opt.py.git
cd unfake-opt

# Install with pip (includes Rust compilation)
pip install .

# Or for development
pip install -e .
```

### From precompiled wheel (after release)

> **Note**: This fork is not yet published on PyPI. Install from source to access all new features.

### Requirements

- Python 3.10+  
- Rust toolchain (for building from source)  
- OpenCV Python bindings  
- Pillow  
- NumPy  
- scikit-learn (for KMeans in `dominant` method)

---

## Usage

### Command Line

```bash
# Basic usage with auto-detection
unfake input.png

# Specify output file
unfake input.png -o output.png

# Control color palette size
unfake input.png -c 16                    # Maximum 16 colors
unfake input.png --auto-colors            # Auto-detect optimal color count

# Force specific scale
unfake input.png --scale 4                # Force 4x downscaling

# Choose downscaling method (NEW: hybrid!)
unfake input.png -m dominant              # Dominant color (KMeans-enhanced, default)
unfake input.png -m content-adaptive      # High-quality, now 10–40× faster
unfake input.png -m hybrid                # NEW: best of dominant + content-adaptive

# Enable new preprocessing/postprocessing
unfake input.png --pre-filter             # Apply light blur before quantization
unfake input.png --edge-preserve          # Preserve sharp edges during downscaling
unfake input.png --post-sharpen           # Experimental sharpening after quantization, not recommended for now
unfake input.png --iterations 5           # Refine dominant threshold over 5 iterations

# Enable cleanup operations
unfake input.png --cleanup morph,jaggy    # Morphological + jaggy edge cleanup

# Use fixed color palette
unfake input.png --palette palette.txt    # File with hex colors, one per line

# Adjust processing parameters
unfake input.png --alpha-threshold 200    # Higher threshold for alpha binarization
unfake input.png --threshold 0.1          # Initial dominant color threshold (0.0–1.0)
unfake input.png --no-snap                # Disable grid snapping

# Verbose output
unfake input.png -v                       # Show detailed processing info
```

### Python API

```python
import unfake

# Basic processing with defaults (now uses KMeans-enhanced dominant)
result = unfake.process_image_sync(
    "input.png",
    max_colors=32,
    detect_method="auto",
    downscale_method="hybrid",            # NEW option!
    cleanup={"morph": False, "jaggy": False},
    snap_grid=True,
    pre_filter=True,                      # NEW
    edge_preserve=True,                   # NEW
    post_sharpen=False,                   # Experimental
    iterations=3                          # NEW: threshold refinement
)

# Access results
processed_image = result['image']        # PIL Image
palette = result['palette']              # List of hex colors
manifest = result['manifest']            # Processing metadata
```

#### Asynchronous API (unchanged, but faster)

```python
import asyncio
import unfake

async def process_image_async():
    result = await unfake.process_image(
        "input.png",
        max_colors=16,
        downscale_method="hybrid",
        pre_filter=True,
        edge_preserve=True
    )
    result["image"].save("output.png")

asyncio.run(process_image_async())
```

---

### New & Updated Processing Options

#### Downscaling Methods
- **`dominant`** (default): Now uses **KMeans clustering** for more accurate dominant color selection — especially effective on textured or gradient pixel-art backgrounds  
- **`content-adaptive`**: Same high-quality algorithm, but **10–40× faster** thanks to Rust optimization  
- **`hybrid`** (**NEW**): Combine best from `dominant` and `content-adaptive` for optimal fidelity  
- **`median` / `mode` / `mean`**: Unchanged, for compatibility

#### New Flags
- `--pre-filter`: Applies a slight Gaussian blur before quantization to reduce noise and improve color coherence  
- `--edge-preserve`: Enhances edge contrast during downscaling to maintain crisp silhouettes  
- `--post-sharpen`: Experimental unsharp masking after quantization (not recommended for now)  
- `--iterations N`: Runs N iterations of threshold tuning for the `dominant` method to find optimal color dominance cutoff  

---

## Algorithm Details

### Dominant Color (Enhanced)
- Uses **KMeans clustering** in RGB space to group similar colors  
- Selects the cluster centroid with the most pixels as the representative color  
- Better handles dithering, gradients, and noisy backgrounds common in AI-generated pixel art  

### Hybrid Downscaling
- For each scale×scale block:
  - Compute results from both `dominant` and `content-adaptive`
  - Combine results based of details frequency (low - dominant, high - adaptive)

### Content Adaptive Downscaling
- Roughly O(num_kernels * num_pixels) => O(num_pixels) per iteration

---

## Versions
- **v1.0.7.1**: Merged changes from original v1.0.7
- **v1.0.4.1**: Initial fork from original v1.0.4, added new options (`hybrid`, `--pre-filter`, `--edge-preserve`, `--iterations`, etc.)

## Credits

This optimized fork builds upon:

- **[unfake.js](https://github.com/jenissimo/unfake.js)** by Eugeniy Smirnov  
- **[image-quantization](https://github.com/ibezkrovnyi/image-quantization)** by Igor Bezkrovnyi  
- **[unfake.py](https://github.com/painebenjamin/unfake.py)** by Benjamin Paine  

Additional references:  
- Wu, Xiaolin. "Efficient Statistical Computations for Optimal Color Quantization" (1992)  
- Kopf, Johannes and Dani Lischinski. "Depixelizing Pixel Art" (2011)  
- Scikit-learn: KMeans implementation for color clustering  

---

## License

MIT License
