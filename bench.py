#!/usr/bin/env python3
"""
bench.py — pipeline speed benchmark for unfake.

Measures each stage of the pipeline individually and reports
mean / min / max / median / stddev / p95 over N warm iterations.

Usage
-----
  python bench.py IMAGE [options]

Examples
  python bench.py samples/orig.png
  python bench.py samples/orig.png -n 20 --scale 4 --colors 32
  python bench.py samples/orig.png -n 10 --method mode --no-rust
  python bench.py samples/orig.png --list-stages
"""

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Silence the unfake INFO logger so pipeline stage output stays clean.
logging.getLogger("unfake.py").setLevel(logging.WARNING)
logging.getLogger("unfake").setLevel(logging.WARNING)


# ── helpers ──────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


def _ms(seconds: float) -> str:
    ms = seconds * 1000
    if ms < 1:
        return f"{ms*1000:.1f} µs"
    if ms < 1000:
        return f"{ms:.2f} ms"
    return f"{ms/1000:.3f} s"


def timeit(fn, *args, **kwargs) -> tuple:
    """Run fn(*args, **kwargs), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def stats(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    mean  = statistics.mean(s)
    stdev = statistics.stdev(s) if n > 1 else 0.0
    p95   = s[int(n * 0.95)] if n >= 20 else s[-1]
    return dict(n=n, mean=mean, min=s[0], max=s[-1],
                median=statistics.median(s), stdev=stdev, p95=p95)


def print_row(label: str, s: dict, baseline: float | None = None) -> None:
    speedup = ""
    if baseline is not None and s["mean"] > 0:
        ratio = baseline / s["mean"]
        color = GREEN if ratio >= 1 else YELLOW
        speedup = "  " + _c(color, f"({ratio:.2f}×)")
    print(
        f"  {BOLD}{label:<28}{RESET}"
        f"  mean {_c(CYAN, _ms(s['mean']))}"
        f"  min {_ms(s['min'])}"
        f"  max {_ms(s['max'])}"
        f"  med {_ms(s['median'])}"
        f"  σ {_ms(s['stdev'])}"
        f"  p95 {_ms(s['p95'])}"
        f"{speedup}"
    )


def print_header(title: str) -> None:
    print()
    print(_c(BOLD + CYAN, f"─── {title} ───"))


# ── benchmark stages ─────────────────────────────────────────────────────────

def bench_scale_detect(img_np: np.ndarray, n: int, rust: bool) -> dict:
    """Stage 1: runs-based scale detection."""
    samples = []
    scale = 1
    if rust:
        try:
            from unfake.unfake import runs_based_detect as _detect
        except ImportError:
            from unfake.pixel_rust_integration import runs_based_detect_accelerated as _detect
    else:
        from unfake.pixel import runs_based_detect as _detect

    # warmup
    for _ in range(2):
        _detect(img_np)
    for _ in range(n):
        scale, elapsed = timeit(_detect, img_np)
        samples.append(elapsed)
    return {"stats": stats(samples), "scale": scale}


def bench_downscale_dominant(img_np: np.ndarray, scale: int,
                              threshold: float, n: int, rust: bool) -> dict:
    """Stage 2a: dominant-colour downscale."""
    samples = []
    result = None
    if rust:
        try:
            from unfake.unfake import downscale_dominant_color as _fn
            call = lambda: _fn(img_np, scale, threshold)
        except ImportError:
            from unfake.pixel_rust_integration import downscale_dominant_color_accelerated as _fn
            call = lambda: _fn(img_np, scale, threshold)
    else:
        from unfake.pixel import downscale_by_dominant_color as _fn
        call = lambda: _fn(img_np, scale, threshold)

    for _ in range(2):
        call()
    for _ in range(n):
        result, elapsed = timeit(call)
        samples.append(elapsed)
    return {"stats": stats(samples), "result": result}


def bench_downscale_mode(img_np: np.ndarray, scale: int,
                          n: int, rust: bool) -> dict:
    """Stage 2b: mode (most-frequent colour) downscale."""
    samples = []
    result = None
    if rust:
        try:
            from unfake.unfake import downscale_mode_method as _fn
            call = lambda: _fn(img_np, scale)
        except ImportError:
            from unfake.pixel_rust_integration import downscale_mode_accelerated as _fn
            call = lambda: _fn(img_np, scale)
    else:
        from unfake.pixel import downscale_block as _py
        call = lambda: _py(img_np, scale, "mode")

    for _ in range(2):
        call()
    for _ in range(n):
        result, elapsed = timeit(call)
        samples.append(elapsed)
    return {"stats": stats(samples), "result": result}


def bench_quantize(img_np: np.ndarray, colors: int, sig_bits: int,
                   n: int, rust: bool) -> dict:
    """Stage 3: Wu colour quantization (build histogram + cut boxes)."""
    samples = []
    palette = []
    if rust:
        try:
            from unfake.pixel_rust_integration import WuQuantizerAccelerated
            def call():
                q = WuQuantizerAccelerated(max_colors=colors, significant_bits=sig_bits)
                return q.quantize(img_np)
        except Exception:
            from unfake.pixel_rust_integration import WuQuantizerAccelerated
            def call():
                q = WuQuantizerAccelerated(max_colors=colors, significant_bits=sig_bits)
                return q.quantize(img_np)
    else:
        from unfake.wu_quantizer import WuQuantizer
        def call():
            q = WuQuantizer(colors, sig_bits)
            return q.quantize(img_np)

    for _ in range(2):
        call()
    for _ in range(n):
        result, elapsed = timeit(call)
        samples.append(elapsed)
        if isinstance(result, tuple):
            palette = result[1] if len(result) > 1 else []
    return {"stats": stats(samples), "palette": palette}


def bench_palette_map(img_np: np.ndarray, palette: list,
                      n: int, rust: bool) -> dict:
    """Stage 4: map every pixel to nearest palette entry."""
    if not palette:
        return {"stats": None}
    samples = []
    if rust:
        try:
            from unfake.pixel_rust_integration import map_pixels_to_palette_accelerated
            call = lambda: map_pixels_to_palette_accelerated(img_np, palette)
        except Exception:
            from unfake.pixel_rust_integration import map_pixels_to_palette_accelerated as _fn
            call = lambda: _fn(img_np, palette)
    else:
        from unfake.pixel_rust_integration import map_pixels_to_palette_accelerated as _fn
        call = lambda: _fn(img_np, palette)

    for _ in range(2):
        call()
    for _ in range(n):
        _, elapsed = timeit(call)
        samples.append(elapsed)
    return {"stats": stats(samples)}


def bench_full_pipeline(img_np: np.ndarray, scale: int, threshold: float,
                         colors: int, sig_bits: int, method: str,
                         n: int, rust: bool) -> dict:
    """Full end-to-end pipeline measured as a single unit."""
    from unfake import process_image_sync
    samples = []
    for _ in range(2):
        try:
            process_image_sync(img_np, scale=scale, colors=colors,
                               method=method, threshold=threshold)
        except Exception:
            pass
    for _ in range(n):
        try:
            _, elapsed = timeit(
                process_image_sync, img_np,
                scale=scale, colors=colors, method=method, threshold=threshold,
            )
        except TypeError:
            # signature varies — try positional
            _, elapsed = timeit(process_image_sync, img_np)
        samples.append(elapsed)
    return {"stats": stats(samples)}


# ── main ─────────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def check_rust() -> bool:
    try:
        import unfake.unfake  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="unfake pipeline benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image", help="Path to test image (PNG/JPEG/WEBP)")
    parser.add_argument("-n", "--iterations", type=int, default=10,
                        help="Number of timed iterations (default: 10)")
    parser.add_argument("--scale", type=int, default=0,
                        help="Downscale factor; 0 = auto-detect (default: 0)")
    parser.add_argument("--colors", type=int, default=16,
                        help="Palette size for quantization (default: 16)")
    parser.add_argument("--sig-bits", type=int, default=5,
                        help="Wu quantizer significant bits 4-7 (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Dominant-color threshold 0-1 (default: 0.5)")
    parser.add_argument("--method", choices=["dominant", "mode", "both"],
                        default="both",
                        help="Downscale method to bench (default: both)")
    parser.add_argument("--no-rust", action="store_true",
                        help="Force Python implementations (skip Rust paths)")
    parser.add_argument("--compare", action="store_true",
                        help="When Rust is available, run both paths and show speedup")
    parser.add_argument("--list-stages", action="store_true",
                        help="Print stage list and exit")
    args = parser.parse_args()

    stages = [
        "1. Scale detection",
        "2a. Downscale (dominant)",
        "2b. Downscale (mode)",
        "3.  Wu quantization",
        "4.  Palette map",
        "5.  Full pipeline",
    ]
    if args.list_stages:
        for s in stages:
            print(s)
        return

    img_path = Path(args.image)
    if not img_path.exists():
        print(_c(RED, f"Error: file not found: {img_path}"), file=sys.stderr)
        sys.exit(1)

    rust_present = check_rust() and not args.no_rust
    use_rust     = rust_present
    n            = args.iterations

    img_np = load_image(img_path)
    h, w = img_np.shape[:2]

    print()
    print(_c(BOLD, "unfake benchmark"))
    print(f"  image      : {img_path.name}  ({w}×{h} px, {w*h/1e6:.2f} Mpx)")
    print(f"  iterations : {n}  (+ 2 warmup per stage)")
    print(f"  colors     : {args.colors}   sig_bits={args.sig_bits}")
    print(f"  method     : {args.method}")
    print(f"  rust       : {'yes' if rust_present else 'no (not installed)'}")
    if args.no_rust:
        print(f"  --no-rust  : forcing Python paths")

    passes = [(use_rust, "Rust" if use_rust else "Python")]
    if args.compare and rust_present:
        passes = [(True, "Rust"), (False, "Python")]

    all_results: dict[str, dict] = {}   # label → {stage → stats_dict}

    for rust, label in passes:
        print_header(label)
        results: dict[str, dict] = {}
        ds_img = None

        # Stage 1: detect
        r = bench_scale_detect(img_np, n, rust)
        scale = args.scale if args.scale > 0 else (r["scale"] if isinstance(r["scale"], int) else 1)
        print_row("scale detect", r["stats"])
        results["scale detect"] = r["stats"]
        if args.scale == 0:
            print(f"  {DIM}→ detected scale = {scale}×{RESET}")

        # Stage 2a / 2b
        if args.method in ("dominant", "both"):
            r2a = bench_downscale_dominant(img_np, scale, args.threshold, n, rust)
            print_row("downscale dominant", r2a["stats"])
            results["downscale dominant"] = r2a["stats"]
            ds_img = r2a["result"]
        if args.method in ("mode", "both"):
            r2b = bench_downscale_mode(img_np, scale, n, rust)
            print_row("downscale mode", r2b["stats"])
            results["downscale mode"] = r2b["stats"]
            if args.method == "mode":
                ds_img = r2b["result"]

        # Stage 3: quantize
        if ds_img is None:
            ds_img = img_np
        r3 = bench_quantize(ds_img, args.colors, args.sig_bits, n, rust)
        print_row("wu quantize", r3["stats"])
        results["wu quantize"] = r3["stats"]

        # Stage 4: palette map
        palette = r3.get("palette", [])
        if palette:
            r4 = bench_palette_map(ds_img, palette, n, rust)
            if r4["stats"]:
                print_row("palette map", r4["stats"])
                results["palette map"] = r4["stats"]

        # Stage 5: full pipeline
        try:
            r5 = bench_full_pipeline(
                img_np, scale, args.threshold,
                args.colors, args.sig_bits, args.method if args.method != "both" else "dominant",
                n, rust,
            )
            print_row("full pipeline", r5["stats"])
            results["full pipeline"] = r5["stats"]
        except Exception as exc:
            print(f"  {DIM}full pipeline: skipped ({exc}){RESET}")

        all_results[label] = results

    # ── Speedup summary table ─────────────────────────────────────────────
    if args.compare and rust_present and "Rust" in all_results and "Python" in all_results:
        rust_r  = all_results["Rust"]
        py_r    = all_results["Python"]
        stages  = [s for s in rust_r if s in py_r]

        print()
        print(_c(BOLD + CYAN, "─── Speedup summary (Python ÷ Rust mean) ───"))
        print(f"  {'Stage':<26}  {'Rust':>10}  {'Python':>10}  {'Speedup':>9}")
        print(f"  {'-'*26}  {'-'*10}  {'-'*10}  {'-'*9}")

        for stage in stages:
            r_mean = rust_r[stage]["mean"]
            p_mean = py_r[stage]["mean"]
            ratio  = p_mean / r_mean if r_mean > 0 else float("inf")
            if ratio >= 10:
                col = GREEN
            elif ratio >= 2:
                col = YELLOW
            elif ratio < 1:
                col = RED
            else:
                col = RESET
            speedup_str = _c(col + BOLD, f"{ratio:6.1f}×")
            print(f"  {stage:<26}  {_ms(r_mean):>10}  {_ms(p_mean):>10}  {speedup_str}")

        # Overall wall-clock comparison (full pipeline if available, else sum of stages)
        if "full pipeline" in rust_r and "full pipeline" in py_r:
            total_r = rust_r["full pipeline"]["mean"]
            total_p = py_r["full pipeline"]["mean"]
        else:
            total_r = sum(v["mean"] for v in rust_r.values())
            total_p = sum(v["mean"] for v in py_r.values())
        overall = total_p / total_r if total_r > 0 else float("inf")
        col = GREEN if overall >= 10 else YELLOW if overall >= 2 else RED
        print(f"  {'-'*26}  {'-'*10}  {'-'*10}  {'-'*9}")
        print(f"  {'OVERALL (pipeline)':<26}  {_ms(total_r):>10}  {_ms(total_p):>10}  "
              f"{_c(col + BOLD, f'{overall:6.1f}×')}")

    print()


if __name__ == "__main__":
    main()
