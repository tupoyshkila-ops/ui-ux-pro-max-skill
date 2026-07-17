#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frames -> alpha-channel animation.

Turns a folder of RGBA PNG frames (a rendered character, an After Effects
export, a Lottie-style sequence, ...) into a single animation file that keeps
straight (non-premultiplied) alpha intact end to end: no baked-in background,
no color fringing on soft edges.

Usage:
  python3 frames_to_alpha_video.py <frames_dir> -o out.mov
  python3 frames_to_alpha_video.py <frames_dir> --fps 30 --format webm -o out.webm
  python3 frames_to_alpha_video.py <frames_dir> --format webp -o out.webp
  python3 frames_to_alpha_video.py <frames_dir> --format apng -o out.png
  python3 frames_to_alpha_video.py <frames_dir> --verify   # sanity-check alpha after encoding
  python3 frames_to_alpha_video.py <frames_dir> --crop --scale 2 -o out.mov
      # crop to the visible content's bounding box, then 2x-upscale with Lanczos.
      # Use --crop when the subject only fills a small area of a much bigger
      # transparent canvas - editors default to showing the clip at its full
      # canvas size, so a small subject on a big canvas reads as "tiny", and
      # zooming it in by hand is what actually causes the pixelation.

Formats (pick the smallest one that still meets your quality bar):
  prores4444 (default, .mov)  Apple ProRes 4444, 10-bit 4:4:4, 16-bit alpha
                              plane. Visually lossless. Use for editing
                              (After Effects/Premiere/Final Cut) or Safari/
                              WKWebView <video> with a transparent background.
                              Large files (~150-200KB/frame at 1080p).
  webm (VP9)                  Smallest web-friendly file. Chrome/Firefox/Edge
                              play it as a normal <video> with real
                              transparency. Alpha is 8-bit 4:2:0 - fine for
                              UI chrome/characters, can show faint banding on
                              very soft/feathered edges. Tune with --quality
                              (lower = better, 0-63, default 30).
  webp                        Animated WebP, drop-in <img>/background
                              replacement for APNG. Lossless (always) and
                              pixel-perfect; the encoder may merge
                              back-to-back duplicate frames and stretch their
                              duration instead (identical playback, fewer
                              bytes) - see the --verify note about frame count.
  apng                        Animated PNG. Same alpha model as the source
                              PNGs, byte-for-byte lossless, biggest files.

Why alpha quality breaks in practice:
  - Premultiplied vs straight alpha: if the source frames were exported with
    premultiplied alpha, RGB "bleeds" through at low-alpha edges and shows up
    as dark/light fringing once composited on a different background. This
    script assumes straight alpha (the PNG/ffmpeg default) and does not
    convert between the two - if you see fringing, fix it at the export step.
  - Chroma subsampling: yuva420p (webm and most video codecs) shares chroma
    samples across a 2x2 pixel block, softening color detail right at the
    alpha edge. yuva444p (prores4444) and rgba (webp/apng) don't have this
    problem, which is why they look sharper on hard edges.
  - Lossy alpha compression: the alpha plane is data, not just a mask - a
    quality setting that's "good enough" for color can still be too
    aggressive for a feathered edge. --quality on --format webm/prores4444
    lets you push the encoder harder; --verify gives a quick numeric check.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGE_EXTS = {".png"}

DEFAULT_EXT = {
    "prores4444": ".mov",
    "webm": ".webm",
    "webp": ".webp",
    "apng": ".png",
}


def find_ffmpeg():
    if shutil.which("ffmpeg"):
        return
    raise SystemExit(
        "ffmpeg not found on PATH.\n"
        "Install it, then re-run this script:\n"
        "  macOS:         brew install ffmpeg\n"
        "  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg\n"
        "  Windows:       winget install Gyan.FFmpeg"
    )


def discover_frames(frames_dir: Path):
    frames = sorted(
        (p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name,
    )
    if len(frames) < 2:
        raise SystemExit(f"Found {len(frames)} PNG frame(s) in {frames_dir} - need at least 2.")
    return frames


def guess_fps(frames_dir: Path):
    m = re.search(r"(\d+)\s*fps", frames_dir.name, re.IGNORECASE)
    return float(m.group(1)) if m else None


def check_alpha_present(frame: Path):
    """Returns the source pix_fmt (e.g. 'rgba' or 'rgba64be') so callers can size the
    alpha plane to match - most PNG exports are 8-bit, and asking prores_ks for a
    16-bit alpha plane on top of that just doubles the file for no real precision."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "default=noprint_wrappers=1:nokey=1", str(frame)],
        capture_output=True, text=True, check=True,
    )
    pix_fmt = result.stdout.strip()
    if "a" not in pix_fmt:
        print(f"WARNING: source frame {frame.name} decodes as '{pix_fmt}' - no alpha plane. "
              "Output will be fully opaque.")
    return pix_fmt


def _frame_size(png_path: Path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(png_path)],
        capture_output=True, text=True, check=True,
    )
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def _alpha_bbox(raw: bytes, w: int, h: int, threshold: int = 10):
    """Bounding box (left, top, right, bottom) of pixels with alpha > threshold,
    or None if the frame is fully transparent. Pure stdlib: bytes slicing + max()
    do the heavy lifting in C, no numpy needed."""
    alpha = raw[3::4]
    top = bottom = left = right = None
    for y in range(h):
        if max(alpha[y * w:(y + 1) * w]) > threshold:
            top = y
            break
    if top is None:
        return None
    for y in range(h - 1, top - 1, -1):
        if max(alpha[y * w:(y + 1) * w]) > threshold:
            bottom = y
            break
    for x in range(w):
        if max(alpha[x::w]) > threshold:
            left = x
            break
    for x in range(w - 1, left - 1, -1):
        if max(alpha[x::w]) > threshold:
            right = x
            break
    return left, top, right, bottom


def detect_crop_box(frames, margin_ratio=0.08, sample_count=12, threshold=10):
    """Union the alpha bounding box across a sample of frames (plus a margin) to
    find the smallest rectangle containing all visible content. Frames are often
    exported on a much bigger transparent canvas than the subject actually uses -
    cropping to this box means a video editor's default 100% scale already shows
    the subject close to full-frame, instead of tiny in a sea of transparency."""
    w, h = _frame_size(frames[0])
    step = max(1, len(frames) // sample_count)
    sample = frames[::step]
    min_x, min_y, max_x, max_y = w, h, 0, 0
    found = False
    for frame in sample:
        raw = _read_rgba(frame)
        bbox = _alpha_bbox(raw, w, h, threshold)
        if bbox is None:
            continue
        left, top, right, bottom = bbox
        min_x, min_y = min(min_x, left), min(min_y, top)
        max_x, max_y = max(max_x, right), max(max_y, bottom)
        found = True
    if not found:
        return None
    box_w, box_h = max_x - min_x, max_y - min_y
    margin_x = max(2, int(box_w * margin_ratio))
    margin_y = max(2, int(box_h * margin_ratio))
    x0 = max(0, min_x - margin_x)
    y0 = max(0, min_y - margin_y)
    x1 = min(w, max_x + margin_x)
    y1 = min(h, max_y + margin_y)
    # Even dimensions - required by yuv420p (webm) and safest for the others too.
    cw = (x1 - x0) // 2 * 2
    ch = (y1 - y0) // 2 * 2
    return f"crop={cw}:{ch}:{x0}:{y0}"


def build_command(fmt, frames_dir: Path, fps: float, output: Path, quality, vf_filters=None, alpha_bits=8):
    # -pattern_type glob sorts matches lexicographically, so this works for any
    # zero-padded, consistently-named sequence without needing a printf pattern
    # or a hand-built frame list (and doesn't suffer the concat demuxer's
    # last-frame-duration quirk, which used to leak an extra duplicate frame).
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-framerate", str(fps), "-pattern_type", "glob", "-i", str(frames_dir / "*.png")]
    if vf_filters:
        base += ["-vf", ",".join(vf_filters)]

    if fmt == "prores4444":
        q = 9 if quality is None else quality
        # alpha_bits should match the source's real bit depth - PNGs are almost
        # always 8-bit, and asking for 16-bit alpha on top of an 8-bit source
        # roughly doubles the file for zero extra precision.
        codec = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                 "-alpha_bits", str(alpha_bits), "-vendor", "apl0", "-qscale:v", str(q), "-f", "mov"]
    elif fmt == "webm":
        crf = 30 if quality is None else quality
        codec = ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
                 "-b:v", "0", "-crf", str(crf), "-row-mt", "1",
                 "-metadata:s:v:0", "alpha_mode=1", "-f", "webm"]
    elif fmt == "webp":
        codec = ["-c:v", "libwebp_anim", "-pix_fmt", "bgra", "-lossless", "1",
                 "-loop", "0", "-compression_level", "6", "-f", "webp"]
    elif fmt == "apng":
        codec = ["-c:v", "apng", "-pix_fmt", "rgba", "-plays", "0", "-f", "apng"]
    else:
        raise SystemExit(f"Unknown format: {fmt}")

    return base + codec + [str(output)]


def _read_rgba(png_path: Path) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.rgba"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(png_path), "-pix_fmt", "rgba", "-f", "rawvideo", str(raw)],
            check=True,
        )
        return raw.read_bytes()


def _read_alpha(png_path: Path):
    alpha = _read_rgba(png_path)[3::4]
    return min(alpha), max(alpha), sum(alpha) / len(alpha)


def verify_alpha(fmt, output: Path, frames):
    """Decode the encoded file back to frames and sanity-check the middle one's
    alpha channel against the matching source frame. Not a full diff - webp/apng
    may merge duplicate frames - just a quick "did transparency survive" check."""
    decoder = ["-c:v", "libvpx-vp9"] if fmt == "webm" else []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *decoder,
                 "-i", str(output), "-pix_fmt", "rgba", str(tmp / "f_%05d.png")],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            if fmt == "webp":
                print("Skipping --verify: this ffmpeg build can't decode animated WebP back to "
                      "frames. Install libwebp's tools (e.g. `apt install webp` / `brew install webp`) "
                      "and check with `webpmux -info` / `anim_dump` instead - the file itself is fine.")
            else:
                print(f"Skipping --verify: ffmpeg could not decode {output} back to frames.")
            return
        decoded = sorted(tmp.glob("f_*.png"))
        if not decoded:
            print("WARNING: could not decode any frames back out of the output file.")
            return
        idx = min(len(frames), len(decoded)) // 2
        src_alpha = _read_alpha(frames[idx])
        out_alpha = _read_alpha(decoded[idx])

    print(f"\nverify (frame {idx}):")
    print(f"  source  alpha min/max/mean = {src_alpha[0]}/{src_alpha[1]}/{src_alpha[2]:.1f}")
    print(f"  encoded alpha min/max/mean = {out_alpha[0]}/{out_alpha[1]}/{out_alpha[2]:.1f}")
    if out_alpha[0] == out_alpha[1] == 255 and not (src_alpha[0] == src_alpha[1] == 255):
        print("WARNING: encoded frame is fully opaque but the source has transparency - "
              "the alpha channel was likely lost. Try a different --format.")
    if len(frames) != len(decoded):
        print(f"Note: source had {len(frames)} frames, encoded file decodes to {len(decoded)} "
              "(some codecs merge back-to-back duplicate frames and stretch duration instead - "
              "this does not affect playback timing).")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble a folder of RGBA PNG frames into an alpha-channel animation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("frames_dir", type=Path, help="Folder of sequentially named RGBA PNG frames")
    parser.add_argument("-o", "--output", type=Path, default=None,
                         help="Output file (default: <frames_dir>.<ext>)")
    parser.add_argument("--fps", type=float, default=None,
                         help="Frame rate (default: parsed from the folder name, e.g. '..._30fps_...', else 30)")
    parser.add_argument("--format", "-f", choices=list(DEFAULT_EXT), default="prores4444",
                         help="Output format (default: prores4444)")
    parser.add_argument("--quality", type=int, default=None,
                         help="prores4444: -qscale:v (1-32, lower=better, default 9). "
                              "webm: -crf (0-63, lower=better, default 30). Ignored for webp/apng (always lossless).")
    parser.add_argument("--verify", action="store_true",
                         help="Decode a frame back out and sanity-check the alpha channel after encoding")
    parser.add_argument("--crop", action="store_true",
                         help="Auto-crop to the union bounding box of visible (alpha>10) content across a "
                              "sample of frames, plus an 8%% margin. Use this when the subject only occupies "
                              "a small area of a much bigger transparent canvas - a video editor's default "
                              "100%% scale will then already show it close to full-frame, and you'll need "
                              "far less manual zoom (which is what causes visible pixelation).")
    parser.add_argument("--scale", type=float, default=None,
                         help="Upscale factor applied with high-quality Lanczos filtering before encoding "
                              "(e.g. 2 for 2x). This does not invent detail - it only avoids relying on your "
                              "video editor's own (often lower-quality/real-time) resize. For real detail "
                              "gain you need a higher-resolution source render or an AI upscaler.")
    args = parser.parse_args()

    find_ffmpeg()

    frames_dir = args.frames_dir
    if not frames_dir.is_dir():
        raise SystemExit(f"Not a directory: {frames_dir}")

    frames = discover_frames(frames_dir)
    source_pix_fmt = check_alpha_present(frames[0])
    alpha_bits = 16 if "64" in source_pix_fmt else 8

    fps = args.fps
    if fps is None:
        fps = guess_fps(frames_dir)
        if fps:
            print(f"Detected {fps:g}fps from folder name '{frames_dir.name}'.")
        else:
            fps = 30.0
            print("No --fps given and none detected from the folder name; assuming 30fps.")

    output = args.output or frames_dir.with_suffix(DEFAULT_EXT[args.format])
    output.parent.mkdir(parents=True, exist_ok=True)

    vf_filters = []
    if args.crop:
        box = detect_crop_box(frames)
        if box:
            vf_filters.append(box)
            print(f"Auto-crop: {box}")
        else:
            print("Auto-crop: no non-transparent content found in the sample; skipping.")
    if args.scale:
        vf_filters.append(f"scale=iw*{args.scale}:ih*{args.scale}:flags=lanczos")
        print(f"Scaling {args.scale:g}x with Lanczos filtering (no new detail, just avoids a lower-quality "
              "resize downstream).")

    print(f"{len(frames)} frames -> {output} ({args.format}, {fps:g}fps)")

    cmd = build_command(args.format, frames_dir, fps, output, args.quality, vf_filters, alpha_bits)
    subprocess.run(cmd, check=True)

    size_kb = output.stat().st_size / 1024
    print(f"Wrote {output} ({size_kb:,.0f} KB)")

    if args.verify:
        try:
            verify_alpha(args.format, output, frames)
        except subprocess.CalledProcessError:
            print(f"Skipping --verify: ffmpeg could not decode {output} back to frames.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f"ffmpeg failed (exit {exc.returncode}). Re-run with a smaller --quality value "
                  "or check that all frames are valid RGBA PNGs.")
