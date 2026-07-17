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
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "default=noprint_wrappers=1:nokey=1", str(frame)],
        capture_output=True, text=True, check=True,
    )
    pix_fmt = result.stdout.strip()
    if "a" not in pix_fmt:
        print(f"WARNING: source frame {frame.name} decodes as '{pix_fmt}' - no alpha plane. "
              "Output will be fully opaque.")


def build_command(fmt, frames_dir: Path, fps: float, output: Path, quality):
    # -pattern_type glob sorts matches lexicographically, so this works for any
    # zero-padded, consistently-named sequence without needing a printf pattern
    # or a hand-built frame list (and doesn't suffer the concat demuxer's
    # last-frame-duration quirk, which used to leak an extra duplicate frame).
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-framerate", str(fps), "-pattern_type", "glob", "-i", str(frames_dir / "*.png")]

    if fmt == "prores4444":
        q = 9 if quality is None else quality
        codec = ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                 "-alpha_bits", "16", "-vendor", "apl0", "-qscale:v", str(q), "-f", "mov"]
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


def _read_alpha(png_path: Path):
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.rgba"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(png_path), "-pix_fmt", "rgba", "-f", "rawvideo", str(raw)],
            check=True,
        )
        data = raw.read_bytes()
    alpha = data[3::4]
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
    args = parser.parse_args()

    find_ffmpeg()

    frames_dir = args.frames_dir
    if not frames_dir.is_dir():
        raise SystemExit(f"Not a directory: {frames_dir}")

    frames = discover_frames(frames_dir)
    check_alpha_present(frames[0])

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

    print(f"{len(frames)} frames -> {output} ({args.format}, {fps:g}fps)")

    cmd = build_command(args.format, frames_dir, fps, output, args.quality)
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
