"""Video composition utilities.

Builds a real cinematic teaser MP4 from keyframe images:
  1. A 2.5s TITLE card (black, title + logline, fade-in)
  2. The scene slideshow (per-scene duration, xfade transitions)
  3. A 2.5s CREDITS card (director / VGO Cinema branding)
  4. A procedurally-generated ambient BGM mixed at -20dB under everything

No external audio assets are required — the BGM is synthesised in pure
Python/numpy (A-minor pad with a slow LFO and gentle white noise), then
muxed into the final mp4 with the bundled ffmpeg's AAC encoder. We don't
use ffmpeg's `aevalsrc` because the imageio-ffmpeg essentials build does
not include that lavfi source.
a slow-evolving chord pad so the result feels like a film teaser out
of the box.
"""
from __future__ import annotations

import logging
import math
import os
import random
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from PIL import Image, ImageDraw, ImageFilter, ImageFont

logger = logging.getLogger(__name__)

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # noqa: BLE001
    FFMPEG_BIN = "ffmpeg"


_DEFAULT_OUTPUT_DIR = "/data/outputs"  # canonical in Docker
# On any other platform (Windows / bare-metal Linux), fall back to a local
# ./data/outputs relative to the project root so the pipeline keeps working
# even when VGO_OUTPUT_DIR is not set.
if not Path(_DEFAULT_OUTPUT_DIR).parent.exists():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_OUTPUT_DIR = str(_PROJECT_ROOT / "data" / "outputs")
OUTPUT_DIR = Path(os.environ.get("VGO_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- SSRF protection ------------------------------------------------
def _is_safe_url(url: str) -> bool:
    """Validate that a URL is safe to fetch from the server (SSRF guard).

    Only allows HTTP(S) URLs whose host resolves to public (non-private,
    non-loopback, non-link-local) IP addresses. Blocks access to internal
    services (127.0.0.1, 10.x, 172.16-31.x, 192.168.x, 169.254.x, ::1,
    cloud metadata endpoints, etc.).
    """
    import ipaddress
    import socket
    import urllib.parse

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:  # noqa: BLE001
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    host = parsed.hostname
    if not host:
        return False

    # Block obvious private / loopback / link-local hostnames.
    _BLOCKED_HOSTS = {
        "localhost", "127.0.0.1", "::1", "0.0.0.0",
        "metadata", "metadata.google.internal",
    }
    if host.lower() in _BLOCKED_HOSTS:
        return False

    # Resolve the host and check every returned IP.
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return False
    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return False
    return True


# ---------- Output dimensions ------------------------------------------------
#
# Every clip in the xfade chain must be the exact same resolution
# (ffmpeg's xfade filter refuses mixed sizes). The scene clips are
# normalised to `image_width × image_height` in `_normalize_images`,
# so the title/credits cards have to be rendered at those same
# dimensions — not a hard-coded 1280x720. We pull the values from
# Settings on demand; defaults match the rest of the pipeline.
def _output_dimensions(width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int, int]:
    try:
        from .config import get_settings
        s = get_settings()
        w = width if width is not None else int(s.image_width)
        h = height if height is not None else int(s.image_height)
        return w, h
    except Exception:  # noqa: BLE001
        return width or 1280, height or 720


# ---------- Typography helpers ----------------------------------------------

def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Try a few common font paths; fall back to PIL's default bitmap."""
    candidates = []
    if bold:
        candidates += [
            "C:/Windows/Fonts/arialbd.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    candidates += [
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ---------- Card generators --------------------------------------------------

BG = (10, 10, 12)
AMBER = (245, 200, 100)
AMBER_DIM = (180, 140, 70)
TEXT_LIGHT = (235, 220, 200)
TEXT_MUTED = (140, 130, 115)


def _grain_overlay(img: Image.Image, opacity: float = 0.04) -> Image.Image:
    """Add a subtle film grain so the cards don't look like flat solids.

    Uses numpy for speed: generate noise at half resolution, upscale
    with bilinear interpolation, then apply a Gaussian blur and blend.
    Falls back to the pure-Python loop if numpy is unavailable.
    """
    try:
        import numpy as np
        w, h = img.size
        # Generate noise at half resolution for speed, then upscale.
        noise = np.random.randint(0, 256, (h // 2, w // 2), dtype=np.uint8)
        noise_img = Image.fromarray(noise).resize(img.size, Image.BILINEAR)
        noise_img = noise_img.filter(ImageFilter.GaussianBlur(0.6))
        return Image.blend(img, noise_img, opacity)
    except ImportError:
        rng = random.Random(42)
        noise = Image.new("RGB", img.size)
        px = noise.load()
        for y in range(0, img.size[1], 2):
            for x in range(0, img.size[0], 2):
                v = rng.randint(0, 255)
                px[x, y] = (v, v, v)
        noise = noise.filter(ImageFilter.GaussianBlur(0.6))
        return Image.blend(img, noise, opacity)


def _vignette(img: Image.Image) -> Image.Image:
    """Add a soft radial darkening at the edges."""
    w, h = img.size
    mask = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(mask)
    d.ellipse(
        (-w * 0.2, -h * 0.2, w * 1.2, h * 1.2),
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(120))
    black = Image.new("RGB", img.size, (0, 0, 0))
    return Image.composite(img, black, mask)


def render_title_card(title: str, logline: str, out: Path,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> Path:
    W, H = _output_dimensions(width, height)
    img = Image.new("RGB", (W, H), BG)

    # subtle horizontal rule with a "key" marker
    draw = ImageDraw.Draw(img)
    title_font = _load_font(64, bold=True)
    line_font = _load_font(22)
    small_font = _load_font(18)

    # Top label
    label = "V G O   C I N E M A   P R E S E N T S"
    bbox = draw.textbbox((0, 0), label, font=small_font)
    lw = bbox[2] - bbox[0]
    draw.text(((W - lw) / 2, 130), label, fill=AMBER_DIM, font=small_font)

    # Title
    safe_title = (title or "UNTITLED")[:60]
    bbox = draw.textbbox((0, 0), safe_title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) / 2, 250), safe_title, fill=AMBER, font=title_font)

    # Decorative line
    line_y = 360
    draw.line([(W / 2 - 80, line_y), (W / 2 + 80, line_y)], fill=AMBER, width=2)

    # Logline (wrap to 2 lines)
    safe_log = (logline or "").strip()
    if safe_log:
        words = safe_log.split()
        line1, line2 = "", ""
        for w in words:
            cand = (line1 + " " + w).strip()
            if draw.textlength(cand, font=line_font) < W * 0.7:
                line1 = cand
            else:
                line2 = (line2 + " " + w).strip()
        for i, ln in enumerate([line1, line2]):
            if not ln:
                continue
            bbox = draw.textbbox((0, 0), ln, font=line_font)
            lw_ = bbox[2] - bbox[0]
            draw.text(((W - lw_) / 2, 410 + i * 36), ln, fill=TEXT_LIGHT, font=line_font)

    img = _vignette(img)
    img = _grain_overlay(img, 0.05)
    img.save(out, "JPEG", quality=92)
    return out


def render_credits_card(title: str, prompt: str, out: Path,
                        roles: Optional[List[Tuple[str, str]]] = None) -> Path:
    """Render the closing credits card. `roles` is a list of
    `(label, credit)` pairs to display. Defaults to the cinematic
    five-line roster when omitted."""
    W, H = _output_dimensions()
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    title_font = _load_font(36, bold=True)
    role_font = _load_font(16)
    name_font = _load_font(22, bold=True)
    foot_font = _load_font(14)

    if not roles:
        roles = [
            ("Director",         "AI Director · meta/llama-3.3-70b-instruct"),
            ("Cinematographer",  "Image Prompt Engineer · meta/llama-3.1-8b-instruct"),
            ("Composer",         "Narrative Editor · meta/llama-3.3-70b-instruct"),
            ("Storyboard",       "4 cinematic keyframes · ffmpeg xfade"),
            ("Original Score",   "Procedural ambient pad · synthesized in-pipeline"),
        ]

    y = 140
    for role, credit in roles:
        draw.text((W / 2 - 240, y), role.upper(), fill=AMBER_DIM, font=role_font)
        bbox = draw.textbbox((0, 0), credit, font=name_font)
        cw = bbox[2] - bbox[0]
        draw.text(((W - cw) / 2, y), credit, fill=TEXT_LIGHT, font=name_font)
        y += 56

    # final tagline
    bbox = draw.textbbox((0, 0), f"« {title} »", font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) / 2, H - 140), f"« {title} »", fill=AMBER, font=title_font)

    draw.text((W / 2 - 60, H - 60), "vgo.cinema", fill=TEXT_MUTED, font=foot_font)

    img = _vignette(img)
    img = _grain_overlay(img, 0.05)
    img.save(out, "JPEG", quality=92)
    return out


# ---------- Image prep -------------------------------------------------------

def _burn_caption(img: Image.Image, caption: str) -> Image.Image:
    if not caption:
        return img
    try:
        draw = ImageDraw.Draw(img)
        try:
            font = _load_font(max(18, img.height // 22), bold=True)
        except OSError:
            font = ImageFont.load_default()
        margin = max(10, img.width // 80)
        x, y = margin + 8, img.height - margin - 36
        draw.text((x + 1, y + 1), caption, fill=(0, 0, 0), font=font)
        draw.text((x, y), caption, fill=AMBER, font=font)
    except Exception as exc:  # noqa: BLE001
        logger.warning("burn_caption: %s", exc)
    return img


def _download_image(url: str, dest: Path, *, timeout: int = 30) -> Optional[Path]:
    if not _is_safe_url(url):
        logger.warning("SSRF guard: blocked URL %s", url[:80])
        return None
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", url) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        return dest
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to download %s: %s", url[:80], exc)
        return None


def _normalize_images(urls: List[str], captions: List[str], out_dir: Path,
                      width: Optional[int] = None,
                      height: Optional[int] = None) -> List[Path]:
    """Download (or copy from local path) the keyframes, normalise to a
    common width, burn the caption and return the local file paths.

    `urls` is allowed to be either an http(s) URL or a local file path;
    the latter is what `_generate_image_for_scene` returns when the
    configured image provider is one of the NIM FLUX/SD3.5 endpoints
    (we save the bytes directly into the worker output dir).
    """
    paths: List[Path] = []
    target_w, target_h = _output_dimensions(width, height)
    for idx, (url, cap) in enumerate(zip(urls, captions)):
        if not url:
            continue
        try:
            if url.startswith(("http://", "https://")):
                src = _download_image(url, out_dir / f"src_{idx:02d}.jpg")
            else:
                # Local file (NIM-generated bytes) — copy into the
                # compose tmp dir so the rest of the pipeline is path-
                # agnostic.
                src_path = Path(url)
                src = out_dir / f"src_{idx:02d}{src_path.suffix.lower() or '.jpg'}"
                if src_path.resolve() != src.resolve():
                    src.write_bytes(src_path.read_bytes())
                src = src
            if src is None:
                continue
            with Image.open(src) as im:
                im = im.convert("RGB")
                # Resize to the configured target dimensions, preserving
                # aspect ratio. Pad to exact size with black so the
                # xfade filter sees a uniform input.
                im.thumbnail((target_w, target_h), Image.LANCZOS)
                canvas = Image.new("RGB", (target_w, target_h), BG)
                x = (target_w - im.width) // 2
                y = (target_h - im.height) // 2
                canvas.paste(im, (x, y))
                im = canvas
                im = _burn_caption(im, f"{idx + 1:02d}  {cap[:48]}")
                out = out_dir / f"frame_{idx:02d}.jpg"
                im.save(out, "JPEG", quality=88)
                paths.append(out)
        except Exception as exc:  # noqa: BLE001
            logger.warning("normalize %s: %s", str(url)[:80], exc)
    return paths


# ---------- ffmpeg helpers ---------------------------------------------------

def _run(cmd: List[str], *, what: str) -> bool:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # Log the failure details from the original run. We don't
        # re-run with verbose logging — that would double the cost
        # of an already-failed (and slow) ffmpeg command. If debugging
        # is needed, set LOG_LEVEL=DEBUG and run manually.
        logger.warning("%s failed (rc=%d)", what, r.returncode)
        logger.warning("cmd: %s", " ".join(cmd[:8]) + " …")
        logger.warning("stderr tail:\n%s", r.stderr[-1500:])
        return False
    return True


def _build_card_clip(image: Path, out_clip: Path, duration: float, fps: int) -> bool:
    target_w, target_h = _output_dimensions()
    cmd = [
        FFMPEG_BIN, "-y", "-loglevel", "error",
        "-loop", "1", "-t", f"{duration:.3f}",
        "-i", str(image),
        "-vf", f"fps={fps},format=yuv420p,scale={target_w}:{target_h}:flags=lanczos",
        "-pix_fmt", "yuv420p",
        "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
        str(out_clip),
    ]
    return _run(cmd, what=f"card_clip({image.name})")


def _build_slideshow_clips(
    frames: List[Path], durations: List[float], out_dir: Path, fps: int
) -> Optional[List[Path]]:
    target_w, target_h = _output_dimensions()
    clips: List[Path] = []
    for idx, (frame, dur) in enumerate(zip(frames, durations[: len(frames)])):
        clip = out_dir / f"scene_{idx:02d}.mp4"
        ok = _run(
            [
                FFMPEG_BIN, "-y", "-loglevel", "error",
                "-loop", "1", "-t", f"{max(0.5, dur):.3f}",
                "-i", str(frame),
                "-vf", f"fps={fps},format=yuv420p,scale={target_w}:{target_h}:flags=lanczos",
                "-pix_fmt", "yuv420p",
                "-an",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
                str(clip),
            ],
            what=f"scene_clip[{idx}]",
        )
        if not ok:
            return None
        clips.append(clip)
    return clips


def _xfade_chain(clips: List[Path], durations: List[float], transition_sec: float, out_clip: Path) -> bool:
    """Stitch a list of clips with `transition_sec` crossfades.

    For each step, the crossfade `offset` is the cumulative time
    on the upstream stream (not on the current clip) at which the
    transition begins. The standard pattern is:
        offset[i] = sum(durations[0..i-1]) + (durations[i] - transition_sec)
    so we accumulate `durations[i] - transition_sec` per step.

    Works for either still-frame clips (built by _build_slideshow_clips)
    or real MP4 clips produced by upstream video generation APIs — ffmpeg's
    xfade filter accepts both as long as the resolution matches.

    If input clips have mixed resolutions ffmpeg rejects them; in that
    case `_build_normalised_clip_chain` is the safe alternative.
    """
    if not clips:
        return False
    if len(clips) == 1:
        out_clip.write_bytes(clips[0].read_bytes())
        return True

    xfades: List[str] = []
    prev = "0:v"
    offset = 0.0
    for i in range(len(clips) - 1):
        offset += max(0.1, durations[i] - transition_sec)
        nxt = f"v{i + 1}"
        xfades.append(
            f"[{prev}][{i + 1}:v]xfade=transition=fade:duration={transition_sec}:offset={offset:.3f}[{nxt}]"
        )
        prev = nxt
    fc = ";".join(xfades)
    inputs: List[str] = []
    for c in clips:
        inputs += ["-i", str(c)]
    return _run(
        [
            FFMPEG_BIN, "-y", "-loglevel", "error",
            *inputs, "-filter_complex", fc,
            "-map", f"[{prev}]",
            "-an",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
            str(out_clip),
        ],
        what="xfade_chain",
    )


# ---------- BGM synthesis ----------------------------------------------------

# A slow-evolving A-minor pad: A2 + C3 + E3 + A3 with a slow LFO on the
# bass. Synthesised in pure Python/numpy because the bundled ffmpeg (essentials
# build) does not include the `aevalsrc` lavfi source.
BGM_DURATION_FALLBACK = 30.0
BGM_FREQ = {
    "A2": 110.0, "C3": 130.81, "E3": 164.81, "G3": 196.00,
    "A3": 220.0, "B3": 246.94, "C4": 261.63, "D4": 293.66,
    "E4": 329.63, "F4": 349.23, "G4": 392.00, "A4": 440.0,
}
BGM_SAMPLE_RATE = 44100


def _synthesize_bgm(duration_sec: float, out_wav: Path) -> bool:
    """Render an ambient A-minor pad as 16-bit stereo WAV.

    Mixes three sines (root, fifth, third) with a slow LFO on the bass and
    a hint of white noise for film-grain texture. Fades in over 1.2s and
    out over 1.5s so the BGM never has a hard cut.
    """
    try:
        import numpy as np
        import wave
    except ImportError as exc:  # pragma: no cover
        logger.error("numpy/wave unavailable: %s", exc)
        return False

    dur = max(8.0, float(duration_sec))
    n = int(dur * BGM_SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / BGM_SAMPLE_RATE

    a2, e3, a3 = BGM_FREQ["A2"], BGM_FREQ["E3"], BGM_FREQ["A3"]
    sig = (
        0.18 * np.sin(2 * np.pi * a2 * t)
        + 0.12 * np.sin(2 * np.pi * e3 * t)
        + 0.10 * np.sin(2 * np.pi * a3 * t)
        + 0.04 * np.sin(2 * np.pi * 0.07 * t)             # LFO
        + 0.05 * (np.random.default_rng(42).random(n).astype(np.float32) - 0.5)
    )

    fade_in = np.minimum(1.0, t / 1.2)
    fade_out = np.minimum(1.0, (dur - t) / 1.5)
    sig = sig * fade_in * fade_out * 0.55

    # Soft clip to keep the mix under -1dBFS
    sig = np.tanh(sig) * 0.95

    pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype(np.int16)
    stereo = np.stack([pcm, pcm], axis=1)  # duplicate mono into both channels

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(BGM_SAMPLE_RATE)
        w.writeframes(stereo.tobytes())
    return out_wav.exists() and out_wav.stat().st_size > 0


# ---- BGM cache ---------------------------------------------------------------
# The ambient pad is always the same audio; only the duration varies.
# We cache a 60-second base WAV once and reuse it (copying + trimming
# via ffmpeg) for short videos, avoiding repeated numpy synthesis.
_BGM_CACHE: Optional[Path] = None
_BGM_BASE_DUR = 60.0
_BGM_LOCK = threading.Lock()


def _get_bgm_wav(duration_sec: float, out_wav: Path) -> bool:
    """Return a BGM WAV of at least `duration_sec` seconds at `out_wav`.

    Uses a cached 60s base file when possible; only synthesises fresh
    when the requested duration exceeds the cache length.
    """
    global _BGM_CACHE
    if _BGM_CACHE is None or not _BGM_CACHE.exists():
        with _BGM_LOCK:
            # Double-check after acquiring the lock — another thread may
            # have already populated the cache while we waited.
            if _BGM_CACHE is None or not _BGM_CACHE.exists():
                cache_path = OUTPUT_DIR / "_bgm_base.wav"
                if _synthesize_bgm(_BGM_BASE_DUR, cache_path):
                    _BGM_CACHE = cache_path
                else:
                    # Fall back to direct synthesis
                    return _synthesize_bgm(duration_sec, out_wav)
    if duration_sec <= _BGM_BASE_DUR:
        import shutil
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_BGM_CACHE, out_wav)
        return out_wav.exists() and out_wav.stat().st_size > 0
    # Requested duration is longer than the cache — synthesise fresh.
    return _synthesize_bgm(duration_sec, out_wav)


def _mux_bgm(video: Path, bgm: Path, out: Path) -> bool:
    """Mix the BGM under the video at ~-20dB and emit a final mp4."""
    cmd = [
        FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(bgm),
        "-filter_complex", "[1:a]volume=0.10[mix]",
        "-map", "0:v", "-map", "[mix]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    return _run(cmd, what="mux_bgm")


def _concat_voiceover_for_scenes(voiceover_paths: List[Optional[str]],
                                  scene_offsets: List[float],
                                  scene_durations: List[float],
                                  out_wav: Path) -> Optional[Path]:
    """Concatenate per-scene voice-over WAVs onto a single timeline
    using `scene_offsets[i]` as the placement of scene `i`'s audio.

    The title/credits region is left silent.
    """
    import wave
    SR = 44100
    if not any(voiceover_paths):
        return None
    total_dur = sum(scene_durations) + 5.0  # title 2.5 + credits 2.5
    n_total = int(total_dur * SR)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    track = bytearray(n_total * 2)

    for p, off in zip(voiceover_paths, scene_offsets):
        if not p or not Path(p).exists():
            continue
        try:
            with wave.open(p, "rb") as w:
                pcm = w.readframes(w.getnframes())
                sr = w.getframerate()
            # Resample if needed (most voice-overs will already be 44.1k).
            # Vectorised with numpy so a 4-voice 30s job stops bottlenecking
            # on per-sample list comprehensions.
            if sr != SR:
                import numpy as _np
                src = _np.frombuffer(pcm, dtype="<i2").astype(_np.float32)
                n_dst = int(round(src.shape[0] * SR / sr))
                # Linear interpolation re-sample.
                idx = _np.linspace(0, src.shape[0] - 1, n_dst).astype(_np.float32)
                i0 = _np.floor(idx).astype(_np.int32)
                i1 = _np.minimum(i0 + 1, src.shape[0] - 1)
                frac = (idx - i0).astype(_np.float32)
                resampled = src[i0] * (1 - frac) + src[i1] * frac
                pcm = _np.clip(resampled, -32768, 32767).astype("<i2").tobytes()
            start_byte = int(off * SR) * 2
            end_byte = start_byte + len(pcm)
            if start_byte >= len(track):
                continue
            if end_byte > len(track):
                pcm = pcm[: len(track) - start_byte]
                end_byte = len(track)
            track[start_byte:end_byte] = pcm
        except Exception as exc:  # noqa: BLE001
            logger.warning("voiceover place: %s: %s", p, exc)

    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(track))
    return out_wav


def _mux_bgm_and_voiceover(video: Path, bgm: Path, voice: Path, out: Path) -> bool:
    """Three-input ffmpeg: BGM ducked under voice-over, voice-over on
    top, all mixed and re-encoded to AAC stereo."""
    # [1:a] BGM at -23dB  -> bgm
    # [2:a] VO at  -3dB   -> vo
    # [bgm][vo] amix -> mix
    cmd = [
        FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(bgm), "-i", str(voice),
        "-filter_complex",
        "[1:a]volume=0.07[bgm];"
        "[2:a]volume=2.0,aresample=44100[vo];"
        "[bgm][vo]amix=inputs=2:duration=first:dropout_transition=0[mix]",
        "-map", "0:v", "-map", "[mix]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    return _run(cmd, what="mux_bgm_and_voiceover")


# ---------- Public entry point ------------------------------------------------

def compose_slideshow(
    image_urls: List[str],
    captions: List[str],
    durations: List[float],
    task_id: str,
    *,
    fps: int = 24,
    transition_sec: float = 0.4,
    title: str = "Untitled",
    logline: str = "",
    prompt: str = "",
    voiceover_paths: Optional[List[Optional[str]]] = None,
    credits_roles: Optional[List[Tuple[str, str]]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[str]:
    """Build the final cinematic teaser MP4 and return its public URL.

    Title-card (2.5s) → slideshow (~N×4s) → credits-card (2.5s), with
    xfade transitions.  Audio:
      * ambient BGM (numpy synthesised, ~-23dB under the mix)
      * per-scene voice-over from NVIDIA Magpie TTS (~0dB on top of BGM)
    """
    if not image_urls:
        return None
    voiceover_paths = voiceover_paths or []

    with tempfile.TemporaryDirectory(prefix=f"vgo_{task_id}_") as tmp:
        tmp_dir = Path(tmp)

        # 1. Download + normalise the keyframes
        frames = _normalize_images(image_urls, captions, tmp_dir,
                                  width=width, height=height)
        if not frames:
            logger.warning("compose_slideshow: no frames for %s", task_id)
            return None

        # 2. Render the title + credits cards
        title_img = render_title_card(title, logline, tmp_dir / "title.jpg",
                                          width=width, height=height)
        credits_img = render_credits_card(title, prompt, tmp_dir / "credits.jpg",
                                          roles=credits_roles)

        title_dur = 2.5
        credits_dur = 2.5

        # 3. Build individual clips
        title_clip = tmp_dir / "00_title.mp4"
        credits_clip = tmp_dir / "99_credits.mp4"
        if not _build_card_clip(title_img, title_clip, title_dur, fps):
            return None
        if not _build_card_clip(credits_img, credits_clip, credits_dur, fps):
            return None

        scene_clips = _build_slideshow_clips(frames, durations, tmp_dir, fps)
        if not scene_clips:
            return None

        # 4. Chain: title -> scene 1 ... scene N -> credits, with xfades
        chain_clips = [title_clip] + scene_clips + [credits_clip]
        chain_durations = [title_dur] + list(durations[: len(scene_clips)]) + [credits_dur]

        body_clip = tmp_dir / "body.mp4"
        if not _xfade_chain(chain_clips, chain_durations, transition_sec, body_clip):
            return None

        # 5. Synthesise BGM to match the body duration
        body_dur = sum(chain_durations) - transition_sec * (len(chain_clips) - 1)
        bgm_wav = tmp_dir / "bgm.wav"
        if not _get_bgm_wav(body_dur, bgm_wav):
            logger.warning("BGM synthesis failed for %s; continuing without audio", task_id)
            out_path = OUTPUT_DIR / f"{task_id}.mp4"
            out_path.write_bytes(body_clip.read_bytes())
            return f"/api/outputs/{task_id}.mp4"

        out_path = OUTPUT_DIR / f"{task_id}.mp4"

        # 6. If we have per-scene voice-overs, place them at the right
        #    offsets (title_dur + transitions + previous scene durations)
        #    and mix BGM + VO together.
        if voiceover_paths and any(voiceover_paths):
            scene_offsets: List[float] = []
            cursor = title_dur
            for i, clip_dur in enumerate(durations[: len(scene_clips)]):
                scene_offsets.append(cursor)
                # advance by clip_dur; the xfade overlap eats into the
                # next clip by transition_sec, which is fine.
                cursor += clip_dur

            vo_wav = tmp_dir / "voiceover_timeline.wav"
            placed = _concat_voiceover_for_scenes(
                voiceover_paths, scene_offsets, durations, vo_wav
            )
            if placed is not None and placed.exists() and placed.stat().st_size > 100:
                if _mux_bgm_and_voiceover(body_clip, bgm_wav, placed, out_path):
                    kb = out_path.stat().st_size / 1024
                    logger.info("compose_slideshow: wrote %s (%.1f KB, ~%.1fs, BGM+VO)", out_path, kb, body_dur)
                    return f"/api/outputs/{task_id}.mp4"
                logger.warning("BGM+VO mux failed; falling back to BGM-only")

        # 7. Fallback: BGM only
        if not _mux_bgm(body_clip, bgm_wav, out_path):
            out_path.write_bytes(body_clip.read_bytes())

        kb = out_path.stat().st_size / 1024
        logger.info("compose_slideshow: wrote %s (%.1f KB, ~%.1fs, BGM-mixed)", out_path, kb, body_dur)
        return f"/api/outputs/{task_id}.mp4"


def compose_real_videos(
    video_paths: List[str],
    durations: List[float],
    task_id: str,
    *,
    transition_sec: float = 0.4,
    title: str = "Untitled",
    logline: str = "",
    voiceover_paths: Optional[List[Optional[str]]] = None,
    credits_roles: Optional[List[Tuple[str, str]]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[str]:
    """Stitch a list of *real* MP4 clips (from upstream video-generation
    APIs like Agnes `agnes-video-v2.0`) into one final teaser.

    Pipeline:
        1. Concatenate (or xfade-chain) the input clips into a single body.
        2. Lay a title-card + credits-card around it.
        3. Mix in scene-level VO.

    Returns the public URL of the resulting MP4 (`/api/outputs/<task>.mp4`).
    """
    if not video_paths:
        return None
    voiceover_paths = voiceover_paths or []

    with tempfile.TemporaryDirectory(prefix=f"vgo_{task_id}_") as tmp:
        tmp_dir = Path(tmp)

        # 1. Build a chain of pre-normalised clips so ffmpeg sees one
        #    consistent resolution + colour space throughout. We transcode
        #    each input through a 2-pass chain (no fancy scaling needed —
        #    the upstream generator usually gives consistent clips).
        body_clips: List[Path] = []
        for idx, (vpath, dur) in enumerate(zip(video_paths, durations)):
            cp = tmp_dir / f"clip_{idx:02d}.mp4"
            ok = _run(
                [
                    FFMPEG_BIN, "-y", "-loglevel", "error",
                    "-i", vpath,
                    "-t", f"{max(0.5, dur):.3f}",
                    "-vf", "format=yuv420p",
                    "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
                    "-an",
                    str(cp),
                ],
                what=f"trim_clip[{idx}]",
            )
            if not ok:
                return None
            body_clips.append(cp)

        # 2. Render title + credits cards and build their clips
        title_img = render_title_card(title, logline, tmp_dir / "title.jpg",
                                      width=width, height=height)
        credits_img = render_credits_card(title, logline, tmp_dir / "credits.jpg",
                                          roles=credits_roles)

        title_dur = 2.5
        credits_dur = 2.5
        fps = 24
        title_clip = tmp_dir / "00_title.mp4"
        credits_clip = tmp_dir / "99_credits.mp4"
        if not _build_card_clip(title_img, title_clip, title_dur, fps):
            return None
        if not _build_card_clip(credits_img, credits_clip, credits_dur, fps):
            return None

        # 3. xfade chain: title → video clips → credits
        chain_clips = [title_clip] + body_clips + [credits_clip]
        chain_durations = [title_dur] + list(durations) + [credits_dur]
        body_clip = tmp_dir / "body.mp4"
        if not _xfade_chain(chain_clips, chain_durations, transition_sec, body_clip):
            return None

        # 4. Concatenate VO + BGM
        # Write to a temp path first; atomically rename to the final location
        # only after all ffmpeg steps succeed — prevents corrupting or
        # overwriting an existing good mp4 if a re-compose fails mid-way.
        final_path = OUTPUT_DIR / f"{task_id}.mp4"
        out_path = tmp_dir / f"{task_id}.mp4"
        body_dur = max(sum(chain_durations) - transition_sec * (len(chain_clips) - 1), 12.0)
        bgm_wav = tmp_dir / "bgm.wav"
        _get_bgm_wav(body_dur, bgm_wav) if voiceover_paths else None  # warm path
        if voiceover_paths:
            merged_vo = tmp_dir / "merged_vo.wav"
            # Place each VO at the cumulative offset of its scene in the
            # body, accounting for the title card at the start and the
            # xfade transitions between clips.
            scene_offsets: List[float] = []
            cursor = title_dur
            for d in durations:
                scene_offsets.append(cursor)
                cursor += d
            merged_ok = _concat_voiceover_for_scenes(
                voiceover_paths, scene_offsets, durations, merged_vo
            ) is not None
            if merged_ok and bgm_wav.exists():
                if _mux_bgm_and_voiceover(body_clip, bgm_wav, merged_vo, out_path):
                    logger.info(
                        "compose_real_videos: wrote %s (%d clips, ~%.1fs, BGM+VO)",
                        out_path, len(body_clips), body_dur,
                    )
                    os.replace(out_path, final_path)
                    return f"/api/outputs/{task_id}.mp4"
            # VO-only (skip BGM) or VO+BGM failed → try BGM-only.
            if _mux_bgm(body_clip, bgm_wav if bgm_wav.exists() else None, out_path):
                logger.info(
                    "compose_real_videos: wrote %s (BGM fallback)",
                    out_path,
                )
                os.replace(out_path, final_path)
                return f"/api/outputs/{task_id}.mp4"
            if merged_ok:
                # Pure-VO fallback: copy body + VO pair using ffmpeg.
                if _run(
                    [FFMPEG_BIN, "-y", "-loglevel", "error",
                     "-i", str(body_clip), "-i", str(merged_vo),
                     "-map", "0:v", "-map", "1:a",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                     "-shortest", "-movflags", "+faststart", str(out_path)],
                    what="mux_voice_only",
                ):
                    logger.info(
                        "compose_real_videos: wrote %s (VO-only fallback)",
                        out_path,
                    )
                    os.replace(out_path, final_path)
                    return f"/api/outputs/{task_id}.mp4"

        # 4. No audio path: copy body verbatim.
        out_path.write_bytes(body_clip.read_bytes())
        os.replace(out_path, final_path)
        logger.info("compose_real_videos: wrote %s (silent, %d clips)", final_path, len(body_clips))
        return f"/api/outputs/{task_id}.mp4"
