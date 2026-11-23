"""Image generation provider.

Image providers (set `IMAGE_PROVIDER=...`):

1. `flux_dev` (default) — NVIDIA NIM FLUX.1-dev. ~28 steps, cfg 3.5,
   base64 JPEG. Best prompt adherence, ~$0.012/image on the standard
   NIM tier. Hit on `https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev`
   using the same NVIDIA_API_KEY as the LLMs.

2. `flux_schnell` — NVIDIA NIM FLUX.1-schnell. Distilled, so cfg_scale
   must be 0 and 1-4 steps suffice. Returns base64 JPEG.

3. `sd35` — NVIDIA NIM Stable Diffusion 3.5 Large. Higher base
   quality on a wider range of styles; the same Visual GenAI endpoint
   pattern.

4. `picsum` — Lorem Picsum placeholder (no key, no prompt match).
   Still useful for offline dev / smoke tests.

5. `none` — skip image generation; the worker just records empty
   image URLs and the UI shows the "AI Director is preparing"
   placeholder.

The two useful entry points are:

* `build_image_url(prompt, seed=...)` — returns a string identifier
  (URL for picsum, NIM model name + payload id for the NIM providers).
  For the NIM providers we return a `nim://flux-dev/<hash>` style
  marker so the worker's image-fetch step knows to call
  `generate_image_bytes()` instead of `httpx.get()`.

* `generate_image_bytes(prompt, seed=...)` — returns the raw JPEG/PNG
  bytes, raising on failure. This is what the worker actually
  downloads; for picsum we go through `httpx` to mirror the same shape.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import urllib.parse
import uuid
from typing import Optional, Tuple

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)


# ------------------------------ NIM providers ---------------------------------

# Models exposed via NVIDIA NIM Visual GenAI (`/v1/genai/<pub>/<model>`).
# Each entry: (publisher/model, payload field names, default steps/cfg).
NIM_IMAGE_MODELS = {
    "flux_dev": {
        "url": "black-forest-labs/flux.1-dev",
        "steps_field": "steps",
        "cfg_field": "cfg_scale",
        "default_steps": 28,
        "default_cfg": 3.5,
    },
    "flux_schnell": {
        "url": "black-forest-labs/flux.1-schnell",
        "steps_field": "steps",
        "cfg_field": "cfg_scale",
        "default_steps": 4,
        "default_cfg": 0.0,  # distilled, no classifier-free guidance
    },
    "sd35": {
        "url": "stabilityai/stable-diffusion-3.5-large",
        "steps_field": "steps",
        "cfg_field": "cfg_scale",
        "default_steps": 30,
        "default_cfg": 4.5,
    },
}


# ------------------------------ Public API ------------------------------------


def build_image_url(prompt: str, *, seed: Optional[str] = None) -> str:
    """Return a deterministic image identifier for `prompt`.

    NOTE: NIM providers used to return a `nim://` pseudo-URL marker here
    so an out-of-band fetcher could resolve them. That abstraction is
    dead — `_generate_image_for_scene` in workflow.py calls
    `generate_image_bytes()` directly now and never re-checks the URL
    scheme. Today this helper exists only for the legacy picsum /
    pollinations / "none" providers; it raises for the NIM providers.
    """
    s = get_settings()
    provider = (s.image_provider or "picsum").lower()
    seed = seed or uuid.uuid4().hex[:12]

    if provider in NIM_IMAGE_MODELS:
        # The caller (`workflow._generate_image_for_scene`) handles NIM
        # providers by calling `generate_image_bytes` directly, so we
        # never reach this branch in normal operation.  Return an empty
        # string as a defensive marker.
        return ""
    if provider == "pollinations":
        return _build_pollinations_url(prompt, seed, s)
    if provider == "picsum":
        return _build_picsum_url(prompt, seed, s)
    if provider == "none":
        return ""
    logger.warning("Unknown IMAGE_PROVIDER=%r, falling back to picsum", provider)
    return _build_picsum_url(prompt, seed, s)


def generate_image_bytes(prompt: str, *, seed: Optional[str] = None,
                          timeout: float = 300.0,
                          width: Optional[int] = None,
                          height: Optional[int] = None) -> Tuple[bytes, str]:
    """Return (image_bytes, content_type) for `prompt`.

    Dispatches on the configured `IMAGE_PROVIDER`:
    * NIM providers — POST to `ai.api.nvidia.com/v1/genai/<pub>/<model>`,
      decode `artifacts[0].base64` from the JSON response.
    * `picsum` — GET the placeholder URL, return the JPEG bytes.
    * `none` — raise.

    Raises `RuntimeError` on failure with a useful message.
    """
    s = get_settings()
    provider = (s.image_provider or "picsum").lower()
    seed = seed or uuid.uuid4().hex[:12]
    img_w = width if width is not None else s.image_width
    img_h = height if height is not None else s.image_height

    if provider in NIM_IMAGE_MODELS:
        return _generate_via_nim(prompt, seed, s, provider, timeout=timeout,
                                 width=img_w, height=img_h)
    if provider == "picsum":
        return _generate_via_picsum(prompt, seed, s, timeout=timeout,
                                    width=img_w, height=img_h)
    if provider == "none":
        raise RuntimeError("IMAGE_PROVIDER=none: skip image generation")
    raise RuntimeError(f"Unknown IMAGE_PROVIDER={provider!r}")


# ------------------------------ Implementations ------------------------------


def _generate_via_nim(prompt: str, seed: str, s, provider: str, *,
        timeout: float,
        width: Optional[int] = None,
        height: Optional[int] = None) -> Tuple[bytes, str]:
    spec = NIM_IMAGE_MODELS[provider]
    api_key = (s.nvidia_api_key or "").strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required for the FLUX/SD3.5 NIM providers")

    width = _snap_to_nim_width(width if width is not None else s.image_width)
    height = _snap_to_nim_height(height if height is not None else s.image_height)
    payload = {
        "prompt": prompt,
        spec["steps_field"]: s.image_inference_steps or spec["default_steps"],
        spec["cfg_field"]: s.image_cfg_scale if s.image_cfg_scale is not None else spec["default_cfg"],
        "width": width,
        "height": height,
    }
    url = f"{s.nvidia_image_base_url.rstrip('/')}/{spec['url']}"
    logger.info("NIM image: provider=%s url=%s steps=%d cfg=%s size=%dx%d",
                provider, url, payload[spec["steps_field"]], payload[spec["cfg_field"]],
                width, height)
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
            )
        if r.status_code >= 400:
            raise RuntimeError(
                f"NIM image HTTP {r.status_code}: {r.text[:300]}"
            )
        data = r.json()
        b64 = data["artifacts"][0]["base64"]
        # NIM returns JPEG for FLUX, PNG for SD; we don't have a content-type
        # in the response, so we sniff the magic bytes.
        raw = base64.b64decode(b64)
        ctype = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        logger.info("NIM image OK: %d bytes (%s)", len(raw), ctype)
        return raw, ctype
    except httpx.HTTPError as exc:
        raise RuntimeError(f"NIM image request failed: {exc}") from exc


def _generate_via_picsum(prompt: str, seed: str, s, *,
                          timeout: float,
                          width: Optional[int] = None,
                          height: Optional[int] = None) -> Tuple[bytes, str]:
    digest = hashlib.sha256((prompt + "|" + seed).encode("utf-8")).hexdigest()[:16]
    img_w = width if width is not None else s.image_width
    img_h = height if height is not None else s.image_height
    url = f"https://picsum.photos/seed/{digest}/{img_w}/{img_h}"
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url)
    r.raise_for_status()
    return r.content, "image/jpeg"


def _build_picsum_url(prompt: str, seed: str, s) -> str:
    digest = hashlib.sha256((prompt + "|" + seed).encode("utf-8")).hexdigest()[:16]
    return f"https://picsum.photos/seed/{digest}/{s.image_width}/{s.image_height}"


def _build_pollinations_url(prompt: str, seed: str, s) -> str:
    encoded = urllib.parse.quote(prompt, safe="")
    return (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={s.image_width}&height={s.image_height}"
        f"&seed={seed}&nologo=true&enhance=true"
    )


# NIM FLUX accepts only widths/heights divisible by 16 within a tight
# range. Snap to the nearest valid value so we don't get a 422.
_NIM_VALID = [768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344]


def _snap_to_nim_width(w: int) -> int:
    return min(_NIM_VALID, key=lambda v: abs(v - w))


def _snap_to_nim_height(h: int) -> int:
    return min(_NIM_VALID, key=lambda v: abs(v - h))
