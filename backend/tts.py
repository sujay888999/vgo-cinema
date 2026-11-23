"""Text-to-speech via NVIDIA Magpie TTS (NVCF gRPC).

Magpie TTS Multilingual is the only NVIDIA-hosted TTS marked as
"Free Endpoint" on build.nvidia.com. The gRPC API is exposed via
NVCF, and the Python client is `nvidia-riva-client`.

We expose two entry points:

* `synthesize_speech(text, out_path, voice=...)` — render text to a
  raw 16-bit PCM mono WAV file on disk. This is what the worker
  calls for each scene's voice-over.

* `is_tts_available()` — feature-detection; if `nvidia-riva-client`
  isn't installed or no key is configured, we just return False so
  the worker can degrade gracefully (silent BGM only).

The audio is then mixed with the ambient BGM in `video_compose.py`
using the existing ffmpeg-based audio pipeline.
"""
from __future__ import annotations

import logging
import os
import wave
from pathlib import Path
from typing import Optional

from .config import get_settings

logger = logging.getLogger(__name__)


# Magpie Multilingual voice IDs (9 languages). The full list from
# the build.nvidia.com model card. We default to the English Aria
# voice; users can switch via the TTS_VOICE env var.
MAGPIE_VOICES = {
    "Magpie-Multilingual.EN-US.Sofia",
    "Magpie-Multilingual.EN-US.Aria",
    "Magpie-Multilingual.EN-US.Jason",
    "Magpie-Multilingual.EN-US.Leo",
    "Magpie-Multilingual.EN-US.JohnVanStan",
    "Magpie-Multilingual.ES-ES.Sofia",
    "Magpie-Multilingual.DE-DE.Sofia",
    "Magpie-Multilingual.FR-FR.Sofia",
    "Magpie-Multilingual.IT-IT.Sofia",
    "Magpie-Multilingual.ZH-CN.Sofia",
    "Magpie-Multilingual.JA-JP.Sofia",
    "Magpie-Multilingual.HI-IN.Sofia",
    "Magpie-Multilingual.VI-VN.Sofia",
}


def is_tts_available() -> bool:
    """True iff Magpie TTS can be used right now."""
    s = get_settings()
    if (s.tts_provider or "").lower() != "magpie":
        return False
    if not s.has_nvidia_key():
        return False
    try:
        import riva.client  # noqa: F401
        return True
    except ImportError:
        return False


def _language_code_for(voice: str, default: str) -> str:
    """Pull the language code out of a Magpie voice id, e.g.
    'Magpie-Multilingual.ZH-CN.Sofia' -> 'zh-CN'.

    Magpie's gRPC endpoint expects the lower-case form (e.g.
    'en-US' for English, 'zh-CN' for Chinese), so we always
    return the token as `<lower>-<upper>` regardless of how
    it's spelled in the voice id."""
    parts = voice.split(".")
    for p in parts:
        if "-" in p and len(p) <= 6 and p.upper() == p:
            token = p.replace("_", "-")
            head, _, tail = token.partition("-")
            return f"{head.lower()}-{tail.upper()}"
    # Normalize the default as well
    head, _, tail = default.partition("-")
    return f"{head.lower()}-{tail.upper()}" if head else default


def _clean_for_magpie(text: str) -> str:
    """Magpie Multilingual's Triton backend chokes on some tokens
    (it returns "Mapping failed" for a handful of common English
    words like "silence"). Strip out anything that isn't a basic
    ASCII letter, digit, whitespace, or a small set of safe
    punctuation marks. The result reads slightly stiffer but is
    always pronounceable."""
    import re
    safe = ".,'!?-"
    cleaned = re.sub(rf"[^A-Za-z0-9\s{re.escape(safe)}]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_mapping_error(exc: Exception) -> bool:
    """The Magpie Triton ensemble intermittently raises
    'Mapping failed' on perfectly valid text. It's a server-side
    warmup/loader bug, not a content issue, so a short retry
    usually clears it."""
    msg = str(exc)
    return "Mapping failed" in msg or "Triton model failed" in msg


def synthesize_speech(text: str, out_wav: Path, *,
                      voice: Optional[str] = None,
                      language: Optional[str] = None,
                      sample_rate: Optional[int] = None,
                      max_retries: int = 3) -> Optional[float]:
    """Render `text` to a 16-bit mono WAV at `out_wav`.

    Returns the duration of the resulting audio in seconds, or None
    on failure (so the caller can fall back to silent narration).
    """
    if not text or not text.strip():
        return None
    if not is_tts_available():
        logger.info("TTS unavailable (no key or riva-client missing)")
        return None

    import riva.client
    import time as _time

    s = get_settings()
    voice = voice or s.tts_voice
    sample_rate = sample_rate or s.tts_sample_rate
    language = language or _language_code_for(voice, s.tts_language)

    cleaned = _clean_for_magpie(text)
    if not cleaned:
        return None

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Two-stage: try the configured voice first, fall back to a
    # well-known safe voice (Sofia) if the first attempt errors
    # with the "Mapping failed" Triton message. Each voice also
    # gets up to `max_retries` attempts with backoff, because the
    # Magpie ensemble intermittently raises "Mapping failed"
    # on perfectly normal text (Triton warmup bug).
    voices_to_try = [voice]
    fallback = "Magpie-Multilingual.EN-US.Sofia"
    if fallback not in voices_to_try:
        voices_to_try.append(fallback)

    last_error = None
    for v in voices_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                auth = riva.client.Auth(
                    uri=s.tts_nvcf_uri,
                    use_ssl=True,
                    metadata_args=[
                        ["function-id", s.tts_function_id],
                        ["authorization", f"Bearer {s.nvidia_api_key}"],
                    ],
                )
                tts = riva.client.SpeechSynthesisService(auth)
                logger.info("Magpie TTS: voice=%s lang=%s text_len=%d attempt=%d/%d",
                            v, language, len(cleaned), attempt, max_retries)
                resp = tts.synthesize(
                    text=cleaned,
                    voice_name=v,
                    language_code=language,
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    sample_rate_hz=int(sample_rate),
                )
                pcm = resp.audio
                if not pcm:
                    last_error = "empty audio"
                    continue

                with wave.open(str(out_wav), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(int(sample_rate))
                    w.writeframes(pcm)
                duration = len(pcm) / (2 * int(sample_rate))
                logger.info("Magpie TTS OK (voice=%s): %.2fs, %d bytes -> %s",
                            v, duration, len(pcm), out_wav)
                return duration
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)[:120]
                logger.warning("Magpie TTS attempt %d failed (voice=%s): %s",
                               attempt, v, exc)
                if _is_mapping_error(exc) and attempt < max_retries:
                    backoff = min(2 ** attempt, 8)
                    logger.info("Retrying in %ds (transient Triton bug)…", backoff)
                    _time.sleep(backoff)
                    continue
                break  # non-transient error -> move to next voice
        else:
            continue
    logger.error("Magpie TTS exhausted voices. last_error=%s", last_error)
    return None
