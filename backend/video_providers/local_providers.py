"""Local-only providers: NullProvider + MockProvider.

These are explicitly *not* video generators — they exercise the
provenance / autosplit paths so you can develop the full pipeline
on a host without an upstream key.

  NullProvider  — always fails with `VideoUnavailable`. The pipeline
                  flow drops to Ken Burns derivation end-to-end.
                  Useful when you want to *test* the fallback path
                  without spending API quota.

  MockProvider  — synthesises a short MP4 locally via ffmpeg using
                  a Ken Burns effect, but marks the result with
                  `provenance="mock"`. Lets the front-end see
                  what the pipeline looks like with zero
                  API spend, while still distinguishing it from real video.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from . import ProviderError, VideoResult, VideoUnavailable

logger = logging.getLogger(__name__)


class NullProvider:
    name = "null"

    def submit(self, *, prompt: str, image_url: Optional[str] = None,
               seconds: int = 5, size: str = "1280x704") -> str:
        raise VideoUnavailable("NullProvider: no upstream configured")

    def wait(self, task_id: str, *, timeout: float) -> VideoResult:
        raise VideoUnavailable("NullProvider cannot wait")


class MockProvider:
    """Returns a synthesised local MP4 marked as 'mock' provenance.

    The MP4 is a Ken Burns clip on the supplied `image_url` (which
    must be a local file path). Unlike real video providers, this
    is a local synthesis — we mark `provenance="mock"` so the UI
    can distinguish it from genuine model-generated video.

    In text-to-video mode the router passes `image_url=None`; this
    provider then raises `ProviderError` so the router falls through
    to the next provider (or the workflow's Ken Burns fallback, which
    uses the locally-generated image directly). This is intentional —
    MockProvider needs a real image file to synthesise a clip.
    """

    name = "mock"

    def __init__(self, *, output_dir: Path):
        self.output_dir = output_dir

    def submit(self, *, prompt: str, image_url: Optional[str] = None,
               seconds: int = 5, size: str = "1280x704") -> str:
        if not image_url:
            # Text-to-video mode: MockProvider can't synthesise a Ken
            # Burns clip without an image. Raise so the router falls
            # through; the workflow's Ken Burns fallback (which uses
            # the locally-generated image) handles the final fallback.
            raise ProviderError("MockProvider requires image_url (local path)")
        src = Path(image_url)
        if not src.exists():
            raise ProviderError(f"MockProvider: source image missing: {src}")
        # Encode a short Ken Burns clip. Done eagerly so `wait` can
        # return the URL right away in this single-shot mock.
        from ..video_compose import _build_card_clip
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dest = self.output_dir / f"mock_{src.stem}.mp4"
        ok = _build_card_clip(src, dest, max(1.0, float(seconds) + 0.5), fps=24)
        if not ok:
            raise ProviderError("MockProvider: ffmpeg encode failed")
        # Return a synthesised "task id" that we'll resolve in wait.
        return f"mock://{dest.name}"

    def wait(self, task_id: str, *, timeout: float) -> VideoResult:
        # Already encoded in submit(); just unwrap the path.
        if not task_id.startswith("mock://"):
            raise ProviderError(f"MockProvider: bad task id {task_id!r}")
        path = self.output_dir / task_id[len("mock://"):]
        if not path.exists() or path.stat().st_size <= 0:
            return VideoResult(status="FAILED", url=None,
                               error="MockProvider: no output file",
                               source=self.name,
                               provenance="mock")
        # The compose stage wants a URL it can normalise; use a
        # file:// URL so any prefetch/download path can stay the same.
        return VideoResult(status="SUCCESS", url=str(path), source=self.name,
                           provenance="mock")
