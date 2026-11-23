"""Video-provider abstraction.

A `VideoProvider` is a single endpoint that can produce a short
text- or image-to-video clip. The router (see
`backend.video_providers.router`) builds a primary+fallback chain so
the pipeline can keep delivering real video even when one upstream
is rate-limited or unprovisioned.

Each provider exposes two operations:

  submit(prompt=..., image_url=..., seconds=..., size=...)
       -> str  (provider-side task id)
  wait(task_id=, timeout=)
       -> VideoResult  (dataclass with status, url, error,
                        source, provenance)

The router maps provider errors to `VideoUnavailable` (caller
falls back / drops to Ken Burns) versus `ProviderError` (raised
as-is so an in-flight retry can react differently). For most
provider wrappers the distinction is the same: any unrecoverable
HTTP failure → `VideoUnavailable`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class ProviderError(Exception):
    """Raised on unrecoverable provider failure (bad key, malformed
    payload, etc.). Surfaces to the caller as-is.
    """


class VideoUnavailable(Exception):
    """Raised when the provider is *transiently* unavailable —
    rate-limited, queue-not-scheduled, etc. The caller is expected
    to retry the provider or fall back to a secondary provider.
    """


@dataclass
class VideoResult:
    status: str          # "SUCCESS" / "IN_PROGRESS" / "NOT_START" / "FAILED"
    url: Optional[str]   # public MP4 URL, only set when status == "SUCCESS"
    error: Optional[str] = None
    source: Optional[str] = None  # provider name, useful for logging
    provenance: Optional[str] = None  # "video" (real model) | "kenburns" | "mock"

    def is_done(self) -> bool:
        return self.status in ("SUCCESS", "FAILED")

    def is_success(self) -> bool:
        return self.status == "SUCCESS" and bool(self.url)


class VideoProvider(Protocol):
    """The interface every concrete provider must satisfy."""

    name: str

    def submit(
        self,
        *,
        prompt: str,
        image_url: Optional[str] = None,
        seconds: int = 5,
        size: str = "1280x704",
    ) -> str:
        """Submit a generation job. Returns the provider-specific task id."""

    def wait(self, task_id: str, *, timeout: float) -> VideoResult:
        """Block until the job lands in SUCCESS/FAILED. Raises
        VideoUnavailable on transient issues, ProviderError on hard
        ones. `timeout` is a hint; the impl may extend it.
        """
