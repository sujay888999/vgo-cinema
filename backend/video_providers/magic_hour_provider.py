"""Magic Hour video provider (real AI video generation).

Implements the ``VideoProvider`` protocol in :mod:`backend.video_providers`
using the Magic Hour REST API (no extra dependency, stdlib urllib only):

* ``POST /v1/text-to-video``  -> returns a job ``id`` (credits are charged
  when the job is queued; failed jobs are refunded by Magic Hour).
* ``GET /v1/video-projects/{id}`` -> poll ``status`` until ``complete``;
  the finished render is in ``downloads[0].url`` (a signed, expiring URL).

Model/resolution come from settings (``MAGIC_HOUR_MODEL`` /
``MAGIC_HOUR_RESOLUTION``), defaulting to the cheapest usable combination
``ltx-2.5`` at ``480p`` (24 credits/sec on the free plan). The ``image_url``
argument is intentionally ignored: the workflow may pass a local image path
(which Magic Hour cannot fetch), so we always do text-to-video from the
scene prompt. The returned clip is a genuine AI-generated video, and the
``VideoResult`` is marked ``provenance="video"`` so the UI never confuses
it with the local Ken Burns fallback.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from . import ProviderError, VideoResult, VideoUnavailable

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.magichour.ai"
_POLL_INTERVAL_S = 10.0


def _aspect_ratio(size: str) -> str:
    try:
        width, height = (int(part) for part in str(size).lower().split("x", 1))
    except (ValueError, TypeError):
        return "16:9"
    if width > height:
        return "16:9"
    if height > width:
        return "9:16"
    return "1:1"


class MagicHourVideoProvider:
    """Text-to-video provider backed by the Magic Hour API."""

    name = "magic-hour"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "ltx-2.5",
        resolution: str = "480p",
        base_url: str = _DEFAULT_BASE_URL,
        audio: bool = False,
        output_dir: Optional[str] = None,
    ):
        key = api_key or os.getenv("MAGIC_HOUR_API_KEY") or ""
        if not key.strip():
            raise ProviderError(
                "MAGIC_HOUR_API_KEY is not configured; cannot use magic-hour provider"
            )
        self.api_key = key.strip()
        self.model = model
        self.resolution = resolution
        self.base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self.audio = audio
        self.output_dir = output_dir
        self.last_credits_charged: Optional[int] = None
        self.last_job_id: Optional[str] = None

    # -- HTTP helper -----------------------------------------------------
    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read() or b"{}"
                return json.loads(payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code == 401:
                raise ProviderError(
                    "Magic Hour API key rejected (HTTP 401); check MAGIC_HOUR_API_KEY"
                ) from exc
            if exc.code == 402:
                raise ProviderError(
                    "Magic Hour credits exhausted (HTTP 402); top up the account"
                ) from exc
            raise ProviderError(
                f"Magic Hour {method} {path} failed: HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Magic Hour {method} {path} unreachable: {exc}") from exc

    # -- VideoProvider protocol ------------------------------------------
    def submit(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        seconds: float = 5,
        size: str = "1280x704",
    ) -> str:
        if image_url:
            logger.info("magic-hour: ignoring image_url, using text-to-video")
        body = {
            "name": f"vgo-cinema-{float(seconds):g}s",
            "model": self.model,
            "resolution": self.resolution,
            "aspect_ratio": _aspect_ratio(size),
            "end_seconds": float(seconds),
            "audio": self.audio,
            "style": {"prompt": prompt},
        }
        result = self._request("POST", "/v1/text-to-video", body)
        job_id = result.get("id")
        if not job_id:
            raise ProviderError(f"Magic Hour submit returned no job id: {result}")
        credits = result.get("credits_charged")
        if isinstance(credits, (int, float)):
            self.last_credits_charged = int(credits)
        self.last_job_id = job_id
        logger.info(
            "magic-hour: submitted job %s model=%s %s %ss (est. %s credits)",
            job_id, self.model, self.resolution, seconds, credits,
        )
        return str(job_id)

    def wait(self, task_id: str, timeout: float = 360) -> VideoResult:
        deadline = time.time() + max(float(timeout), 1.0)
        while True:
            detail = self._request("GET", f"/v1/video-projects/{task_id}")
            status = str(detail.get("status", "")).lower()
            credits = detail.get("credits_charged")
            if isinstance(credits, (int, float)):
                self.last_credits_charged = int(credits)
            if status == "complete":
                url = self._download_url(detail)
                if not url:
                    raise ProviderError(
                        f"Magic Hour job {task_id} complete but no download url: {detail}"
                    )
                logger.info(
                    "magic-hour: job %s complete (%s credits): %s",
                    task_id, self.last_credits_charged, url[:80],
                )
                return VideoResult(
                    status="SUCCESS",
                    url=url,
                    source=self.name,
                    provenance="video",
                )
            if status in ("error", "failed", "canceled", "cancelled"):
                raise ProviderError(
                    f"Magic Hour job {task_id} ended with status={status}: "
                    f"{str(detail.get('error')) or detail}"
                )
            if time.time() >= deadline:
                raise VideoUnavailable(
                    f"Magic Hour job {task_id} did not finish within {timeout}s "
                    f"(last status={status})"
                )
            logger.info("magic-hour: job %s status=%s, waiting...", task_id, status)
            time.sleep(_POLL_INTERVAL_S)

    @staticmethod
    def _download_url(detail: dict) -> Optional[str]:
        downloads = detail.get("downloads") or []
        for item in downloads:
            if isinstance(item, str) and item:
                return item
            if isinstance(item, dict):
                for key in ("url", "file_url", "download_url"):
                    if item.get(key):
                        return str(item[key])
        for key in ("url", "video_url", "file_url", "output_url"):
            if detail.get(key):
                return str(detail[key])
        return None
