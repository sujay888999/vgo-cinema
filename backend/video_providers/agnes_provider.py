"""Agnes-AI provider.

Reuses the existing `AgnesClient` for chat/image/video traffic. The
provider implements the `VideoProvider` protocol by translating the
client's "submit task → poll → poll → URL" flow into the unified
`submit / wait` shape, while keeping the per-process rate-limit
quota — released to the rest of the system as
`QuotaState` so the router can serialise multi-provider retries.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

from . import ProviderError, VideoResult, VideoUnavailable

logger = logging.getLogger(__name__)


# Quota per-minute string parsed from server error messages.
_PER_MINUTE_RE = re.compile(r"allows\s+(\d+)\s+requests?\s+per\s+(\d+)\s+minute", re.I)
_TRY_IN_RE = re.compile(r"try again in\s+(\d+)\s*seconds?", re.I)


def parse_rate_limit_hint(exc: Exception) -> Optional[float]:
    """Best-effort Retry-After hint out of an Agnes 429 / 503 message.

    Returns seconds-to-wait (`float`) or `None` if nothing usable was
    found. Falls through both documented ("X requests per Y minute")
    and chatty ("try again in N seconds") phrasings.
    """
    msg = str(exc) if exc else ""
    m = _PER_MINUTE_RE.search(msg)
    if m:
        n, window = int(m.group(1)), int(m.group(2))
        per_req = max(1.0, (window * 60.0) / max(1, n))
        return per_req + 2.0
    m = _TRY_IN_RE.search(msg)
    if m:
        return float(m.group(1)) + 1.0
    return None


class AgnesVideoProvider:
    """Adapter over `AgnesClient.generate_video` + `get_video_status`.

    Shares the rate-limit-aware quota primitives with the older
    `workflow_agnes._submit_video_serial` so behaviour doesn't
    regress.
    """

    name = "agnes"

    def __init__(
        self,
        *,
        agnes_client,
        cooldown_after_accept: float = 2.0,
    ):
        self.client = agnes_client
        self._cooldown_after_accept = cooldown_after_accept

    # ----- submit -----------------------------------------------------
    def submit(
        self,
        *,
        prompt: str,
        image_url: Optional[str] = None,
        seconds: int = 5,
        size: str = "1280x704",
    ) -> str:
        """Submit a video generation job.

        Retries 429 / 503 / "service busy" automatically per the
        `max_attempts` budget. Anything else is surfaced as
        ProviderError so the router can decide whether to fall back.
        """
        from ..agnes_client import AgnesAPIError

        delay = 1.5
        last_err: Optional[Exception] = None
        for attempt in range(1, 5):  # mirror previous budget
            try:
                task_id = self.client.generate_video(
                    prompt=prompt,
                    image_url=image_url,
                    seconds=seconds,
                    size=size,
                )
                time.sleep(self._cooldown_after_accept)
                logger.info(
                    "AgnesVideoProvider: accept task %s (attempt %d)", task_id, attempt,
                )
                return task_id
            except AgnesAPIError as exc:
                last_err = exc
                msg = str(exc).lower()
                if "503" not in msg and "429" not in msg and "rate limit" not in msg \
                   and "service busy" not in msg:
                    raise ProviderError(f"Agnes video hard error: {exc}") from exc
                hint = parse_rate_limit_hint(exc)
                if hint is not None:
                    delay = max(15.0, hint)
                else:
                    delay = min(15.0 * (2 ** (attempt - 1)), 75.0)
                logger.warning(
                    "AgnesVideoProvider: transient %s (attempt %d) — backoff %.1fs",
                    exc, attempt, delay,
                )
                time.sleep(delay)
        raise VideoUnavailable(f"AgnesVideoProvider: retries exhausted: {last_err}")

    # ----- wait -------------------------------------------------------
    def wait(self, task_id: str, *, timeout: float) -> VideoResult:
        """Poll until the task lands in SUCCESS/FAILED.

        Uses a generous NOT_START grace window because Agnes queue
        latency commonly spans 30-60 s. Raises `VideoUnavailable`
        (server not scheduling) vs `ProviderError` (FAILED).
        """
        from ..agnes_client import AgnesAPIError

        end = time.time() + timeout
        not_start_streak = 0
        not_start_bail = 6  # 30 s grace

        while time.time() < end:
            try:
                info = self.client.get_video_status(task_id)
            except AgnesAPIError as exc:
                raise ProviderError(str(exc)) from exc

            status = info["status"]
            if status == "SUCCESS" and info.get("url"):
                return VideoResult(status="SUCCESS", url=info["url"], source=self.name)
            if status == "FAILED":
                return VideoResult(status="FAILED", url=None,
                                   error=str(info.get("error")), source=self.name)
            if status == "NOT_START":
                not_start_streak += 1
                if not_start_streak >= not_start_bail:
                    raise VideoUnavailable(
                        f"Agnes video task {task_id} stuck in NOT_START for "
                        f"{not_start_streak} consecutive polls."
                    )
            else:
                not_start_streak = 0
            time.sleep(5.0)

        # Hit the timeout without the work landing. Surface as
        # transient so the router can decide to retry with a
        # fallback provider or simply abort. Most projects time out
        # at 360 s.
        raise VideoUnavailable(f"Agnes video task {task_id} timed out after {timeout}s")
