"""Video-provider router.

Builds a primary→fallback chain driven entirely by config:

    [video_providers]
    primary = magic-hour
    fallback = mock,null   # tried in order

If `primary` is unavailable the router tries the next item in
`fallback`, etc. The same per-offset rate-limit bucket
(`_next_submit_at`) is shared across providers so a stuck Agnes
attempt won't accidentally bypass its own throttle by switching
into MockProvider between calls — provider switches are at the
job level, not the request level.

The router exposes `submit_for_job(prompt, image_url, ...)`
returning `(provider_name, task_id)` or raising
`VideoUnavailable` if every provider refused. Calling
`wait_for_job(provider_name, task_id)` polls the right provider.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

from . import ProviderError, VideoProvider, VideoResult, VideoUnavailable
from .agnes_provider import AgnesVideoProvider
from .magic_hour_provider import MagicHourVideoProvider
from .local_providers import MockProvider, NullProvider

logger = logging.getLogger(__name__)

# ----- Cross-provider rate-limit bucket -----------------------------------
# Agnes caps at "1 request per 1 minute" — that constraint applies
# regardless of which provider is named in the URL, so the bucket
# lives in the router, not in the provider.
_VIDEO_RATE_LIMIT_SEC = 60.0
_VIDEO_BEGIN_COOLDOWN = 2.0
_bucket_lock = threading.Lock()
_next_submit_at: float = 0.0


def _wait_for_quota() -> float:
    with _bucket_lock:
        now = time.monotonic()
        if now >= _next_submit_at:
            return 0.0
        sleep = _next_submit_at - now
    time.sleep(sleep)
    return sleep


def _reserve_next_slot() -> None:
    global _next_submit_at
    with _bucket_lock:
        base = max(time.monotonic(), _next_submit_at)
        _next_submit_at = base + _VIDEO_RATE_LIMIT_SEC


# ----- Provider registry --------------------------------------------------
def _build_providers(chain: Iterable[str], *, agnes_client,
                     output_dir) -> List[VideoProvider]:
    """Instantiate provider objects in declaration order.

    Unknown provider names land on NullProvider (so the slot is
    reserved and the chain still terminates with a clean
    "all-buckets-exhausted" error rather than a TypeError).
    """
    out: List[VideoProvider] = []
    for name in chain:
        n = (name or "").lower().strip()
        if n == "agnes":
            # H4: Agnes is explicitly UNAVAILABLE (AGNES_API_KEY is empty
            # or a placeholder). Never instantiate the Agnes provider;
            # substitute NullProvider so the chain keeps its shape.
            logger.warning(
                "video_router: 'agnes' in chain is unavailable "
                "(AGNES_API_KEY is empty or a placeholder); "
                "substituting with NullProvider"
            )
            out.append(NullProvider())
        elif n == "magic-hour":
            from ..config import get_settings
            settings = get_settings()
            out.append(
                MagicHourVideoProvider(
                    api_key=settings.magic_hour_api_key or None,
                    model=settings.magic_hour_model,
                    resolution=settings.magic_hour_resolution,
                    base_url=settings.magic_hour_base_url,
                )
            )
        elif n == "mock":
            out.append(MockProvider(output_dir=output_dir))
        elif n == "null":
            out.append(NullProvider())
        else:
            logger.warning("video_providers: unknown provider %r — treating as null", name)
            out.append(NullProvider())
    if not out:
        out.append(NullProvider())
    return out


class VideoRouter:
    """Primary + fallback router.

    Submit / wait semantics:
      - `submit_for_job` tries providers in order. If a provider
        raises `VideoUnavailable`, we fall back to the next one. If
        a provider raises `ProviderError` we *also* fall back — that
        exception class means the request itself was malformed, and
        a fallback provider is unlikely to fix it but cheap to try.
      - Submission waits for a free quota slot *inside* the
        provider chain (the slot is global across providers), so
        a switch from Agnes→Mock arrives after the next 1-minute
        window.
      - Returns the provider name + task id so `wait_for_job` knows
        which provider to poll.
    """

    def __init__(self, chain: Iterable[str], *, agnes_client=None,
                 output_dir=None):
        if output_dir is None:
            from ..video_compose import OUTPUT_DIR
            output_dir = OUTPUT_DIR
        self.chain: List[VideoProvider] = _build_providers(
            chain, agnes_client=agnes_client, output_dir=output_dir,
        )
        self.name_lookup: Dict[str, VideoProvider] = {
            p.name: p for p in self.chain
        }

    @property
    def primary(self) -> VideoProvider:
        return self.chain[0]

    def submit_for_job(
        self,
        *,
        prompt: str,
        image_url: Optional[str] = None,
        seconds: int = 5,
        size: str = "1280x704",
    ) -> Tuple[str, str]:
        """Walk the chain. Returns (provider_name, task_id)."""
        last_unavail: Optional[Exception] = None
        for provider in self.chain:
            try:
                # Only Agnes has an upstream rate limit (1 req/min).
                # Mock and Null providers generate locally and don't
                # consume external quota, so they skip the wait entirely.
                # This prevents the global bucket from serialising parallel
                # scene sub-tasks that ultimately use local providers.
                if provider.name == "agnes":
                    waited = _wait_for_quota()
                    if waited > 0:
                        logger.info("video_router: quota wait %.1fs before submit", waited)
                task_id = provider.submit(
                    prompt=prompt, image_url=image_url,
                    seconds=seconds, size=size,
                )
                if provider.name == "agnes":
                    _reserve_next_slot()
                    time.sleep(_VIDEO_BEGIN_COOLDOWN)
                return provider.name, task_id
            except VideoUnavailable as exc:
                last_unavail = exc
                logger.info(
                    "video_router: provider %s unavailable (%s); trying next",
                    provider.name, exc,
                )
            except ProviderError as exc:
                # Treat as fallback-eligible. Most cases are 4xx
                # upstream payloads.
                last_unavail = exc
                logger.info(
                    "video_router: provider %s hard error (%s); trying next",
                    provider.name, exc,
                )
        raise VideoUnavailable(
            f"video_router: all providers refused the job"
            f"{f' — last error: {last_unavail}' if last_unavail else ''}"
        )

    def wait_for_job(self, provider_name: str, task_id: str,
                     *, timeout: float = 360.0) -> VideoResult:
        provider = self.name_lookup.get(provider_name)
        if provider is None:
            raise ProviderError(f"video_router: unknown provider {provider_name!r}")
        return provider.wait(task_id, timeout=timeout)


# ----- Singleton accessor -----------------------------------------------
_ROUTER_SINGLETON: Optional[VideoRouter] = None
_ROUTER_LOCK = threading.Lock()


def get_router(*, force_reload: bool = False) -> VideoRouter:
    """Build / return a singleton router driven by settings.

    Chain order:
      ["<primary>", "<fallback-1>", "<fallback-2>", ...]
    Pulled from config:
      video_provider_primary    (default "magic-hour")
      video_provider_fallback   (csv of names, default "mock,null")
    """
    global _ROUTER_SINGLETON
    with _ROUTER_LOCK:
        if _ROUTER_SINGLETON is not None and not force_reload:
            return _ROUTER_SINGLETON
        from ..config import get_settings
        settings = get_settings()
        primary = getattr(settings, "video_provider_primary", "magic-hour") or "magic-hour"
        fallback_csv = getattr(settings, "video_provider_fallback", "mock,null") or ""
        chain = [primary] + [p.strip() for p in fallback_csv.split(",") if p.strip()]
        # Strip empties + duplicates while keeping order.
        seen = set()
        chain = [c for c in chain if not (c in seen or seen.add(c))]
        # H4: Agnes is explicitly UNAVAILABLE (no real key is configured;
        # AGNES_API_KEY is a placeholder). Never build an Agnes client;
        # drop "agnes" from the chain instead of substituting silently.
        if "agnes" in chain:
            logger.warning(
                "video_router: 'agnes' was requested in the video chain "
                "but Agnes is explicitly unavailable (AGNES_API_KEY is "
                "empty or a placeholder); removing it from the chain."
            )
            chain = [c for c in chain if c != "agnes"]
        agnes_client = None
        if "magic-hour" in chain and not settings.has_magic_hour_key():
            logger.info("video_router: magic-hour in chain but MAGIC_HOUR_API_KEY unset or a placeholder; substituting with NullProvider")
            chain = [c if c != "magic-hour" else "null" for c in chain]
        _ROUTER_SINGLETON = VideoRouter(chain, agnes_client=agnes_client)
        logger.info("video_router: initialised chain = %s",
                    [p.name for p in _ROUTER_SINGLETON.chain])
        return _ROUTER_SINGLETON
