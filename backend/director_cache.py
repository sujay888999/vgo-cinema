"""Director script cache.

For identical user prompts we reuse the previously-generated cinematic
script (title / logline / scenes) so the user gets an instant response on
repeat submissions and we save a paid LLM roundtrip.

The cache is keyed by sha256(prompt + '|' + style + '|' + max_scenes) and
stored as a Redis string with a configurable TTL. Worker (and any other
process that can reach Redis) populates and reads it transparently.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Optional, Tuple

import redis

from .config import get_settings

from . import disk_cache  # noqa: E402

logger = logging.getLogger(__name__)

_KEY_PREFIX = "vgo:director-cache:"
_TTL_SEC = 86400 * 7   # 7 days; longer is fine for an LLM-response cache.

_redis_client: Optional[redis.Redis] = None
_redis_checked: bool = False
_lock = threading.Lock()


def _get_redis() -> Optional[redis.Redis]:
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    # Don't even attempt a connection when Redis is disabled — mirrors
    # the behaviour of task_store._try_connect().
    if not get_settings().enable_redis:
        with _lock:
            _redis_client = None
            _redis_checked = True
        return None
    with _lock:
        if _redis_checked:
            return _redis_client
        try:
            client = redis.Redis.from_url(
                get_settings().redis_url,
                socket_timeout=2,
                decode_responses=True,
            )
            client.ping()
            _redis_client = client
        except (redis.RedisError, OSError) as exc:
            logger.warning("director_cache: Redis unavailable: %s", exc)
            _redis_client = None
        _redis_checked = True
    return _redis_client


def _cache_key(prompt: str, style: Optional[str], max_scenes: int) -> str:
    raw = f"{prompt.strip()}|{style or ''}|{max_scenes}"
    return _KEY_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(prompt: str, style: Optional[str], max_scenes: int) -> Optional[dict]:
    """Look up a cached director response. Returns the dict on hit, else None.

    Lookup order:
      1. Redis (fastest; only populated when ENABLE_REDIS=true).
      2. Disk (`data/cache/director/<sha>.json`) — always present.
    Hits are mirrored: a Redis hit copies itself into the disk cache so
    a later no-Redis deploy still benefits.
    """
    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(_cache_key(prompt, style, max_scenes))
            if raw:
                try:
                    cached_dispatch, hits = raw.split("\0", 1) if "\0" in raw else ("1", raw)
                    if json.loads(cached_dispatch):
                        data = json.loads(hits if "\0" in raw else raw)
                        # Mirror into disk so a later redis-less run
                        # still finds it.
                        disk_cache.put_director(prompt, style, max_scenes, data)
                        return data
                except (ValueError, TypeError) as exc:
                    logger.warning("director_cache: malformed cache value: %s", exc)
        except (redis.RedisError, OSError) as exc:
            logger.warning("director_cache: GET failed: %s", exc)
    return disk_cache.get_director(prompt, style, max_scenes)


def put(prompt: str, style: Optional[str], max_scenes: int, response: dict) -> None:
    """Store a director response in Redis (best-effort) AND on disk.

    Disk is the source of truth across processes / restarts; Redis
    is purely an accelerator.
    """
    disk_cache.put_director(prompt, style, max_scenes, response)
    r = _get_redis()
    if r is None:
        return
    payload = json.dumps(True) + "\0" + json.dumps(response)
    try:
        r.set(_cache_key(prompt, style, max_scenes), payload, ex=_TTL_SEC)
    except (redis.RedisError, OSError) as exc:
        logger.warning("director_cache: PUT failed: %s", exc)


def stats() -> Tuple[int, int]:
    """Return (entries, ttl_seconds) for diagnostics. Best-effort."""
    r = _get_redis()
    if r is None:
        return (0, 0)
    try:
        keys = list(r.scan_iter(match=_KEY_PREFIX + "*", count=200))
        return (len(keys), _TTL_SEC)
    except redis.RedisError:
        return (0, 0)
