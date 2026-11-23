"""Local-disk LRU cache for hot lookups.

Two payloads go through this cache:

  1. Director scripts (`backend.director_cache`) — the saved JSON
     response for a (prompt, style, max_scenes) key.
  2. Generated images from upstream APIs (Agnes etc.) — `Path`-keyed
     fetches so a repeat prompt doesn't pull the same URL twice.

Both back-ends (Redis in production, disk in dev/no-Redis deploys)
write to the same code paths. When `ENABLE_REDIS=false` we *still*
get a working cache — just stored in `data/cache/...` instead of
Redis. When Redis comes online later, we use whichever backend is
healthy at the moment of the request (Redis takes priority so the
multi-process scheduler keeps one consistent view).

The on-disk format is intentionally boring:
  data/cache/director/{sha256}.json      — full director JSON
  data/cache/images/{sha256}.{ext}      — original raw bytes

The lookups are O(1) (`mtime`-sorted + capped at 1000 entries) so a
multi-day running host doesn't accidentally grow the directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_ROOT = _PROJECT_ROOT / "data" / "cache"
_DIR_CACHE = _CACHE_ROOT / "director"
_IMG_CACHE = _CACHE_ROOT / "images"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_DIR_CACHE.mkdir(parents=True, exist_ok=True)
_IMG_CACHE.mkdir(parents=True, exist_ok=True)

_MAX_ENTRIES = 1000   # soft cap per cache to keep directory small
_lock = threading.Lock()


def director_key(prompt: str, style: Optional[str], max_scenes: int) -> str:
    raw = f"{prompt.strip()}|{style or ''}|{max_scenes}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def image_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


# ----- Director ----------------------------------------------------------
def get_director(prompt: str, style: Optional[str], max_scenes: int
                 ) -> Optional[dict]:
    """Disk-backed director look-up. Returns the dict on hit, else None."""
    p = _DIR_CACHE / f"{director_key(prompt, style, max_scenes)}.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Touch mtime for LRU eviction accounting.
        _touch(p)
        return data
    except (OSError, ValueError) as exc:
        logger.warning("disk_cache: corrupt director entry %s: %s", p, exc)
        try:
            p.unlink()
        except OSError:
            pass
        return None


def put_director(prompt: str, style: Optional[str], max_scenes: int,
                 response: dict) -> None:
    p = _DIR_CACHE / f"{director_key(prompt, style, max_scenes)}.json"
    try:
        tmp = p.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        tmp.replace(p)
        _evict_dir()
    except OSError as exc:
        logger.warning("disk_cache: director write failed: %s", exc)


# ----- Image URL cache ---------------------------------------------------
def get_image_path(url: str) -> Optional[Path]:
    """Return local Path of cached image bytes (or None on miss)."""
    parsed = _split_url(url)
    p = _IMG_CACHE / f"{image_key(url)}.{parsed}"
    if not p.exists():
        return None
    _touch(p)
    return p


def put_image(url: str, data: bytes) -> Path:
    parsed = _split_url(url)
    p = _IMG_CACHE / f"{image_key(url)}.{parsed}"
    try:
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        _evict_img()
        return p
    except OSError as exc:
        logger.warning("disk_cache: image write failed: %s", exc)
        return p  # may or may not exist; caller decides what to do.


# ----- Helpers -----------------------------------------------------------
def _split_url(url: str) -> str:
    # Use lowercase extension without query string. Examples:
    #   https://.../foo.PNG?x=1 → "png"
    #   /local/path/SCENE_00.png → "png"
    try:
        stem = url.split("?")[0]
    except (AttributeError, IndexError):
        stem = url
    ext = os.path.splitext(stem)[1].lstrip(".").lower() or "bin"
    # Whitelist: only 1-4 char alphabetic extensions.
    if not ext.isalpha() or not (1 <= len(ext) <= 4):
        ext = "bin"
    return ext


def _touch(p: Path) -> None:
    try:
        os.utime(p, None)
    except OSError:
        pass


def _evict_dir() -> None:
    with _lock:
        _evict(_DIR_CACHE)


def _evict_img() -> None:
    with _lock:
        _evict(_IMG_CACHE)


def _evict(directory: Path) -> None:
    try:
        items = list(directory.iterdir())
    except OSError:
        return
    if len(items) <= _MAX_ENTRIES:
        return
    items.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
    # Drop the oldest (max - cap) entries so the next write doesn't have to.
    drop = max(1, len(items) - _MAX_ENTRIES)
    for p in items[:drop]:
        try:
            p.unlink()
        except OSError:
            pass


def stats() -> dict:
    return {
        "director_entries": sum(1 for _ in _DIR_CACHE.iterdir()) if _DIR_CACHE.exists() else 0,
        "image_entries": sum(1 for _ in _IMG_CACHE.iterdir()) if _IMG_CACHE.exists() else 0,
    }
