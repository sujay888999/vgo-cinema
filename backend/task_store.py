"""Task store backed by Redis Hash fields with per-task TTL.

Single key per task (vgo:task:<id>) stored as a Redis Hash.  Only the
fields that actually change are sent to Redis (HSET/HINCRBY).  A small
process-level LRU cache of the most recently touched tasks is kept so
that read-heavy workloads do not hit Redis on every call.

All public functions are thread-safe.

Also publishes change events to a Redis pub/sub channel
(`vgo:task-events:<id>`) so the SSE endpoint can broadcast to live
browsers without requiring the frontend to poll.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import redis

from .config import get_settings
from .schemas import Scene, TaskStatus, VideoTask

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_tasks_lru: OrderedDict[str, VideoTask] = OrderedDict()
_MAX_LRU = 100           # per-process hot cache cap

# Per-task locks — serialise read-modify-write in update_task so
# concurrent scene sub-tasks (Agnes path) don't clobber each other's
# scene updates.  One lock per task_id; cleaned up when no longer needed.
_task_locks: "OrderedDict[str, threading.Lock]" = OrderedDict()
_task_locks_guard = threading.Lock()
_TTL_SEC = 86400 * 2     # 48 h TTL for completed tasks (leaves headroom for logs)

_redis_client: Optional[redis.Redis] = None
REDIS_KEY_PREFIX = "vgo:task:"


def sanitize_error(message: str, max_len: int = 300) -> str:
    """Sanitize an error message before storing it in task.error / warnings.

    Strips sensitive details (file paths, URLs, API keys/tokens, Redis
    connection strings) so internal implementation details are not exposed
    to unauthenticated API consumers. The full unsanitized error is
    always available in server logs via ``logger.exception``.
    """
    import re
    if not message:
        return "unknown error"
    msg = str(message)
    # Strip file paths (Windows + Unix style).
    msg = re.sub(r"[A-Za-z]:\\[^\s]+", "[path]", msg)
    msg = re.sub(r"/[\w./-]+/[\w./-]+", "[path]", msg)
    # Strip URLs.
    msg = re.sub(r"https?://[^\s]+", "[url]", msg)
    # Strip common secret patterns: long hex/base64-ish strings (>20 chars).
    msg = re.sub(r"[A-Za-z0-9+/]{20,}={0,2}", "[token]", msg)
    # Collapse whitespace and truncate.
    msg = " ".join(msg.split())
    return msg[:max_len]
EVENTS_CHANNEL_PREFIX = "vgo:task-events:"


def _get_redis() -> Optional[redis.Redis]:
    """Self-healing Redis singleton.

    Probes on first call. If the probe fails (e.g. WSL cold start with
    2-3 s initial latency), returns None — but a subsequent caller that
    *needs* Redis can call `try_redis_once()` to attempt again. Public
    callers should re-call `_get_redis()` periodically, or themselves
    handle the None return.
    """
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    return _try_connect()


def _try_connect() -> Optional[redis.Redis]:
    """Lazy-singleton Redis client. Short timeouts are key: any single Redis
    operation should fail fast (≤ 1 s) so HTTP requests through this
    backend don't hang when the WSL2 NAT bridge flaps.

    The singleton is reset on any RedisError via `reset_redis()` so the
    next caller either reconnects or stays offline permanently.

    If `ENABLE_REDIS=false` (the default for local dev / single-
    container deploys), short-circuit and return None — no connection
    attempt, no warning spam in the log.
    """
    global _redis_client
    with _lock:
        if _redis_client is not None:
            return _redis_client
        if not get_settings().enable_redis:
            return None
        try:
            client = redis.Redis.from_url(
                get_settings().redis_url,
                # Aggressive deadlines — Redis is supposed to be LAN-local
                # (172.27.x.x inside WSL2); if a read takes >1 s something
                # is wrong and the request should fall through the LRU
                # cache instead of blocking the API.
                socket_timeout=1,
                socket_connect_timeout=2,
                socket_keepalive=True,
                health_check_interval=15,
                retry_on_timeout=False,   # we own the retry decision
                retry_on_error=[],
                decode_responses=True,
            )
            client.ping()
            _redis_client = client
            logger.info("task_store: connected to Redis")
        except (redis.RedisError, OSError, TimeoutError) as exc:
            _redis_client = None
            logger.warning("task_store: Redis unavailable (%s), using in-memory only", exc)
        return _redis_client


def reset_redis() -> None:
    """Discard the singleton so the next `_get_redis()` probes again.

    Used by the main.py middleware to recover after transient Redis
    hiccups (e.g. WSL2 NAT flap). Cheap; safe to call any time.
    """
    global _redis_client
    with _lock:
        if _redis_client is not None:
            try:
                _redis_client.close()
            except Exception:  # noqa: BLE001
                pass
            _redis_client = None


def _task_redis_key(task_id: str) -> str:
    return REDIS_KEY_PREFIX + task_id


def _events_channel(task_id: str) -> str:
    return EVENTS_CHANNEL_PREFIX + task_id


def publish_event(task_id: str, payload: Dict) -> None:
    """Best-effort publish of a small event dict on the per-task channel.

    Failures (Redis down, type error) are logged at warning level — the
    task store must remain available offline. The SSE endpoint catches up
    via a follow-up `get_task` if it misses an event.
    """
    r = _get_redis()
    if r is None:
        return
    try:
        r.publish(_events_channel(task_id), json.dumps(payload, default=str))
    except (redis.RedisError, OSError) as exc:
        logger.warning("task_store: publish_event(%s) failed: %s", task_id, exc)


def subscribe_events(task_id: str):
    """Generator-style helper returning a `pubsub` subscribed to one task.

    Returns None when Redis is unreachable so the SSE endpoint can fall
    back to plain polling. Caller is responsible for unsubscribing
    (`pubsub.close()`).
    """
    r = _get_redis()
    if r is None:
        return None
    try:
        pubsub = r.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(_events_channel(task_id))
        return pubsub
    except (redis.RedisError, OSError) as exc:
        # Redis went away between _get_redis() and .subscribe(). Reset
        # the singleton so the next call probes again.
        logger.warning("task_store: subscribe_events(%s) failed: %s", task_id, exc)
        reset_redis()
        return None

# ---------------------------------------------------------------------------
# Redis Hash helpers
# ---------------------------------------------------------------------------


# --- fields that need JSON special handling ---
_JSON_FIELDS = {"scenes", "storyboard", "video_urls", "cost_breakdown",
                "stage_durations_sec", "scene_provenance", "warnings"}


def _model_to_fields(obj: VideoTask) -> Dict[str, str]:
    """Convert a full VideoTask to flat string values for Redis Hash."""
    base = {
        "id": obj.id,
        "prompt": obj.prompt,
        "status": obj.status.value,
        "script": obj.script or "",
        "storyboard": json.dumps(obj.storyboard),
        "video_urls": json.dumps(obj.video_urls),
        "final_video_url": obj.final_video_url or "",
        "title": obj.title or "",
        "logline": obj.logline or "",
        "duration_sec": str(obj.duration_sec),
        "error": obj.error or "",
        "created_at": obj.created_at or "",
        "style": obj.style or "",
        "video_mode": obj.video_mode or "auto",
        "estimated_cost_usd": str(obj.estimated_cost_usd),
        "cost_breakdown": json.dumps(obj.cost_breakdown),
        "scenes": json.dumps([s.model_dump() for s in obj.scenes]),
        "stage_durations_sec": json.dumps(obj.stage_durations_sec),
        "current_stage_started_at": obj.current_stage_started_at or "",
        "scene_provenance": json.dumps({str(k): v for k, v in (obj.scene_provenance or {}).items()}),
        "render_quality": obj.render_quality or "",
        "warnings": json.dumps(list(obj.warnings or [])),
        "retry_pass_count": str(obj.retry_pass_count or 0),
    }
    return base


def _fields_to_dict(fields: Dict[str, str]) -> Dict:
    """Redis Hash flat dict -> Python dict for VideoTask."""
    out = dict(fields)
    out["status"] = out.get("status", TaskStatus.PENDING)
    out["duration_sec"] = float(out.get("duration_sec", "0"))
    out["estimated_cost_usd"] = float(out.get("estimated_cost_usd", "0"))
    out["storyboard"] = json.loads(out.get("storyboard", "[]"))
    out["video_urls"] = json.loads(out.get("video_urls", "[]"))
    out["cost_breakdown"] = json.loads(out.get("cost_breakdown", "{}"))
    out["scenes"] = [Scene.model_validate(s) for s in json.loads(out.get("scenes", "[]"))]
    sd_raw = out.get("stage_durations_sec")
    if sd_raw and sd_raw.strip():
        try:
            out["stage_durations_sec"] = json.loads(sd_raw)
        except (ValueError, TypeError):
            out["stage_durations_sec"] = {}
    else:
        out["stage_durations_sec"] = {}
    out["current_stage_started_at"] = out.get("current_stage_started_at") or None
    sp_raw = out.get("scene_provenance")
    if sp_raw and sp_raw.strip():
        try:
            parsed = json.loads(sp_raw)
            out["scene_provenance"] = {int(k): v for k, v in parsed.items()}
        except (ValueError, TypeError):
            out["scene_provenance"] = {}
    else:
        out["scene_provenance"] = {}
    out["render_quality"] = out.get("render_quality") or "ok"
    out["warnings"] = json.loads(out.get("warnings") or "[]")
    try:
        out["retry_pass_count"] = int(out.get("retry_pass_count") or 0)
    except (TypeError, ValueError):
        out["retry_pass_count"] = 0
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_task(prompt: str, *, style: Optional[str] = None,
                video_mode: str = "auto") -> VideoTask:
    import datetime as _dt
    task_id = str(uuid.uuid4())
    task = VideoTask(
        id=task_id,
        prompt=prompt,
        status=TaskStatus.PENDING,
        storyboard=[],
        scenes=[],
        video_urls=[],
        created_at=_dt.datetime.utcnow().isoformat() + "Z",
        style=style,
        video_mode=video_mode,
    )
    # --- write to Redis (self-healing; never raises). We MUST not block
    # the API path on Redis — the WSL2 NAT bridge behind the host stack
    # flaps every few seconds. Each Redis op is wrapped with a hard
    # timeout and any hiccup routes through the in-process LRU cache
    # rather than hanging the request.
    r = _get_redis()
    if r is not None:
        try:
            key = _task_redis_key(task_id)
            with _lock:
                # ``r.pipeline().execute()`` shares the underlying socket's
                # timeout settings; FAILED HSETs bubble up as
                # ``redis.RedisError`` and we fall back. The hard cap on
                # pipe.execute() is the socket_timeout on the client.
                pipe = r.pipeline()
                pipe.hset(key, mapping=_model_to_fields(task))
                pipe.expire(key, _TTL_SEC)
                pipe.execute()
        except (redis.RedisError, OSError, TimeoutError) as exc:
            logger.warning("create_task: Redis HSET failed (%s); falling back to LRU", exc)
            reset_redis()
    # --- LRU cache ---
    with _lock:
        _tasks_lru[task_id] = task
        if len(_tasks_lru) > _MAX_LRU:
            _tasks_lru.popitem(last=False)
    return task


def get_task(task_id: str) -> Optional[VideoTask]:
    # --- fast path: local LRU ---
    with _lock:
        if task_id in _tasks_lru:
            # move to newest
            task = _tasks_lru.pop(task_id)
            _tasks_lru[task_id] = task
            return task
    # --- fallback: Redis ---
    r = _get_redis()
    if r is not None:
        fields = r.hgetall(_task_redis_key(task_id))
        if fields:
            try:
                task = VideoTask.model_validate(_fields_to_dict(fields))
                with _lock:
                    _tasks_lru[task_id] = task
                return task
            except Exception as exc:
                logger.warning("task_store: failed to decode task %s: %s", task_id, exc)
    return None


def _get_task_live(task_id: str) -> Optional[VideoTask]:
    """Read directly from Redis, bypassing the LRU cache.

    Used by the SSE endpoint so a late subscriber joining mid-task gets
    the *current* state rather than the stale `pending` snapshot the
    cache might hold from create-time.
    """
    r = _get_redis()
    if r is None:
        return get_task(task_id)
    fields = r.hgetall(_task_redis_key(task_id))
    if not fields:
        return None
    try:
        return VideoTask.model_validate(_fields_to_dict(fields))
    except Exception as exc:
        logger.warning("task_store: failed to live-decode task %s: %s", task_id, exc)
        return None


def list_tasks() -> List[VideoTask]:
    r = _get_redis()
    if r is not None:
        keys: List[str] = []
        try:
            keys = list(r.scan_iter(match=REDIS_KEY_PREFIX + "*", count=200))
        except redis.RedisError as exc:
            logger.warning("task_store: redis SCAN failed: %s", exc)
        tasks: List[VideoTask] = []
        for key in keys:
            try:
                fields = r.hgetall(key)
                if fields:
                    tasks.append(VideoTask.model_validate(_fields_to_dict(fields)))
            except Exception as exc:
                logger.warning("task_store: failed to decode %s: %s", key, exc)
        return tasks
    # fallback: in-memory only
    with _lock:
        return list(_tasks_lru.values())



def _get_task_lock(task_id: str) -> threading.Lock:
    """Return (or lazily create) the per-task update lock.

    Serialises read-modify-write in `update_task` so concurrent
    scene sub-tasks don't race on the full `scenes` list.
    """
    with _task_locks_guard:
        lk = _task_locks.get(task_id)
        if lk is None:
            lk = threading.Lock()
            _task_locks[task_id] = lk
        return lk


def update_task(
    task_id: str,
    status: Optional[TaskStatus] = None,
    final_video_url: Optional[str] = None,
    video_urls: Optional[List[str]] = None,
    storyboard: Optional[List[str]] = None,
    error: Optional[str] = None,
    script: Optional[str] = None,
    title: Optional[str] = None,
    logline: Optional[str] = None,
    scenes: Optional[List[Scene]] = None,
    duration_sec: Optional[float] = None,
    estimated_cost_usd: Optional[float] = None,
    cost_breakdown: Optional[dict] = None,
    style: Optional[str] = None,
    scene_update: Optional[dict] = None,
    storyboard_idx: Optional[int] = None,
    video_urls_idx: Optional[int] = None,
    stage_durations_sec: Optional[dict] = None,
    current_stage_started_at: Optional[str] = None,
    scene_provenance: Optional[Dict[int, str]] = None,
    render_quality: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    retry_pass_count: Optional[int] = None,
) -> Optional[VideoTask]:
    """Incrementally update a task.

    Only fields that were explicitly passed (not None) are written — this
    is the *whole* point of using HSET on a Redis Hash instead of SET on
    a JSON blob.  Returns the freshly-merged VideoTask.

    Concurrent scene sub-tasks use `scene_update={"scene": <1-indexed>,
    "image_url": "...", "video_url": "..."}` to patch a single scene in
    place without racing each other on the full `scenes` list.
    """
    # ---- 1. Refresh local snapshot from Redis (so concurrent writer is
    #         visible to us).  Fall back to LRU when Redis is down. ----------
    # Acquire per-task lock to serialise the read-modify-write below
    # (concurrent scene sub-tasks must not clobber each other's updates).
    _tlock = _get_task_lock(task_id)
    _tlock.acquire()
    try:
        task = get_task(task_id)
        if task is None:
            return None

        updates: Dict[str, str] = {}

        if status is not None:
            # If the status is changing, log the previous stage's elapsed
            # time into stage_durations_sec so the SSE endpoint can compute
            # a rolling ETA for the *next* stage.
            prev_status = task.status
            if prev_status != status:
                import datetime as _dt
                now_iso = _dt.datetime.utcnow().isoformat() + "Z"
                prev_started = task.current_stage_started_at
                if prev_started and prev_status != TaskStatus.PENDING:
                    try:
                        elapsed = (
                            _dt.datetime.fromisoformat(prev_started.replace("Z", "+00:00"))
                            - _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
                        ).total_seconds()
                        # elapsed is negative because prev_started is in the past;
                        # the magnitude is what we want.
                        task.stage_durations_sec[prev_status.value] = round(-elapsed, 3)
                        updates["stage_durations_sec"] = json.dumps(task.stage_durations_sec)
                    except (ValueError, TypeError):
                        pass
                task.current_stage_started_at = now_iso
                updates["current_stage_started_at"] = now_iso
            task.status = status
            updates["status"] = status.value
        if stage_durations_sec is not None:
            task.stage_durations_sec.update(stage_durations_sec)
            updates["stage_durations_sec"] = json.dumps(task.stage_durations_sec)
        if current_stage_started_at is not None:
            task.current_stage_started_at = current_stage_started_at
            updates["current_stage_started_at"] = current_stage_started_at
        if final_video_url is not None:
            task.final_video_url = final_video_url
            updates["final_video_url"] = final_video_url
        if video_urls is not None:
            task.video_urls = video_urls
            updates["video_urls"] = json.dumps(video_urls)
        if storyboard is not None:
            task.storyboard = storyboard
            updates["storyboard"] = json.dumps(storyboard)
        if script is not None:
            task.script = script
            updates["script"] = script
        if title is not None:
            task.title = title
            updates["title"] = title
        if logline is not None:
            task.logline = logline
            updates["logline"] = logline
        if duration_sec is not None:
            task.duration_sec = duration_sec
            updates["duration_sec"] = str(duration_sec)
        if scenes is not None:
            task.scenes = scenes
            updates["scenes"] = json.dumps([s.model_dump() for s in scenes])
        elif scene_update is not None:
            # Patch one scene in place. The scene dict keys we accept are the
            # Scene-schema fields (without `scene` itself used as 1-indexed match).
            # Work on a copy so we never mutate the caller's dict.
            scene_update_copy = dict(scene_update)
            scene_num = scene_update_copy.pop("scene", None)
            if scene_num is not None:
                # scene_number is 1-indexed (matches the Scene model).
                target_idx = scene_num - 1
                while len(task.scenes) <= target_idx:
                    # Growable list so even race-with-scripts ordering is fine.
                    task.scenes.append(Scene(scene=len(task.scenes) + 1,
                                              title="", description=""))
                scene_obj = task.scenes[target_idx]
                for k, v in scene_update_copy.items():
                    if v is not None and hasattr(scene_obj, k):
                        setattr(scene_obj, k, v)
                updates["scenes"] = json.dumps([s.model_dump() for s in task.scenes])
            # (No need to restore the original dict — we used a copy.)

        # Concurrent-safe appends for `storyboard` (image URLs) and
        # `video_urls` (clip URLs). storyboard_idx / video_urls_idx are
        # 0-indexed positions to overwrite, not append.
        if storyboard_idx is not None and storyboard is None:
            path = scene_update.get("image_url") if scene_update else None
            if path is not None:
                while len(task.storyboard) <= storyboard_idx:
                    task.storyboard.append("")
                task.storyboard[storyboard_idx] = path
                updates["storyboard"] = json.dumps(task.storyboard)
        if video_urls_idx is not None and video_urls is None:
            path = scene_update.get("video_url") if scene_update else None
            if path is not None:
                while len(task.video_urls) <= video_urls_idx:
                    task.video_urls.append("")
                task.video_urls[video_urls_idx] = path
                updates["video_urls"] = json.dumps(task.video_urls)
        if estimated_cost_usd is not None:
            task.estimated_cost_usd = estimated_cost_usd
            updates["estimated_cost_usd"] = str(estimated_cost_usd)
        if cost_breakdown is not None:
            task.cost_breakdown = cost_breakdown
            updates["cost_breakdown"] = json.dumps(cost_breakdown)
        if style is not None:
            task.style = style
            updates["style"] = style
        if scene_provenance is not None:
            # Merge per-scene provenance so concurrent writers (one per
            # scene sub-task) don't clobber each other's keys. Keys are
            # JSON-stringified ints.
            merged = dict(task.scene_provenance or {})
            for k, v in scene_provenance.items():
                merged[int(k)] = v
            task.scene_provenance = merged
            updates["scene_provenance"] = json.dumps({str(k): v for k, v in merged.items()})
        if render_quality is not None:
            task.render_quality = render_quality
            updates["render_quality"] = render_quality
        if warnings is not None:
            # Append — never overwrite. The UI surfaces the union of all
            # warnings produced over the task lifetime.
            merged = list(task.warnings or [])
            for w in warnings:
                if w and w not in merged:
                    merged.append(w)
            task.warnings = merged
            updates["warnings"] = json.dumps(merged)
        if retry_pass_count is not None:
            task.retry_pass_count = max(task.retry_pass_count or 0, int(retry_pass_count))
            updates["retry_pass_count"] = str(task.retry_pass_count)
        if error is not None:
            task.error = error
            updates["error"] = error
        elif status == TaskStatus.FAILED and task.error is None:
            task.error = "unknown error"
            updates["error"] = "unknown error"

        # ---- 2. Push only the changed fields to Redis --------------------------
        if updates:
            r = _get_redis()
            if r is not None:
                key = _task_redis_key(task_id)
                try:
                    pipe = r.pipeline()
                    pipe.hset(key, mapping=updates)
                    # Bump TTL on every write so active tasks stay alive.
                    pipe.expire(key, _TTL_SEC)
                    pipe.execute()
                except (redis.RedisError, OSError) as exc:
                    logger.warning("task_store: redis HSET failed: %s", exc)
                    reset_redis()

        # ---- 3. Local cache: refresh + LRU eviction ----------------------------
        with _lock:
            _tasks_lru[task_id] = task
            if len(_tasks_lru) > _MAX_LRU:
                _tasks_lru.popitem(last=False)

        # ---- 4. Best-effort: notify SSE subscribers ---------------------------
        # We only forward changed scalar fields; the frontend merges into its
        # already-cached copy so the payload stays tiny (~200 bytes per emit).
        if updates:
            emit: Dict = {"type": "update", "ts": time.time()}
            # Always include status (the frontend timeline runs the show).
            emit["status"] = task.status.value if hasattr(task.status, "value") else str(task.status)
            # Stable, cheap-to-render fields the UI cares about:
            for k in ("final_video_url", "title", "logline", "duration_sec",
                      "estimated_cost_usd", "script"):
                v = getattr(task, k, None)
                if v is not None:
                    emit[k] = v
            # Lists: only forward the count so the SSE stream stays small.
            # The frontend re-fetches the full task via getTask(id) when it
            # needs the actual array contents.
            emit["scenes_count"] = len(task.scenes)
            emit["storyboard_count"] = len(task.storyboard or [])
            emit["video_urls_count"] = len(task.video_urls or [])
            # ETA in seconds: historical average per remaining stage minus
            # what's been spent in the current stage. First run uses a coarse
            # default so the UI isn't blank; subsequent runs of similar
            # prompts get tighter estimates as the dict grows.
            status_value = emit["status"]
            avg = task.stage_durations_sec or {}
            if status_value == "pending":
                eta = avg.get("scripting", 30.0) + avg.get("image_gen", 90.0) + avg.get("video_gen", 180.0) + avg.get("composing", 25.0)
            elif status_value in ("completed", "failed"):
                eta = 0.0
            else:
                remaining_lookup = {
                    "scripting": 0,
                    "image_gen": avg.get("image_gen", 90.0),
                    "video_gen": avg.get("video_gen", 180.0),
                    "composing": avg.get("scripting", 25.0),
                }
                remaining = remaining_lookup.get(status_value, 60.0)
                spent_here = 0.0
                if task.current_stage_started_at:
                    try:
                        import datetime as _dt
                        started = _dt.datetime.fromisoformat(
                            task.current_stage_started_at.replace("Z", "+00:00")
                        )
                        spent_here = (_dt.datetime.now(_dt.timezone.utc) - started).total_seconds()
                    except (ValueError, TypeError):
                        spent_here = 0.0
                eta = max(0.0, remaining - spent_here)
            emit["eta_seconds"] = round(eta, 1)
            emit["stage_durations_sec"] = dict(task.stage_durations_sec)
            # Per-scene readiness flags so the UI can animate each scene
            # chip independently as the parent chord fans out 4 sub-tasks.
            # Cheap to compute (~16 bytes per scene) and idempotent if a
            # scene's `image_url` etc. haven't changed since the prior frame.
            emit["scenes_state"] = [
                {
                    "scene": s.scene,
                    "image_ready": bool(s.image_url),
                    "video_ready": bool(s.video_url),
                    "audio_ready": bool(s.audio_path),
                }
                for s in task.scenes
            ]
            publish_event(task_id, emit)

            return task
    finally:
        _tlock.release()
        # Cleanup: when a task reaches a terminal state, remove its lock
        # from the OrderedDict so long-running servers don't accumulate
        # locks indefinitely. The check is safe because terminal status
        # means no further writes are expected (retries create a fresh
        # lock via _get_task_lock).
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            with _task_locks_guard:
                if _task_locks.get(task_id) is _tlock:
                    _task_locks.pop(task_id, None)


def append_warning(task_id: str, message: str) -> Optional[VideoTask]:
    """Append a warning to a task without touching other fields.

    Uses the same per-task lock as `update_task` so concurrent scene
    sub-tasks don't race on the warnings list.
    """
    _tlock = _get_task_lock(task_id)
    _tlock.acquire()
    try:
        task = get_task(task_id)
        if task is None:
            return None
        merged = list(task.warnings or [])
        if message and message not in merged:
            merged.append(message)
        task.warnings = merged
        # Push to Redis if available
        r = _get_redis()
        if r is not None:
            try:
                r.hset(_task_redis_key(task_id), mapping={"warnings": json.dumps(merged)})
                r.expire(_task_redis_key(task_id), _TTL_SEC)
            except (redis.RedisError, OSError) as exc:
                logger.warning("task_store: redis HSET warnings failed: %s", exc)
                reset_redis()
        # Update local cache
        with _lock:
            _tasks_lru[task_id] = task
        # Emit SSE update
        emit = {"type": "update", "ts": time.time(), "status": task.status.value,
                "warnings_count": len(merged)}
        publish_event(task_id, emit)
        return task
    finally:
        _tlock.release()
