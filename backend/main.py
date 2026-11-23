import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator, Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from . import task_store
from .config import get_settings
from .schemas import TaskStatus, VideoTask

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# H4/H5 startup validation: warn (never fail) when a configured provider's
# key is missing or a placeholder, and when the output-signing secret is unset.
for _startup_warning in settings.provider_key_warnings():
    logger.warning(_startup_warning)
logger.info(
    "video chain: provider=%s primary=%s fallback=%s",
    settings.provider,
    settings.video_provider_primary,
    settings.video_provider_fallback,
)

app = FastAPI(title=settings.app_name, version=settings.app_version)


# ---- In-process task pool ---------------------------------------------------
# Bound to the FastAPI process. Concurrency=2 keeps
# aggregate upstream pressure on Agnes / Magpie / FLUX moderate; the
# _run_with_failure_capture wrapper means a single task's crash never
# takes down the executor, and the workflow's own update_task(FAILED)
# keeps the user-visible state correct.
local_pool = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="vgo-worker",
)


def _run_with_failure_capture(task_id: str, prompt: str, style: Optional[str],
                              video_mode: str = "auto") -> None:
    """Top-level worker entry widged for the executor.

    `run_workflow` itself manages retries — by the time the executor
    sees the exception here, retries were exhausted. We just re-mark
    FAILED defensively in case the inner code swallowed.
    """
    from .workflow import run_workflow
    try:
        run_workflow(task_id, prompt, style, video_mode=video_mode)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Workflow top-level crashed for task %s", task_id)
        try:
            task_store.update_task(
                task_id, TaskStatus.FAILED,
                error=task_store.sanitize_error(f"top-level crash: {exc}"),
            )
        except Exception:  # noqa: BLE001
            pass

# CORS: explicit allow-list for prod, regex for dev (covers 3000~3002 / 9000 / 8080).
# CORS: explicit allow-list for prod, regex for dev (covers both
# `localhost:*` and `127.0.0.1:*` plus the previously-allowed ports).
# CORS: explicit allow-list for both prod (via env) and dev. The dev
# regex previously in use had a subtle issue where Starlette refused to
# match `127.0.0.1` even though the regex itself does — using an exact
# list avoids that whole class of issue, and is narrower than the regex
# was meant to be anyway.
DEV_ALLOWED_ORIGINS = (
    f"http://localhost:3000,http://127.0.0.1:3000,"
    f"http://localhost:3001,http://127.0.0.1:3001,"
    f"http://localhost:3002,http://127.0.0.1:3002,"
    f"http://localhost:9000,http://127.0.0.1:9000"
)
_explicit_origins = settings.cors_origins_list() or DEV_ALLOWED_ORIGINS.split(",")
cors_kwargs = {"allow_origins": [o for o in _explicit_origins if o]}

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Api-Key"],
    expose_headers=["Content-Disposition"],
    max_age=600,
    **cors_kwargs,
)


@app.middleware("http")
async def explicit_options_handler(request: Request, call_next):
    """FastAPI's router returns 400 for OPTIONS requests that don't
    match any registered handler. Starlette's CORSMiddleware is supposed
    to short-circuit preflight BEFORE FastAPI's router sees it, but in
    some edge cases (length of `Access-Control-Request-Headers`,
    trailing whitespace, etc.) it falls through and we get a 400 with
    no `Access-Control-Allow-Origin` header. That's the worst case for
    browsers: their preflight fails, so they never send the POST.

    We short-circuit **all** OPTIONS requests that aren't handled by
    the route stack with a 204 + the same CORS headers the middleware
    would emit, sidestepping the bug entirely.
    """
    if request.method == "OPTIONS":
        origin = request.headers.get("origin") or "http://localhost:3000"
        allowed = ("localhost", "127.0.0.1")
        host_port = origin.split("//", 1)[-1].split(":")
        host = host_port[0]
        try:
            port = int(host_port[1])
        except (IndexError, ValueError):
            port = 0
        dev_allowed = {"localhost", "127.0.0.1"}
        if host in dev_allowed and port in (3000, 3001, 3002, 9000):
            return JSONResponse(
                status_code=204,
                content="",
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
                    "Access-Control-Allow-Headers": request.headers.get(
                        "access-control-request-headers", "content-type, x-api-key"
                    ),
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Max-Age": "600",
                },
            )
    return await call_next(request)


# ---- Auth + per-token rate limit ------------------------------------------
# Two concerns here, deliberately:
#
#   1) Authorisation: when `AUTH_REQUIRED=true`, every request to a
#      protected endpoint must carry `X-Api-Key: <token>`. Tokens come
#      from `API_KEYS` (`token:label:limit_per_window` triples).
#   2) Rate limiting: the bucket key is the **token** when present, else
#      the IP. The limit is the per-token `limit_per_window` (default
#      `RATE_LIMIT_DEFAULT`). When auth is off, the IP is used and the
#      default limit applies.
#
# Both endpoints go through Redis ZSET sliding windows for clustered
# correctness; on Redis failure we fall back to in-process deques so the
# endpoint never silently stops working (single-replica correctness only).
RATE_LIMIT_REDIS_KEY = "vgo:rl"
_generate_log: Dict[str, Deque[float]] = defaultdict(deque)
_rl_lock = threading.Lock()

# SSE connection tracking — limits concurrent EventSource streams per
# bucket (IP or API key) so a malicious client cannot exhaust resources
# by opening many long-lived SSE connections.
_SSE_CONN_LIMIT = 5
_sse_conn_count: Dict[str, int] = defaultdict(int)
_sse_lock = threading.Lock()


def _bucket_for_request(request: Request) -> tuple:
    """Resolve (bucket_id, limit) from header + IP for the request.

    `bucket_id` is the per-token string when the X-Api-Key header is
    present and valid; otherwise the IP. `limit` is the configured int.
    """
    ip = request.client.host if request.client else "unknown"
    token = request.headers.get("X-Api-Key", "").strip()
    api_keys = settings.api_keys()
    if token and token in api_keys:
        return f"key:{token}", api_keys[token]["limit"]
    return f"ip:{ip}", settings.rate_limit_default_per_window


def _redis_rate_limit(bucket_id: str, now: float, limit: int) -> Optional[bool]:
    """Sliding-window rate-limit over a Redis ZSET.

    Returns True if the limit was hit (reject), False if it should pass,
    None on any Redis error (caller falls back to in-process).
    """
    from .task_store import _get_redis
    r = _get_redis()
    if r is None:
        return None  # type: ignore[return-value]
    key = f"{RATE_LIMIT_REDIS_KEY}:{bucket_id}"
    cutoff = now - settings.rate_limit_window_sec
    member = f"{now:.6f}"
    try:
        pipe = r.pipeline()
        pipe.zremrangebyscore(key, 0, cutoff)
        pipe.zcard(key)
        pipe.zadd(key, {member: now})
        pipe.expire(key, settings.rate_limit_window_sec + 5)
        _, count, _, _ = pipe.execute()
        return count >= limit
    except Exception:  # noqa: BLE001
        return None  # type: ignore[return-value]


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method

    # ---- Write endpoints: generate + retry-scene ----
    if (path == "/api/generate"
            or path.endswith("/retry-scene")) and method == "POST":
        bucket_id, limit = _bucket_for_request(request)
        token = request.headers.get("X-Api-Key", "").strip()
        if settings.auth_required and not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "X-Api-Key header required (set AUTH_REQUIRED=false to disable)"},
            )
        if token and not settings.auth_required:
            # Auth is opt-in: keys may be sent but we do not require
            # them. We *recognise* known tokens and bucket by them so a
            # well-behaved client gets its own stream + its own limit.
            pass
        now = time.time()
        verdict = _redis_rate_limit(bucket_id, now, limit)
        if verdict is None:
            with _rl_lock:
                bucket = _generate_log[bucket_id]
                cutoff = now - settings.rate_limit_window_sec
                while bucket and bucket[0] < cutoff:
                    bucket.popleft()
                if len(bucket) >= limit:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": f"rate limit ({bucket_id[:20]}…): max {limit} generations per {settings.rate_limit_window_sec}s"},
                    )
                bucket.append(now)
        elif verdict:
            return JSONResponse(
                status_code=429,
                content={"detail": f"rate limit ({bucket_id[:20]}…): max {limit} generations per {settings.rate_limit_window_sec}s"},
            )

    # ---- Read endpoints: task list / task detail / outputs ----
    # Generous limits (10× the write limit) to prevent enumeration and
    # timing attacks without impacting normal UI polling.
    _READ_LIMIT_MULT = 10
    if method == "GET" and (
        path == "/api/tasks"
        or path.startswith("/api/tasks/")
        or path.startswith("/api/outputs/")
    ):
        # Exclude SSE events endpoint — it has its own connection limit.
        if path.endswith("/events"):
            return await call_next(request)
        bucket_id, limit = _bucket_for_request(request)
        read_limit = limit * _READ_LIMIT_MULT
        now = time.time()
        verdict = _redis_rate_limit(bucket_id + ":r", now, read_limit)
        if verdict is None:
            with _rl_lock:
                bucket = _generate_log[bucket_id + ":r"]
                cutoff = now - settings.rate_limit_window_sec
                while bucket and bucket[0] < cutoff:
                    bucket.popleft()
                if len(bucket) >= read_limit:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": f"rate limit ({bucket_id[:20]}…): max {read_limit} reads per {settings.rate_limit_window_sec}s"},
                    )
                bucket.append(now)
        elif verdict:
            return JSONResponse(
                status_code=429,
                content={"detail": f"rate limit ({bucket_id[:20]}…): max {read_limit} reads per {settings.rate_limit_window_sec}s"},
            )

    return await call_next(request)


# ---- Global authentication middleware -------------------------------------
# Protects all /api/* endpoints (except public ones) when AUTH_REQUIRED=true.
# The X-Api-Key header is checked against the configured API_KEYS; for SSE
# streams (where EventSource cannot set custom headers) the key may also be
# supplied as a ?key= query parameter.
# H1: /api/health now requires an API key too; only /api/styles stays public.
_PUBLIC_API_PATHS = {"/api/styles"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Enforce API-key authentication on all /api/* endpoints.

    Only /api/styles stays public. /api/outputs/* accepts a time-limited
    HMAC signature (?sig=&exp=) or a valid API key — the Next.js proxy
    adds the key server-side, so browser <video>/<img> tags keep working.
    SSE endpoints accept the key via ?key= query param since EventSource
    cannot set headers either.
    """
    path = request.url.path
    # Non-API paths pass through (e.g. root, docs).
    if not path.startswith("/api/"):
        return await call_next(request)
    # Public endpoints (H1: /api/health now requires a key too).
    if path in _PUBLIC_API_PATHS:
        return await call_next(request)
    # H5: outputs accept a time-limited HMAC signature (?sig=&exp=) or,
    # equivalently, a valid API key (the Next.js proxy adds it server-side).
    if path.startswith("/api/outputs/"):
        filename = unquote(path[len("/api/outputs/"):])
        sig = request.query_params.get("sig")
        exp = request.query_params.get("exp")
        if sig and exp and _verify_output_signature(filename, sig, exp):
            return await call_next(request)
        # No (valid) signature: fall through to the API-key check below.
    # OPTIONS preflight — handled by CORS middleware, no auth needed.
    if request.method == "OPTIONS":
        return await call_next(request)
    # If auth is disabled, pass through.
    if not settings.auth_required:
        return await call_next(request)
    # Check header first, then query param (for EventSource / SSE).
    token = request.headers.get("X-Api-Key", "").strip()
    if not token:
        token = request.query_params.get("key", "").strip()
    if not token or token not in settings.api_keys():
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required: provide a valid X-Api-Key header or key query parameter"},
        )
    return await call_next(request)


@app.middleware("http")
async def sse_limit_middleware(request: Request, call_next):
    """Limit concurrent SSE event-stream connections per bucket.

    Each SSE stream is a long-lived connection that holds a worker
    thread on the server. Without a cap, a client could open many
    streams and exhaust the thread pool. We allow at most
    `_SSE_CONN_LIMIT` concurrent connections per bucket (IP or key).
    """
    if request.url.path.endswith("/events") and request.method == "GET":
        bucket_id, _ = _bucket_for_request(request)
        with _sse_lock:
            if _sse_conn_count[bucket_id] >= _SSE_CONN_LIMIT:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"too many SSE connections (max {_SSE_CONN_LIMIT} "
                                  f"per {bucket_id[:20]}…); close idle streams and retry"
                    },
                )
            _sse_conn_count[bucket_id] += 1
        try:
            return await call_next(request)
        finally:
            with _sse_lock:
                _sse_conn_count[bucket_id] = max(0, _sse_conn_count[bucket_id] - 1)
    return await call_next(request)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    # "cinematic" (default), "news", or "documentary". Synonyms like
    # "photojournalism" or "trailer" are accepted and normalised inside
    # the workflow module.
    style: Optional[str] = Field(default=None, max_length=64)
    # Video generation mode for this task:
    #   "auto"        (default) — image-to-video when an image is
    #                   available, text-to-video otherwise.
    #   "image2video" — always pass the generated image as image_url.
    #   "text2video"  — always omit image_url; prompt only.
    # Falls back to the server-side VIDEO_MODE config when omitted.
    video_mode: Optional[str] = Field(default=None, max_length=32)

    @field_validator("prompt")
    @classmethod
    def check_prompt(cls, v: str) -> str:
        """Basic content filter — reject prompts containing blocked terms.

        This is a lightweight, configurable denylist; for production
        deployments, consider an LLM-based moderation endpoint instead.
        """
        from .config import get_settings
        blocked = [
            w.strip().lower()
            for w in (get_settings().blocked_prompt_words or "").split(",")
            if w.strip()
        ]
        lower = v.lower()
        for word in blocked:
            if word and word in lower:
                raise ValueError(
                    f"Prompt contains disallowed content: '{word}'"
                )
        return v


class GenerateResponse(BaseModel):
    task_id: str
    status: TaskStatus
    style: Optional[str] = None
    video_mode: str = "auto"


@app.get("/api/styles")
def list_styles():
    """Expose the available style presets so the frontend can build a
    selector and so external clients can discover them."""
    from .workflow import STYLE_PRESETS
    return {
        "presets": [
            {
                "key": k,
                "label": v["label"],
                "description": v["description"],
                "aspect": f"{v['aspect_w']}x{v['aspect_h']}",
            }
            for k, v in STYLE_PRESETS.items()
        ]
    }


@app.get("/api/health")
def health():
    # Best-effort Redis status — never raise, never 500. The fields are
    # advisory; clients should keep working when Redis flaps (the SSE
    # endpoint falls back to in-process polling; the /api/generate
    # endpoint falls back to in-process worker thread).
    redis_ok = False
    try:
        from .task_store import _get_redis, reset_redis
        r = _get_redis()
        if r is not None:
            try:
                redis_ok = bool(r.ping())
            except Exception:
                reset_redis()
        # Cheap second probe to populate the field for the next caller
        r = _get_redis()
        if r is not None and not redis_ok:
            try:
                redis_ok = bool(r.ping())
            except Exception:
                redis_ok = False
    except Exception:  # noqa: BLE001
        redis_ok = False
    return {
        "status": "ok",
        "service": "vgo-cinema",
        "version": app.version,
        "redis": redis_ok,
    }


@app.get("/api/me")
def me(request: Request):
    """Return the request identity + rate-limit budget when one applies.

    Returns `{"identity": "key:..."| "ip:...", "limit": int}` so the
    frontend can show "you have N requests left" before triggering a
    generation. With no auth header set, the IP-bucket is returned.
    """
    bucket_id, limit = _bucket_for_request(request)
    return {"identity": bucket_id, "limit": limit, "window_sec": settings.rate_limit_window_sec}


@app.post("/api/generate", response_model=GenerateResponse, status_code=202)
def generate(req: GenerateRequest):
    # Lazy import keeps startup cold-path small
    from .workflow import _resolve_style, run_workflow
    from .config import get_settings
    resolved_style = _resolve_style(req.style)
    # Resolve video_mode: request value wins, otherwise fall back to
    # the server-side VIDEO_MODE config, normalised to one of the
    # three accepted values.
    settings = get_settings()
    resolved_video_mode = (
        req.video_mode.strip().lower()
        if req.video_mode
        else settings.normalized_video_mode()
    )
    if resolved_video_mode not in ("auto", "image2video", "text2video"):
        resolved_video_mode = "auto"
    task = task_store.create_task(req.prompt, style=resolved_style,
                                  video_mode=resolved_video_mode)

    # Submit the workflow to a local background thread pool.
    # Running in-process gives us:
    #   - immediate enqueue (no broker round-trip),
    #   - bounded concurrency (max_workers=2) so we don't drown the
    #     Agnes / Magpie / FLUX APIs with parallel jobs,
    #   - shared retry bookkeeping via `run_workflow`.
    try:
        local_pool.submit(_run_with_failure_capture, task.id, req.prompt,
                          resolved_style, resolved_video_mode)
    except RuntimeError as exc:  # pool shutting down, etc.
        logger.exception("local pool rejected task %s", task.id)
        task_store.update_task(task.id, TaskStatus.FAILED, error=task_store.sanitize_error(f"local pool rejected: {exc}"))
        raise HTTPException(status_code=503, detail="local task queue unavailable")

    return GenerateResponse(task_id=task.id, status=task.status,
                            style=resolved_style,
                            video_mode=resolved_video_mode)


@app.get("/api/tasks/{task_id}", response_model=VideoTask)
def get_task(task_id: str):
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


class RetrySceneRequest(BaseModel):
    # 1-indexed scene numbers (matches the `Scene.scene` field and the
    # frontend `sceneProvenance` keys).
    scenes: List[int] = Field(default_factory=list)


class RetrySceneResponse(BaseModel):
    accepted: bool
    message: str
    recovered: List[int] = Field(default_factory=list)


def _run_retry_scene_safe(task_id: str, scene_indices: List[int]) -> None:
    """Retry wrapper used by the worker pool. Errors are swallowed so a
    single failure doesn't kill the executor — they're surfaced via
    update_task(warnings=...) inside the workflow itself."""
    from .workflow_agnes import retry_video_only_for_scene, AgnesAPIError
    from .task_store import get_task, update_task
    # Use the task's stored video_mode so retries honour the same mode
    # the original generation used.
    _task = get_task(task_id)
    _video_mode = _task.video_mode if _task else "auto"
    try:
        recovered = []
        for idx in scene_indices:
            try:
                res = retry_video_only_for_scene(task_id, idx,
                                                 video_mode=_video_mode)
                if res.get("provenance") == "video":
                    recovered.append(idx + 1)
            except AgnesAPIError as exc:
                update_task(
                    task_id, warnings=[f"Scene {idx + 1} retry failed: {exc}"],
                )
        if recovered:
            update_task(
                task_id, warnings=[f"Manual retry recovered scenes {recovered}."]
            )
        # Re-stitch the final mp4 with any newly recovered scenes so
        # the user can watch a fuller video right away.
        from .video_compose import compose_real_videos
        from .task_store import get_task as _get
        snap = _get(task_id)
        if snap is not None and snap.scenes:
            video_paths = [s.video_url for s in snap.scenes if s.video_url]
            durations = [s.duration_sec for s in snap.scenes]
            vo = [s.audio_path for s in snap.scenes]
            re_mp4 = compose_real_videos(
                video_paths, durations, task_id,
                voiceover_paths=vo,
                title=snap.title or "Untitled",
                logline=snap.logline or "",
            )
            if re_mp4:
                update_task(task_id, final_video_url=re_mp4)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).exception(
            "[%s] manual retry crashed: %s", task_id, exc,
        )
        update_task(
            task_id, warnings=[f"Manual retry crashed: {exc}"],
        )


@app.post("/api/tasks/{task_id}/retry-scene",
          response_model=RetrySceneResponse)
def retry_scene_endpoint(task_id: str, req: RetrySceneRequest):
    """Re-run the *video* stage for specified scenes of a completed
    task. Re-uses cached images, so it's a cheap path that respects
    the per-process 1 req/min rate limit.

    Body: { "scenes": [1, 3, 4] }     (1-indexed scene numbers)
    Frontends call this when the render comes back with `render_quality`
    of "partial" or "fallback_only". Returns immediately with
    `accepted=true` — the actual retry happens in a background thread
    and progress is observable via the existing SSE stream.
    """
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    if not req.scenes:
        raise HTTPException(status_code=422, detail="scenes is required")
    bad = [s for s in req.scenes if s < 1 or s > len(task.scenes)]
    if bad:
        raise HTTPException(status_code=400, detail=f"out-of-range scene(s): {bad}")

    # Strip down to the actual scene_payload indices (0-based) the
    # worker expects.
    scene_indices = [s - 1 for s in req.scenes]

    try:
        local_pool.submit(_run_retry_scene_safe, task_id, scene_indices)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"retry queue unavailable: {exc}")

    task_store.update_task(
        task_id,
        warnings=[
            f"Manual retry queued for scenes {req.scenes} "
            f"(pass #{(task.retry_pass_count or 0) + 1})."
        ],
        retry_pass_count=(task.retry_pass_count or 0) + 1,
    )
    return RetrySceneResponse(
        accepted=True,
        message="Retry queued in background; watch the SSE stream for updates.",
        recovered=[],
    )


@app.get("/api/tasks", response_model=List[VideoTask])
def list_tasks(limit: Optional[int] = 50):
    items = task_store.list_tasks()
    return items[:limit]


@app.get("/api/tasks/{task_id}/events")
async def stream_task_events(task_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events stream for one task.

    Emits one event per `update_task(...)` from anywhere in the backend
    (Redis pub/sub fan-out). Streams heartbeat pings every 15 s so the
    connection is not silently reaped by intermediaries. Clients should
    close gracefully on `done` or `failed` status events.
    """
    if task_store.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail="task not found")

    async def event_stream() -> AsyncIterator[bytes]:
        import redis as _redis
        from redis import exceptions as _rexc
        pubsub = task_store.subscribe_events(task_id)
        # Local-state fallback: when Redis pub/sub is unavail, deliver
        # periodic current-snapshot frames so the client still sees progress.
        if pubsub is None:
            # Redis pub/sub is unavailable — fall back to polling only.
            # Emit a stream of `ready` and `update` frames so the
            # frontend's EventSource still lights up the chips.
            try:
                snap_initial = task_store._get_task_live(task_id)
            except Exception:  # noqa: BLE001
                snap_initial = None
            yield _format_sse({"type": "ready", "task_id": task_id, "ts": time.time()})
            last_status = None
            last_states: list = []
            if snap_initial is not None:
                last_status = snap_initial.status.value if hasattr(snap_initial.status, "value") else str(snap_initial.status)
                last_states = [
                    {"scene": s.scene, "image_ready": bool(s.image_url),
                     "video_ready": bool(s.video_url),
                     "audio_ready": bool(s.audio_path)}
                    for s in snap_initial.scenes
                ]
                yield _format_sse({
                    "type": "snapshot",
                    "status": last_status,
                    "scenes_count": len(snap_initial.scenes),
                    "storyboard_count": len(snap_initial.storyboard or []),
                    "video_urls_count": len(snap_initial.video_urls or []),
                    "scenes_state": last_states,
                    "title": snap_initial.title or "",
                    "final_video_url": snap_initial.final_video_url or "",
                    "ts": time.time(),
                    **_render_provenance_payload(snap_initial),
                })
                if last_status in ("completed", "failed"):
                    return
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    snap = None
                    try: snap = task_store._get_task_live(task_id)
                    except Exception: snap = None
                    if snap is not None:
                        cur = snap.status.value if hasattr(snap.status, "value") else str(snap.status)
                        cur_states = [
                            {"scene": s.scene, "image_ready": bool(s.image_url),
                             "video_ready": bool(s.video_url),
                             "audio_ready": bool(s.audio_path)}
                            for s in snap.scenes
                        ]
                        if cur != last_status or cur_states != last_states:
                            last_status = cur
                            last_states = cur_states
                            yield _format_sse({
                                "type": "update",
                                "status": cur,
                                "scenes_count": len(snap.scenes),
                                "storyboard_count": len(snap.storyboard or []),
                                "video_urls_count": len(snap.video_urls or []),
                                "scenes_state": cur_states,
                                "title": snap.title or "",
                                "final_video_url": snap.final_video_url or "",
                                "ts": time.time(),
                                **_render_provenance_payload(snap),
                            })
                            if cur in ("completed", "failed"):
                                break
                    await asyncio.sleep(2.0)
            finally:
                pass
            return

        try:
            yield _format_sse({
                "type": "ready",
                "task_id": task_id,
                "ts": time.time(),
            })
            # Initial snapshot — clients wired for SSE-merge can splice
            # straight in without a separate fetch. We bypass the LRU
            # cache here so subscribers that join mid-flight still see
            # the live state from Redis (not a stale `pending` that the
            # cache may still hold from create-time).
            snap_live = task_store._get_task_live(task_id)
            if snap_live is not None:
                yield _format_sse({
                    "type": "snapshot",
                    "status": snap_live.status.value,
                    "scenes_count": len(snap_live.scenes),
                    "storyboard_count": len(snap_live.storyboard or []),
                    "video_urls_count": len(snap_live.video_urls or []),
                    "final_video_url": snap_live.final_video_url or "",
                    "title": snap_live.title or "",
                    "ts": time.time(),
                    **_render_provenance_payload(snap_live),
                })
                if snap_live.status.value in ("completed", "failed"):
                    return
            last_heartbeat = time.time()
            last_status = snap_live.status.value if snap_live is not None else None
            last_scenes_state: list = []
            last_seen_message_at = time.time()
            terminal = last_status in ("completed", "failed") if last_status else False
            while True:
                if await request.is_disconnected():
                    break
                # Poll pubsub with a short timeout so we can interleave
                # heartbeat pings and exit signals.
                msg = None
                try:
                    msg = pubsub.get_message(timeout=1.0)
                except _rexc.ConnectionError:
                    # Worker exited or Redis briefly unavailable — keep the
                    # heartbeat going but stop trying to drain pub/sub.
                    msg = None
                if msg is not None and msg.get("type") == "message":
                    try:
                        payload = msg["data"]
                        if isinstance(payload, bytes):
                            payload = payload.decode("utf-8")
                        yield _format_sse(json.loads(payload))
                        # After emitting, check if it's terminal.
                        data = json.loads(payload)
                        status = data.get("status")
                        if status in ("completed", "failed"):
                            terminal = True
                        if status:
                            last_status = status
                        last_seen_message_at = time.time()
                    except (ValueError, KeyError, TypeError) as exc:
                        logger.warning("events stream: bad message: %s", exc)
                # Polling fallback. Pubsub on WSL2/Windows NAT often
                # flaps without disconnecting the client; we always
                # poll every 2 s as the source of truth (pubsub only
                # *primes* this branch).
                elif time.time() - last_seen_message_at > 1.5:
                    live = task_store._get_task_live(task_id)
                    if live is not None:
                        cur_status = live.status.value if hasattr(live.status, "value") else str(live.status)
                        cur_states = [
                            {"scene": s.scene, "image_ready": bool(s.image_url),
                             "video_ready": bool(s.video_url),
                             "audio_ready": bool(s.audio_path)}
                            for s in live.scenes
                        ]
                        if cur_status != last_status or cur_states != last_scenes_state:
                            last_status = cur_status
                            last_scenes_state = cur_states
                            yield _format_sse({
                                "type": "update",
                                "status": cur_status,
                                "scenes_count": len(live.scenes),
                                "storyboard_count": len(live.storyboard or []),
                                "video_urls_count": len(live.video_urls or []),
                                "scenes_state": cur_states,
                                "title": live.title or "",
                                "final_video_url": live.final_video_url or "",
                                "ts": time.time(),
                                **_render_provenance_payload(live),
                            })
                        if cur_status in ("completed", "failed"):
                            terminal = True
                now = time.time()
                if now - last_heartbeat >= 15.0:
                    yield b": heartbeat\n\n"
                    last_heartbeat = now
                if terminal:
                    break
        finally:
            try:
                pubsub.unsubscribe()
                pubsub.close()
            except Exception:  # noqa: BLE001
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


def _format_sse(data: dict, *, max_bytes: int = 12 * 1024) -> bytes:
    """Serialise one SSE event.

    Each frame is *always* terminated with `\\n\\n` so any consumer
    (EventSource / fetch reader / curl) sees a complete event. We do
    *not* split an event — splitting would force the reducer to
    re-assemble partial frames, which the bridged `data: ...\\n\\n`
    contract doesn't tolerate. Instead, we drop the heavy fields
    (`scenes` with prompts, full `script`) on large frames; the
    reducer can fetch them via the polling fallback when their size
    would push the SSE channel over Starlette's internal
    Content-Length estimate.

    Threshold (`max_bytes` = 12 KB) is well under Starlette's
    internal chunk sizing but large enough for typical SSE updates.
    """
    payload = json.dumps(data, default=str)
    body = f"data: {payload}\n\n".encode("utf-8")
    if len(body) <= max_bytes:
        return body
    # Trim heavy fields that the reducer merges from polling anyway.
    trimmed = dict(data)
    trimmed.pop("scenes", None)
    trimmed.pop("script", None)
    trimmed["truncated"] = True
    body = f"data: {json.dumps(trimmed, default=str)}\n\n".encode("utf-8")
    if len(body) > max_bytes:
        # Last-ditch: drop the asset-heavy fields one by one.
        for k in ("storyboard", "video_urls", "cost_breakdown",
                  "stage_durations_sec", "scene_provenance", "warnings"):
            trimmed.pop(k, None)
            body = f"data: {json.dumps(trimmed, default=str)}\n\n".encode("utf-8")
            if len(body) <= max_bytes:
                break
    return body


def _render_provenance_payload(snap) -> dict:
    """Pull provenance / quality / warnings fields off the live or
    in-flight snapshot. They land on the task itself, not the scenes
    list, so we need a small helper rather than rebuilding inside every
    SSE emit.

    Scene keys come back as ints in the model but JSON requires strings,
    so we stringify here — the frontend reducer normalises them back.
    """
    raw = getattr(snap, "scene_provenance", {}) or {}
    return {
        "scene_provenance": {str(k): v for k, v in raw.items()},
        "render_quality": getattr(snap, "render_quality", "ok") or "ok",
        "warnings": list(getattr(snap, "warnings", []) or []),
        "retry_pass_count": int(getattr(snap, "retry_pass_count", 0) or 0),
    }


# ---- Generated asset delivery ------------------------------------------------

# Honour an explicit env override; otherwise resolve to the same path that
# video_compose uses, falling back to <project_root>/data/outputs on
# non-Docker platforms (Windows / bare-metal Linux).
if "VGO_OUTPUT_DIR" in os.environ:
    OUTPUT_DIR = Path(os.environ["VGO_OUTPUT_DIR"])
else:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = _PROJECT_ROOT / "data" / "outputs"
    # Keep Docker behaviour if /data exists (the compose mount target).
    if Path("/data").exists():
        OUTPUT_DIR = Path("/data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fallback path used by the host-side venv worker when it writes a file
# that the Docker backend can't see in the named volume (the bind-mount
# in docker-compose.yml mount the host's data/outputs as /host_outputs;
# when that path exists we look there if the primary directory misses).
HOST_OUTPUT_FALLBACK = Path("/host_outputs") if Path("/host_outputs").exists() else None


_OUTPUT_SIGN_TTL_SEC = 3600  # H5: signed output URLs are valid for 1 hour.


def _sign_output(filename: str, exp: int) -> str:
    """HMAC-SHA256 signature for a time-limited output URL (H5)."""
    secret = settings.output_signing_secret or ""
    payload = f"{filename}:{exp}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _verify_output_signature(filename: str, sig: str, exp: str) -> bool:
    """True only for a fresh, correctly signed output URL (H5)."""
    secret = settings.output_signing_secret or ""
    if not secret or not sig or not exp:
        return False
    try:
        exp_int = int(exp)
    except (TypeError, ValueError):
        return False
    if exp_int < int(time.time()):
        return False
    expected = _sign_output(filename, exp_int)
    return hmac.compare_digest(expected, sig)


def _signed_output_url(filename: str, ttl_sec: int = _OUTPUT_SIGN_TTL_SEC) -> str:
    """Relative signed URL for an output file (H5)."""
    exp = int(time.time()) + ttl_sec
    return f"/api/outputs/{filename}?sig={_sign_output(filename, exp)}&exp={exp}"


def _resolve_output(filename: str) -> Optional[Path]:
    """Find a generated asset across both Docker and host output trees.

    H5: containment is enforced with resolve() + is_relative_to(): a
    filename may not escape its output directory via "..", an absolute
    path, or a symlink. Returns None for anything outside the trees.
    """
    if not (filename.endswith(".mp4") or filename.endswith(".jpg") or filename.endswith(".png")):
        return None
    for base in (OUTPUT_DIR, HOST_OUTPUT_FALLBACK):
        if base is None:
            continue
        root = base.resolve()
        candidate = (root / filename).resolve()
        if candidate.is_relative_to(root) and candidate.exists():
            return candidate
    return None


@app.get("/api/outputs/{filename}/signed")
async def sign_output_endpoint(filename: str):
    """Mint a time-limited signed URL for an output file (H5).

    Requires a valid API key (the Next.js proxy adds it server-side).
    The returned URL can be used directly by a browser <video> tag.
    """
    path = _resolve_output(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Output not found")
    return {
        "filename": filename,
        "signed_url": _signed_output_url(filename),
        "expires_in": _OUTPUT_SIGN_TTL_SEC,
    }


@app.get("/api/outputs/{filename}")
def serve_output(filename: str):
    """Serve a generated MP4 (or scene image) across the Docker-volume +
    host-bind-mount output trees (see HOST_OUTPUT_FALLBACK above).

    Filenames are constrained to <task_id><suffix> so this can never
    escape the outputs directory. The media type is sniffed from the
    extension; for `.mp4` we also emit `Accept-Ranges: bytes` so the
    browser can seek the video.
    """
    # Decode URL-encoded characters first (e.g. %2e%2e)
    filename = unquote(filename)
    # Strict whitelist: alphanumeric, dash, underscore, dot, then .mp4/.jpg/.png
    if not re.match(r"^[\w.-]+\.(mp4|jpg|png)$", filename):
        raise HTTPException(status_code=400, detail="invalid filename")
    path = _resolve_output(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="not found")
    ctype = {
        ".mp4": "video/mp4",
        ".jpg": "image/jpeg",
        ".png": "image/png",
    }[path.suffix.lower()]
    headers = {"Cache-Control": "public, max-age=3600"}
    if ctype == "video/mp4":
        headers["Accept-Ranges"] = "bytes"
    return FileResponse(path, media_type=ctype, headers=headers)
