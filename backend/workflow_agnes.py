"""Agnes-AI-backed cinematic video generation workflow.

Differences from the NVIDIA pipeline (workflow.py):
  - Chat models: agnes-2.0-flash (director/composer), agnes-1.5-flash (prompt enhancer)
  - Image: agnes-image-2.1-flash → public URL (needs download to local disk)
  - Video: agnes-video-v2.0 → async, polled. Supports both text-to-video
    (prompt only) and image-to-video (prompt + image_url).
    Two modes are available, controlled by `VIDEO_MODE` config or the
    per-task `video_mode` request field:
      * "image2video" — pass the generated scene image as image_url.
        Higher quality, preferred by default.
      * "text2video"  — omit image_url; prompt only. Useful when the
        image stage is rate-limited or you want pure text-driven motion.
      * "auto"        — image-to-video when an image is available,
        text-to-video otherwise.
  - Canvas: 1280×704 (Agnes video native resolution) with black letter-box
    for the 2.39:1 cinematic look, or full 16:9 for news/documentary.

The pipeline maps 1-to-1 onto the 6 stages used by workflow.py, so the
worker, store and frontend require zero changes.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

from .agnes_client import AgnesClient, AgnesAPIError
from .config import get_settings
from .director_cache import get as cache_get, put as cache_put
from .schemas import Scene, TaskStatus
from .task_store import update_task, sanitize_error
from .tts import is_tts_available, synthesize_speech
from .video_compose import compose_slideshow, compose_real_videos, OUTPUT_DIR as _OUTPUT_DIR, _build_card_clip

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Cost estimates (USD) — these are *rough* figures based on the
# public pricing Agnes lists on their dashboard. They may drift, so
# treat the numbers as indicative only.
# ----------------------------------------------------------------
COST_AGNES_CHAT_USD = 0.002       # agnes-2.0-flash, ~1 k tokens
COST_AGNES_IMAGE_USD = 0.012      # agnes-image-2.1-flash
COST_AGNES_VIDEO_USD = 0.080      # agnes-video-v2.0, 5 s clip
COST_TTS_SEC_USD = 0.0005        # Magpie TTS (unchanged)

# ------------------------------------------------------------------
# Prompt templates — same structure as NVIDIA but tuned for the Agnes
# models, which handle concise system prompts well.
# ------------------------------------------------------------------
DIRECTOR_SYSTEM = """You are an AI Director. Given a user prompt, break it into
3-5 visually striking scenes for a short film teaser.

For each scene produce:
- scene: integer
- title: short title (max 8 words)
- description: vivid visual description (what the camera SEES).
- shot_type: one of Wide Shot, Medium Shot, Close-up, Extreme Close-up, Aerial
- lighting: one or two words (e.g., "Golden Hour", "Neon", "Volumetric")
- narration: ONE sentence of voice-over (max 22 words). Movie-trailer tone.

Respond with JSON exactly:
{
  "title": "...",
  "logline": "...",
  "scenes": [{"scene":1,"title":"...","description":"...","shot_type":"...","lighting":"...","narration":"..."}]
}
No markdown fences."""

PROMPT_ENHANCER_SYSTEM = """Rewrite a scene description as a single rich English prompt
for an image-generation model. Include subject, setting, lighting,
camera angle, and mood. 1-3 sentences. Output ONLY the prompt."""

COMPOSER_SYSTEM = """You are a film editor. Given the original request, scene breakdown,
and image URLs, write a 3-5 sentence narrative tying the scenes together.
Reference each URL naturally. Plain prose only."""


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _download_to_disk(url: str, dest: Path) -> None:
    """Download a public Agnes asset (image or video) to local disk.

    If `url` is HTTP(S) and the disk cache already has a copy keyed
    on the URL, we copy-rename from cache to `dest` and skip the
    network roundtrip. For HTTP(S) misses we also write through to
    the cache so a future identical URL is free.
    """
    import httpx
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith(("http://", "https://")):
        from .video_compose import _is_safe_url
        if not _is_safe_url(url):
            raise ValueError(f"URL not allowed (SSRF guard): {url[:80]}")
        from . import disk_cache
        cached = disk_cache.get_image_path(url)
        if cached is not None and cached.exists():
            try:
                dest.write_bytes(cached.read_bytes())
                return
            except OSError:
                # fall through and re-download
                pass
        r = httpx.get(url, timeout=60.0, follow_redirects=True)
        r.raise_for_status()
        blob = r.content
        # Persist into the disk cache so the next caller can copy it.
        try:
            disk_cache.put_image(url, blob)
        except Exception:
            pass
        dest.write_bytes(blob)
        return
    # Non-HTTP (MockProvider or local path): straight copy.
    # Validate the resolved path stays within the allowed output/cache
    # directories to prevent path-traversal reads of arbitrary files.
    src_path = Path(url).resolve()
    _allowed_dirs = (_OUTPUT_DIR.resolve(),
                     (Path(__file__).parent.parent / "data" / "cache").resolve())
    if not any(str(src_path).startswith(str(d)) for d in _allowed_dirs):
        raise ValueError(f"Local path not allowed: {url[:80]}")
    if not src_path.exists():
        raise FileNotFoundError(url)
    dest.write_bytes(src_path.read_bytes())


class VideoUnavailable(Exception):
    """Backward-compat alias kept for any caller referencing it.

    The router package (`backend.video_providers`) defines its own
    `VideoUnavailable`; this re-export ensures legacy callers don't
    crash on import. New code should import from `video_providers`.
    """
    pass


def _validate_script(obj: dict, max_scenes: int) -> List[dict]:
    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Director output missing scenes")
    cleaned = []
    for i, s in enumerate(scenes[:max_scenes], 1):
        if not isinstance(s, dict):
            continue
        cleaned.append({
            "scene": int(s.get("scene", i)),
            "title": str(s.get("title", f"Scene {i}"))[:80],
            "description": str(s.get("description", "")).strip(),
            "shot_type": str(s.get("shot_type", "Wide Shot")),
            "lighting": str(s.get("lighting", "Natural")),
            "narration": (str(s.get("narration", "")).strip() or None),
        })
    if not cleaned:
        raise ValueError("No usable scenes")
    return cleaned


def _enhance_prompt(client: AgnesClient, scene: dict) -> str:
    """Use agnes-1.5-flash to rewrite a scene for FLUX/Agnes image gen."""
    s = get_settings()
    user_msg = (
        f"Scene: {scene['description']}\n"
        f"Shot: {scene['shot_type']}\n"
        f"Lighting: {scene['lighting']}\n"
        f"Style: high-quality photorealistic image, detailed"
    )
    try:
        return client.chat(
            model="agnes-1.5-flash",
            messages=[
                {"role": "system", "content": PROMPT_ENHANCER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
    except AgnesAPIError as exc:
        logger.warning("Agnes prompt enhancer failed: %s", exc)
        return f"{scene['description']}, {scene['shot_type']}, detailed photorealistic image"


# ------------------------------------------------------------------
# Main worker entrypoint
# ------------------------------------------------------------------
def start_generation_workflow_agnes(task_id: str, prompt: str,
                                    style: Optional[str] = None,
                                    video_mode: str = "auto") -> None:
    """Entrypoint for the Agnes-backed pipeline.

    This function runs once per user request and only does:
    1. Director LLM → scenes
    2. Persist scenes + enqueue 4 parallel scene tasks
    3. Wait for every scene task to complete
    4. Compose LLM + TTS + final ffmpeg stitch
    5. Mark COMPLETED with `final_video_url`

    It does NOT do any per-scene image/video work directly — that lives in
    `generate_scene_task` so multiple scenes can run in parallel.
    """
    settings = get_settings()
    max_scenes = max(1, min(settings.max_scenes_per_video, 8))

    # Initialise the Agnes client once for the Director + Composer LLMs.
    # Per-scene tasks create their own clients.
    client = AgnesClient(
        api_key=settings.agnes_api_key,
        base_url=settings.agnes_base_url,
    )

    cost_breakdown = {
        "director_llm": 0.0,
        "prompt_enhancer_llm": 0.0,
        "composer_llm": 0.0,
        "image_gen": 0.0,
        "video_gen": 0.0,
        "tts": 0.0,
        "video_compose": 0.0,
    }

    try:
        # ----- Stage 1: SCRIPTING -----
        update_task(task_id, TaskStatus.SCRIPTING)
        logger.info("[%s] Agnes Director (%s) writing script...", task_id, settings.agnes_chat_model)

        # Director-cache fast path: identical (prompt, style, max_scenes)
        # was used for a prior task — skip the paid LLM call and return
        # the pre-computed script. Filter responses to `valid JSON dict`.
        cached = cache_get(prompt, style, max_scenes)
        if cached and isinstance(cached, dict) and isinstance(cached.get("scenes"), list):
            logger.info("[%s] Director cache HIT — reusing previous script", task_id)
            director_obj = cached
            cache_hit = True
        else:
            director_obj = client.chat_json(
                model=settings.agnes_chat_model,
                system=DIRECTOR_SYSTEM,
                user=f"User request: {prompt}\n\nProduce a JSON object with up to {max_scenes} scenes.",
                temperature=0.7,
                max_tokens=1800,
            )
            cache_hit = False
            cache_put(prompt, style, max_scenes, director_obj)

        cost_breakdown["director_llm"] = 0.0 if cache_hit else COST_AGNES_CHAT_USD

        title = str(director_obj.get("title", "Untitled"))[:120]
        logline = str(director_obj.get("logline", prompt))[:300]
        raw_scenes = _validate_script(director_obj, max_scenes=max_scenes)

        per_scene_seconds = max(2.0, 16.0 / max(1, len(raw_scenes)))
        scenes: List[Scene] = [
            Scene(
                scene=raw["scene"],
                title=raw["title"],
                description=raw["description"],
                shot_type=raw["shot_type"],
                lighting=raw["lighting"],
                duration_sec=per_scene_seconds,
                narration=(raw.get("narration") or "")[:300] or None,
            )
            for raw in raw_scenes
        ]

        for s in scenes:
            if s.narration and len(s.narration) > settings.tts_max_chars_per_scene:
                s.narration = s.narration[: settings.tts_max_chars_per_scene - 1].rstrip() + "…"

        update_task(
            task_id, TaskStatus.SCRIPTING,
            scenes=[s.model_copy() for s in scenes],
            title=title, logline=logline,
            cost_breakdown=cost_breakdown,
        )

        # ----- Stage 2: IMAGE + VIDEO (parallel, scene-level sub-tasks) -----
        update_task(task_id, TaskStatus.IMAGE_GEN)
        # Snapshot scene dicts for sub-tasks. We deliberately PATCH
        # image_url / video_url / image_prompt back into this queryset
        # as each sub-task completes so the parent's view of progress
        # stays consistent.
        scene_payloads: List[dict] = [s.model_dump() for s in scenes]

        results = _run_scenes_in_parallel(task_id, scene_payloads, video_mode=video_mode)
        scene_provenance: Dict[int, str] = {}
        for idx, payload in enumerate(results):
            scenes[idx].image_url = payload.get("image_url") or scenes[idx].image_url
            scenes[idx].video_url = payload.get("video_url") or scenes[idx].video_url
            scenes[idx].image_prompt = payload.get("image_prompt") or scenes[idx].image_prompt
            scenes[idx].audio_path = payload.get("audio_path") or scenes[idx].audio_path
            sceneno = scenes[idx].scene or (idx + 1)
            scene_provenance[sceneno] = (
                payload.get("provenance")
                or ("video" if scenes[idx].video_url else "missing")
            )

        # Persist the final per-scene state once all 4 are done.
        storyboard = [scenes[i].image_url for i in range(len(scenes))]
        video_urls = [scenes[i].video_url for i in range(len(scenes))]

        # Coarse pipeline health roll-up. Drives the "Render quality"
        # banner in the UI: ok / partial / fallback_only / missing.
        # `missing` is its own bucket because there's no usable asset
        # at all (neither real video nor Ken Burns) — distinct from
        # kenburns, which at least renders something.
        provenance_values = list(scene_provenance.values())
        if not provenance_values:
            render_quality = "missing"
        elif any(p == "missing" for p in provenance_values):
            render_quality = "missing"
        elif all(p == "video" for p in provenance_values):
            render_quality = "ok"
        elif all(p == "kenburns" for p in provenance_values):
            render_quality = "fallback_only"
        elif any(p == "video" for p in provenance_values):
            render_quality = "partial"
        else:
            # edge: empty / unknown — default to missing so the UI
            # banner surfaces rather than silently rendering as ok.
            render_quality = "missing"

        update_task(
            task_id, TaskStatus.VIDEO_GEN,
            scenes=[s.model_copy() for s in scenes],
            storyboard=storyboard,
            video_urls=video_urls,
            scene_provenance=scene_provenance,
            render_quality=render_quality,
            cost_breakdown=cost_breakdown,
        )

        # ----- Stage 3: COMPOSER + TTS + Final compose -----
        update_task(task_id, TaskStatus.COMPOSING)
        composer_input = (
            f"Original request: {prompt}\n"
            f"Title: {title}\nLogline: {logline}\n\nScenes:\n"
            + "\n".join(f"  Scene {s.scene} ({s.title}) → {s.image_url}"
                        for s in scenes)
        )
        try:
            narrative = client.chat(
                model=settings.agnes_chat_model,
                messages=[
                    {"role": "system", "content": COMPOSER_SYSTEM},
                    {"role": "user", "content": composer_input},
                ],
                temperature=0.6,
                max_tokens=600,
            )
            cost_breakdown["composer_llm"] = COST_AGNES_CHAT_USD
        except AgnesAPIError:
            narrative = f"{title}. {logline}"

        # Per-scene TTS (sequential; ~5–10 s each)
        if is_tts_available():
            for idx, scene in enumerate(scenes):
                if not scene.narration:
                    continue
                wav_path = _OUTPUT_DIR / f"{task_id}_voiceover{idx:02d}.wav"
                dur = synthesize_speech(scene.narration, wav_path)
                if dur and dur > 0:
                    scene.duration_sec = max(scene.duration_sec, dur + 0.5)
                    scene.audio_path = str(wav_path)
                    cost_breakdown["tts"] += dur * COST_TTS_SEC_USD
                    logger.info("[%s] Scene %d VO: %.2fs", task_id, idx + 1, dur)

        # ----- Stage 4: FINAL COMPOSE -----
        durations = [s.duration_sec for s in scenes]
        voiceover_paths = [s.audio_path for s in scenes]
        video_paths = [s.video_url for s in scenes if s.video_url]
        mp4_path = compose_real_videos(
            video_paths, durations, task_id,
            title=title, logline=logline,
            voiceover_paths=voiceover_paths,
        )
        final_url = mp4_path if mp4_path else (video_paths[0] if video_paths else None)
        cost_breakdown["video_compose"] = 0.0

        total_cost = round(sum(cost_breakdown.values()), 4)
        update_task(
            task_id, TaskStatus.COMPOSING if render_quality != "ok" else TaskStatus.COMPLETED,
            final_video_url=final_url,
            video_urls=video_paths,
            scenes=[s.model_copy() for s in scenes],
            scene_provenance=scene_provenance,
            render_quality=render_quality,
            script=narrative,
            title=title,
            logline=logline,
            duration_sec=sum(durations),
            estimated_cost_usd=total_cost,
            cost_breakdown=cost_breakdown,
        )
        logger.info(
            "[%s] Compose done. total_cost=$%.4f render_quality=%s",
            task_id, total_cost, render_quality,
        )

        # Auto-retry pass: when render_quality is partial / fallback_only,
        # try one extra round of video-only retries before declaring the
        # task COMPLETED. This catches the most common failure mode —
        # 1-2 scenes being rate-limited by Agnes in the fan-out — and
        # closes the gap to a real-video deliverable without the user
        # having to manually retry. We don't loop if the retry round
        # didn't fix everything; the user can hit "Retry" themselves.
        if render_quality in ("partial", "fallback_only", "missing"):
            try:
                update_task(
                    task_id, TaskStatus.COMPOSING,
                    render_quality=render_quality,
                    warnings=[f"Initial render was {render_quality}; "
                              "re-trying video pass for affected scenes."],
                )
                auto_recovered = par_retry_partial_video(task_id, video_mode=video_mode)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] auto-retry pass failed: %s", task_id, exc)
                update_task(task_id, warnings=[f"Auto-retry failed: {exc}"])
                auto_recovered = []

            # Re-stitch the final mp4 with any newly-real scenes.
            if auto_recovered:
                from .task_store import get_task as _get
                refreshed = _get(task_id)
                if refreshed is not None:
                    refreshed_video_paths = [s.video_url for s in refreshed.scenes
                                             if s.video_url]
                    refreshed_durations = [s.duration_sec for s in refreshed.scenes]
                    refreshed_vo = [s.audio_path for s in refreshed.scenes]
                    re_mp4 = compose_real_videos(
                        refreshed_video_paths, refreshed_durations, task_id,
                        title=title, logline=logline,
                        voiceover_paths=refreshed_vo,
                    )
                    if re_mp4:
                        final_url = re_mp4
                        update_task(
                            task_id, TaskStatus.COMPOSING,
                            final_video_url=re_mp4,
                            scenes=[s.model_copy() for s in refreshed.scenes],
                            warnings=[
                                f"Recovered real video for scenes {auto_recovered}; "
                                "re-stitched final mp4."
                            ],
                        )

        # Mark COMPLETED only at the very end, so the SSE stream
        # doesn't emit a terminal event before the auto-retry pass
        # could potentially succeed.
        from .task_store import get_task as _get
        final_snap = _get(task_id)
        final_quality = (final_snap.render_quality if final_snap else "ok") or "ok"
        # When no real video was produced at all (render_quality="missing"),
        # the task cannot be considered successfully completed.
        final_status = (TaskStatus.COMPLETED
                        if final_quality not in ("missing",)
                        else TaskStatus.FAILED)
        final_error = None
        if final_status == TaskStatus.FAILED:
            final_error = "No real video could be generated for any scene; " \
                          "output may be empty or a placeholder."
        update_task(
            task_id, final_status,
            final_video_url=final_url,
            scenes=[s.model_copy() for s in scenes]
            if (final_snap is None or not final_snap.scenes) else
            [s.model_copy() for s in final_snap.scenes],
            render_quality=final_quality,
            error=final_error,
        )
        logger.info(
            "[%s] Done. total_cost=$%.4f render_quality=%s",
            task_id, total_cost, final_quality,
        )

    except Exception as exc:
        logger.exception("[%s] Agnes workflow failed", task_id)
        update_task(
            task_id, TaskStatus.FAILED,
            error=sanitize_error(f"Agnes pipeline error: {exc}"),
        )
        raise


def _run_scenes_in_parallel(task_id: str, scene_payloads: List[dict],
                             video_mode: str = "auto") -> List[dict]:
    """Run one scene's image+video generation per scene, in parallel.

    We spawn a private `ThreadPoolExecutor` to drive each scene concurrently.
    Each thread does its own httpx polling against `apihub.agnes-ai.com`,
    so this is the lightweight way to overlap the slow `IN_PROGRESS` waits.

    Returns a list of scene payloads with their `image_url` / `video_url`
    populated for the parent to fold into the final VideoTask state.

    We deliberately use an in-process ThreadPoolExecutor here for two reasons:
      1. avoids cycle-imports between modules (no separate task process).
      2. each scene only does HTTP/IO; a 4-thread pool is enough and never
         holds the GIL long enough to starve other browser polling.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not scene_payloads:
        return []

    with ThreadPoolExecutor(max_workers=max(1, len(scene_payloads))) as ex:
        futures = {
            ex.submit(_generate_scene_inline, task_id, idx, payload, video_mode): idx
            for idx, payload in enumerate(scene_payloads)
        }
        results: List[dict] = [{} for _ in scene_payloads]
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result() or {}
            except Exception as exc:
                logger.exception("[%s] Scene %d inline-run failed", task_id, idx + 1)
                # Record the failure in warnings rather than marking the whole
                # task FAILED — the parent function will decide the final state
                # after all scenes complete. This prevents FAILED → COMPLETED
                # state contradiction in the SSE stream.
                from .task_store import append_warning, sanitize_error
                append_warning(task_id, sanitize_error(f"Scene {idx + 1}: {type(exc).__name__}: {exc}"))
                results[idx] = {}
    return results


def _generate_scene_inline(task_id: str, scene_idx: int, scene_payload: dict,
                           video_mode: str = "auto") -> dict:
    """Run one scene's image+video generation synchronously.

    Lifted out of `_generate_scene_task` so the parent can call it
    in-process via a ThreadPoolExecutor — no separate task process
    or broker round-trip needed.

    `video_mode` controls whether the generated image is passed to the
    upstream video provider:
      - "image2video": always pass image_url (prompt + image)
      - "text2video":  never pass image_url (prompt only)
      - "auto":        pass image_url when the local image exists,
                       text-only otherwise
    """
    from .config import get_settings
    settings = get_settings()

    payload = dict(scene_payload)
    payload.pop("image_url", None)
    payload.pop("video_url", None)
    payload.pop("image_prompt", None)
    payload.pop("audio_path", None)

    # Local Agnes client — one per thread keeps the connection pool
    # independent so we don't trip the "host egress mapped to one IP" risk.
    client = AgnesClient(
        api_key=settings.agnes_api_key,
        base_url=settings.agnes_base_url,
    )

    # ----- 2a. Image generation -----
    enhanced = _enhance_prompt(client, scene_payload)
    img_urls = client.generate_image(
        model=settings.agnes_image_model,
        prompt=enhanced,
        n=1,
        size="1024x1024",
    )
    img_url = img_urls[0]
    img_ext = Path(img_url).suffix or ".jpg"
    local_img = _OUTPUT_DIR / f"{task_id}_scene{scene_idx:02d}{img_ext}"
    _download_to_disk(img_url, local_img)

    payload["image_url"] = str(local_img)
    payload["image_prompt"] = enhanced
    update_task(
        task_id, TaskStatus.IMAGE_GEN,
        storyboard_idx=scene_idx,
        scene_update={"scene": scene_idx + 1, "image_url": str(local_img),
                      "image_prompt": enhanced},
    )

    # ----- 2b. Video generation via the configured router chain ------
    # `video_mode` decides whether to feed the generated image to the
    # upstream provider. image2video/auto pass image_url (higher quality);
    # text2video sends the prompt only. MockProvider can still synthesise
    # a Ken Burns clip from `local_img` regardless of mode because the
    # image is always generated for the storyboard.
    video_prompt = (
        f"{enhanced}. Cinematic motion, dynamic camera movement, "
        f"film still, high quality video footage"
    )
    # Decide what image_url to send to the video provider.
    if video_mode == "text2video":
        video_image_url: Optional[str] = None
    elif video_mode == "image2video":
        video_image_url = str(local_img)
    else:  # auto — use image when available
        video_image_url = str(local_img) if local_img.exists() else None

    logger.info("[%s] Scene %d launching video via router (mode=%s): %s",
                task_id, scene_idx + 1, video_mode, video_prompt[:80])
    local_vid = _OUTPUT_DIR / f"{task_id}_scene{scene_idx:02d}.mp4"

    cost_source = "kenburns_fallback"
    provenance = "kenburns"

    try:
        from .video_providers.router import get_router
        from .video_providers import VideoUnavailable as RouterUnavailable, ProviderError
        router = get_router()
        provider_name, task_id_remote = router.submit_for_job(
            prompt=video_prompt,
            image_url=video_image_url,
            seconds=settings.agnes_video_seconds,
            size="1280x704",
        )
        result = router.wait_for_job(provider_name, task_id_remote, timeout=360.0)
        if not result.is_success() or not result.url:
            # Provider returned FAILED. Drop to Ken Burns as before.
            raise RouterUnavailable(
                f"video provider {provider_name} did not succeed: {result.error}"
            )
        # If the URL is local (MockProvider produces file://-style
        # paths), just copy / alias into `local_vid`.
        video_url = result.url
        if video_url.startswith(("mock://", "file://", "/")):
            src_path = Path(
                video_url[len("mock://"):] if video_url.startswith("mock://")
                else video_url[len("file://"):] if video_url.startswith("file://")
                else video_url
            )
            if src_path.exists() and src_path.resolve() != local_vid.resolve():
                try:
                    local_vid.write_bytes(src_path.read_bytes())
                except Exception:
                    # If we can't copy, just point at the src.
                    local_vid = src_path
        else:
            _download_to_disk(video_url, local_vid)
        cost_source = f"{provider_name}_video"
        provenance = "video"
    except RouterUnavailable as exc:
        logger.warning(
            "[%s] Scene %d video router unavailable (%s); falling back to Ken Burns",
            task_id, scene_idx + 1, exc,
        )
        if not _build_card_clip(Path(local_img), local_vid,
                                duration=settings.agnes_video_seconds + 0.5,
                                fps=24):
            raise AgnesAPIError(
                f"Ken Burns fallback failed for scene {scene_idx + 1}"
            )
    except ProviderError as exc:
        # Should be unreachable in practice because router swaps on
        # ProviderError too, but keep a defensive barrier in case
        # someone adds a future provider that raises ProviderError
        # *for the only chain item*.
        logger.warning(
            "[%s] Scene %d video provider error (%s); falling back to Ken Burns",
            task_id, scene_idx + 1, exc,
        )
        if not _build_card_clip(Path(local_img), local_vid,
                                duration=settings.agnes_video_seconds + 0.5,
                                fps=24):
            raise AgnesAPIError(
                f"Ken Burns fallback failed for scene {scene_idx + 1}"
            )

    payload["video_url"] = str(local_vid)
    payload["video_cost_source"] = cost_source
    payload["provenance"] = provenance
    payload["video_provider"] = cost_source.removesuffix("_video")
    update_task(
        task_id, TaskStatus.IMAGE_GEN,
        storyboard_idx=scene_idx,
        video_urls_idx=scene_idx,
        scene_update={"scene": scene_idx + 1, "video_url": str(local_vid)},
    )
    # Also persist provenance in the task-level dict so the UI sees it
    # alongside the scene row; this is cheap (one HSET) and avoids a
    # second pass at the end of the run.
    update_task(
        task_id, TaskStatus.IMAGE_GEN,
        scene_provenance={scene_idx + 1: provenance},
    )
    return payload


def _regenerate_scene_video_only(task_id: str, scene_idx: int,
                                  payload: dict,
                                  video_mode: str = "auto") -> dict:
    """Re-run only the *video* stage for one scene. Used by the manual
    retry endpoint after a `partial` / `fallback_only` render so the
    pipeline can recover without paying for image / LLM calls again.

    `video_mode` mirrors `_generate_scene_inline`: "image2video" uses
    the cached image, "text2video" sends the prompt only, "auto" uses
    the image when available and falls back to text-only otherwise.

    Returns a small dict the caller merges into the running task.
    """
    from .config import get_settings
    settings = get_settings()

    # Re-derive the prompt the same way `_generate_scene_inline` does —
    # we don't have the original enhanced prompt persisting on the task
    # model, so we settle for the scene description. Note `scene` is a
    # *scene number* (int) in the Scene model, not a sub-dict — the
    # payload we hand in is the scene dump, so the fallback is
    # `_regenerate_scene_video_only`'s caller, not the int.
    description = payload.get("description", "")
    shot_type = payload.get("shot_type", "")
    lighting = payload.get("lighting", "")
    video_prompt = (
        f"{description}. {shot_type}, {lighting} lighting. "
        "Cinematic motion, dynamic camera movement, film still, high quality video footage"
    )

    local_img = payload.get("image_url")
    img_exists = bool(local_img) and Path(local_img).exists()

    # Decide whether to require the cached image and what to send.
    if video_mode == "image2video" and not img_exists:
        raise AgnesAPIError(
            f"Scene {scene_idx + 1}: missing cached image, retry must regenerate "
            f"the full pipeline."
        )
    if video_mode == "text2video":
        video_image_url: Optional[str] = None
    elif video_mode == "image2video":
        video_image_url = str(local_img)
    else:  # auto
        video_image_url = str(local_img) if img_exists else None

    local_vid = _OUTPUT_DIR / f"{task_id}_scene{scene_idx:02d}.mp4"

    update_task(
        task_id, TaskStatus.IMAGE_GEN,
        scene_update={"scene": scene_idx + 1},
    )
    # Video routing through the same router the initial pass uses, so
    # a forgotten Mock fallback in the chain also covers retries.
    from .video_providers.router import get_router
    from .video_providers import VideoUnavailable as RouterUnavailable, ProviderError
    router = get_router()
    provider_name, task_id_remote = router.submit_for_job(
        prompt=video_prompt,
        image_url=video_image_url,
        seconds=settings.agnes_video_seconds,
        size="1280x704",
    )
    result = router.wait_for_job(provider_name, task_id_remote, timeout=360.0)
    if not result.is_success() or not result.url:
        raise RouterUnavailable(
            f"video provider {provider_name} did not succeed: {result.error}"
        )
    video_url = result.url
    if video_url.startswith(("mock://", "file://", "/")):
        src_path = Path(
            video_url[len("mock://"):] if video_url.startswith("mock://")
            else video_url[len("file://"):] if video_url.startswith("file://")
            else video_url
        )
        if src_path.exists() and src_path.resolve() != local_vid.resolve():
            try:
                local_vid.write_bytes(src_path.read_bytes())
            except Exception:
                local_vid = src_path
    else:
        _download_to_disk(video_url, local_vid)
    provenance = "video"

    update_task(
        task_id, TaskStatus.IMAGE_GEN,
        video_urls_idx=scene_idx,
        scene_update={"scene": scene_idx + 1, "video_url": str(local_vid)},
    )
    update_task(
        task_id, TaskStatus.IMAGE_GEN,
        scene_provenance={scene_idx + 1: provenance},
    )
    logger.info(
        "[%s] Scene %d retry produced real video via %s",
        task_id, scene_idx + 1, provider_name,
    )
    return {"video_url": str(local_vid), "provenance": provenance,
            "provider": provider_name}


def retry_video_only_for_scene(task_id: str, scene_idx: int,
                                video_mode: str = "auto") -> dict:
    """Public wrapper: re-runs the video stage for one scene index.

    Used both by the auto-retry pass (called from
    `start_generation_workflow_agnes` when `render_quality` ends up as
    `partial`) and by the manual `POST /api/tasks/<id>/retry-scene`
    endpoint.
    """
    from .task_store import get_task
    task = get_task(task_id)
    if task is None:
        raise AgnesAPIError(f"task {task_id} not found")
    if scene_idx < 0 or scene_idx >= len(task.scenes):
        raise AgnesAPIError(f"scene index {scene_idx} out of range")
    scene = task.scenes[scene_idx]
    payload = scene.model_dump()
    payload["image_url"] = scene.image_url
    return _regenerate_scene_video_only(task_id, scene_idx, payload,
                                        video_mode=video_mode)


def par_retry_partial_video(task_id: str, video_mode: str = "auto") -> List[int]:
    """Auto-retry hook: scan the completed task's provenance, kick
    `retry_video_only_for_scene` for every kenburns / missing scene
    (1-based), and return which scenes actually recovered to real video.

    Called once after the main pipeline completes if render_quality is
    not 'ok'. Bounded to at most one pass per task so a perpetually
    rate-limited upstream doesn't hammer us; further retries must be
    triggered by the user via the retry endpoint.
    """
    from .task_store import get_task, update_task
    task = get_task(task_id)
    if task is None:
        return []
    recovered: List[int] = []
    for scene_idx, scene in enumerate(task.scenes):
        prov = (task.scene_provenance or {}).get(scene.scene)
        if prov in (None, "video"):
            continue
        # In image2video mode we need the cached image; in text2video
        # or auto mode we can retry without it.
        if video_mode == "image2video" and (not scene.image_url or not Path(scene.image_url).exists()):
            continue
        try:
            res = _regenerate_scene_video_only(task_id, scene_idx, scene.model_dump(),
                                               video_mode=video_mode)
            if res.get("provenance") == "video":
                recovered.append(scene.scene)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] auto-retry scene %d failed: %s",
                task_id, scene.scene, exc,
            )
    # Bump retry_pass_count and re-aggregate render_quality.
    new_count = (task.retry_pass_count or 0) + 1
    update_task(task_id, retry_pass_count=new_count)
    from .task_store import get_task as _get
    new_task = _get(task_id)
    if new_task is not None:
        new_prov = new_task.scene_provenance or {}
        if not new_prov:
            new_quality = "missing"
        elif any(p == "missing" for p in new_prov.values()):
            new_quality = "missing"
        elif all(p == "video" for p in new_prov.values()):
            new_quality = "ok"
        elif all(p == "kenburns" for p in new_prov.values()):
            new_quality = "fallback_only"
        elif any(p == "video" for p in new_prov.values()):
            new_quality = "partial"
        else:
            new_quality = "missing"

        update_task(task_id, render_quality=new_quality)
        if new_quality == "ok":
            update_task(task_id, warnings=[
                f"Auto-retry pass #{new_count} recovered all scenes "
                f"(saved: {recovered})."
            ])
        elif recovered:
            update_task(task_id, warnings=[
                f"Auto-retry pass #{new_count} recovered scenes {recovered}; "
                f"some still using fallback; quality={new_quality}."
            ])
        else:
            update_task(task_id, warnings=[
                f"Auto-retry pass #{new_count} could not recover any scenes; "
                f"quality={new_quality}."
            ])
    return recovered
