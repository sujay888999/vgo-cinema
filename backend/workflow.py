"""Cinematic video generation workflow.

This is the heart of VGO Cinema. Stages:

    1. SCRIPTING   — director LLM breaks the prompt into 3-5 scenes with
                     titles, descriptions, shot types, lighting, and a
                     per-scene voice-over narration line.
    2. IMAGE_GEN   — a small prompt-enhancer LLM rewrites each scene as a
                     rich image prompt; NVIDIA NIM FLUX.1-dev renders the
                     keyframe (real diffusion model, ~$0.012/image).
    3. VIDEO_GEN   — per-scene motion description (logged for now; FLUX
                     image-to-video is not exposed in NIM free tier).
    4. COMPOSING   — composer LLM writes a Director's Cut narrative and
                     we call NVIDIA Magpie TTS (NVCF gRPC, free endpoint)
                     for each scene's voice-over. ffmpeg stitches the
                     slideshow, title card, credits card, BGM and voice-
                     over into the final MP4.

All model calls go through the NIM API using the same `NVIDIA_API_KEY`
the user provided. Cost is tracked per-stage and surfaced to the UI so
the user can see exactly what the run cost.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .config import get_settings
from .image_gen import (
    build_image_url,
    generate_image_bytes,
)
from .nvidia_client import NvidiaAPIError, get_nvidia_client
from .schemas import Scene, TaskStatus
from .task_store import update_task, sanitize_error
from .tts import is_tts_available, synthesize_speech
from .video_compose import compose_slideshow

logger = logging.getLogger(__name__)


# Rough 2026 USD pricing per NIM call. Used for the per-run cost
# estimate. These are conservative — actual NIM pricing varies by
# contract, but they are within the order of magnitude NVIDIA lists
# on build.nvidia.com.
COST_PER_LLM_CALL_USD = 0.002     # ~70B model, ~1k tokens, free tier 0 but reported
COST_PER_LLM_SMALL_USD = 0.0005   # 8B model
COST_PER_FLUX_IMAGE_USD = 0.012   # FLUX.1-dev on NIM
COST_PER_FLUX_SCHNELL_USD = 0.003
COST_PER_SD35_IMAGE_USD = 0.013
COST_PER_TTS_SEC_USD = 0.0005     # Magpie TTS, free tier but reported
COST_PER_PICSUM_USD = 0.0         # placeholder


def _cost_for_image(prompt: str) -> float:
    s = get_settings()
    p = (s.image_provider or "picsum").lower()
    if p == "flux_dev":
        return COST_PER_FLUX_IMAGE_USD
    if p == "flux_schnell":
        return COST_PER_FLUX_SCHNELL_USD
    if p == "sd35":
        return COST_PER_SD35_IMAGE_USD
    return COST_PER_PICSUM_USD


# ---------- Style presets ---------------------------------------------------
#
# VGO Cinema supports three content styles. Each preset drives the
# director's tone, the FLUX image-prompt vocabulary, the aspect ratio
# (and therefore the canvas size we render at), and the credits card
# wording. The default is "cinematic"; pick "news" for photojournalism
# or "documentary" for observational footage.
#
#   - cinematic   : HDR anamorphic widescreen, 1280×768 (≈2.39:1)
#   - news        : 16:9 broadcast stills, Reuters/AP photojournalism
#   - documentary : 16:9 cinema-vérité, available light, observational
#
# All presets share the same six-stage pipeline; only the prompt
# templates and output canvas differ.
STYLE_PRESETS: dict = {
    "cinematic": {
        "label": "Cinematic",
        "description": "Widescreen film trailer with anamorphic HDR framing",
        "aspect_w": 1280,
        "aspect_h": 768,
        "director_system": """You are the AI Director of a cinematic short-film studio.
Given a user prompt, break it into a small number (3 to 5) of distinct,
visually striking scenes suitable for a 15-second cinematic teaser.

For each scene produce:
- scene: integer (1, 2, 3, ...)
- title: short, evocative title (max 8 words)
- description: one or two sentences of vivid visual description (what the
  audience SEES — composition, lighting, mood, action). Do NOT include
  dialogue or sound.
- shot_type: one of "Wide Shot", "Establishing Shot", "Medium Shot",
  "Close-up", "Extreme Close-up", "Over-the-shoulder", "POV", "Aerial"
- lighting: one or two words (e.g. "Neon", "Golden Hour", "Volumetric",
  "High Key", "Chiaroscuro")
- narration: ONE sentence of voice-over for the scene (max 22 words).
  Should sound like a movie trailer narrator — short, atmospheric, with
  strong verbs. Do not use quotes. Do not refer to "we" or "the viewer".

Respond with a single JSON object matching this schema:
{
  "title": "<short film title>",
  "logline": "<one sentence pitch>",
  "scenes": [ {scene, title, description, shot_type, lighting, narration}, ... ]
}

No prose outside the JSON. No markdown fences.""",
        "enhancer_suffix": "Film style: cinematic, high dynamic range, anamorphic widescreen, 2.39:1 aspect ratio, color graded for film print.",
        "composer_role": "lead editor of a short cinematic film",
        "composer_opening": "You are the lead editor of a short cinematic film.",
        "credits_roles": [
            ("Director",         "AI Director · meta/llama-3.3-70b-instruct"),
            ("Cinematographer",  "Image Prompt Engineer · meta/llama-3.1-8b-instruct"),
            ("Composer",         "Narrative Editor · meta/llama-3.3-70b-instruct"),
            ("Storyboard",       "4 cinematic keyframes · ffmpeg xfade"),
            ("Original Score",   "Procedural ambient pad · synthesized in-pipeline"),
        ],
    },
    "news": {
        "label": "News",
        "description": "Photojournalism in the style of Reuters / AP / National Geographic",
        "aspect_w": 1280,
        "aspect_h": 720,
        "director_system": """You are the AI producer of a short television news segment.
Given a user prompt, break it into 3 to 5 short visual beats suitable
for a 15-second news package (b-roll + stills). Each beat is a
self-contained "shot" a viewer might see in a 30-second news brief.

For each beat produce:
- scene: integer (1, 2, 3, ...)
- title: short factual headline (max 10 words, e.g. "Protesters gather downtown")
- description: ONE or TWO sentences of visual description of what the
  camera CAPTURES — the subject, the setting, the action. Do NOT invent
  quotes, sound, or off-screen narration. Stick to observable facts.
- shot_type: one of "Wide Shot", "Medium Shot", "Close-up", "Extreme
  Close-up", "Over-the-shoulder", "Aerial", "Handheld", "Press Photo"
- lighting: one or two words (e.g. "Natural", "Mixed", "Flash", "Low
  Key", "Outdoor Daylight", "Newsroom")
- narration: ONE factual sentence of voice-over (max 20 words). Should
  sound like a TV news anchor — clear, neutral, no adjectives. No
  speculation, no quotes, no first-person. State who/what/where.

Respond with a single JSON object matching this schema:
{
  "title": "<short factual headline>",
  "logline": "<one sentence summary of the story>",
  "scenes": [ {scene, title, description, shot_type, lighting, narration}, ... ]
}

No prose outside the JSON. No markdown fences.""",
        "enhancer_suffix": "Photojournalism style: Reuters / AP newswire framing, 35mm documentary lens, available natural light, candid moment, real people, no stylized cinematic color grading, 16:9 broadcast aspect ratio.",
        "composer_role": "TV news editor writing a 30-second package script",
        "composer_opening": "You are a TV news editor. Given the original user request, a beat-by-beat breakdown, and a list of generated photo URLs, write a 3-4 sentence anchor read that ties the beats together as a coherent news package. Reference each beat URL naturally as if you were describing the b-roll rolling under your voice.",
        "credits_roles": [
            ("Producer",        "AI Producer · meta/llama-3.3-70b-instruct"),
            ("Field Reporter",  "AI Reporter · meta/llama-3.1-8b-instruct"),
            ("Photo Editor",    "Narrative Editor · meta/llama-3.3-70b-instruct"),
            ("Wire Photos",     "4 photojournalism stills · FLUX.1-dev via NIM"),
            ("Original Score",  "Procedural ambient pad · synthesized in-pipeline"),
        ],
    },
    "documentary": {
        "label": "Documentary",
        "description": "Cinema-vérité observational footage with available light",
        "aspect_w": 1280,
        "aspect_h": 720,
        "director_system": """You are the AI director of a short documentary segment.
Given a user prompt, break it into 3 to 5 observational scenes suitable
for a 15-second documentary vignette. Each scene should feel candid,
real, and human — the camera is observing, not staging.

For each scene produce:
- scene: integer (1, 2, 3, ...)
- title: short, observational (max 8 words, e.g. "Hands at the loom")
- description: one or two sentences of visual description of what the
  camera CAPTURES in real time. No melodrama, no cinematic stylization.
  Use concrete verbs and concrete nouns. Avoid adjectives like
  "luminous", "ethereal", "dramatic".
- shot_type: one of "Wide Shot", "Medium Shot", "Close-up", "Tracking
  Shot", "Observational", "Handheld", "Interview Frame", "Cutaway"
- lighting: one or two words (e.g. "Natural", "Available", "Window
  Light", "Tungsten", "Mixed")
- narration: ONE calm, observational sentence (max 20 words). Sounds
  like a documentary narrator — present tense, factual, gently
  reflective. No quotes, no dramatization.

Respond with a single JSON object matching this schema:
{
  "title": "<observational title>",
  "logline": "<one sentence that frames the subject in context>",
  "scenes": [ {scene, title, description, shot_type, lighting, narration}, ... ]
}

No prose outside the JSON. No markdown fences.""",
        "enhancer_suffix": "Documentary cinematography: cinema-vérité, observational, available light, handheld camera, real locations, naturalistic color grade, 16:9 broadcast aspect ratio.",
        "composer_role": "documentary narrator writing a short reflective script",
        "composer_opening": "You are a documentary narrator. Given the original user request, a beat-by-beat breakdown, and a list of generated image URLs, write a 3-4 sentence reflective narration that ties the scenes together as a single observational piece. Reference each URL naturally as if you were describing the footage under your voice.",
        "credits_roles": [
            ("Director",       "AI Director · meta/llama-3.3-70b-instruct"),
            ("DP",              "Image Prompt Engineer · meta/llama-3.1-8b-instruct"),
            ("Field Producer",  "Narrative Editor · meta/llama-3.3-70b-instruct"),
            ("Storyboard",      "4 observational stills · FLUX.1-dev via NIM"),
            ("Original Score",  "Procedural ambient pad · synthesized in-pipeline"),
        ],
    },
}

# Default to cinematic for backwards compatibility.
DEFAULT_STYLE = "cinematic"


def _resolve_style(name: Optional[str]) -> str:
    """Map an arbitrary user-supplied style string to a preset key.

    Falls back to the default if the name is unknown. Matching is
    case-insensitive and tolerant of synonyms (e.g. "photojournalism"
    maps to "news")."""
    if not name:
        return DEFAULT_STYLE
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in STYLE_PRESETS:
        return key
    synonyms = {
        "photo": "news", "photojournalism": "news", "journalism": "news",
        "wire": "news", "broadcast": "news", "tv": "news",
        "doc": "documentary", "docu": "documentary", "verite": "documentary",
        "vérité": "documentary", "cinema_verite": "documentary",
        "film": "cinematic", "movie": "cinematic", "trailer": "cinematic",
    }
    return synonyms.get(key, DEFAULT_STYLE)


def get_style_preset(name: Optional[str]) -> dict:
    """Return the preset dict for a style name (always returns one)."""
    return STYLE_PRESETS[_resolve_style(name)]


# Backwards-compatible alias used by code that still imports the old
# constants. The new code uses `get_style_preset(style)['director_system']`
# etc. so changing the style at request time is a single line.
DIRECTOR_SYSTEM = STYLE_PRESETS[DEFAULT_STYLE]["director_system"]
PROMPT_ENHANCER_SYSTEM = """You rewrite a scene description as a single,
high-quality English prompt for FLUX.1-dev. The prompt should be 1-3
sentences, sensory, and include: subject, setting, lighting, camera
angle, and mood. Do not include dialogue. Output ONLY the prompt,
no preamble."""
COMPOSER_SYSTEM = """You are the lead editor of a short cinematic film.
You are given the original user request, a scene-by-scene breakdown, and a
list of generated image URLs. Write a 3-5 sentence "Director's Cut"
narrative that ties the scenes together as one cohesive film. Mention
each scene URL naturally. Output plain prose, no JSON, no markdown."""


# ---------- Helpers ----------------------------------------------------------

def _validate_script(obj: dict, max_scenes: int) -> List[dict]:
    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Director output missing 'scenes' array")
    cleaned: List[dict] = []
    for i, s in enumerate(scenes[:max_scenes], start=1):
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
        raise ValueError("No usable scenes in director output")
    return cleaned


def _enhance_prompt(nano, scene: dict, task_id: str, scene_index: int, *,
                    style: str = DEFAULT_STYLE) -> str:
    """Use the small prompt-enhancer LLM to rewrite a scene for FLUX."""
    preset = get_style_preset(style)
    user_msg = (
        f"Original scene: {scene['description']}\n"
        f"Shot: {scene['shot_type']}\n"
        f"Lighting: {scene['lighting']}\n"
        f"{preset['enhancer_suffix']}"
    )
    try:
        return nano.chat(
            model=get_settings().nvidia_prompt_model,
            messages=[
                {"role": "system", "content": PROMPT_ENHANCER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=1000,
        )
    except NvidiaAPIError as exc:
        logger.warning("Prompt enhancer failed for %s scene %d: %s", task_id, scene_index, exc)
        suffix = preset["enhancer_suffix"].replace("Film style: ", "").replace("Photojournalism style: ", "").replace("Documentary cinematography: ", "")
        return (
            f"{scene['description']} {scene['shot_type']}, {scene['lighting']} lighting, "
            f"{suffix}"
        )


def _generate_image_for_scene(prompt: str, task_id: str, scene_index: int,
                                seed: str, work_dir: Path,
                                width: Optional[int] = None,
                                height: Optional[int] = None) -> tuple[str, float]:
    """Render one keyframe. Returns (public_image_url, cost_usd).

    For NIM providers we generate the bytes, save them to a path
    inside the work dir, and return a `file://` style URL that
    `video_compose.py` and the frontend can read. (We can't use the
    public `/api/outputs/...` route for image gen because the file
    wasn't written there — we could move it, but the simplest path is
    to return the local worker file path; the compose step uses
    PIL.Image.open to load it.)

    For picsum we return the picsum URL as before.
    """
    s = get_settings()
    if (s.image_provider or "").lower() in ("flux_dev", "flux_schnell", "sd35"):
        # Download via NIM, save under the worker output dir, return path.
        from .video_compose import OUTPUT_DIR  # same volume
        bytes_, ctype = generate_image_bytes(prompt, seed=seed, width=width, height=height)
        ext = ".png" if "png" in ctype else ".jpg"
        local = OUTPUT_DIR / f"{task_id}_scene{scene_index:02d}{ext}"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(bytes_)
        cost = _cost_for_image(prompt)
        logger.info("[%s] Scene %d FLUX: %s (%d bytes, %s)",
                    task_id, scene_index + 1, local.name, len(bytes_), ctype)
        # Return both the local path and the cost; the caller stores
        # the local path as image_url, and video_compose knows how to
        # open it directly.
        return str(local), cost
    # picsum / pollinations / none: just return the remote URL.
    return build_image_url(prompt, seed=seed), 0.0


def _script_to_text(scenes: List, title: str, logline: str) -> str:
    """Render a human-readable script. Accepts either dicts (legacy) or
    Scene Pydantic models."""
    parts = [f"Title: {title}", f"Logline: {logline}", ""]
    for s in scenes:
        if isinstance(s, dict):
            scene_no, scene_title, shot, light, desc, narr = (
                s.get("scene"), s.get("title"), s.get("shot_type"),
                s.get("lighting"), s.get("description", ""),
                s.get("narration", "") or "",
            )
        else:
            scene_no, scene_title, shot, light, desc, narr = (
                s.scene, s.title, s.shot_type, s.lighting, s.description,
                s.narration or "",
            )
        parts.append(
            f"Scene {scene_no} — {scene_title}  ({shot}, {light})\n"
            f"  {desc}\n"
            f"  Voice-over: {narr}" if narr else
            f"Scene {scene_no} — {scene_title}  ({shot}, {light})\n  {desc}"
        )
    return "\n\n".join(parts)


# ---------- Worker task ------------------------------------------------------
#
# Inline background-pool entry. The workflow runs in-process via
# `backend.main.local_pool` (ThreadPoolExecutor) — no external broker
# or Celery dependency. This avoids broker round-trip latency and
# keeps the deployment simple (single process, no separate worker).

_MAX_WORKFLOW_RETRIES = 2
_RETRY_DELAY_SEC = 30


def run_workflow(task_id: str, prompt: str, style: Optional[str] = None,
                 video_mode: str = "auto") -> None:
    """Run the workflow inside a background thread.

    Mirrors the retry semantics that the workflow used to
    have, minus the broker. Up to `_MAX_WORKFLOW_RETRIES` additional
    attempts are made; a sane first-attempt FAILED is reverted to
    `running` before the retry so the UI doesn't flicker.
    """
    import time as _time
    for attempt in range(_MAX_WORKFLOW_RETRIES + 1):
        try:
            start_generation_workflow(task_id, prompt, style, video_mode=video_mode)
            return
        except Exception as exc:  # noqa: BLE001
            if attempt >= _MAX_WORKFLOW_RETRIES:
                # Final failure — the workflow already marked it FAILED,
                # we just stop re-trying here.
                logger.warning("[%s] out of retries after %d attempts: %s",
                               task_id, attempt + 1, exc)
                return
            # Reset status from FAILED to PENDING so the UI doesn't show
            # "failed" while the retry is in-flight. The retry will
            # immediately set SCRIPTING on the next attempt.
            from .task_store import get_task, update_task
            t = get_task(task_id)
            if t is not None:
                update_task(
                    task_id,
                    TaskStatus.PENDING,
                    error="",
                )
            logger.info(
                "[%s] retrying workflow in %ds (attempt %d)",
                task_id, _RETRY_DELAY_SEC, attempt + 1,
            )
            _time.sleep(_RETRY_DELAY_SEC)


def start_generation_workflow(task_id: str, prompt: str,
                              style: Optional[str] = None,
                              video_mode: str = "auto") -> None:
    settings = get_settings()

    # ---- Provider routing: Agnes vs NVIDIA -----------------------------
    if settings.use_agnes():
        from .workflow_agnes import start_generation_workflow_agnes
        start_generation_workflow_agnes(task_id, prompt, style, video_mode=video_mode)
        return

    nvidia = get_nvidia_client()
    max_scenes = max(1, min(settings.max_scenes_per_video, 8))
    style = _resolve_style(style)
    preset = get_style_preset(style)
    # Pin the output canvas to this style's aspect ratio for the whole
    # run. We use local variables (not settings mutation) to avoid
    # corrupting the shared singleton when tasks with different styles
    # run concurrently.
    canvas_w = preset["aspect_w"]
    canvas_h = preset["aspect_h"]
    logger.info("[%s] Style: %s (%dx%d) — %s",
                task_id, preset["label"], preset["aspect_w"], preset["aspect_h"],
                preset["description"])

    cost_breakdown = {
        "director_llm": 0.0,
        "prompt_enhancer_llm": 0.0,
        "composer_llm": 0.0,
        "image_gen": 0.0,
        "tts": 0.0,
        "video_compose": 0.0,
    }

    try:
        # ----- Stage 1: SCRIPTING -----
        update_task(task_id, TaskStatus.SCRIPTING)
        logger.info("[%s] Director (%s) writing script...", task_id, settings.nvidia_director_model)

        director_obj = nvidia.chat_json(
            model=settings.nvidia_director_model,
            system=preset["director_system"],
            user=(
                f"User request: {prompt}\n\n"
                f"Produce a JSON object with up to {max_scenes} scenes."
            ),
            temperature=0.7,
            max_tokens=1800,
        )
        cost_breakdown["director_llm"] = COST_PER_LLM_CALL_USD

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

        # Truncate narration to TTS budget.
        for s in scenes:
            if s.narration and len(s.narration) > settings.tts_max_chars_per_scene:
                s.narration = s.narration[: settings.tts_max_chars_per_scene - 1].rstrip() + "…"

        update_task(
            task_id, TaskStatus.SCRIPTING,
            title=title, logline=logline,
            scenes=list(scenes),
            duration_sec=sum(s.duration_sec for s in scenes),
            script=_script_to_text(scenes, title, logline),
            estimated_cost_usd=sum(cost_breakdown.values()),
            cost_breakdown=dict(cost_breakdown),
        )
        logger.info("[%s] Script: %d scenes, title=%r", task_id, len(scenes), title)

        # ----- Stage 2: IMAGE_GEN -----
        update_task(task_id, TaskStatus.IMAGE_GEN)
        image_urls: List[str] = []
        # NOTE: image gen stays strictly serial to honour the NIM 40 RPM
        # free-tier rate cap. We only write the task back to the store
        # once after the last keyframe renders (Redis Hash field-level
        # update keeps the payload small).  This was previously called
        # once per scene (~20 KB JSON dump × N).
        for idx, scene in enumerate(scenes):
            enhanced = _enhance_prompt(nvidia, scene.model_dump(), task_id, idx,
                                        style=style)
            cost_breakdown["prompt_enhancer_llm"] += COST_PER_LLM_SMALL_USD
            seed = uuid.uuid5(uuid.NAMESPACE_URL, f"{task_id}:img:{idx}").hex[:12]

            url, img_cost = _generate_image_for_scene(enhanced, task_id, idx, seed, None,
                                     width=canvas_w, height=canvas_h)
            cost_breakdown["image_gen"] += img_cost

            scene.image_prompt = enhanced
            scene.image_url = url
            image_urls.append(url)
            logger.info("[%s] Scene %d prompt: %s", task_id, idx + 1, enhanced[:90])

        # Single combined write at the end of image generation.
        update_task(
            task_id, TaskStatus.IMAGE_GEN,
            storyboard=list(image_urls),
            scenes=[s.model_copy() for s in scenes],
            estimated_cost_usd=sum(cost_breakdown.values()),
            cost_breakdown=dict(cost_breakdown),
        )

        # ----- Stage 3: VIDEO_GEN (motion descriptions) -----
        # NVIDIA NIM free tier has no video-generation model, so each
        # scene uses a Ken Burns fallback (static image + motion). We
        # mark provenance as "kenburns" so the UI can distinguish this
        # from real generated video (Agnes path).
        update_task(task_id, TaskStatus.VIDEO_GEN)
        video_urls: List[str] = []
        scene_provenance: Dict[int, str] = {}
        for idx, (scene, img_url) in enumerate(zip(scenes, image_urls)):
            motion_prompt = (
                f"slow cinematic motion of: {scene.description} "
                f"({scene.shot_type}, {scene.lighting} lighting)"
            )
            logger.info("[%s] Scene %d motion: %s", task_id, idx + 1, motion_prompt[:90])
            scene.motion = motion_prompt
            scene.video_url = img_url
            video_urls.append(img_url)
            scene_provenance[scene.scene] = "kenburns"
        update_task(
            task_id, TaskStatus.VIDEO_GEN,
            video_urls=list(video_urls),
            scenes=[s.model_copy() for s in scenes],
            scene_provenance=scene_provenance,
            render_quality="fallback_only",
            warnings=[
                "NVIDIA NIM free tier has no video-generation model; "
                "all scenes use Ken Burns fallback (static image + motion)."
            ],
            estimated_cost_usd=sum(cost_breakdown.values()),
            cost_breakdown=dict(cost_breakdown),
        )

        # ----- Stage 4a: TTS voice-over per scene -----
        # Render each scene's narration with Magpie TTS into a per-scene
        # WAV. The compose step stitches them back-to-back and ducks the
        # BGM under them.
        if is_tts_available():
            from .video_compose import OUTPUT_DIR
            scene_audios: List[Path] = []
            for idx, scene in enumerate(scenes):
                if not scene.narration:
                    scene_audios.append(None)
                    continue
                wav_path = OUTPUT_DIR / f"{task_id}_voiceover{idx:02d}.wav"
                dur = synthesize_speech(scene.narration, wav_path)
                if dur and dur > 0:
                    # Extend scene duration to fit the narration + a
                    # half-second pad so it doesn't get cut off.
                    scene.duration_sec = max(scene.duration_sec, dur + 0.5)
                    scene.audio_path = str(wav_path)
                    scene_audios.append(wav_path)
                    cost_breakdown["tts"] += dur * COST_PER_TTS_SEC_USD
                    logger.info("[%s] Scene %d voice-over: %.2fs", task_id, idx + 1, dur)
                else:
                    scene_audios.append(None)
            update_task(
                task_id, TaskStatus.VIDEO_GEN,
                scenes=[s.model_copy() for s in scenes],
                estimated_cost_usd=sum(cost_breakdown.values()),
                cost_breakdown=dict(cost_breakdown),
            )
        else:
            logger.info("[%s] TTS unavailable; voice-over skipped", task_id)

        # ----- Stage 4b: COMPOSING (narrative + video) -----
        update_task(task_id, TaskStatus.COMPOSING)
        composer_input = (
            f"Original request: {prompt}\n\n"
            f"Title: {title}\nLogline: {logline}\n\n"
            f"Scenes:\n"
            + "\n".join(
                f"  Scene {s.scene} ({s.title}) → {s.image_url}"
                for s in scenes
            )
        )
        # Build a composer system prompt tailored to the active style.
        composer_system = preset["composer_opening"] + (
            " Output plain prose, no JSON, no markdown." if "Output plain prose" not in preset["composer_opening"] else ""
        )
        try:
            narrative = nvidia.chat(
                model=settings.nvidia_composer_model,
                messages=[
                    {"role": "system", "content": composer_system},
                    {"role": "user", "content": composer_input},
                ],
                temperature=0.6,
                max_tokens=600,
            )
            cost_breakdown["composer_llm"] = COST_PER_LLM_CALL_USD
        except NvidiaAPIError as exc:
            logger.warning("[%s] Composer failed: %s", task_id, exc)
            narrative = (
                f"{title}. {logline} A {style} story across "
                f"{len(scenes)} scenes."
            )

        captions = [f"{s.title}" for s in scenes]
        durations = [s.duration_sec for s in scenes]
        voiceover_paths = [s.audio_path for s in scenes]
        mp4_path = compose_slideshow(
            image_urls, captions, durations, task_id,
            title=title, logline=logline, prompt=prompt,
            voiceover_paths=voiceover_paths,
            credits_roles=preset["credits_roles"],
            width=canvas_w, height=canvas_h,
        )
        if mp4_path:
            final_url = mp4_path
        else:
            final_url = image_urls[0] if image_urls else None
        cost_breakdown["video_compose"] = 0.0  # local ffmpeg, no API cost

        total_cost = round(sum(cost_breakdown.values()), 4)

        update_task(
            task_id, TaskStatus.COMPLETED,
            final_video_url=final_url,
            video_urls=video_urls,
            storyboard=image_urls,
            scenes=[s.model_copy() for s in scenes],
            title=title, logline=logline,
            duration_sec=sum(s.duration_sec for s in scenes),
            script=(
                f"## {title}\n\n{logline}\n\n{narrative}\n\n"
                f"### Storyboard\n" + "\n".join(f"- Scene {i+1}: {u}" for i, u in enumerate(image_urls))
            ),
            estimated_cost_usd=total_cost,
            cost_breakdown={k: round(v, 4) for k, v in cost_breakdown.items()},
            scene_provenance=scene_provenance,
            render_quality="fallback_only",
        )
        logger.info("[%s] Done. total_cost=$%.4f", task_id, total_cost)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Workflow failed for task %s", task_id)
        # We are currently *inside* a worker thread.
        # The retry bookkeeping lives in `run_workflow`; we just mark the
        # task FAILED here so the user-visible state is always correct,
        # and let the caller decide whether to retry.
        update_task(
            task_id, TaskStatus.FAILED,
            error=sanitize_error(f"{type(exc).__name__}: {exc}"),
            cost_breakdown={k: round(v, 4) for k, v in cost_breakdown.items()},
        )
        raise
