"""Centralised settings backed by pydantic-settings.

Reads from environment / .env file and validates types. Importing
`get_settings()` is cached by @lru_cache so the file is parsed once.

The .env lookup is anchored to this file's directory (backend/) so the
app works regardless of the current working directory.
"""
import os
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_KEY_MARKERS = (
    "your-", "your_", "placeholder", "xxx", "xxxx", "changeme",
    "test-", "test_", "demo-", "demo_", "example", "insert-",
    "replace-", "todo", "sample",
)


def is_placeholder_key(value: str) -> bool:
    """True when a configured key looks like a placeholder, not a real key.

    Placeholders such as "your-agnes-api-key" must never be treated as a
    configured credential: has_*_key() rejects them and the startup
    validation warns about them. Only the shape is inspected; the value
    itself is never logged.
    """
    if not value:
        return True
    v = value.strip().lower()
    if v in ("", "key", "api-key", "api_key", "none", "null", "todo"):
        return True
    return any(m in v for m in PLACEHOLDER_KEY_MARKERS)


def has_valid_key(value: str) -> bool:
    """Non-empty and not a placeholder."""
    return bool(value and value.strip()) and not is_placeholder_key(value)


_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Infrastructure ---
    # When `enable_redis` is False (default for local dev / single-
    # container deploys) the task-store skips Redis entirely and falls
    # back to a per-process LRU + in-memory pub/sub. Set True via the
    # `ENABLE_REDIS=true` env var when running against a real Redis.
    enable_redis: bool = Field(default=False, alias="ENABLE_REDIS")
    # H2: in production the URL must carry a password, e.g.
    # redis://:<REDIS_PASSWORD>@127.0.0.1:6379/0 (set in backend/.env).
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    # Empty by default → backend per-host list (DEV_ALLOWED_ORIGINS). Set
    # `CORS_ALLOW_ORIGINS=<csv>` in `.env` to override in production.
    cors_allow_origins: str = Field(default="", alias="CORS_ALLOW_ORIGINS")

    # --- App ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_scenes_per_video: int = Field(default=4, alias="MAX_SCENES_PER_VIDEO")
    app_name: str = "VGO Cinema API"
    app_version: str = "0.3.0"

    # --- NVIDIA NIM (LLM provider) ---
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL"
    )
# All three use a reasoning model (nemotron-3.5-lightning-30b-a3b) - keep max_tokens generous so reasoning does not eat the visible output budget.
    # (nemotron-*, etc.) return their answer in `reasoning_content`
    # rather than `content`, which is not what we need.
    nvidia_director_model: str = Field(
        default="nvidia/nemotron-3.5-lightning-30b-a3b", alias="NVIDIA_DIRECTOR_MODEL"
    )
    nvidia_composer_model: str = Field(
    # nemotron-3.5-lightning-30b-a3b is a reasoning model; keep max_tokens generous.
        default="nvidia/nemotron-3.5-lightning-30b-a3b", alias="NVIDIA_COMPOSER_MODEL"
    )
    nvidia_prompt_model: str = Field(
        default="nvidia/nemotron-3.5-lightning-30b-a3b", alias="NVIDIA_PROMPT_MODEL"
    )
    nvidia_request_timeout: int = Field(default=60, alias="NVIDIA_REQUEST_TIMEOUT")

    # --- Image generation ---
    # Provider choices:
    #   `flux_dev` (default) — real NVIDIA NIM FLUX.1-dev. Best prompt
    #       adherence, ~$0.012/image on the standard tier. Returns a
    #       base64 JPEG via the NIM Visual GenAI REST endpoint.
    #   `flux_schnell` — real NVIDIA NIM FLUX.1-schnell (distilled,
    #       cfg_scale=0, 1-4 steps). Cheaper and faster than dev.
    #   `sd35` — real NVIDIA NIM Stable Diffusion 3.5 Large.
    #   `picsum` — Lorem Picsum placeholder (no key, no prompt match).
    #   `none` — skip image generation.
    image_provider: str = Field(default="flux_dev", alias="IMAGE_PROVIDER")
    image_width: int = Field(default=1280, alias="IMAGE_WIDTH")
    image_height: int = Field(default=768, alias="IMAGE_HEIGHT")
    image_inference_steps: int = Field(default=28, alias="IMAGE_INFERENCE_STEPS")
    image_cfg_scale: float = Field(default=3.5, alias="IMAGE_CFG_SCALE")
    nvidia_image_base_url: str = Field(
        default="https://ai.api.nvidia.com/v1/genai", alias="NVIDIA_IMAGE_BASE_URL"
    )

    # --- Voice (NVIDIA Magpie TTS via NVCF gRPC) ---
    # Magpie Multilingual is the only NVIDIA-hosted TTS that is
    # exposed as a "Free Endpoint" on build.nvidia.com (function-id
    # 877104f7-e885-42b9-8de8-f6e4c6303969). 9 languages.
    tts_provider: str = Field(default="magpie", alias="TTS_PROVIDER")
    tts_voice: str = Field(
        default="Magpie-Multilingual.EN-US.Aria", alias="TTS_VOICE"
    )
    tts_language: str = Field(default="en-US", alias="TTS_LANGUAGE")
    tts_sample_rate: int = Field(default=44100, alias="TTS_SAMPLE_RATE")
    tts_function_id: str = Field(
        default="877104f7-e885-42b9-8de8-f6e4c6303969", alias="TTS_FUNCTION_ID"
    )
    tts_nvcf_uri: str = Field(
        default="grpc.nvcf.nvidia.com:443", alias="TTS_NVCF_URI"
    )
    # Per-scene narration budget (chars). Long enough for one or two
    # sentences of voice-over per scene.
    tts_max_chars_per_scene: int = Field(default=220, alias="TTS_MAX_CHARS_PER_SCENE")

    # --- Agnes AI Integration ---
    # Agnes provides chat, image and video models via an OpenAI-compatible
    # API (https://apihub.agnes-ai.com/v1). Setting agnes_api_key makes
    # the pipeline prefer Agnes-native models when available.
    agnes_api_key: str = Field(default="", alias="AGNES_API_KEY")
    agnes_base_url: str = Field(
        default="https://apihub.agnes-ai.com/v1", alias="AGNES_BASE_URL"
    )
    agnes_chat_model: str = Field(
        default="agnes-2.0-flash", alias="AGNES_CHAT_MODEL"
    )
    agnes_image_model: str = Field(
        default="agnes-image-2.1-flash", alias="AGNES_IMAGE_MODEL"
    )
    agnes_video_model: str = Field(
        default="agnes-video-v2.0", alias="AGNES_VIDEO_MODEL"
    )
    # How many seconds each video clip generated by Agnes should be.
    # The model supports 5s clips natively.
    agnes_video_seconds: int = Field(default=5, alias="AGNES_VIDEO_SECONDS")

    # --- Magic Hour (real AI video generation) ---
    magic_hour_api_key: str = Field(default="", alias="MAGIC_HOUR_API_KEY")
    magic_hour_model: str = Field(default="ltx-2.5", alias="MAGIC_HOUR_MODEL")
    magic_hour_resolution: str = Field(default="480p", alias="MAGIC_HOUR_RESOLUTION")
    magic_hour_base_url: str = Field(default="https://api.magichour.ai", alias="MAGIC_HOUR_BASE_URL")
    # Provider routing: "agnes" or "nvidia" — which provider drives
    # the legacy single-provider pipeline.
    # "agnes"  → uses Agnes models for chat, image and video (default).
    # "nvidia" → uses NVIDIA NIM for chat + image (legacy).
    provider: str = Field(default="nvidia", alias="PROVIDER")

    # --- Video provider router (P2.1) --------------------------------
    # `video_provider_primary` and `video_provider_fallback` decide the
    # submit chain for the *video* stage only. Chat / image are still
    # routed through `provider` above. Supported names:
    #   "agnes" — submit through AgnesClient.generate_video + poll
    #   "mock"  — synthesise an MP4 locally (no API calls)
    #   "null"  — always refuse (forces Ken Burns fallback)
    # Any unknown name degrades to NullProvider; the chain still terminates.
    video_provider_primary: str = Field(default="magic-hour", alias="VIDEO_PROVIDER_PRIMARY")
    video_provider_fallback: str = Field(
        default="mock,null", alias="VIDEO_PROVIDER_FALLBACK",
    )

    # --- Video generation mode ------------------------------------------
    # Controls how the video stage feeds the upstream model:
    #   "auto"        (default) — image-to-video when an image is available,
    #                              text-to-video otherwise. Best quality/cost
    #                              balance for most workflows.
    #   "image2video" — always pass the generated image as image_url.
    #                   Requires a successful image generation step.
    #   "text2video"  — always omit image_url; prompt only. Useful when
    #                   img2vid is rate-limited or you want pure text-driven
    #                   motion without depending on the image stage.
    # The mode can also be overridden per-task via the API request body.
    video_mode: str = Field(default="auto", alias="VIDEO_MODE")

    def normalized_video_mode(self) -> str:
        """Validate + normalise `video_mode` into one of the three
        accepted values. Unknown values fall back to "auto"."""
        m = (self.video_mode or "auto").strip().lower()
        if m in ("image2video", "image-to-video", "img2vid", "img2video"):
            return "image2video"
        if m in ("text2video", "text-to-video", "t2v", "t2vid"):
            return "text2video"
        return "auto"

    # --- Future integrations (placeholders; not yet consumed) ---
    claude_api_key: str = Field(default="", alias="CLAUDE_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    runway_api_key: str = Field(default="", alias="RUNWAY_API_KEY")
    luma_api_key: str = Field(default="", alias="LUMA_API_KEY")

    # --- Auth + per-key rate limiting -------------------------------------
    # When `auth_required` is true (secure default), every /api/* endpoint
    # except public ones (/api/styles only; /api/outputs/* needs a signature or a key) must
    # carry a valid X-Api-Key header (matched against `api_keys`).
    # Multiple comma-separated `key:label:limit_per_window` triples can be set,
    # e.g. `sk-prod:demo:10,sk-test:local:60`. When AUTH_REQUIRED=false (local
    # dev only), auth is off and limits are applied to a shared IP bucket.
    auth_required: bool = Field(default=True, alias="AUTH_REQUIRED")
    api_keys_raw: str = Field(default="", alias="API_KEYS")
    # H5: secret for HMAC-signing time-limited /api/outputs/* URLs.
    # Server-side only (backend + Next proxy); never expose to the browser.
    output_signing_secret: str = Field(default="", alias="OUTPUT_SIGNING_SECRET")
    rate_limit_window_sec: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SEC")
    rate_limit_default_per_window: int = Field(default=10, alias="RATE_LIMIT_DEFAULT")
    # Comma-separated denylist of terms that will be rejected in user
    # prompts. Empty by default (no filtering); set to enable basic
    # content moderation, e.g. "term1,term2,term3".
    blocked_prompt_words: str = Field(default="", alias="BLOCKED_PROMPT_WORDS")
    @field_validator("api_keys_raw", mode="before")
    @classmethod
    def _parse_api_keys(cls, v):  # noqa: D401
        """`API_KEYS` is read verbatim; structured parsing happens in
        `api_keys()` so this validator stays a no-op."""
        return v if v is not None else ""

    def api_keys(self) -> Dict[str, Dict]:
        """Parse `API_KEYS` raw value into a dict keyed by token.

        Each entry: `{token: {"label": str, "limit": int}}`.
        Tokens are case-sensitive.
        """
        out: Dict[str, Dict] = {}
        if not self.api_keys_raw:
            return out
        for entry in self.api_keys_raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) < 2:
                continue
            token, label = parts[0], parts[1]
            try:
                limit = int(parts[2]) if len(parts) >= 3 else self.rate_limit_default_per_window
            except ValueError:
                limit = self.rate_limit_default_per_window
            out[token] = {"label": label, "limit": limit}
        return out

    def cors_origins_list(self) -> List[str]:
        if not self.cors_allow_origins:
            return []
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    def has_nvidia_key(self) -> bool:
        # H4: a placeholder value is NOT a configured key.
        return has_valid_key(self.nvidia_api_key)

    def has_agnes_key(self) -> bool:
        # H4: AGNES_API_KEY is currently a placeholder ("your-..."), so
        # Agnes is treated as NOT configured. Do not invent a key here.
        return has_valid_key(self.agnes_api_key)

    def has_magic_hour_key(self) -> bool:
        # H4: same placeholder rule as the other providers.
        return has_valid_key(self.magic_hour_api_key)

    def use_agnes(self) -> bool:
        # H4: Agnes is only used with a real (non-placeholder) key.
        return self.provider.lower() == "agnes" and self.has_agnes_key()

    def provider_key_warnings(self) -> List[str]:
        """Startup validation (H4): one warning per configured provider
        whose key is missing or a placeholder. Never includes key values.
        Call at startup and log every returned line as a warning."""
        warnings: List[str] = []
        keyed = {
            "agnes": ("AGNES_API_KEY", self.agnes_api_key),
            "nvidia": ("NVIDIA_API_KEY", self.nvidia_api_key),
            "magic-hour": ("MAGIC_HOUR_API_KEY", self.magic_hour_api_key),
        }
        configured = [("legacy provider", self.provider)]
        configured.append(("video primary", self.video_provider_primary))
        for fb in (self.video_provider_fallback or "").split(","):
            fb = fb.strip()
            if fb:
                configured.append(("video fallback", fb))
        for label, name in configured:
            n = (name or "").strip().lower()
            if n in ("", "null", "mock"):
                continue
            env_name, raw = keyed.get(n, (None, ""))
            if env_name is None:
                warnings.append(f"H4: unknown provider '{n}' configured as {label}; it has no known API-key setting and will be unavailable.")
            elif not (raw or "").strip():
                warnings.append(f"H4: provider '{n}' ({label}) has no API key configured ({env_name} is empty); it will be unavailable.")
            elif is_placeholder_key(raw):
                warnings.append(f"H4: provider '{n}' ({label}) key in {env_name} looks like a placeholder (e.g. starts with 'your-'); treating it as NOT configured. Replace it with a real key from the provider console.")
        if not self.has_agnes_key():
            warnings.append("H4: Agnes is explicitly UNAVAILABLE (AGNES_API_KEY is empty or a placeholder); the Agnes video chain and the legacy Agnes pipeline are disabled. Effective video chain: " f"primary={self.video_provider_primary}, fallback={self.video_provider_fallback}.")
        if not self.output_signing_secret:
            warnings.append("H5: OUTPUT_SIGNING_SECRET is empty; signed output URLs cannot be minted until it is set in backend/.env.")
        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
