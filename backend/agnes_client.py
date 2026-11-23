"""Agnes AI API client.

Thin wrapper around the Agnes AI OpenAI-compatible API. Supports:
  - Chat completions (agnes-2.0-flash, agnes-1.5-flash)
  - Image generations (agnes-image-2.1-flash, agnes-image-2.0-flash)
  - Video generations (agnes-video-v2.0) — async with polling

Ref: https://apihub.agnes-ai.com/v1
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
DEFAULT_TIMEOUT = 300.0
# Video endpoints are queued server-side and can take >60s to start. Use a
# dedicated longer timeout so we don't trip UNEXPECTED_EOF_WHILE_READING on
# the first byte.
VIDEO_START_TIMEOUT = 600.0

CHAT_MODELS = {
    "agnes-2.0-flash",
    "agnes-1.5-flash",
}
IMAGE_MODELS = {
    "agnes-image-2.0-flash",
    "agnes-image-2.1-flash",
}
VIDEO_MODEL = "agnes-video-v2.0"


class AgnesAPIError(Exception):
    """Raised when the Agnes API returns a non-2xx or an unexpected body."""
    pass


class AgnesClient:
    """Thread-safe synchronous client for Agnes AI."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key
        # Honour the user-provided base URL but allow callers to fall back
        # to long-form OpenAI-compatible hosts like
        # "https://apihub.agnes-ai.com/v1" without trailing-path mistakes.
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Disable HTTP/2 — Agnes upstream occasionally mid-closes h2 streams
        # which surfaces as ssl.SSL_ERROR_EOF on Windows clients. Force
        # HTTP/1.1 (the safer default for a single-tenant backend).
        import httpx as _httpx
        limits = _httpx.Limits(
            max_connections=8,
            # keep-alive = False: force a fresh TLS handshake per request.
            # The Agnes Cloudflare edge times out stale keep-alive sockets
            # mid-stream (~8 s) which Python surfaces as
            # `ssl.SSL_ERROR_EOF`. New socket per call avoids the trap.
            max_keepalive_connections=0,
            keepalive_expiry=0.0,
        )
        self._client = _httpx.Client(
            timeout=timeout,
            http2=False,
            limits=limits,
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, json: dict, timeout: Optional[float] = None) -> dict:
        url = f"{self.base_url}{path}"
        to = timeout if timeout is not None else self.timeout
        resp = self._client.post(url, headers=self._headers(), json=json, timeout=to)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            logger.error("Agnes API %s error: %s — body: %s", path, exc.response.status_code, body)
            raise AgnesAPIError(f"HTTP {exc.response.status_code}: {body}") from exc
        data = resp.json()
        if not isinstance(data, dict):
            raise AgnesAPIError(f"Unexpected response from {path}: {data}")
        return data

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        resp = self._client.get(url, headers=self._headers())
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            logger.error("Agnes API %s error: %s — body: %s", path, exc.response.status_code, body)
            raise AgnesAPIError(f"HTTP {exc.response.status_code}: {body}") from exc
        data = resp.json()
        if not isinstance(data, dict):
            raise AgnesAPIError(f"Unexpected response from {path}: {data}")
        return data

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def chat(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Simple chat completion. Returns the assistant's text."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._post("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise AgnesAPIError(f"Malformed chat response: {data}") from exc

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> dict:
        """Chat with a system + user message, parse response as JSON."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._post("/chat/completions", payload)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise AgnesAPIError(f"Malformed chat response: {data}") from exc

        import json
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Sometimes the model wraps JSON in markdown fences
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            return json.loads(cleaned)

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------
    def generate_image(
        self, *,
        model: str = "agnes-image-2.1-flash",
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
    ) -> List[str]:
        """Generate image(s) and return a list of public HTTP URLs."""
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        data = self._post("/images/generations", payload)
        try:
            return [d["url"] for d in data["data"]]
        except (KeyError, TypeError) as exc:
            raise AgnesAPIError(f"Malformed image response: {data}") from exc

    # ------------------------------------------------------------------
    # Video (async)
    # ------------------------------------------------------------------
    def generate_video(
        self, *,
        model: str = VIDEO_MODEL,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        seconds: int = 5,
        size: str = "1280x704",
    ) -> str:
        """Start a video generation job. Returns the task_id for polling.

        Provide *either* `prompt` (text-to-video) or `image_url` (image-to-video).
        """
        if not prompt and not image_url:
            raise ValueError("Either prompt or image_url must be provided for video generation")
        payload: dict = {
            "model": model,
            "seconds": seconds,
            "size": size,
        }
        if prompt:
            payload["prompt"] = prompt
        if image_url:
            payload["image_url"] = image_url
        # Agnes expects seconds as a STRING, not an int.
        payload["seconds"] = str(seconds)
        # Video START endpoint is slow (queued server-side). Override the
        # short default timeout for this single call so we don't trip a
        # premature close on Windows + antivirus that masquerades as
        # SSL UNEXPECTED_EOF_WHILE_READING.
        try:
            data = self._post("/video/generations", payload, timeout=VIDEO_START_TIMEOUT)
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            raise AgnesAPIError(
                f"Agnes video start hung ({type(exc).__name__}) — likely server-side queue. Retry once."
            ) from exc
        try:
            return data["task_id"]
        except KeyError:
            raise AgnesAPIError(f"Malformed video start response: {data}") from None

    def get_video_status(self, task_id: str) -> dict:
        """Poll a video task. Returns a dict with keys:
        - status: "IN_PROGRESS" | "SUCCESS" | "FAILED"
        - progress: str e.g. "30%"
        - url: the final MP4 URL (when status is SUCCESS)
        """
        data = self._get(f"/video/generations/{task_id}")
        task_data = data.get("data", {})
        inner = task_data.get("data", {})

        result = {
            "status": task_data.get("status", "UNKNOWN"),
            "progress": task_data.get("progress", "0%"),
            "url": None,
            "error": None,
            "seconds": inner.get("seconds"),
            "size": inner.get("size"),
        }

        # The final video URL may be in different fields depending on
        # API version — check the common candidates in priority order.
        if result["status"] == "SUCCESS":
            result["url"] = (
                inner.get("remixed_from_video_id")
                or inner.get("video_url")
                or inner.get("url")
                or task_data.get("url")
            )
            if not result["url"]:
                logger.warning("Agnes video SUCCESS but no URL field found: %s", inner)
        elif inner.get("error"):
            result["error"] = inner["error"]

        return result

    def wait_for_video(
        self,
        task_id: str,
        *,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> str:
        """Poll until the video task completes. Returns the MP4 URL."""
        start = time.time()
        while time.time() - start < timeout:
            info = self.get_video_status(task_id)
            status = info["status"]
            logger.info("Agnes video task %s: %s (progress %s)", task_id, status, info["progress"])
            if status == "SUCCESS" and info["url"]:
                return info["url"]
            if status == "FAILED":
                raise AgnesAPIError(f"Video generation failed: {info.get('error') or 'unknown error'}")
            time.sleep(poll_interval)
        raise AgnesAPIError(f"Video generation timed out after {timeout}s")
