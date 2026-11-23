"""Thin wrapper around the NVIDIA NIM OpenAI-compatible API.

NVIDIA NIM exposes an OpenAI-compatible chat-completions endpoint, so we use
the official `openai` SDK with a custom base_url. Failures are translated to
`NvidiaAPIError` so callers can handle them uniformly.
"""
from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from openai import OpenAI

from .config import get_settings

logger = logging.getLogger(__name__)


class NvidiaAPIError(RuntimeError):
    """Raised when a NVIDIA NIM call fails or returns malformed output."""


class NvidiaClient:
    """Synchronous client. Cheap to instantiate; share one per process."""

    def __init__(self) -> None:
        s = get_settings()
        if not s.has_nvidia_key():
            raise NvidiaAPIError(
                "NVIDIA_API_KEY is not set. Add it to backend/.env "
                "or set the NVIDIA_API_KEY environment variable."
            )
        self._client = OpenAI(
            api_key=s.nvidia_api_key,
            base_url=s.nvidia_base_url,
            timeout=s.nvidia_request_timeout,
            max_retries=0,
        )
        self._settings = s

    def chat(
        self,
        *,
        model: str,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        extra: Optional[dict] = None,
    ) -> str:
        """Single-turn chat completion. Returns the assistant text content."""
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if extra:
            kwargs.update(extra)
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise NvidiaAPIError(f"NVIDIA call failed: {exc}") from exc

        if not resp.choices:
            raise NvidiaAPIError(f"NVIDIA returned no choices: {resp!r}")
        content = resp.choices[0].message.content
        if content is None:
            raise NvidiaAPIError("NVIDIA returned empty content")
        return content.strip()

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.6,
        max_tokens: int = 1500,
    ) -> Any:
        """Chat completion that asks the model to respond with a JSON object.

        Tries to parse the response as JSON. Strips Markdown code fences
        (```json ... ```) if present. Raises NvidiaAPIError on failure.
        """
        text = self.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _parse_json_text(text)


def _parse_json_text(text: str) -> Any:
    """Best-effort JSON extraction from a model response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip leading ``` or ```json
        first_nl = cleaned.find("\n")
        if first_nl != -1:
            cleaned = cleaned[first_nl + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: locate the first {...} or [...] block
        for opener, closer in (("{", "}"), ("[", "]")):
            i = cleaned.find(opener)
            j = cleaned.rfind(closer)
            if i != -1 and j > i:
                snippet = cleaned[i : j + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
        raise NvidiaAPIError(f"Could not parse JSON from model output: {text[:300]!r}")


_singleton: Optional[NvidiaClient] = None


def get_nvidia_client() -> NvidiaClient:
    """Lazy singleton — first call validates the API key."""
    global _singleton
    if _singleton is None:
        _singleton = NvidiaClient()
    return _singleton
