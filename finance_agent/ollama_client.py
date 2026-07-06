"""Small fail-safe HTTP client for local Ollama structure interpretation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:30b-a3b"


class OllamaError(RuntimeError):
    """Raised when a local Ollama request cannot return a usable response."""


@dataclass
class OllamaClient:
    """Reusable client for strict-JSON calls to the local Ollama API."""

    endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    model: str = DEFAULT_OLLAMA_MODEL
    timeout_seconds: float = 90.0

    def _request(
        self,
        path: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one JSON request to Ollama and decode its JSON envelope.

        Inputs: API path, HTTP method, and optional JSON payload.
        Outputs: decoded Ollama response object.
        Assumptions: Ollama is local and its API uses UTF-8 JSON.
        """

        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.endpoint.rstrip('/')}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except HTTPError as exc:
            raise OllamaError(
                f"Ollama returned HTTP {exc.code} for {path}."
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaError(f"Could not reach Ollama at {self.endpoint}: {exc}") from exc

        try:
            decoded = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned a non-JSON API response.") from exc
        if not isinstance(decoded, dict):
            raise OllamaError("Ollama returned an unexpected API response shape.")
        return decoded

    def is_available(self) -> bool:
        """Check whether the configured local Ollama endpoint responds.

        Inputs: configured endpoint and timeout.
        Outputs: True when the tags endpoint returns successfully, otherwise False.
        Assumptions: availability does not guarantee the configured model is installed.
        """

        try:
            self._request("/api/tags", method="GET")
        except OllamaError:
            return False
        return True

    def generate(self, prompt: str) -> str:
        """Ask Ollama for one non-streaming strict-JSON response.

        Inputs: compact task-specific prompt.
        Outputs: model response text from Ollama's response envelope.
        Assumptions: callers validate the model-authored JSON before using it.
        """

        response = self._request(
            "/api/generate",
            method="POST",
            payload={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                # Qwen3 otherwise may place the requested JSON in its separate
                # thinking field and leave the normal response field empty.
                "think": False,
                # Structure classification should be stable rather than creative.
                "options": {"temperature": 0},
            },
        )
        generated_text = response.get("response")
        if not isinstance(generated_text, str) or not generated_text.strip():
            # Older/newer Ollama combinations may ignore think=False. Returning
            # this text is still safe because the caller applies strict JSON
            # parsing and the full allowlist before accepting any suggestion.
            generated_text = response.get("thinking")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise OllamaError("Ollama response did not contain generated text.")
        return generated_text
