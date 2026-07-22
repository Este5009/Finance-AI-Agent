"""Small fail-safe HTTP client for local Ollama structure interpretation."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:30b-a3b"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_READ_TIMEOUT_SECONDS = 600.0
DEFAULT_KEEP_ALIVE = "15m"


class OllamaError(RuntimeError):
    """Raised when a local Ollama request cannot return a usable response."""

    def __init__(self, message: str, *, category: str = "ollama_error") -> None:
        """Store an Ollama error with a stable category.

        Inputs: human-readable message and machine-readable category.
        Outputs: initialized exception.
        Assumptions: callers may surface the category in telemetry/UI.
        """

        super().__init__(message)
        self.category = category


@dataclass
class OllamaClient:
    """Reusable client for strict-JSON calls to the local Ollama API."""

    endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    model: str = DEFAULT_OLLAMA_MODEL
    timeout_seconds: float | None = None
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: float = DEFAULT_READ_TIMEOUT_SECONDS
    reasoning_enabled: bool = False
    response_format: str | dict[str, Any] = "json"
    keep_alive: str = DEFAULT_KEEP_ALIVE

    def effective_read_timeout_seconds(self) -> float:
        """Return the read/inference timeout used for HTTP responses.

        Inputs: configured read timeout and legacy timeout option.
        Outputs: numeric timeout in seconds.
        Assumptions: ``timeout_seconds`` remains backward compatible by
        overriding the read timeout when explicitly supplied.
        """

        return float(
            self.timeout_seconds
            if self.timeout_seconds is not None
            else self.read_timeout_seconds
        )

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
        started = time.perf_counter()
        request = Request(
            f"{self.endpoint.rstrip('/')}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            # urllib exposes one socket timeout that covers both connect and
            # subsequent reads. For clear diagnostics we first perform a tiny
            # explicit TCP connection check with the connect timeout, then use
            # the longer read timeout for model loading/generation.
            self._check_tcp_connect(path)
            with urlopen(request, timeout=self.effective_read_timeout_seconds()) as response:
                raw_response = response.read().decode("utf-8")
        except HTTPError as exc:
            raise OllamaError(
                f"Ollama returned HTTP {exc.code} for {path}.",
                category="service_unavailable" if exc.code >= 500 else "http_error",
            ) from exc
        except TimeoutError as exc:
            elapsed = time.perf_counter() - started
            raise OllamaError(
                "Ollama inference/read timed out after "
                f"{elapsed:.1f}s while waiting for {path}.",
                category="inference_timeout",
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            category = _classify_url_error(exc)
            message = _error_message_for_category(
                category,
                endpoint=self.endpoint,
                path=path,
                exc=exc,
            )
            raise OllamaError(message, category=category) from exc

        try:
            decoded = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise OllamaError(
                "Ollama returned a malformed non-JSON API response.",
                category="malformed_response",
            ) from exc
        if not isinstance(decoded, dict):
            raise OllamaError(
                "Ollama returned an unexpected API response shape.",
                category="malformed_response",
            )
        return decoded

    def _check_tcp_connect(self, path: str) -> None:
        """Verify the endpoint accepts TCP connections using connect timeout.

        Inputs: request path used only for diagnostics.
        Outputs: returns None when the socket connects.
        Assumptions: Ollama is normally HTTP(S); non-HTTP endpoints are rejected.
        """

        parsed = urlparse(self.endpoint)
        host = parsed.hostname
        if host is None:
            raise OllamaError(
                f"Invalid Ollama endpoint: {self.endpoint}",
                category="configuration_error",
            )
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection(
                (host, port),
                timeout=float(self.connect_timeout_seconds),
            ):
                return
        except socket.timeout as exc:
            raise OllamaError(
                "Ollama connection timed out after "
                f"{self.connect_timeout_seconds:.1f}s for {self.endpoint}{path}.",
                category="connection_timeout",
            ) from exc
        except ConnectionRefusedError as exc:
            raise OllamaError(
                f"Ollama service is unavailable at {self.endpoint}: connection refused.",
                category="service_unavailable",
            ) from exc
        except OSError as exc:
            raise OllamaError(
                f"Ollama service is unavailable at {self.endpoint}: {exc}",
                category="service_unavailable",
            ) from exc

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

    def health(self) -> dict[str, Any]:
        """Return detailed Ollama health status.

        Inputs: configured endpoint.
        Outputs: status dictionary with availability and error category.
        Assumptions: model availability is checked by the caller if needed.
        """

        started = time.perf_counter()
        try:
            self._request("/api/tags", method="GET")
        except OllamaError as exc:
            return {
                "available": False,
                "error": str(exc),
                "error_category": exc.category,
                "elapsed_seconds": time.perf_counter() - started,
            }
        return {
            "available": True,
            "error": None,
            "error_category": None,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def warm_up(self, *, prompt: str = "Responde únicamente JSON: {\"ok\": true}") -> dict[str, Any]:
        """Warm the configured model and keep it resident for later stages.

        Inputs: optional tiny prompt.
        Outputs: generation envelope with telemetry.
        Assumptions: callers use warm-up for benchmarks/uncached live runs only.
        """

        previous_response_format = self.response_format
        try:
            self.response_format = "json"
            return self.generate_with_metadata(prompt)
        finally:
            self.response_format = previous_response_format

    def generate_with_metadata(self, prompt: str) -> dict[str, Any]:
        """Ask Ollama for one response plus timing metadata.

        Inputs: compact task-specific prompt.
        Outputs: dictionary with generated text and Ollama timing telemetry.
        Assumptions: callers validate the model-authored JSON before using it.
        """

        request_started = datetime.now(timezone.utc)
        perf_started = time.perf_counter()
        try:
            response = self._request(
            "/api/generate",
            method="POST",
            payload={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": self.response_format,
                "keep_alive": self.keep_alive,
                # Structure fallback stays non-thinking for strict schema mapping,
                # while planner/analysis clients can explicitly enable reasoning.
                "think": self.reasoning_enabled,
                # Structure classification should be stable rather than creative.
                "options": {"temperature": 0},
            },
        )
            error_category = None
        except OllamaError as exc:
            telemetry = {
                "model": self.model,
                "reasoning_enabled": self.reasoning_enabled,
                "request_start_utc": request_started.isoformat(),
                "request_end_utc": datetime.now(timezone.utc).isoformat(),
                "http_elapsed_time_seconds": time.perf_counter() - perf_started,
                "prompt_characters": len(prompt),
                "prompt_token_estimate": max(1, len(prompt) // 4) if prompt else 0,
                "connect_timeout_seconds": self.connect_timeout_seconds,
                "read_timeout_seconds": self.effective_read_timeout_seconds(),
                "keep_alive": self.keep_alive,
                "timeout_error_category": exc.category,
            }
            exc.telemetry = telemetry  # type: ignore[attr-defined]
            raise
        generated_text = response.get("response")
        if not isinstance(generated_text, str) or not generated_text.strip():
            # Older/newer Ollama combinations may ignore think=False. Returning
            # this text is still safe because the caller applies strict JSON
            # parsing and the full allowlist before accepting any suggestion.
            generated_text = response.get("thinking")
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise OllamaError(
                "Ollama response did not contain generated text.",
                category="malformed_response",
            )

        def seconds_from_nanoseconds(field_name: str) -> float:
            """Convert an Ollama duration field to seconds.

            Inputs: Ollama response field name.
            Outputs: seconds, or 0.0 when the field is absent.
            Assumptions: Ollama duration metadata is reported in nanoseconds.
            """

            value = response.get(field_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value) / 1_000_000_000
            return 0.0

        telemetry = {
            "model": self.model,
            "reasoning_enabled": self.reasoning_enabled,
            "request_start_utc": request_started.isoformat(),
            "request_end_utc": datetime.now(timezone.utc).isoformat(),
            "http_elapsed_time_seconds": time.perf_counter() - perf_started,
            "prompt_characters": len(prompt),
            "prompt_token_estimate": max(1, len(prompt) // 4) if prompt else 0,
            "model_load_time_seconds": seconds_from_nanoseconds("load_duration"),
            "prompt_evaluation_time_seconds": seconds_from_nanoseconds(
                "prompt_eval_duration"
            ),
            "generation_time_seconds": seconds_from_nanoseconds("eval_duration"),
            "total_ollama_time_seconds": seconds_from_nanoseconds("total_duration"),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "generation_eval_count": response.get("eval_count"),
            "thinking_characters": len(str(response.get("thinking", ""))),
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.effective_read_timeout_seconds(),
            "keep_alive": self.keep_alive,
            "timeout_error_category": error_category,
        }
        return {"response": generated_text, "telemetry": telemetry}

    def generate(self, prompt: str) -> str:
        """Ask Ollama for one non-streaming strict-JSON response.

        Inputs: compact task-specific prompt.
        Outputs: model response text from Ollama's response envelope.
        Assumptions: this compatibility wrapper discards timing metadata.
        """

        return str(self.generate_with_metadata(prompt)["response"])


def _classify_url_error(exc: BaseException) -> str:
    """Classify urllib/socket exceptions into stable Ollama categories.

    Inputs: exception raised during URL request.
    Outputs: error category string.
    Assumptions: urllib wraps many socket errors in ``URLError.reason``.
    """

    reason = getattr(exc, "reason", exc)
    if isinstance(reason, socket.timeout) or isinstance(exc, TimeoutError):
        return "inference_timeout"
    if isinstance(reason, ConnectionRefusedError):
        return "service_unavailable"
    text = str(exc).casefold()
    if "timed out" in text:
        return "inference_timeout"
    if "refused" in text or "actively refused" in text:
        return "service_unavailable"
    return "service_unavailable"


def _error_message_for_category(
    category: str,
    *,
    endpoint: str,
    path: str,
    exc: BaseException,
) -> str:
    """Build a precise user-facing Ollama error message.

    Inputs: category, endpoint, path, and original exception.
    Outputs: concise diagnostic message.
    Assumptions: read timeouts should not be described as unreachable service.
    """

    if category == "inference_timeout":
        return f"Ollama inference/read timed out while waiting for {path}: {exc}"
    if category == "connection_timeout":
        return f"Ollama connection timed out while connecting to {endpoint}: {exc}"
    return f"Ollama service is unavailable at {endpoint}: {exc}"
