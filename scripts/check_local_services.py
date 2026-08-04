"""Bounded local service checks for the Finance AI Agent development workflow.

The checker never starts services. It only reports whether the expected local
Streamlit and Ollama endpoints are already reachable, with a total runtime
budget so Codex and CI tasks cannot hang on environment checks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_STREAMLIT_URL = "http://localhost:8501"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:30b-a3b"


@dataclass
class ServiceStatus:
    """Structured status for one bounded service probe."""

    name: str
    url: str
    reachable: bool
    detail: str
    elapsed_seconds: float


@dataclass
class LocalServiceSummary:
    """Combined local service check result."""

    streamlit: ServiceStatus
    ollama: ServiceStatus
    model: str
    model_available: bool | None
    model_detail: str
    total_elapsed_seconds: float


def _remaining_timeout(deadline: float) -> float:
    """Return the remaining bounded timeout in seconds for the current probe."""

    return max(0.1, min(2.0, deadline - time.monotonic()))


def _http_get_json(url: str, deadline: float) -> tuple[bool, Any, str, float]:
    """Perform a bounded HTTP GET request and parse JSON when possible."""

    started = time.monotonic()
    try:
        request = Request(url, headers={"User-Agent": "finance-agent-service-check/1.0"})
        with urlopen(request, timeout=_remaining_timeout(deadline)) as response:
            raw = response.read(512_000)
            elapsed = time.monotonic() - started
            content_type = response.headers.get("Content-Type", "")
            if "json" in content_type.lower() or raw[:1] in (b"{", b"["):
                return True, json.loads(raw.decode("utf-8")), "OK", elapsed
            return True, raw.decode("utf-8", errors="replace")[:200], "OK", elapsed
    except HTTPError as exc:
        return False, None, f"HTTP {exc.code}: {exc.reason}", time.monotonic() - started
    except URLError as exc:
        return False, None, f"No disponible: {exc.reason}", time.monotonic() - started
    except TimeoutError:
        return False, None, "Tiempo de espera agotado", time.monotonic() - started
    except Exception as exc:  # pragma: no cover - defensive diagnostics.
        return False, None, f"Error: {exc}", time.monotonic() - started


def check_streamlit(url: str, deadline: float) -> ServiceStatus:
    """Check whether the Streamlit UI is already reachable."""

    reachable, _payload, detail, elapsed = _http_get_json(url, deadline)
    return ServiceStatus("Streamlit", url, reachable, detail, elapsed)


def check_ollama(base_url: str, model: str, deadline: float) -> tuple[ServiceStatus, bool | None, str]:
    """Check Ollama reachability and configured model availability."""

    tags_url = base_url.rstrip("/") + "/api/tags"
    reachable, payload, detail, elapsed = _http_get_json(tags_url, deadline)
    status = ServiceStatus("Ollama", base_url, reachable, detail, elapsed)
    if not reachable:
        return status, None, "No se verificó el modelo porque Ollama no respondió."

    names: set[str] = set()
    if isinstance(payload, dict):
        for item in payload.get("models", []):
            if isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]))
    available = model in names
    if available:
        return status, True, f"Modelo configurado disponible: {model}"
    return status, False, f"Modelo configurado no encontrado: {model}"


def run_checks(
    *,
    timeout_seconds: float = 10.0,
    streamlit_url: str = DEFAULT_STREAMLIT_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
) -> LocalServiceSummary:
    """Run all local checks within a bounded overall timeout budget."""

    started = time.monotonic()
    deadline = started + max(1.0, timeout_seconds)
    streamlit = check_streamlit(streamlit_url, deadline)
    ollama, model_available, model_detail = check_ollama(ollama_url, model, deadline)
    return LocalServiceSummary(
        streamlit=streamlit,
        ollama=ollama,
        model=model,
        model_available=model_available,
        model_detail=model_detail,
        total_elapsed_seconds=time.monotonic() - started,
    )


def format_summary(summary: LocalServiceSummary) -> str:
    """Format the bounded service result for humans."""

    def line(status: ServiceStatus) -> str:
        state = "disponible" if status.reachable else "no disponible"
        return f"- {status.name}: {state} ({status.detail}) [{status.elapsed_seconds:.2f}s] {status.url}"

    model_state = (
        "no verificado"
        if summary.model_available is None
        else "disponible"
        if summary.model_available
        else "no disponible"
    )
    return "\n".join(
        [
            f"Verificación local acotada ({summary.total_elapsed_seconds:.2f}s)",
            line(summary.streamlit),
            line(summary.ollama),
            f"- Modelo Ollama {summary.model}: {model_state}. {summary.model_detail}",
        ]
    )


def main() -> int:
    """Parse CLI arguments, run bounded checks, and print a readable summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=10.0, help="Tiempo máximo total en segundos.")
    parser.add_argument("--streamlit-url", default=DEFAULT_STREAMLIT_URL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--json", action="store_true", help="Imprime salida JSON estructurada.")
    args = parser.parse_args()

    summary = run_checks(
        timeout_seconds=args.timeout,
        streamlit_url=args.streamlit_url,
        ollama_url=args.ollama_url,
        model=args.model,
    )
    if args.json:
        print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
