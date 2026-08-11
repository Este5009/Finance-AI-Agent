"""Canonical Windows/macOS startup orchestration for the desktop application."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from finance_agent.desktop.config import DesktopConfig
from finance_agent.desktop.paths import APP_DATA_ENV, RESOURCE_ROOT_ENV, AppDataPaths, app_data_paths, resource_path, resource_root
from finance_agent.llm.ollama_client import OllamaClient
from finance_agent.llm.ollama_readiness import check_ollama_readiness
from finance_agent.memory import initialize_database


LOGGER_NAME = "finance_ai_agent.desktop"


class StartupState(str, Enum):
    """User-facing first-run and runtime startup states."""

    INITIALIZING = "initializing"
    OLLAMA_MISSING = "ollama_missing"
    OLLAMA_STOPPED = "ollama_stopped"
    MODEL_MISSING = "model_missing"
    MODEL_DOWNLOAD_REQUIRED = "model_download_required"
    AI_HEALTH_FAILED = "ai_health_failed"
    AI_READY = "ai_ready"
    APPLICATION_READY = "application_ready"


@dataclass(frozen=True)
class StartupStatus:
    """One Spanish user-facing desktop startup status."""

    state: StartupState
    message_es: str
    technical_detail: str = ""


class DesktopStartupError(RuntimeError):
    """Expected startup failure carrying a readable Spanish message."""

    def __init__(self, status: StartupStatus) -> None:
        super().__init__(status.message_es)
        self.status = status


ConsentCallback = Callable[[str, str], bool]
StatusCallback = Callable[[StartupStatus], None]
ProcessFactory = Callable[..., subprocess.Popen[bytes]]
ClientFactory = Callable[..., OllamaClient]


def find_free_port(address: str, preferred_port: int) -> int:
    """Select the preferred local port or ask the OS for a free alternative.

    Inputs: loopback address and preferred port.
    Outputs: currently available TCP port.
    Assumptions: Streamlit binds immediately after this bounded probe.
    """

    for candidate in (int(preferred_port), 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((address, candidate))
            except OSError:
                continue
            return int(probe.getsockname()[1])
    raise DesktopStartupError(
        StartupStatus(StartupState.INITIALIZING, "No se encontró un puerto local disponible para abrir la aplicación.")
    )


def find_ollama_executable(
    *, platform: str | None = None, which: Callable[[str], str | None] = shutil.which
) -> Path | None:
    """Locate Ollama without relying solely on a developer-configured PATH.

    Inputs: optional platform and command lookup function for tests.
    Outputs: executable path or None.
    Assumptions: standard Ollama install locations cover supported installers.
    """

    discovered = which("ollama")
    if discovered:
        return Path(discovered)
    active_platform = platform or sys.platform
    candidates: list[Path] = []
    if active_platform.startswith("win"):
        local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates.append(local_app_data / "Programs" / "Ollama" / "ollama.exe")
    elif active_platform == "darwin":
        candidates.extend((Path("/Applications/Ollama.app/Contents/Resources/ollama"), Path("/usr/local/bin/ollama")))
    return next((path for path in candidates if path.is_file()), None)


def _streamlit_command(port: int, address: str) -> list[str]:
    """Build a source or frozen Streamlit child command with no system Python dependency."""

    app = resource_path("finance_agent", "ui", "streamlit_app.py")
    common = [
        str(app),
        "--global.developmentMode",
        "false",
        "--server.port",
        str(port),
        "--server.address",
        address,
        "--server.headless",
        "true",
    ]
    if getattr(sys, "frozen", False):
        return [sys.executable, "--streamlit-child", *common]
    return [sys.executable, "-m", "streamlit", "run", *common]


def _is_http_ready(url: str, timeout_seconds: float = 1.0) -> bool:
    """Return whether a local HTTP endpoint responds within a small bound."""

    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_http(url: str, timeout_seconds: float, *, poll_interval: float = 0.2) -> bool:
    """Poll a local readiness endpoint until success or a hard timeout.

    Inputs: URL, total timeout, and poll interval.
    Outputs: True when ready, otherwise False.
    Assumptions: individual HTTP probes use at most one second.
    """

    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if _is_http_ready(url):
            return True
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    return False


def _configure_logging(paths: AppDataPaths) -> logging.Logger:
    """Configure bounded rotating file logging for technical diagnostics."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.handlers.RotatingFileHandler(
            paths.log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _default_status_callback(status: StartupStatus) -> None:
    """Write readable startup status for developer runs and installer logs."""

    print(status.message_es, flush=True)


def _desktop_consent(title: str, message: str) -> bool:
    """Request explicit graphical consent before downloading a large model."""

    try:
        import tkinter.messagebox as messagebox

        return bool(messagebox.askyesno(title, message))
    except Exception:
        if sys.stdin and sys.stdin.isatty():
            answer = input(f"{message}\n¿Descargar ahora? (s/N): ").strip().casefold()
            return answer in {"s", "si", "sí", "y", "yes"}
        return False


def _show_error(message: str) -> None:
    """Show a readable desktop error while keeping tracebacks in the log."""

    try:
        import tkinter.messagebox as messagebox

        messagebox.showerror("Finance AI Agent", message)
    except Exception:
        print(message, file=sys.stderr)


class DesktopRuntime:
    """Coordinate app data, Ollama, Streamlit, browser, and shutdown."""

    def __init__(
        self,
        *,
        paths: AppDataPaths | None = None,
        config: DesktopConfig | None = None,
        status_callback: StatusCallback = _default_status_callback,
        consent_callback: ConsentCallback = _desktop_consent,
        process_factory: ProcessFactory = subprocess.Popen,
        client_factory: ClientFactory = OllamaClient,
        ollama_executable: Path | None = None,
    ) -> None:
        """Create a dependency-injectable runtime suitable for unit testing."""

        self.paths = paths or app_data_paths()
        self.config = config
        self.status_callback = status_callback
        self.consent_callback = consent_callback
        self.process_factory = process_factory
        self.client_factory = client_factory
        self.ollama_executable = ollama_executable
        self.ollama_process: subprocess.Popen[bytes] | None = None
        self.streamlit_process: subprocess.Popen[bytes] | None = None
        self.logger: logging.Logger | None = None

    def emit(self, state: StartupState, message: str, technical_detail: str = "") -> StartupStatus:
        """Publish one readable status and log its technical detail."""

        status = StartupStatus(state, message, technical_detail)
        self.status_callback(status)
        if self.logger:
            self.logger.info("state=%s message=%s detail=%s", state.value, message, technical_detail)
        return status

    def prepare(self) -> DesktopConfig:
        """Initialize persistent paths, config, database, environment, and logging."""

        self.paths.initialize()
        self.logger = _configure_logging(self.paths)
        self.emit(StartupState.INITIALIZING, "Preparando Finance AI Agent…")
        try:
            active_config = self.config or DesktopConfig.load(self.paths.config_file)
            active_config.validate()
            initialize_database(self.paths.memory_database)
        except Exception as exc:
            if self.logger:
                self.logger.exception("Desktop initialization failed")
            raise DesktopStartupError(
                StartupStatus(StartupState.INITIALIZING, "No se pudo preparar el espacio de trabajo de la aplicación.", str(exc))
            ) from exc
        self.config = active_config
        os.environ[APP_DATA_ENV] = str(self.paths.root)
        os.environ[RESOURCE_ROOT_ENV] = str(resource_root())
        os.environ["FINANCE_AI_OUTPUT_DIR"] = str(self.paths.outputs)
        os.environ["FINANCE_AI_UPLOAD_DIR"] = str(self.paths.imports)
        os.environ["FINANCE_AI_MEMORY_DB"] = str(self.paths.memory_database)
        return active_config

    def ensure_ollama_ready(self) -> StartupStatus:
        """Detect/start Ollama, verify the model, and run the shared health check."""

        config = self.config or self.prepare()
        executable = self.ollama_executable or find_ollama_executable()
        if executable is None:
            raise DesktopStartupError(
                self.emit(
                    StartupState.OLLAMA_MISSING,
                    "Ollama no está instalado. Instálelo desde ollama.com y vuelva a abrir Finance AI Agent.",
                )
            )
        self.ollama_executable = executable
        client = self.client_factory(
            endpoint=config.ollama_endpoint,
            model=config.model,
            connect_timeout_seconds=config.connect_timeout_seconds,
            read_timeout_seconds=config.health_read_timeout_seconds,
        )
        if not client.is_available():
            self.emit(StartupState.OLLAMA_STOPPED, "Ollama está instalado pero detenido. Intentando iniciarlo…")
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
            with self.paths.log_file.open("ab") as log_handle:
                self.ollama_process = self.process_factory(
                    [str(executable), "serve"],
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            if not wait_for_http(
                f"{config.ollama_endpoint.rstrip('/')}/api/tags",
                config.ollama_start_timeout_seconds,
            ):
                raise DesktopStartupError(
                    self.emit(StartupState.AI_HEALTH_FAILED, "Ollama no pudo iniciarse. Revise el registro técnico e intente nuevamente.")
                )
        try:
            installed_models = client.list_models()
        except Exception as exc:
            raise DesktopStartupError(
                self.emit(StartupState.AI_HEALTH_FAILED, "No se pudo verificar los modelos locales de Ollama.", str(exc))
            ) from exc
        if config.model not in installed_models:
            self.emit(StartupState.MODEL_MISSING, f"El modelo configurado {config.model} no está instalado.")
            consent_message = (
                f"Finance AI Agent necesita descargar el modelo {config.model}. "
                "La descarga puede ocupar varios gigabytes y requiere conexión a Internet."
            )
            self.emit(StartupState.MODEL_DOWNLOAD_REQUIRED, consent_message)
            if not self.consent_callback("Configurar modelo de IA", consent_message):
                raise DesktopStartupError(
                    StartupStatus(
                        StartupState.MODEL_DOWNLOAD_REQUIRED,
                        "La descarga fue cancelada. Puede cambiar el modelo en la configuración o volver a intentarlo después.",
                    )
                )
            completed = subprocess.run(
                [str(executable), "pull", config.model],
                check=False,
                timeout=60 * 60,
                stdout=None,
                stderr=None,
            )
            if completed.returncode != 0 or not client.model_exists(config.model):
                raise DesktopStartupError(
                    self.emit(StartupState.MODEL_MISSING, "No se pudo descargar o verificar el modelo configurado.")
                )
        readiness = check_ollama_readiness(
            client,
            model=config.model,
            connect_timeout_seconds=config.connect_timeout_seconds,
            read_timeout_seconds=config.health_read_timeout_seconds,
            stage_timeout_seconds=config.health_stage_timeout_seconds,
        )
        if not readiness.is_ready:
            raise DesktopStartupError(
                self.emit(StartupState.AI_HEALTH_FAILED, readiness.message_es, "; ".join(readiness.issues))
            )
        return self.emit(StartupState.AI_READY, readiness.message_es)

    def start_streamlit(self) -> tuple[str, subprocess.Popen[bytes]]:
        """Start Streamlit on a free local port and wait for bounded readiness."""

        config = self.config or self.prepare()
        port = find_free_port(config.address, config.preferred_port)
        url = f"http://{config.address}:{port}"
        command = _streamlit_command(port, config.address)
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        with self.paths.log_file.open("ab") as log_handle:
            self.streamlit_process = self.process_factory(
                command,
                cwd=str(resource_root()),
                env=os.environ.copy(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        if not wait_for_http(f"{url}/_stcore/health", config.startup_timeout_seconds):
            self.shutdown()
            raise DesktopStartupError(
                self.emit(StartupState.INITIALIZING, "La interfaz no pudo iniciarse a tiempo. Revise el registro técnico.")
            )
        self.emit(StartupState.APPLICATION_READY, "Finance AI Agent está listo.")
        if config.open_browser:
            webbrowser.open(url, new=1)
        return url, self.streamlit_process

    def shutdown(self) -> None:
        """Terminate only child processes started by this runtime.

        Inputs: tracked Streamlit and optional Ollama child processes.
        Outputs: none.
        Assumptions: an independently running Ollama service is never stopped.
        """

        for process in (self.streamlit_process, self.ollama_process):
            if process is None or process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def run(self) -> int:
        """Run the full desktop lifecycle until the UI exits or receives a signal."""

        self.prepare()
        self.ensure_ollama_ready()
        _, process = self.start_streamlit()

        def stop_handler(_signum: int, _frame: object) -> None:
            self.shutdown()

        for signal_name in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signal_name, stop_handler)
        try:
            return int(process.wait())
        finally:
            self.shutdown()


def _run_streamlit_child(arguments: Sequence[str]) -> int:
    """Execute Streamlit inside a PyInstaller binary child process."""

    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", *arguments]
    streamlit_cli.main(prog_name="streamlit")
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Create command-line options used by wrappers and packaged entry points."""

    parser = argparse.ArgumentParser(description="Finance AI Agent desktop runtime")
    parser.add_argument("--streamlit-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("streamlit_arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the packaged/source desktop entry point without exposing tracebacks."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--streamlit-child":
        return _run_streamlit_child(arguments[1:])
    runtime = DesktopRuntime()
    try:
        return runtime.run()
    except DesktopStartupError as exc:
        if runtime.logger and exc.status.technical_detail:
            runtime.logger.error(exc.status.technical_detail)
        _show_error(exc.status.message_es)
        return 1
    except Exception as exc:  # noqa: BLE001 - desktop users must not see tracebacks.
        if runtime.logger:
            runtime.logger.exception("Unexpected desktop startup error")
        _show_error("Finance AI Agent no pudo iniciarse. Revise el registro técnico e intente nuevamente.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
