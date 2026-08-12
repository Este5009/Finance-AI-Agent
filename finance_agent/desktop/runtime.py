"""Canonical Windows/macOS startup orchestration for the desktop application."""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
import uuid
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


def _pid_is_alive(pid: int) -> bool:
    """Return whether a process ID still exists without changing it."""

    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        try:
            import ctypes

            synchronize = 0x00100000
            wait_timeout = 0x00000102
            handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
            if not handle:
                return False
            try:
                return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == wait_timeout
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _process_command(pid: int) -> str:
    """Return a bounded process command used to validate stale ownership."""

    if sys.platform.startswith("win"):
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "try { "
                        f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\").CommandLine "
                        "} catch { '' }"
                    ),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return result.stdout.strip()
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


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


def _streamlit_app_path() -> Path:
    """Return the bundled Streamlit app script path used by both launch modes."""

    return resource_path("finance_agent", "ui", "streamlit_app.py")


def _streamlit_helper_executable() -> Path:
    """Return the frozen helper executable path for the active platform.

    Inputs: the current PyInstaller executable path from ``sys.executable``.
    Outputs: sibling helper executable path.
    Assumptions: Windows frozen helpers require the explicit ``.exe`` suffix;
    macOS/Linux helpers are extensionless files inside the collected bundle.
    """

    helper_name = "Finance AI Agent Streamlit.exe" if sys.platform.startswith("win") else "Finance AI Agent Streamlit"
    return Path(sys.executable).with_name(helper_name)


def _streamlit_environment() -> dict[str, str]:
    """Build a UTF-8-safe child environment for packaged Streamlit startup."""

    environment = os.environ.copy()
    # PyInstaller/Windows child output is written to the technical log. These
    # variables prevent Spanish status text from becoming mojibake in that log.
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    return environment


def _streamlit_command(port: int, address: str, *, session_id: str = "", parent_pid: int | None = None) -> list[str]:
    """Build a source or frozen Streamlit child command with no system Python dependency."""

    app = _streamlit_app_path()
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
        helper = _streamlit_helper_executable()
        return [
            str(helper),
            "--desktop-session",
            session_id,
            "--parent-pid",
            str(parent_pid or os.getpid()),
            *common,
        ]
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


def _tail_text(path: Path, *, max_characters: int = 5000) -> str:
    """Return the end of a technical log for startup diagnostics."""

    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-max_characters:].decode("utf-8", errors="replace")


def _wait_for_streamlit_ready(
    *,
    process: subprocess.Popen[bytes],
    health_url: str,
    timeout_seconds: float,
    log_path: Path,
    logger: logging.Logger | None = None,
    poll_interval: float = 0.2,
) -> tuple[bool, str]:
    """Wait for Streamlit readiness while detecting early child crashes.

    Inputs: owned helper process, health URL, timeout, and technical log path.
    Outputs: ``(ready, detail)`` where detail distinguishes timeout from crash.
    Assumptions: Streamlit is considered ready when ``/_stcore/health`` answers.
    """

    started = time.monotonic()
    deadline = started + float(timeout_seconds)
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            elapsed = time.monotonic() - started
            detail = (
                f"streamlit_child_exit exit_code={exit_code} elapsed_seconds={elapsed:.2f} "
                f"log_tail={_tail_text(log_path)!r}"
            )
            if logger:
                logger.error(detail)
            return False, detail
        if _is_http_ready(health_url):
            elapsed = time.monotonic() - started
            if logger:
                logger.info("streamlit_ready elapsed_seconds=%.2f health_url=%s", elapsed, health_url)
            return True, f"streamlit_ready elapsed_seconds={elapsed:.2f}"
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    detail = (
        f"streamlit_readiness_timeout timeout_seconds={timeout_seconds} "
        f"alive={process.poll() is None} health_url={health_url} log_tail={_tail_text(log_path)!r}"
    )
    if logger:
        logger.error(detail)
    return False, detail


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
        self.session_id = uuid.uuid4().hex
        self.launcher_pid = os.getpid()
        self.selected_port: int | None = None
        self.current_url: str | None = None
        self.ollama_started_by_session = False
        self._launcher_lock_acquired = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False
        self._stop_requested = threading.Event()

    def _log(self, message: str, *args: object, level: int = logging.INFO) -> None:
        """Write one session-correlated lifecycle record."""

        if self.logger:
            self.logger.log(
                level,
                "session=%s launcher_pid=%s " + message,
                self.session_id,
                self.launcher_pid,
                *args,
            )

    def _write_session_state(self, phase: str, **extra: object) -> None:
        """Atomically record transient ownership without touching user data."""

        payload = {
            "session_id": self.session_id,
            "launcher_pid": self.launcher_pid,
            "launcher_command": _process_command(self.launcher_pid),
            "streamlit_pid": self.streamlit_process.pid if self.streamlit_process else None,
            "port": self.selected_port,
            "phase": phase,
            "updated_at": time.time(),
            **extra,
        }
        temporary = self.paths.active_session_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.paths.active_session_file)

    def _acquire_launcher_lock(self) -> None:
        """Acquire a stale-safe single-launcher lock before slow startup work.

        Inputs: current launcher PID and writable runtime directory.
        Outputs: a lock file owned by this process.
        Assumptions: this guards only Finance AI Agent launchers, not the
        shared Ollama service or unrelated user processes.
        """

        if self._launcher_lock_acquired:
            return
        lock_path = self.paths.launcher_lock_file
        lock_payload = {
            "session_id": self.session_id,
            "launcher_pid": self.launcher_pid,
            "created_at": time.time(),
        }
        while True:
            try:
                descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    existing = json.loads(lock_path.read_text(encoding="utf-8"))
                    existing_pid = int(existing.get("launcher_pid") or 0)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    existing_pid = 0
                command = _process_command(existing_pid) if existing_pid and _pid_is_alive(existing_pid) else ""
                if existing_pid and "Finance AI Agent" in command:
                    active_url = ""
                    try:
                        active_state = json.loads(self.paths.active_session_file.read_text(encoding="utf-8"))
                        active_url = str(active_state.get("url") or "")
                    except (OSError, json.JSONDecodeError):
                        active_url = ""
                    if active_url and _is_http_ready(f"{active_url.rstrip('/')}/_stcore/health"):
                        try:
                            webbrowser.open(active_url, new=1)
                        except Exception:
                            pass
                        raise DesktopStartupError(
                            StartupStatus(
                                StartupState.APPLICATION_READY,
                                "Finance AI Agent ya está abierto. Se abrirá la sesión existente.",
                                f"existing_launcher_pid={existing_pid} url={active_url}",
                            )
                        )
                    raise DesktopStartupError(
                        StartupStatus(
                            StartupState.INITIALIZING,
                            "Finance AI Agent ya se está iniciando. Espere a que termine la apertura actual.",
                            f"existing_launcher_pid={existing_pid}",
                        )
                    )
                lock_path.unlink(missing_ok=True)
                continue
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(lock_payload, handle)
                self._launcher_lock_acquired = True
                self._log("launcher_lock_acquired path=%s", lock_path)
                return

    def _recover_stale_session(self) -> None:
        """Clean only a verifiably owned helper from a dead launcher session."""

        path = self.paths.active_session_file
        if not path.is_file():
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            old_session = str(state.get("session_id") or "")
            launcher_pid = int(state.get("launcher_pid") or 0)
            streamlit_pid = int(state.get("streamlit_pid") or 0)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            self._log("removed_invalid_session_state path=%s", path)
            return
        launcher_command = _process_command(launcher_pid) if _pid_is_alive(launcher_pid) else ""
        if launcher_pid and "Finance AI Agent" in launcher_command:
            raise DesktopStartupError(
                StartupStatus(
                    StartupState.INITIALIZING,
                    "Finance AI Agent ya está abierto. Use la ventana de control existente.",
                    f"active session {old_session} launcher PID {launcher_pid}",
                )
            )
        command = _process_command(streamlit_pid) if _pid_is_alive(streamlit_pid) else ""
        if old_session and "Finance AI Agent Streamlit" in command and old_session in command:
            try:
                os.kill(streamlit_pid, signal.SIGTERM)
                deadline = time.monotonic() + 3
                while _pid_is_alive(streamlit_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if _pid_is_alive(streamlit_pid):
                    os.kill(streamlit_pid, signal.SIGKILL)
                self._log("recovered_stale_streamlit session=%s streamlit_pid=%s", old_session, streamlit_pid)
            except OSError as exc:
                self._log("stale_streamlit_cleanup_error pid=%s error=%s", streamlit_pid, exc, level=logging.WARNING)
        path.unlink(missing_ok=True)

    def emit(self, state: StartupState, message: str, technical_detail: str = "") -> StartupStatus:
        """Publish one readable status and log its technical detail."""

        status = StartupStatus(state, message, technical_detail)
        self.status_callback(status)
        self._log("state=%s message=%s detail=%s", state.value, message, technical_detail)
        return status

    def prepare(self) -> DesktopConfig:
        """Initialize persistent paths, config, database, environment, and logging."""

        self.paths.initialize()
        self.logger = _configure_logging(self.paths)
        self._log("session_start resource_root=%s", resource_root())
        self._acquire_launcher_lock()
        self._recover_stale_session()
        self._write_session_state("preparing")
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
            self._log("ollama_detected executable=%s running=false", executable)
            self.emit(StartupState.OLLAMA_STOPPED, "Ollama está instalado pero detenido. Intentando iniciarlo…")
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
            with self.paths.log_file.open("ab") as log_handle:
                self.ollama_process = self.process_factory(
                    [str(executable), "serve"],
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            self.ollama_started_by_session = True
            self._log("ollama_started pid=%s lifecycle=shared_leave_running", self.ollama_process.pid)
            if not wait_for_http(
                f"{config.ollama_endpoint.rstrip('/')}/api/tags",
                config.ollama_start_timeout_seconds,
            ):
                raise DesktopStartupError(
                    self.emit(StartupState.AI_HEALTH_FAILED, "Ollama no pudo iniciarse. Revise el registro técnico e intente nuevamente.")
                )
        else:
            self._log("ollama_detected executable=%s running=true lifecycle=shared", executable)
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

        if self._stop_requested.is_set():
            raise DesktopStartupError(StartupStatus(StartupState.INITIALIZING, "Inicio cancelado."))
        config = self.config or self.prepare()
        port = find_free_port(config.address, config.preferred_port)
        self.selected_port = port
        url = f"http://{config.address}:{port}"
        self.current_url = url
        command = _streamlit_command(
            port,
            config.address,
            session_id=self.session_id,
            parent_pid=self.launcher_pid,
        )
        app_path = _streamlit_app_path()
        if not app_path.is_file():
            raise DesktopStartupError(
                self.emit(
                    StartupState.INITIALIZING,
                    "No se encontró la interfaz empaquetada de Finance AI Agent.",
                    f"missing_streamlit_app path={app_path}",
                )
            )
        if getattr(sys, "frozen", False):
            helper_path = Path(command[0])
            if not helper_path.is_file():
                raise DesktopStartupError(
                    self.emit(
                        StartupState.INITIALIZING,
                        "No se encontró el ejecutable empaquetado de la interfaz.",
                        f"missing_streamlit_helper path={helper_path}",
                    )
                )
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0
        with self.paths.streamlit_log_file.open("ab") as child_log:
            self.streamlit_process = self.process_factory(
                command,
                cwd=str(resource_root()),
                env=_streamlit_environment(),
                stdout=child_log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        self._write_session_state("streamlit_starting", command=command)
        self._log("streamlit_started streamlit_pid=%s port=%s command=%s", self.streamlit_process.pid, port, command)
        ready, detail = _wait_for_streamlit_ready(
            process=self.streamlit_process,
            health_url=f"{url}/_stcore/health",
            timeout_seconds=config.startup_timeout_seconds,
            log_path=self.paths.streamlit_log_file,
            logger=self.logger,
        )
        if not ready:
            self.shutdown()
            raise DesktopStartupError(
                self.emit(StartupState.INITIALIZING, "La interfaz no pudo iniciarse. Revise el registro técnico.", detail)
            )
        self.emit(StartupState.APPLICATION_READY, "Finance AI Agent está listo.")
        self._write_session_state("ready", url=url)
        if config.open_browser:
            webbrowser.open(url, new=1)
        return url, self.streamlit_process

    def shutdown(self, *, reason: str = "requested") -> None:
        """Terminate only child processes started by this runtime.

        Inputs: tracked Streamlit and optional Ollama child processes.
        Outputs: none.
        Assumptions: an independently running Ollama service is never stopped.
        """

        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._stop_requested.set()
            self._log("shutdown_start reason=%s", reason)
            process = self.streamlit_process
            if process is not None and process.poll() is None:
                self._log("streamlit_terminate streamlit_pid=%s port=%s", process.pid, self.selected_port)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._log("streamlit_kill streamlit_pid=%s", process.pid, level=logging.WARNING)
                    process.kill()
                    process.wait(timeout=2)
            if self.ollama_started_by_session and self.ollama_process is not None:
                self._log("ollama_left_running pid=%s lifecycle=shared", self.ollama_process.pid)
            try:
                if self.paths.active_session_file.is_file():
                    state = json.loads(self.paths.active_session_file.read_text(encoding="utf-8"))
                    if state.get("session_id") == self.session_id:
                        self.paths.active_session_file.unlink(missing_ok=True)
                if self._launcher_lock_acquired and self.paths.launcher_lock_file.is_file():
                    state = json.loads(self.paths.launcher_lock_file.read_text(encoding="utf-8"))
                    if state.get("session_id") == self.session_id:
                        self.paths.launcher_lock_file.unlink(missing_ok=True)
            except (OSError, json.JSONDecodeError):
                self._log("session_state_cleanup_failed", level=logging.WARNING)
            self._shutdown_complete = True
            self._log("shutdown_complete reason=%s port_released=%s", reason, self.selected_port)

    def startup(self) -> tuple[str, subprocess.Popen[bytes]]:
        """Run bounded initialization and return the owned Streamlit session."""

        self.prepare()
        if self._stop_requested.is_set():
            raise DesktopStartupError(StartupStatus(StartupState.INITIALIZING, "Inicio cancelado."))
        self.ensure_ollama_ready()
        if self._stop_requested.is_set():
            raise DesktopStartupError(StartupStatus(StartupState.INITIALIZING, "Inicio cancelado."))
        return self.start_streamlit()

    def run(self) -> int:
        """Run the full desktop lifecycle until the UI exits or receives a signal."""

        _, process = self.startup()

        def stop_handler(_signum: int, _frame: object) -> None:
            self.shutdown(reason=f"signal_{_signum}")

        for signal_name in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signal_name, stop_handler)
        try:
            return int(process.wait())
        finally:
            self.shutdown(reason="streamlit_exit")


def _watch_parent(parent_pid: int) -> None:
    """Exit an orphaned helper immediately when its desktop launcher dies."""

    while True:
        if _parent_watch_should_exit(parent_pid):
            os._exit(0)
        time.sleep(0.5)


def _parent_watch_should_exit(parent_pid: int) -> bool:
    """Return whether the Streamlit helper should self-terminate.

    Inputs: explicit launcher PID provided in helper arguments.
    Outputs: True when the launcher is gone or, on POSIX, the helper was
    re-parented away from that launcher.
    Assumptions: Windows PyInstaller bootloader parentage can be unreliable, so
    Windows relies on the explicit parent PID's liveness instead.
    """

    parent_mismatch = not sys.platform.startswith("win") and os.getppid() != parent_pid
    return parent_mismatch or not _pid_is_alive(parent_pid)


def streamlit_helper_main(arguments: Sequence[str] | None = None) -> int:
    """Run the dedicated bundled Streamlit helper with parent-death protection."""

    values = list(sys.argv[1:] if arguments is None else arguments)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--desktop-session", required=True)
    parser.add_argument("--parent-pid", required=True, type=int)
    internal, streamlit_arguments = parser.parse_known_args(values)
    threading.Thread(target=_watch_parent, args=(internal.parent_pid,), daemon=True).start()
    return _run_streamlit_child(streamlit_arguments)


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
    if arguments and arguments[0] == "--streamlit-child":  # backward compatibility for old bundles.
        return _run_streamlit_child(arguments[1:])
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        from finance_agent.desktop.macos_controller import run_macos_controller

        return run_macos_controller()
    runtime = DesktopRuntime()
    try:
        return runtime.run()
    except DesktopStartupError as exc:
        if runtime.logger and exc.status.technical_detail:
            runtime.logger.error(exc.status.technical_detail)
        _show_error(exc.status.message_es)
        runtime.shutdown(reason="startup_error")
        return 1
    except Exception as exc:  # noqa: BLE001 - desktop users must not see tracebacks.
        if runtime.logger:
            runtime.logger.exception("Unexpected desktop startup error")
        _show_error("Finance AI Agent no pudo iniciarse. Revise el registro técnico e intente nuevamente.")
        runtime.shutdown(reason="unexpected_startup_error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
