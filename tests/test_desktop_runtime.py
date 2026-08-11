"""Tests for desktop distribution paths and startup orchestration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from finance_agent.desktop.config import DesktopConfig
from finance_agent.desktop.paths import app_data_paths, app_data_root, resource_path
from finance_agent.desktop.runtime import (
    DesktopRuntime,
    DesktopStartupError,
    StartupState,
    _streamlit_command,
    find_free_port,
)


@pytest.fixture(autouse=True)
def restore_desktop_environment() -> object:
    """Prevent runtime environment variables from leaking between tests."""

    names = ("FINANCE_AI_APP_DATA", "FINANCE_AI_RESOURCE_ROOT", "FINANCE_AI_OUTPUT_DIR", "FINANCE_AI_UPLOAD_DIR", "FINANCE_AI_MEMORY_DB")
    original = {name: os.environ.get(name) for name in names}
    yield
    for name, value in original.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class FakeProcess:
    """Minimal controllable subprocess double."""

    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        """Return the configured process state."""

        return self.returncode

    def terminate(self) -> None:
        """Record graceful termination."""

        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        """Record forced termination."""

        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        """Return the final process code."""

        return int(self.returncode or 0)


class FakeClient:
    """Ollama client double supporting the shared readiness contract."""

    def __init__(self, *, available: bool = True, models: tuple[str, ...] = ("qwen3:30b-a3b",), health_ok: bool = True, **_: object) -> None:
        self.available = available
        self.models = models
        self.health_ok = health_ok

    def is_available(self) -> bool:
        """Return fake service availability."""

        return self.available

    def list_models(self) -> list[str]:
        """Return fake installed models."""

        return list(self.models)

    def model_exists(self, model: str) -> bool:
        """Return exact fake model presence."""

        return model in self.models

    def health_prompt(self) -> dict[str, object]:
        """Return or reject one fake health prompt."""

        if not self.health_ok:
            raise RuntimeError("health failed")
        return {"response": {"ok": True}, "telemetry": {}}


def test_app_data_path_resolution_windows_and_macos(tmp_path: Path) -> None:
    """Windows/macOS paths follow the required production conventions."""

    windows = app_data_root(platform="win32", environ={"LOCALAPPDATA": str(tmp_path / "Local")}, home=tmp_path)
    macos = app_data_root(platform="darwin", environ={}, home=tmp_path)
    assert windows == tmp_path / "Local" / "FinanceAIAgent"
    assert macos == tmp_path / "Library" / "Application Support" / "FinanceAIAgent"


def test_resource_path_is_independent_of_current_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bundled resources resolve from the supplied bundle root, not cwd."""

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert resource_path("finance_agent", "memory", "schema.sql", frozen_root=bundle) == bundle / "finance_agent" / "memory" / "schema.sql"


def test_runtime_prepares_outside_repository_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production writable paths initialize correctly from an unrelated cwd."""

    elsewhere = tmp_path / "unrelated"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    paths = app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path / "app-data")})
    runtime = DesktopRuntime(paths=paths, config=DesktopConfig(open_browser=False))
    runtime.prepare()
    assert paths.memory_database.is_file()
    assert Path(os.environ["FINANCE_AI_OUTPUT_DIR"]) == paths.outputs
    assert not str(paths.root).startswith(str(Path(__file__).resolve().parents[1]))


def test_initialize_preserves_existing_user_data(tmp_path: Path) -> None:
    """Repeated app-data initialization never deletes database/config contents."""

    paths = app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path / "app")})
    paths.initialize()
    paths.memory_database.write_bytes(b"existing-db")
    paths.config_file.write_text('{"model":"custom"}', encoding="utf-8")
    paths.initialize()
    assert paths.memory_database.read_bytes() == b"existing-db"
    assert json.loads(paths.config_file.read_text(encoding="utf-8"))["model"] == "custom"


def test_port_collision_selects_free_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    """A busy preferred port causes safe local fallback selection."""

    class FakeSocket:
        """Socket double that rejects the preferred port once."""

        calls = 0

        def __init__(self, *_args: object) -> None:
            """Accept the normal socket constructor arguments."""

        def __enter__(self) -> "FakeSocket":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def setsockopt(self, *_args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            FakeSocket.calls += 1
            if address[1] == 8501:
                raise OSError("busy")

        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", 49152)

    monkeypatch.setattr("finance_agent.desktop.runtime.socket.socket", FakeSocket)
    assert find_free_port("127.0.0.1", 8501) == 49152


def test_ollama_missing_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Ollama is reported as a readable first-run state."""

    runtime = DesktopRuntime(paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}), config=DesktopConfig(open_browser=False))
    runtime.prepare()
    monkeypatch.setattr("finance_agent.desktop.runtime.find_ollama_executable", lambda: None)
    with pytest.raises(DesktopStartupError) as caught:
        runtime.ensure_ollama_ready()
    assert caught.value.status.state == StartupState.OLLAMA_MISSING


def test_model_installed_and_ai_ready(tmp_path: Path) -> None:
    """A running service with the configured model reaches AI_READY."""

    statuses = []
    runtime = DesktopRuntime(
        paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}),
        config=DesktopConfig(open_browser=False),
        status_callback=statuses.append,
        ollama_executable=tmp_path / "ollama",
        client_factory=lambda **kwargs: FakeClient(**kwargs),
    )
    runtime.prepare()
    status = runtime.ensure_ollama_ready()
    assert status.state == StartupState.AI_READY
    assert statuses[-1].state == StartupState.AI_READY


def test_ollama_stopped_is_started_then_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed but stopped Ollama service is started before readiness."""

    class StartingClient(FakeClient):
        """Return stopped once and running for the shared readiness check."""

        checks = 0

        def is_available(self) -> bool:
            StartingClient.checks += 1
            return StartingClient.checks > 1

    fake_process = FakeProcess()
    statuses = []
    monkeypatch.setattr("finance_agent.desktop.runtime.wait_for_http", lambda *_args, **_kwargs: True)
    runtime = DesktopRuntime(
        paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}),
        config=DesktopConfig(open_browser=False),
        status_callback=statuses.append,
        ollama_executable=tmp_path / "ollama",
        client_factory=lambda **kwargs: StartingClient(**kwargs),
        process_factory=lambda *_args, **_kwargs: fake_process,
    )
    runtime.prepare()
    assert runtime.ensure_ollama_ready().state == StartupState.AI_READY
    assert StartupState.OLLAMA_STOPPED in {status.state for status in statuses}
    runtime.shutdown()
    assert fake_process.terminated


def test_model_missing_requires_explicit_consent(tmp_path: Path) -> None:
    """A multi-GB model is never downloaded when consent is declined."""

    runtime = DesktopRuntime(
        paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}),
        config=DesktopConfig(open_browser=False),
        ollama_executable=tmp_path / "ollama",
        client_factory=lambda **kwargs: FakeClient(models=(), **kwargs),
        consent_callback=lambda _title, _message: False,
    )
    runtime.prepare()
    with pytest.raises(DesktopStartupError) as caught:
        runtime.ensure_ollama_ready()
    assert caught.value.status.state == StartupState.MODEL_DOWNLOAD_REQUIRED


def test_ai_health_failure_is_readable(tmp_path: Path) -> None:
    """Failed bounded inference is classified without exposing a traceback."""

    runtime = DesktopRuntime(
        paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}),
        config=DesktopConfig(open_browser=False),
        ollama_executable=tmp_path / "ollama",
        client_factory=lambda **kwargs: FakeClient(health_ok=False, **kwargs),
    )
    runtime.prepare()
    with pytest.raises(DesktopStartupError) as caught:
        runtime.ensure_ollama_ready()
    assert caught.value.status.state == StartupState.AI_HEALTH_FAILED


def test_startup_timeout_terminates_streamlit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamlit readiness timeout stops the owned child process."""

    fake = FakeProcess()
    monkeypatch.setattr("finance_agent.desktop.runtime.wait_for_http", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("finance_agent.desktop.runtime.find_free_port", lambda *_args, **_kwargs: 8501)
    runtime = DesktopRuntime(
        paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}),
        config=DesktopConfig(open_browser=False, startup_timeout_seconds=0.01),
        process_factory=lambda *_args, **_kwargs: fake,
    )
    runtime.prepare()
    with pytest.raises(DesktopStartupError):
        runtime.start_streamlit()
    assert fake.terminated


def test_graceful_shutdown_stops_only_owned_children(tmp_path: Path) -> None:
    """Graceful shutdown terminates tracked live child processes."""

    runtime = DesktopRuntime(paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}), config=DesktopConfig(open_browser=False))
    runtime.streamlit_process = FakeProcess()  # type: ignore[assignment]
    runtime.ollama_process = FakeProcess()  # type: ignore[assignment]
    runtime.shutdown()
    assert runtime.streamlit_process.terminated
    assert runtime.ollama_process.terminated


def test_frozen_streamlit_command_uses_packaged_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Packaged startup does not require a separately installed system Python."""

    monkeypatch.setattr("finance_agent.desktop.runtime.sys.frozen", True, raising=False)
    command = _streamlit_command(8501, "127.0.0.1")
    assert command[:2] == [str(Path(__import__("sys").executable)), "--streamlit-child"]
    assert "-m" not in command
    assert command[command.index("--global.developmentMode") + 1] == "false"


def test_packaging_config_excludes_private_runtime_data() -> None:
    """PyInstaller source config must not bundle financial/runtime directories."""

    root = Path(__file__).resolve().parents[1]
    spec = (root / "packaging" / "finance_ai_agent.spec").read_text(encoding="utf-8")
    assert 'ROOT / "data"' not in spec
    assert 'ROOT / "outputs"' not in spec
    assert ".env" in spec
    assert "schema.sql" in spec
    assert "streamlit_app.py" in spec
    assert 'collect_submodules("finance_agent")' in spec
