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
    _parent_watch_should_exit,
    _streamlit_environment,
    _streamlit_helper_executable,
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

    next_pid = 41000

    def __init__(self, returncode: int | None = None) -> None:
        FakeProcess.next_pid += 1
        self.pid = FakeProcess.next_pid
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

    def __init__(self, *, available: bool = True, models: tuple[str, ...] = ("qwen3:8b",), health_ok: bool = True, **_: object) -> None:
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
    runtime.shutdown(reason="test_cleanup")
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


def test_legacy_desktop_default_migrates_to_balanced_model(tmp_path: Path) -> None:
    """The auto-generated legacy 30B config migrates without overriding QUALITY."""

    path = tmp_path / "config.json"
    path.write_text('{"model":"qwen3:30b-a3b","open_browser":false}', encoding="utf-8")

    config = DesktopConfig.load(path)

    assert config.model == "qwen3:8b"
    assert config.model_tier == "BALANCED"
    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "qwen3:8b"


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
    runtime.shutdown(reason="test_cleanup")


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
    assert not fake_process.terminated


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
    runtime.shutdown(reason="test_cleanup")


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
    runtime.shutdown(reason="test_cleanup")


def test_startup_timeout_terminates_streamlit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamlit readiness timeout stops the owned child process."""

    fake = FakeProcess()
    monkeypatch.setattr("finance_agent.desktop.runtime._is_http_ready", lambda *_args, **_kwargs: False)
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


def test_streamlit_child_exit_is_reported_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashed packaged helper is not misreported as a generic startup timeout."""

    fake = FakeProcess(returncode=3)
    monkeypatch.setattr("finance_agent.desktop.runtime._is_http_ready", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("finance_agent.desktop.runtime.find_free_port", lambda *_args, **_kwargs: 8501)
    runtime = DesktopRuntime(
        paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}),
        config=DesktopConfig(open_browser=False, startup_timeout_seconds=3),
        process_factory=lambda *_args, **_kwargs: fake,
    )
    runtime.prepare()
    runtime.paths.streamlit_log_file.write_text("Traceback: helper failed to import streamlit\n", encoding="utf-8")

    with pytest.raises(DesktopStartupError) as caught:
        runtime.start_streamlit()

    assert "exit_code=3" in caught.value.status.technical_detail
    assert "helper failed" in caught.value.status.technical_detail


def test_streamlit_child_output_uses_separate_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamlit stdout must not corrupt structured launcher lifecycle records."""

    fake = FakeProcess()
    captured: dict[str, object] = {}

    def fake_factory(*_args: object, **kwargs: object) -> FakeProcess:
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("finance_agent.desktop.runtime._is_http_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("finance_agent.desktop.runtime.find_free_port", lambda *_args, **_kwargs: 8501)
    runtime = DesktopRuntime(
        paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}),
        config=DesktopConfig(open_browser=False),
        process_factory=fake_factory,
    )
    runtime.prepare()
    runtime.start_streamlit()

    stream = captured["stdout"]
    assert getattr(stream, "name", "") == str(runtime.paths.streamlit_log_file)
    runtime.shutdown(reason="test_cleanup")


def test_graceful_shutdown_stops_streamlit_but_leaves_ollama(tmp_path: Path) -> None:
    """Shutdown stops the session helper and leaves shared Ollama available."""

    runtime = DesktopRuntime(paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)}), config=DesktopConfig(open_browser=False))
    runtime.streamlit_process = FakeProcess()  # type: ignore[assignment]
    runtime.ollama_process = FakeProcess()  # type: ignore[assignment]
    runtime.ollama_started_by_session = True
    runtime.shutdown()
    assert runtime.streamlit_process.terminated
    assert not runtime.ollama_process.terminated


def test_frozen_streamlit_command_uses_packaged_executable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Packaged startup does not require a separately installed system Python."""

    monkeypatch.setattr("finance_agent.desktop.runtime.sys.frozen", True, raising=False)
    command = _streamlit_command(8501, "127.0.0.1", session_id="abc123", parent_pid=99)
    expected_name = "Finance AI Agent Streamlit.exe" if os.name == "nt" else "Finance AI Agent Streamlit"
    assert Path(command[0]).name == expected_name
    assert command[1:5] == ["--desktop-session", "abc123", "--parent-pid", "99"]
    assert "-m" not in command
    assert command[command.index("--global.developmentMode") + 1] == "false"


def test_windows_frozen_helper_path_uses_exe_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows PyInstaller helpers must be invoked through the concrete .exe file."""

    monkeypatch.setattr("finance_agent.desktop.runtime.sys.platform", "win32")
    monkeypatch.setattr("finance_agent.desktop.runtime.sys.executable", r"C:\App\Finance AI Agent.exe")

    assert str(_streamlit_helper_executable()).endswith(r"Finance AI Agent Streamlit.exe")


def test_windows_parent_watch_ignores_pyinstaller_ppid_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frozen Windows helper must not exit solely because bootloader parentage differs."""

    monkeypatch.setattr("finance_agent.desktop.runtime.sys.platform", "win32")
    monkeypatch.setattr("finance_agent.desktop.runtime.os.getppid", lambda: 111)
    monkeypatch.setattr("finance_agent.desktop.runtime._pid_is_alive", lambda pid: pid == 222)

    assert _parent_watch_should_exit(222) is False
    assert _parent_watch_should_exit(333) is True


def test_frozen_startup_fails_before_timeout_when_helper_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad packaged helper path is reported directly instead of as readiness timeout."""

    bundle = tmp_path / "bundle"
    app = bundle / "finance_agent" / "ui" / "streamlit_app.py"
    app.parent.mkdir(parents=True)
    app.write_text("# bundled app", encoding="utf-8")
    executable = tmp_path / "Finance AI Agent.exe"
    executable.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("FINANCE_AI_RESOURCE_ROOT", str(bundle))
    monkeypatch.setattr("finance_agent.desktop.runtime.sys.frozen", True, raising=False)
    monkeypatch.setattr("finance_agent.desktop.runtime.sys.platform", "win32")
    monkeypatch.setattr("finance_agent.desktop.runtime.sys.executable", str(executable))
    monkeypatch.setattr("finance_agent.desktop.runtime.find_free_port", lambda *_args, **_kwargs: 8501)

    runtime = DesktopRuntime(
        paths=app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path / "app-data")}),
        config=DesktopConfig(open_browser=False),
    )
    runtime.prepare()

    with pytest.raises(DesktopStartupError) as caught:
        runtime.start_streamlit()

    assert "missing_streamlit_helper" in caught.value.status.technical_detail
    runtime.shutdown(reason="test_cleanup")


def test_streamlit_child_environment_forces_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    """Packaged child logs keep Spanish text readable on Windows consoles."""

    monkeypatch.delenv("PYTHONUTF8", raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)

    environment = _streamlit_environment()

    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"


def test_duplicate_launcher_is_blocked_before_streamlit_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second double-click during slow startup does not create a competing session."""

    paths = app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)})
    paths.initialize()
    paths.launcher_lock_file.write_text(
        json.dumps({"session_id": "first", "launcher_pid": 1200, "created_at": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr("finance_agent.desktop.runtime._pid_is_alive", lambda pid: pid == 1200)
    monkeypatch.setattr("finance_agent.desktop.runtime._process_command", lambda _pid: "Finance AI Agent.exe")

    runtime = DesktopRuntime(paths=paths, config=DesktopConfig(open_browser=False))

    with pytest.raises(DesktopStartupError) as caught:
        runtime.prepare()

    assert "ya se está iniciando" in caught.value.status.message_es


def test_stale_launcher_lock_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashed launcher leaves a lock that the next startup can safely replace."""

    paths = app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)})
    paths.initialize()
    paths.launcher_lock_file.write_text(
        json.dumps({"session_id": "dead", "launcher_pid": 1200, "created_at": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr("finance_agent.desktop.runtime._pid_is_alive", lambda _pid: False)

    runtime = DesktopRuntime(paths=paths, config=DesktopConfig(open_browser=False))
    runtime.prepare()

    lock = json.loads(paths.launcher_lock_file.read_text(encoding="utf-8"))
    assert lock["session_id"] == runtime.session_id
    runtime.shutdown(reason="test_cleanup")


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
    assert 'packaging" / "streamlit_helper.py"' in spec
    assert 'name="Finance AI Agent Streamlit"' in spec


def test_repeated_launch_shutdown_creates_fresh_sessions_and_releases_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three sessions select ports anew and remove their ownership state."""

    ports = iter((49101, 49102, 49103))
    processes: list[FakeProcess] = []
    monkeypatch.setattr("finance_agent.desktop.runtime.find_free_port", lambda *_args: next(ports))
    monkeypatch.setattr("finance_agent.desktop.runtime._is_http_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("finance_agent.desktop.runtime.webbrowser.open", lambda *_args, **_kwargs: True)
    paths = app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)})
    sessions: list[str] = []
    selected: list[int | None] = []
    for _ in range(3):
        process = FakeProcess()
        processes.append(process)
        runtime = DesktopRuntime(
            paths=paths,
            config=DesktopConfig(open_browser=False),
            process_factory=lambda *_args, _process=process, **_kwargs: _process,
        )
        runtime.prepare()
        runtime.start_streamlit()
        sessions.append(runtime.session_id)
        selected.append(runtime.selected_port)
        assert paths.active_session_file.is_file()
        runtime.shutdown(reason="test_cycle")
        assert process.terminated
        assert not paths.active_session_file.exists()
    assert len(set(sessions)) == 3
    assert selected == [49101, 49102, 49103]


def test_stale_session_recovers_only_verified_streamlit_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead launcher permits cleanup only for its matching named helper."""

    paths = app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)})
    paths.initialize()
    paths.active_session_file.write_text(
        json.dumps({"session_id": "old-session", "launcher_pid": 1200, "streamlit_pid": 1300}),
        encoding="utf-8",
    )
    alive = {1300: True}
    killed: list[tuple[int, object]] = []

    def fake_kill(pid: int, signum: object) -> None:
        killed.append((pid, signum))
        alive[pid] = False

    monkeypatch.setattr("finance_agent.desktop.runtime._pid_is_alive", lambda pid: alive.get(pid, False))
    monkeypatch.setattr(
        "finance_agent.desktop.runtime._process_command",
        lambda pid: "Finance AI Agent Streamlit --desktop-session old-session" if pid == 1300 else "",
    )
    monkeypatch.setattr("finance_agent.desktop.runtime.os.kill", fake_kill)
    runtime = DesktopRuntime(paths=paths, config=DesktopConfig(open_browser=False))
    runtime.prepare()
    assert killed and killed[0][0] == 1300
    assert json.loads(paths.active_session_file.read_text(encoding="utf-8"))["session_id"] == runtime.session_id


def test_stale_state_does_not_kill_unrelated_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PID reuse cannot cause an unrelated process to be terminated."""

    paths = app_data_paths(environ={"FINANCE_AI_APP_DATA": str(tmp_path)})
    paths.initialize()
    paths.active_session_file.write_text(
        json.dumps({"session_id": "old-session", "launcher_pid": 1200, "streamlit_pid": 1300}),
        encoding="utf-8",
    )
    monkeypatch.setattr("finance_agent.desktop.runtime._pid_is_alive", lambda pid: pid == 1300)
    monkeypatch.setattr("finance_agent.desktop.runtime._process_command", lambda _pid: "/usr/bin/unrelated")
    monkeypatch.setattr("finance_agent.desktop.runtime.os.kill", lambda *_args: pytest.fail("must not kill unrelated PID"))
    runtime = DesktopRuntime(paths=paths, config=DesktopConfig(open_browser=False))
    runtime.prepare()
    assert json.loads(paths.active_session_file.read_text(encoding="utf-8"))["session_id"] == runtime.session_id
