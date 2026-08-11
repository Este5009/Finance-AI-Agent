"""Tests for the repository's bounded development workflow helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run a command from the repository root with bounded execution."""

    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def test_agents_persistent_process_policy_is_documented() -> None:
    """AGENTS.md must contain the permanent anti-hang rules."""

    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "Persistent Process and Timeout Policy" in text
    assert "Codex must never call `Start-Process`" in text
    assert "Validate server availability only when it is already running" in text
    assert "Starting Streamlit is not part of Codex task completion." in text
    assert "Never stage `outputs/`, databases, uploaded files, caches" in text


def test_windows_launcher_is_foreground_and_parseable() -> None:
    """The Windows launcher must avoid Start-Process and have valid PowerShell syntax."""

    script = ROOT / "scripts" / "start_streamlit_windows.ps1"
    text = script.read_text(encoding="utf-8")
    assert "Start-Process" not in text
    assert "-m streamlit run" in text
    assert "--server.port" in text
    assert "http://localhost:$port" in text

    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is not available in this environment.")

    command = [
        powershell,
        "-NoProfile",
        "-Command",
        (
            "$errors = $null; "
            f"[System.Management.Automation.PSParser]::Tokenize((Get-Content -Raw '{script}'), [ref]$errors) | Out-Null; "
            "if ($errors.Count -gt 0) { $errors | ForEach-Object { Write-Output $_.Message }; exit 1 }"
        ),
    ]
    result = _run(command, timeout=10)
    assert result.returncode == 0, result.stdout


def test_macos_launcher_is_foreground_and_parseable() -> None:
    """The macOS/Linux launcher must be syntax-valid and foreground-only."""

    script = ROOT / "scripts" / "start_streamlit_macos.sh"
    text = script.read_text(encoding="utf-8")
    assert "nohup" not in text
    assert "Start-Process" not in text
    assert "exec \"${python_executable}\" -m streamlit run" in text
    assert "http://localhost:${port}" in text

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available in this environment.")

    result = _run([bash, "-n", str(script)], timeout=10)
    assert result.returncode == 0, result.stdout


def test_manual_app_launchers_include_ollama_readiness_and_parse() -> None:
    """Manual app launchers should delegate to the canonical desktop runtime."""

    windows_script = ROOT / "scripts" / "start_app_windows.ps1"
    macos_script = ROOT / "scripts" / "start_app_macos.sh"
    windows_text = windows_script.read_text(encoding="utf-8")
    macos_text = macos_script.read_text(encoding="utf-8")

    assert "-m finance_agent.desktop" in windows_text
    assert "-m finance_agent.desktop" in macos_text
    assert "ollama" not in windows_text.casefold()
    assert "ollama" not in macos_text.casefold()
    assert "streamlit run" not in windows_text
    assert "streamlit run" not in macos_text
    assert "$LASTEXITCODE" in windows_text
    assert "exec" in macos_text
    assert "estÃ" not in windows_text + macos_text
    assert "Â¿" not in windows_text + macos_text

    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is not None:
        command = [
            powershell,
            "-NoProfile",
            "-Command",
            (
                "$errors = $null; "
                f"[System.Management.Automation.PSParser]::Tokenize((Get-Content -Raw '{windows_script}'), [ref]$errors) | Out-Null; "
                "if ($errors.Count -gt 0) { $errors | ForEach-Object { Write-Output $_.Message }; exit 1 }"
            ),
        ]
        result = _run(command, timeout=10)
        assert result.returncode == 0, result.stdout

    bash = shutil.which("bash")
    if bash is not None:
        result = _run([bash, "-n", str(macos_script)], timeout=10)
        assert result.returncode == 0, result.stdout


def test_check_local_services_finishes_within_budget() -> None:
    """The local service checker should report status without starting services."""

    started = time.monotonic()
    result = _run([sys.executable, "scripts/check_local_services.py", "--timeout", "3"], timeout=10)
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stdout
    assert elapsed < 10
    assert "Streamlit" in result.stdout
    assert "Ollama" in result.stdout


def test_ui_test_runner_can_run_a_single_focus_test() -> None:
    """The UI test helper should import the app and run focused tests without launching a server."""

    result = _run(
        [
            sys.executable,
            "scripts/run_ui_tests.py",
            "--timeout",
            "60",
            "--pytest-args",
            "tests/test_streamlit_ui.py::test_streamlit_ui_imports_without_streamlit_dependency",
            "-q",
        ],
        timeout=90,
    )
    assert result.returncode == 0, result.stdout


def test_project_test_runner_timeout_behavior(tmp_path: Path) -> None:
    """The full-suite helper should terminate pytest cleanly when its timeout is reached."""

    slow_test = tmp_path / "test_slow_timeout_contract.py"
    slow_test.write_text(
        "import time\n\n"
        "def test_slow_timeout_contract():\n"
        "    time.sleep(2)\n",
        encoding="utf-8",
    )

    result = _run(
        [
            sys.executable,
            "scripts/run_project_tests.py",
            "--timeout",
            "1",
            "--pytest-args",
            str(slow_test),
            "-q",
        ],
        timeout=10,
    )
    assert result.returncode == 124
    assert "timed out after 1 seconds" in result.stdout
