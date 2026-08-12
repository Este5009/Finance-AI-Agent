"""Regression tests for Windows packaged-build cleanup safety."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from finance_agent.desktop.build_cleanup import (
    ProcessInfo,
    is_project_packaged_process,
    parse_cim_processes,
    project_packaged_processes,
)


def test_project_packaged_process_matching_is_path_scoped(tmp_path: Path) -> None:
    """Verify only repo-owned packaged executables are build-cleanup targets."""

    repo = tmp_path / "repo"
    owned = repo / "dist" / "Finance AI Agent" / "Finance AI Agent.exe"
    unrelated = tmp_path / "other" / "Finance AI Agent.exe"
    processes = [
        ProcessInfo(101, "Finance AI Agent.exe", str(owned), str(owned)),
        ProcessInfo(102, "Finance AI Agent.exe", str(unrelated), str(unrelated)),
        ProcessInfo(103, "Finance AI Agent Streamlit.exe", str(repo / "build" / "finance_ai_agent" / "Finance AI Agent Streamlit.exe"), ""),
    ]

    matched = project_packaged_processes(repo, processes)

    assert [process.pid for process in matched] == [101, 103]
    assert is_project_packaged_process(processes[1], repo) is False


def test_cleanup_never_matches_ollama_python_or_streamlit(tmp_path: Path) -> None:
    """Verify generic service names cannot be terminated by build cleanup."""

    repo = tmp_path / "repo"
    processes = [
        ProcessInfo(201, "ollama.exe", str(repo / "dist" / "Finance AI Agent" / "ollama.exe"), ""),
        ProcessInfo(202, "python.exe", str(repo / ".venv" / "Scripts" / "python.exe"), "python -m streamlit run app"),
        ProcessInfo(203, "streamlit.exe", str(repo / ".venv" / "Scripts" / "streamlit.exe"), "streamlit run app"),
    ]

    assert project_packaged_processes(repo, processes) == []


def test_parse_cim_process_payload_accepts_single_or_multiple_objects() -> None:
    """Verify PowerShell JSON shape differences are normalized."""

    single = '{"ProcessId":301,"Name":"Finance AI Agent.exe","ExecutablePath":"C:/repo/dist/Finance AI Agent/Finance AI Agent.exe","CommandLine":""}'
    multiple = "[" + single + ',{"ProcessId":302,"Name":"python.exe","ExecutablePath":"C:/Python/python.exe","CommandLine":""}]'

    assert parse_cim_processes(single)[0].pid == 301
    assert [process.pid for process in parse_cim_processes(multiple)] == [301, 302]


def test_cleanup_script_imports_from_arbitrary_cwd(tmp_path: Path) -> None:
    """Verify the standalone cleanup script can import the project package."""

    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "cleanup_windows_build_locks.py"),
            "--repo-root",
            str(tmp_path / "isolated_repo"),
            "--timeout-seconds",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "procesos empaquetados" in completed.stdout.casefold()


def test_build_windows_invokes_safe_cleanup_before_pyinstaller() -> None:
    """Verify the Windows build script runs cleanup before PyInstaller."""

    repo = Path(__file__).resolve().parents[1]
    script = (repo / "scripts" / "build_windows.ps1").read_text(encoding="utf-8")

    cleanup_index = script.index("cleanup_windows_build_locks.py")
    pyinstaller_index = script.index("-m PyInstaller")
    assert cleanup_index < pyinstaller_index
    assert "Stop-Process" not in script
    assert "ollama" not in script.casefold()


@pytest.mark.parametrize(
    "name",
    ["Finance AI Agent.exe", "Finance AI Agent Streamlit.exe"],
)
def test_allowed_packaged_process_names_are_exact(tmp_path: Path, name: str) -> None:
    """Verify packaged executable names are the only positive name matches."""

    repo = tmp_path / "repo"
    process = ProcessInfo(401, name, str(repo / "dist" / "Finance AI Agent" / name), "")

    assert is_project_packaged_process(process, repo) is True
