"""Safe cleanup helpers for Windows packaged-build locks.

The build step may need to remove ``dist/Finance AI Agent`` while a previous
packaged launcher/helper is still alive. This module targets only executable
processes produced by this repository's PyInstaller build; it deliberately does
not match generic Python, Streamlit, Ollama, or user-owned services.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PACKAGED_PROCESS_NAMES = frozenset(
    {
        "finance ai agent.exe",
        "finance ai agent streamlit.exe",
    }
)


@dataclass(frozen=True)
class ProcessInfo:
    """Small process record used for safe build-lock decisions.

    Inputs: PID, process name, executable path, and command line.
    Outputs: immutable process metadata.
    Assumptions: executable path may be absent for protected/unreadable
    processes, so command line is used only as a secondary ownership clue.
    """

    pid: int
    name: str
    executable_path: str = ""
    command_line: str = ""


def _norm(value: str | Path) -> str:
    """Normalize a filesystem-ish value for case-insensitive Windows checks."""

    return os.path.normcase(os.path.abspath(os.fspath(value)))


def _is_inside(path: str | Path, root: str | Path) -> bool:
    """Return whether ``path`` is inside ``root`` after normalization.

    Inputs: candidate path and trusted project subdirectory.
    Outputs: True when the candidate belongs to that directory.
    Assumptions: Windows comparisons are case-insensitive.
    """

    candidate = _norm(path)
    base = _norm(root)
    try:
        return os.path.commonpath([candidate, base]) == base
    except ValueError:
        return False


def packaged_roots(repo_root: str | Path) -> tuple[Path, Path]:
    """Return repository-owned packaged artifact roots.

    Inputs: repository root.
    Outputs: dist and build roots that may legitimately contain packaged exes.
    Assumptions: only these roots are safe ownership signals for cleanup.
    """

    root = Path(repo_root).resolve()
    return (root / "dist" / "Finance AI Agent", root / "build" / "finance_ai_agent")


def is_project_packaged_process(process: ProcessInfo, repo_root: str | Path) -> bool:
    """Decide whether a process is a stale packaged process from this project.

    Inputs: process metadata and repository root.
    Outputs: True only for known packaged executable names under project build
        or dist roots.
    Assumptions: unrelated Python, Streamlit, and Ollama processes must never
        match because their process names differ from the packaged executables.
    """

    if process.name.strip().casefold() not in PACKAGED_PROCESS_NAMES:
        return False
    roots = packaged_roots(repo_root)
    if process.executable_path and any(_is_inside(process.executable_path, root) for root in roots):
        return True
    command = process.command_line or ""
    return any(str(root).casefold() in command.casefold() for root in roots)


def parse_cim_processes(payload: str) -> list[ProcessInfo]:
    """Parse PowerShell CIM JSON process data.

    Inputs: JSON emitted by ``ConvertTo-Json``.
    Outputs: process records.
    Assumptions: PowerShell returns either one object or a list depending on
        result count.
    """

    if not payload.strip():
        return []
    data = json.loads(payload)
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        return []
    processes: list[ProcessInfo] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item.get("ProcessId") or item.get("process_id") or 0)
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        processes.append(
            ProcessInfo(
                pid=pid,
                name=str(item.get("Name") or item.get("name") or ""),
                executable_path=str(item.get("ExecutablePath") or item.get("executable_path") or ""),
                command_line=str(item.get("CommandLine") or item.get("command_line") or ""),
            )
        )
    return processes


def discover_processes() -> list[ProcessInfo]:
    """Discover local Windows processes with executable path and command line.

    Inputs: none.
    Outputs: process records from PowerShell CIM.
    Assumptions: build cleanup is Windows-focused; non-Windows callers receive
        an empty list so the helper is harmless in cross-platform tests.
    """

    if os.name != "nt":
        return []
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,Name,ExecutablePath,CommandLine | "
            "ConvertTo-Json -Compress"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    if completed.returncode == 0:
        return parse_cim_processes(completed.stdout)
    fallback_command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "Get-Process | Where-Object { $_.ProcessName -like 'Finance AI Agent*' } | "
            "Select-Object @{Name='ProcessId';Expression={$_.Id}},"
            "@{Name='Name';Expression={$_.ProcessName + '.exe'}},"
            "@{Name='ExecutablePath';Expression={$_.Path}},"
            "@{Name='CommandLine';Expression={''}} | ConvertTo-Json -Compress"
        ),
    ]
    fallback = subprocess.run(fallback_command, capture_output=True, text=True, timeout=15, check=False)
    if fallback.returncode != 0:
        raise RuntimeError(
            "No se pudo inspeccionar procesos de Windows. "
            f"CIM: {completed.stderr.strip()} Get-Process: {fallback.stderr.strip()}"
        )
    return parse_cim_processes(fallback.stdout)


def project_packaged_processes(repo_root: str | Path, processes: Iterable[ProcessInfo] | None = None) -> list[ProcessInfo]:
    """Return only packaged Finance AI Agent processes owned by this repo."""

    inventory = list(discover_processes() if processes is None else processes)
    return [process for process in inventory if is_project_packaged_process(process, repo_root)]


def terminate_project_packaged_processes(
    repo_root: str | Path,
    *,
    timeout_seconds: float = 10.0,
    dry_run: bool = False,
) -> list[ProcessInfo]:
    """Terminate stale packaged processes before rebuilding.

    Inputs: repository root, bounded timeout, and dry-run flag.
    Outputs: list of matched processes.
    Assumptions: termination is restricted by ``is_project_packaged_process``.
    """

    matched = project_packaged_processes(repo_root)
    if dry_run:
        return matched
    for process in matched:
        try:
            os.kill(process.pid, signal.SIGTERM)
        except OSError:
            # The process may have already exited between discovery and kill.
            pass
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        remaining = project_packaged_processes(repo_root)
        if not remaining:
            return matched
        time.sleep(0.25)
    remaining = project_packaged_processes(repo_root)
    details = "; ".join(
        f"pid={process.pid} name={process.name} exe={process.executable_path or 'desconocido'}"
        for process in remaining
    )
    raise RuntimeError(f"No se pudieron cerrar procesos empaquetados del proyecto: {details}")


def clean_build_artifacts(repo_root: str | Path) -> None:
    """Remove stale build/dist artifact roots after process cleanup.

    Inputs: repository root.
    Outputs: none.
    Assumptions: only generated PyInstaller artifact directories are removed.
    """

    for path in packaged_roots(repo_root):
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
        except OSError as error:
            remaining = project_packaged_processes(repo_root)
            process_details = "; ".join(
                f"pid={process.pid} name={process.name} exe={process.executable_path or 'desconocido'}"
                for process in remaining
            )
            raise RuntimeError(
                f"No se pudo limpiar {path}: {error}. "
                f"Procesos empaquetados detectados: {process_details or 'ninguno'}"
            ) from error


def cleanup_for_windows_build(repo_root: str | Path, *, timeout_seconds: float = 10.0) -> list[ProcessInfo]:
    """Perform the complete safe pre-build cleanup.

    Inputs: repository root and bounded process-shutdown timeout.
    Outputs: processes that were matched/terminated.
    Assumptions: callers run this before PyInstaller attempts to collect dist.
    """

    # If neither generated artifact root exists, there is nothing for a stale
    # packaged executable to lock for this repository. This also keeps dry test
    # environments from requiring privileged process enumeration.
    if not any(path.exists() for path in packaged_roots(repo_root)):
        return []
    matched = terminate_project_packaged_processes(repo_root, timeout_seconds=timeout_seconds)
    clean_build_artifacts(repo_root)
    return matched
