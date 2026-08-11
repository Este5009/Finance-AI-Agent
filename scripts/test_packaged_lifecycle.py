"""Bounded smoke test for repeated lifecycle of the actual macOS bundle."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path


def process_alive(pid: int) -> bool:
    """Return whether a process still exists."""

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_until(predicate, timeout: float) -> bool:
    """Poll a bounded condition using monotonic time."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


def main() -> None:
    """Launch and stop the frozen application repeatedly."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--app-data", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    executable = args.app / "Contents" / "MacOS" / "Finance AI Agent"
    config_path = args.app_data / "config" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({
        "model": "qwen3:8b", "model_tier": "BALANCED", "open_browser": False,
        "preferred_port": 8501, "startup_timeout_seconds": 45,
    }), encoding="utf-8")
    env = {**os.environ, "FINANCE_AI_APP_DATA": str(args.app_data)}
    summaries = []
    for cycle in range(1, args.cycles + 1):
        started = time.monotonic()
        launcher = subprocess.Popen([str(executable)], env=env)
        state_path = args.app_data / "runtime" / "active-session.json"

        def ready() -> bool:
            if not state_path.is_file():
                return False
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                return state.get("phase") == "ready"
            except (OSError, json.JSONDecodeError):
                return False

        if not wait_until(ready, 60):
            launcher.terminate()
            raise RuntimeError(f"cycle {cycle}: packaged app did not become ready")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        streamlit_pid = int(state["streamlit_pid"])
        port = int(state["port"])
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
            if response.status != 200:
                raise RuntimeError(f"cycle {cycle}: health returned {response.status}")
        launcher.send_signal(signal.SIGTERM)
        launcher.wait(timeout=15)
        child_gone = wait_until(lambda: not process_alive(streamlit_pid), 10)

        def port_released() -> bool:
            with socket.socket() as probe:
                return probe.connect_ex(("127.0.0.1", port)) != 0

        port_free = wait_until(port_released, 10)
        stale_state = state_path.exists()
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as response:
            ollama_ok = response.status == 200
        summary = {
            "cycle": cycle, "session_id": state.get("session_id"),
            "launcher_pid": launcher.pid, "streamlit_pid": streamlit_pid, "port": port,
            "startup_seconds": time.monotonic() - started, "launcher_exit_code": launcher.returncode,
            "streamlit_gone": child_gone, "port_released": port_free,
            "stale_session_state": stale_state, "ollama_usable": ollama_ok,
        }
        summaries.append(summary)
        if not child_gone or not port_free or stale_state or not ollama_ok:
            raise RuntimeError(json.dumps(summary))
    print(json.dumps({"cycles": summaries}, indent=2))


if __name__ == "__main__":
    main()
