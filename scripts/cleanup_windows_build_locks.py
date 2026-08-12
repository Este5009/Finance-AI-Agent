"""Pre-build cleanup for Windows PyInstaller artifact locks.

This script is intentionally finite and non-interactive. It terminates only
Finance AI Agent packaged executables whose executable path or command line is
inside this repository's generated build/dist roots, then removes those
generated artifact directories before PyInstaller rebuilds them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root_from_file() -> Path:
    """Return the repository root when the script is executed directly."""

    return Path(__file__).resolve().parents[1]


ROOT = _repo_root_from_file()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finance_agent.desktop.build_cleanup import cleanup_for_windows_build  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run safe Windows build cleanup from the command line.

    Inputs: optional CLI arguments.
    Outputs: process exit code.
    Assumptions: non-Windows runs are harmless because no process inventory is
    returned by the cleanup helper.
    """

    parser = argparse.ArgumentParser(description="Clean Finance AI Agent Windows build locks.")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        matched = cleanup_for_windows_build(args.repo_root, timeout_seconds=args.timeout_seconds)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if matched:
        print("Procesos empaquetados cerrados antes de compilar:")
        for process in matched:
            print(f"- pid={process.pid} name={process.name} exe={process.executable_path or 'desconocido'}")
    else:
        print("No se detectaron procesos empaquetados del proyecto bloqueando la compilación.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
