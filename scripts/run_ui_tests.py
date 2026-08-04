"""Run bounded UI validation without launching Streamlit.

This helper is the safe Codex path for Streamlit-related changes. It imports
the app module and runs focused tests/artifact checks, but it never starts a
persistent web server.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_TESTS = [
    "tests/test_streamlit_ui.py",
    "tests/test_report_renderer.py",
]


def _repo_root() -> Path:
    """Return the repository root inferred from this script location."""

    return Path(__file__).resolve().parents[1]


def _run_pytest(test_args: list[str], timeout_seconds: int) -> int:
    """Run pytest with a hard timeout and print the last captured output on timeout."""

    command = [sys.executable, "-m", "pytest", "--rootdir", str(_repo_root()), *test_args]
    process = subprocess.Popen(
        command,
        cwd=_repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()
        print(f"UI tests timed out after {timeout_seconds} seconds.", file=sys.stderr)
        print("\n".join((output or "").splitlines()[-80:]), file=sys.stderr)
        return 124

    print(output or "")
    return process.returncode


def main() -> int:
    """Import the Streamlit app and run focused bounded UI tests."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Importing catches broken module-level dependencies without launching a UI.
    import finance_agent.ui.streamlit_app  # noqa: F401

    test_args = args.pytest_args if args.pytest_args else [*DEFAULT_TESTS, "-q"]
    return _run_pytest(test_args, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
