"""Run the full project test suite with a hard timeout.

The script gives Codex a bounded, repeatable command for validation. If pytest
hangs or exceeds the configured budget, the process is terminated and the last
captured output is printed for diagnosis.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TIMEOUT_SECONDS = 600


def _repo_root() -> Path:
    """Return the repository root inferred from this script location."""

    return Path(__file__).resolve().parents[1]


def run_tests(pytest_args: list[str], timeout_seconds: int) -> int:
    """Run pytest with a hard timeout and return a process-style exit code."""

    command = [sys.executable, "-m", "pytest", "--rootdir", str(_repo_root()), *pytest_args]
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
        _safe_print(f"Project tests timed out after {timeout_seconds} seconds.", stream=sys.stderr)
        _safe_print("\n".join((output or "").splitlines()[-120:]), stream=sys.stderr)
        return 124

    _safe_print(output or "")
    return process.returncode


def _safe_print(text: str, *, stream: object | None = None) -> None:
    """Print captured test output without crashing on Windows code pages.

    Inputs: text and optional output stream.
    Outputs: writes text to stdout/stderr.
    Assumptions: diagnostic output may contain replacement characters or
    Unicode from tests; failing to print it should never mask pytest's result.
    """

    target = stream or sys.stdout
    try:
        print(text, file=target)  # type: ignore[arg-type]
    except UnicodeEncodeError:
        encoding = getattr(target, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe, file=target)  # type: ignore[arg-type]


def main() -> int:
    """Parse CLI arguments and run the bounded project test suite."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    pytest_args = args.pytest_args if args.pytest_args else ["-q"]
    return run_tests(pytest_args, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
