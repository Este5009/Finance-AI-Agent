"""Initialize the Finance AI Agent SQLite memory database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.memory import initialize_database  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for memory database initialization.

    Inputs: none.
    Outputs: configured argument parser.
    Assumptions: default path lives under data/memory.
    """

    parser = argparse.ArgumentParser(description="Initialize Finance AI memory DB.")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "memory" / "finance_memory.db",
    )
    return parser


def main() -> None:
    """Initialize the SQLite memory database and print its path.

    Inputs: optional database path.
    Outputs: initialized schema on disk.
    Assumptions: sqlite3 is available in the Python standard library.
    """

    args = build_argument_parser().parse_args()
    path = initialize_database(args.database)
    print(f"Initialized memory database: {path}")


if __name__ == "__main__":
    main()
