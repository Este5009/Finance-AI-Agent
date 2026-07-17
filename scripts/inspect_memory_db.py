"""Inspect Finance AI Agent SQLite memory database counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.memory.repository import MemoryRepository  # noqa: E402


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for memory DB inspection.

    Inputs: none.
    Outputs: configured parser.
    Assumptions: inspection is read-only except schema initialization if missing.
    """

    parser = argparse.ArgumentParser(description="Inspect Finance AI memory DB.")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "memory" / "finance_memory.db",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    """Print memory database table counts.

    Inputs: optional database path.
    Outputs: table counts to stdout.
    Assumptions: counts are enough for Phase 11A diagnostics.
    """

    args = build_argument_parser().parse_args()
    repository = MemoryRepository(args.database)
    counts = repository.table_counts()
    if args.json:
        print(json.dumps(counts, indent=2, ensure_ascii=False))
        return
    print(f"Database: {repository.database_path}")
    for table, count in counts.items():
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
