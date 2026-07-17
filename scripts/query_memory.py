"""Query the SQLite historical memory database with safe read-only tools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_agent.memory.retrieval import (  # noqa: E402
    get_artifact_references,
    get_full_period_record,
    get_goal_progress,
    get_memory_facts,
    get_metric_history,
    get_period_history,
    get_previous_period,
    get_previous_recommendations,
    get_repeated_anomalies,
    to_json,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for historical memory queries.

    Inputs: none.
    Outputs: configured parser.
    Assumptions: command choices map one-to-one to safe Python retrieval tools.
    """

    parser = argparse.ArgumentParser(description="Query Finance AI historical memory.")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "memory" / "finance_memory.db",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    previous = sub.add_parser("previous-period")
    previous.add_argument("--current-period", required=True)

    history = sub.add_parser("period-history")
    history.add_argument("--limit", type=int, default=6)
    history.add_argument("--before-period")

    metric = sub.add_parser("metric-history")
    metric.add_argument("--metric", required=True)
    metric.add_argument("--periods", type=int, default=6)
    metric.add_argument("--department")
    metric.add_argument("--before-period")

    repeated = sub.add_parser("repeated-anomalies")
    repeated.add_argument("--periods", type=int, default=6)
    repeated.add_argument("--department")
    repeated.add_argument("--min-occurrences", type=int, default=2)
    repeated.add_argument("--before-period")

    recs = sub.add_parser("previous-recommendations")
    recs.add_argument("--periods", type=int, default=6)
    recs.add_argument("--department")
    recs.add_argument("--status")
    recs.add_argument("--before-period")

    goals = sub.add_parser("goal-progress")
    goals.add_argument("--metric")
    goals.add_argument("--periods", type=int, default=6)
    goals.add_argument("--before-period")

    facts = sub.add_parser("memory-facts")
    facts.add_argument("--category")
    facts.add_argument("--subject")
    facts.add_argument("--periods", type=int, default=6)
    facts.add_argument("--before-period")

    full = sub.add_parser("full-period-record")
    full.add_argument("--period", required=True)

    artifacts = sub.add_parser("artifact-references")
    artifacts.add_argument("--period", required=True)
    artifacts.add_argument("--artifact-type")
    return parser


def main() -> None:
    """Execute one read-only memory query and print JSON.

    Inputs: CLI arguments.
    Outputs: JSON result to stdout.
    Assumptions: invalid inputs should fail fast with argparse/ValueError.
    """

    args = build_argument_parser().parse_args()
    database_path = args.database
    if args.command == "previous-period":
        result = get_previous_period(args.current_period, database_path=database_path)
    elif args.command == "period-history":
        result = get_period_history(args.limit, args.before_period, database_path=database_path)
    elif args.command == "metric-history":
        result = get_metric_history(
            args.metric,
            args.periods,
            department=args.department,
            before_period=args.before_period,
            database_path=database_path,
        )
    elif args.command == "repeated-anomalies":
        result = get_repeated_anomalies(
            args.periods,
            department=args.department,
            min_occurrences=args.min_occurrences,
            before_period=args.before_period,
            database_path=database_path,
        )
    elif args.command == "previous-recommendations":
        result = get_previous_recommendations(
            args.periods,
            department=args.department,
            status=args.status,
            before_period=args.before_period,
            database_path=database_path,
        )
    elif args.command == "goal-progress":
        result = get_goal_progress(
            args.metric,
            args.periods,
            before_period=args.before_period,
            database_path=database_path,
        )
    elif args.command == "memory-facts":
        result = get_memory_facts(
            args.category,
            args.subject,
            args.periods,
            before_period=args.before_period,
            database_path=database_path,
        )
    elif args.command == "full-period-record":
        result = get_full_period_record(args.period, database_path=database_path)
    else:
        result = get_artifact_references(
            args.period,
            args.artifact_type,
            database_path=database_path,
        )
    print(to_json(result))


if __name__ == "__main__":
    main()
