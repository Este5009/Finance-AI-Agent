"""Assembly and serialization of the Step 2 financial document model."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from finance_agent.understanding.classification import classify_table
from finance_agent.understanding.document_understanding import understand_workbook
from finance_agent.understanding.features import extract_financial_features
from finance_agent.ingestion.ingestion import load_raw_excel_workbook
from finance_agent.understanding.models import (
    FinancialDocumentModel,
    IntermediateFinancialTable,
)
from finance_agent.understanding.normalization import normalize_detected_table
from finance_agent.ingestion.schema import clean_column_name


MODEL_VERSION = "2.0"
AUTOMATIC_TABLE_CONFIDENCE = 0.75


def _safe_identifier(value: str) -> str:
    """Convert source text to a stable file/table identifier.

    Inputs: workbook, sheet, or table label.
    Outputs: lowercase filesystem-safe identifier.
    Assumptions: clean_column_name provides deterministic ASCII normalization.
    """

    cleaned = clean_column_name(value)
    return cleaned or "unnamed"


def _table_identifier(
    workbook_path: str,
    sheet_name: str,
    table_index: int,
) -> str:
    """Build a globally stable table identifier.

    Inputs: source workbook path, original sheet name, and per-sheet table index.
    Outputs: identifier unique across the processed document set.
    Assumptions: workbook stems and sheet/index combinations are unique inputs.
    """

    return (
        f"{_safe_identifier(Path(workbook_path).stem)}"
        f"__{_safe_identifier(sheet_name)}"
        f"__table_{table_index:02d}"
    )


def build_financial_document_model(
    workbook_paths: Iterable[str | Path],
) -> FinancialDocumentModel:
    """Build the reusable intermediate model from arbitrary Excel workbooks.

    Inputs: ordered paths to one or more Excel workbooks.
    Outputs: in-memory model containing cleaned DataFrames and structural metadata.
    Assumptions: finance calculations and anomaly detection consume this later.
    """

    source_workbooks: list[str] = []
    sheet_analyses: list[dict[str, Any]] = []
    intermediate_tables: list[IntermediateFinancialTable] = []

    for workbook_path in workbook_paths:
        raw_workbook = load_raw_excel_workbook(workbook_path)
        source_workbooks.append(raw_workbook.workbook_path)
        understood_sheets = understand_workbook(raw_workbook)

        for sheet_analysis in understood_sheets:
            sheet_table_ids: list[str] = []
            for raw_table in sheet_analysis.tables:
                normalized = normalize_detected_table(raw_table)
                classification = classify_table(raw_table, normalized)
                dimensions, metrics = extract_financial_features(normalized)
                table_id = _table_identifier(
                    raw_workbook.workbook_path,
                    raw_table.sheet_name,
                    raw_table.table_index,
                )
                sheet_table_ids.append(table_id)

                low_confidence_columns = [
                    mapping
                    for mapping in normalized.column_mappings
                    if mapping.requires_interpretation
                ]
                low_confidence_share = len(low_confidence_columns) / max(
                    1,
                    len(normalized.column_mappings),
                )
                requires_interpretation = (
                    classification.requires_interpretation
                    or classification.confidence < AUTOMATIC_TABLE_CONFIDENCE
                    or raw_table.header_confidence < 0.70
                    or low_confidence_share > 0.50
                )

                intermediate_tables.append(
                    IntermediateFinancialTable(
                        table_id=table_id,
                        source_workbook=raw_workbook.workbook_path,
                        sheet=raw_table.sheet_name,
                        table_title=raw_table.title,
                        detected_type=classification.detected_type,
                        confidence=classification.confidence,
                        header_confidence=raw_table.header_confidence,
                        requires_future_interpretation=requires_interpretation,
                        header_row=raw_table.header_row,
                        start_row=raw_table.start_row,
                        end_row=raw_table.end_row,
                        start_column=raw_table.start_column,
                        end_column=raw_table.end_column,
                        original_columns=normalized.original_columns,
                        normalized_columns=normalized.normalized_columns,
                        column_mappings=normalized.column_mappings,
                        extracted_dimensions=dimensions,
                        extracted_metrics=metrics,
                        cleaned_dataframe=normalized.dataframe,
                    )
                )

            sheet_record = sheet_analysis.to_dict(sheet_table_ids)
            sheet_record["source_workbook"] = raw_workbook.workbook_path
            sheet_analyses.append(sheet_record)

    return FinancialDocumentModel(
        model_version=MODEL_VERSION,
        source_workbooks=source_workbooks,
        sheet_analyses=sheet_analyses,
        tables=intermediate_tables,
    )


def build_feature_summary(model: FinancialDocumentModel) -> dict[str, Any]:
    """Create an aggregate catalog of extracted financial features.

    Inputs: completed intermediate document model.
    Outputs: counts by type plus dimension/metric inventories and review queue.
    Assumptions: summary supports discovery; table CSVs retain row-level values.
    """

    type_counts = Counter(table.detected_type for table in model.tables)
    dimensions_by_type: dict[str, set[str]] = defaultdict(set)
    metrics_by_type: dict[str, set[str]] = defaultdict(set)
    for table in model.tables:
        dimensions_by_type[table.detected_type].update(
            feature.semantic_name for feature in table.extracted_dimensions
        )
        metrics_by_type[table.detected_type].update(
            feature.semantic_name for feature in table.extracted_metrics
        )

    return {
        "model_version": model.model_version,
        "source_workbook_count": len(model.source_workbooks),
        "detected_table_count": len(model.tables),
        "table_type_counts": dict(sorted(type_counts.items())),
        "automatic_table_count": sum(
            not table.requires_future_interpretation for table in model.tables
        ),
        "requires_future_interpretation_count": sum(
            table.requires_future_interpretation for table in model.tables
        ),
        "features_by_table_type": {
            table_type: {
                "dimensions": sorted(dimensions_by_type[table_type]),
                "metrics": sorted(metrics_by_type[table_type]),
            }
            for table_type in sorted(type_counts)
        },
        "tables": [
            {
                "table_id": table.table_id,
                "detected_type": table.detected_type,
                "confidence": table.confidence,
                "requires_future_ollama_interpretation": (
                    table.requires_future_interpretation
                ),
                "dimensions": [
                    feature.semantic_name
                    for feature in table.extracted_dimensions
                ],
                "metrics": [
                    feature.semantic_name
                    for feature in table.extracted_metrics
                ],
            }
            for table in model.tables
        ],
    }


def _write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write a JSON-compatible dictionary with stable readable formatting.

    Inputs: data dictionary and destination path.
    Outputs: UTF-8 JSON file.
    Assumptions: model serializers already converted dates and pandas values.
    """

    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_intermediate_outputs(
    model: FinancialDocumentModel,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Save model metadata, feature summary, and one CSV per detected table.

    Inputs: in-memory model and output directory.
    Outputs: paths to the two JSON files and normalized-table directory.
    Assumptions: old CSVs in this dedicated output directory are stale pipeline artifacts.
    """

    output_dir = Path(output_directory).resolve()
    normalized_dir = output_dir / "normalized_tables"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    # A regenerated model must not leave stale table CSVs that no longer appear
    # in the JSON manifest. Only CSVs in the dedicated output folder are removed.
    for stale_csv in normalized_dir.glob("*.csv"):
        stale_csv.unlink()

    for table in model.tables:
        csv_path = normalized_dir / f"{table.table_id}.csv"
        table.cleaned_dataframe.to_csv(
            csv_path,
            index=False,
            encoding="utf-8",
            date_format="%Y-%m-%d",
        )
        table.normalized_table_file = str(
            Path("normalized_tables") / csv_path.name
        ).replace("\\", "/")

    model_path = output_dir / "financial_document_model.json"
    feature_path = output_dir / "feature_summary.json"
    _write_json(model.to_dict(), model_path)
    _write_json(build_feature_summary(model), feature_path)
    return {
        "financial_document_model": model_path,
        "feature_summary": feature_path,
        "normalized_tables": normalized_dir,
    }
