"""Validated loader for Step 2 intermediate model artifacts.

This module deliberately reads only the intermediate JSON manifest and its
referenced normalized CSV files. It has no Excel or PDF dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class IntermediateModelLoadError(RuntimeError):
    """Raised when an intermediate model or referenced CSV is invalid."""


@dataclass(frozen=True)
class LoadedIntermediateTable:
    """One manifest table and its validated normalized DataFrame."""

    table_id: str
    detected_type: str
    source_workbook: str
    sheet: str
    confidence: float
    csv_path: str
    dataframe: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedIntermediateModel:
    """Structured intermediate model ready for deterministic calculations."""

    model_path: str
    model_version: str
    source_workbooks: list[str]
    tables: tuple[LoadedIntermediateTable, ...]
    manifest: dict[str, Any]


def _read_manifest(model_path: Path) -> dict[str, Any]:
    """Read and minimally validate the intermediate JSON document.

    Inputs: resolved path to financial_document_model.json.
    Outputs: parsed manifest dictionary.
    Assumptions: table-level validation occurs while loading each CSV reference.
    """

    try:
        manifest = json.loads(model_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Intermediate model file does not exist: {model_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise IntermediateModelLoadError(
            f"Unable to read intermediate model '{model_path}': {error}"
        ) from error

    if not isinstance(manifest, dict):
        raise IntermediateModelLoadError("Intermediate model root must be a JSON object.")
    if not isinstance(manifest.get("tables"), list):
        raise IntermediateModelLoadError(
            "Intermediate model must contain a 'tables' list."
        )
    return manifest


def _resolve_csv_reference(model_directory: Path, csv_reference: object) -> Path:
    """Resolve a manifest CSV reference and keep it inside the model directory.

    Inputs: model directory and the manifest reference value.
    Outputs: validated absolute CSV path.
    Assumptions: normalized tables are model-owned sibling artifacts.
    """

    if not isinstance(csv_reference, str) or not csv_reference.strip():
        raise IntermediateModelLoadError(
            "Table metadata contains an empty normalized_table_file reference."
        )

    csv_path = (model_directory / csv_reference).resolve()
    try:
        csv_path.relative_to(model_directory)
    except ValueError as error:
        raise IntermediateModelLoadError(
            f"CSV reference escapes the intermediate model directory: {csv_reference}"
        ) from error
    if not csv_path.exists() or not csv_path.is_file():
        raise IntermediateModelLoadError(
            f"Referenced normalized CSV does not exist: {csv_path}"
        )
    return csv_path


def _load_table(
    table_metadata: object,
    model_directory: Path,
) -> LoadedIntermediateTable:
    """Load and validate one table declared by the intermediate manifest.

    Inputs: table metadata object and model directory.
    Outputs: structured table with a pandas DataFrame.
    Assumptions: normalized columns and row count are authoritative controls.
    """

    if not isinstance(table_metadata, dict):
        raise IntermediateModelLoadError("Every manifest table must be a JSON object.")

    table_id = str(table_metadata.get("table_id") or "").strip()
    detected_type = str(table_metadata.get("detected_type") or "").strip()
    if not table_id or not detected_type:
        raise IntermediateModelLoadError(
            "Each manifest table requires table_id and detected_type."
        )

    csv_path = _resolve_csv_reference(
        model_directory,
        table_metadata.get("normalized_table_file"),
    )
    try:
        dataframe = pd.read_csv(csv_path)
    except Exception as error:
        raise IntermediateModelLoadError(
            f"Unable to read normalized CSV for table '{table_id}': {error}"
        ) from error

    expected_columns = table_metadata.get("normalized_columns")
    if isinstance(expected_columns, list):
        actual_columns = list(dataframe.columns)
        if actual_columns != expected_columns:
            raise IntermediateModelLoadError(
                f"Column mismatch for table '{table_id}'. "
                f"Expected {expected_columns}, received {actual_columns}."
            )

    expected_rows = table_metadata.get("row_count")
    if isinstance(expected_rows, int) and len(dataframe.index) != expected_rows:
        raise IntermediateModelLoadError(
            f"Row-count mismatch for table '{table_id}'. "
            f"Expected {expected_rows}, received {len(dataframe.index)}."
        )

    return LoadedIntermediateTable(
        table_id=table_id,
        detected_type=detected_type,
        source_workbook=str(table_metadata.get("source_workbook") or ""),
        sheet=str(table_metadata.get("sheet") or ""),
        confidence=float(table_metadata.get("confidence") or 0.0),
        csv_path=str(csv_path),
        dataframe=dataframe,
        metadata=table_metadata,
    )


def load_intermediate_model(
    model_path: str | Path,
) -> LoadedIntermediateModel:
    """Load the financial document model and every referenced normalized CSV.

    Inputs: path to outputs/intermediate/financial_document_model.json.
    Outputs: validated structured model and table DataFrames.
    Assumptions: the finance engine must never fall back to raw workbook files.
    """

    resolved_model_path = Path(model_path).expanduser().resolve()
    manifest = _read_manifest(resolved_model_path)
    model_directory = resolved_model_path.parent
    tables = tuple(
        _load_table(table_metadata, model_directory)
        for table_metadata in manifest["tables"]
    )

    source_workbooks = manifest.get("source_workbooks")
    if not isinstance(source_workbooks, list):
        source_workbooks = []
    return LoadedIntermediateModel(
        model_path=str(resolved_model_path),
        model_version=str(manifest.get("model_version") or ""),
        source_workbooks=[str(source) for source in source_workbooks],
        tables=tables,
        manifest=manifest,
    )
