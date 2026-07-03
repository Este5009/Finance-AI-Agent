"""Data models shared by document understanding and feature extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RawSheetData:
    """Raw worksheet values and layout evidence produced by ingestion."""

    sheet_name: str
    values: tuple[tuple[Any, ...], ...]
    max_row: int
    max_column: int
    merged_ranges: tuple[str, ...]


@dataclass(frozen=True)
class RawWorkbookData:
    """Raw workbook contents with original worksheet order preserved."""

    workbook_path: str
    sheet_names: tuple[str, ...]
    sheets: dict[str, RawSheetData]


@dataclass(frozen=True)
class MergedTitleRegion:
    """Merged worksheet region that contains title or contextual text."""

    start_row: int
    end_row: int
    start_column: int
    end_column: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the merged-title region.

        Inputs: this region.
        Outputs: a JSON-compatible coordinate and text dictionary.
        Assumptions: worksheet coordinates are one-based.
        """

        return {
            "start_row": self.start_row,
            "end_row": self.end_row,
            "start_column": self.start_column,
            "end_column": self.end_column,
            "text": self.text,
        }


@dataclass(frozen=True)
class ContextRegion:
    """Non-table worksheet region such as a title, note, or footer."""

    start_row: int
    end_row: int
    region_type: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the contextual region.

        Inputs: this region.
        Outputs: a JSON-compatible row range, type, and text dictionary.
        Assumptions: region_type is descriptive evidence, not financial meaning.
        """

        return {
            "start_row": self.start_row,
            "end_row": self.end_row,
            "region_type": self.region_type,
            "text": self.text,
        }


@dataclass
class DetectedRawTable:
    """Logical table detected from raw worksheet geometry."""

    table_index: int
    sheet_name: str
    header_row: int
    start_row: int
    end_row: int
    start_column: int
    end_column: int
    header_confidence: float
    title: str | None
    original_columns: list[str]
    dataframe: pd.DataFrame


@dataclass
class SheetUnderstanding:
    """Structural interpretation of one worksheet."""

    sheet_name: str
    max_row: int
    max_column: int
    empty_separator_rows: list[int]
    merged_title_regions: list[MergedTitleRegion]
    context_regions: list[ContextRegion]
    tables: list[DetectedRawTable]

    def to_dict(self, table_ids: list[str]) -> dict[str, Any]:
        """Serialize sheet structure without embedding DataFrames.

        Inputs: ordered model table identifiers corresponding to detected tables.
        Outputs: JSON-compatible structural evidence for the worksheet.
        Assumptions: table_ids follow the same order as self.tables.
        """

        return {
            "sheet_name": self.sheet_name,
            "max_row": self.max_row,
            "max_column": self.max_column,
            "empty_separator_rows": self.empty_separator_rows,
            "merged_title_regions": [
                region.to_dict() for region in self.merged_title_regions
            ],
            "context_regions": [region.to_dict() for region in self.context_regions],
            "detected_table_ids": table_ids,
        }


@dataclass(frozen=True)
class ColumnMapping:
    """Original-to-normalized column mapping with deterministic confidence."""

    original_name: str
    normalized_name: str
    confidence: float
    matched_alias: bool
    requires_interpretation: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize one column mapping.

        Inputs: this column mapping.
        Outputs: JSON-compatible mapping and confidence fields.
        Assumptions: confidence is bounded between zero and one.
        """

        return {
            "original_name": self.original_name,
            "normalized_name": self.normalized_name,
            "confidence": self.confidence,
            "matched_alias": self.matched_alias,
            "requires_future_interpretation": self.requires_interpretation,
        }


@dataclass(frozen=True)
class ExtractedFeature:
    """Dimension or metric inferred from a normalized table column."""

    original_column: str
    normalized_column: str
    semantic_name: str
    role: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize one extracted feature.

        Inputs: this feature.
        Outputs: JSON-compatible semantic role and confidence metadata.
        Assumptions: role is either dimension or metric.
        """

        return {
            "original_column": self.original_column,
            "normalized_column": self.normalized_column,
            "semantic_name": self.semantic_name,
            "role": self.role,
            "confidence": self.confidence,
        }


@dataclass
class IntermediateFinancialTable:
    """Normalized bridge table consumed by future deterministic finance logic."""

    table_id: str
    source_workbook: str
    sheet: str
    table_title: str | None
    detected_type: str
    confidence: float
    header_confidence: float
    requires_future_interpretation: bool
    header_row: int
    start_row: int
    end_row: int
    start_column: int
    end_column: int
    original_columns: list[str]
    normalized_columns: list[str]
    column_mappings: list[ColumnMapping]
    extracted_dimensions: list[ExtractedFeature]
    extracted_metrics: list[ExtractedFeature]
    cleaned_dataframe: pd.DataFrame
    normalized_table_file: str = ""

    def to_dict(self, *, sample_size: int = 5) -> dict[str, Any]:
        """Serialize table metadata and a small cleaned sample.

        Inputs: this table and maximum sample size.
        Outputs: JSON-compatible table metadata; full records remain in its CSV.
        Assumptions: normalized CSV is the authoritative row-level interchange.
        """

        sample_json = self.cleaned_dataframe.head(sample_size).to_json(
            orient="records",
            date_format="iso",
        )
        return {
            "table_id": self.table_id,
            "source_workbook": self.source_workbook,
            "sheet": self.sheet,
            "table_title": self.table_title,
            "detected_type": self.detected_type,
            "confidence": self.confidence,
            "header_confidence": self.header_confidence,
            "requires_future_ollama_interpretation": self.requires_future_interpretation,
            "header_row": self.header_row,
            "table_boundaries": {
                "start_row": self.start_row,
                "end_row": self.end_row,
                "start_column": self.start_column,
                "end_column": self.end_column,
            },
            "row_count": int(len(self.cleaned_dataframe.index)),
            "column_count": int(len(self.cleaned_dataframe.columns)),
            "original_columns": self.original_columns,
            "normalized_columns": self.normalized_columns,
            "column_mappings": [
                mapping.to_dict() for mapping in self.column_mappings
            ],
            "extracted_dimensions": [
                feature.to_dict() for feature in self.extracted_dimensions
            ],
            "extracted_metrics": [
                feature.to_dict() for feature in self.extracted_metrics
            ],
            "normalized_table_file": self.normalized_table_file,
            "sample_rows": json.loads(sample_json),
        }


@dataclass
class FinancialDocumentModel:
    """Complete Step 2 intermediate representation for one or more workbooks."""

    model_version: str
    source_workbooks: list[str]
    sheet_analyses: list[dict[str, Any]]
    tables: list[IntermediateFinancialTable] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document model without duplicating full CSV records.

        Inputs: this model.
        Outputs: JSON-compatible source, sheet, and table metadata.
        Assumptions: each normalized_table_file points to a generated CSV.
        """

        return {
            "model_version": self.model_version,
            "purpose": (
                "Intermediate bridge between arbitrary financial workbooks "
                "and deterministic finance calculations."
            ),
            "source_workbooks": self.source_workbooks,
            "sheet_count": len(self.sheet_analyses),
            "table_count": len(self.tables),
            "sheet_analyses": self.sheet_analyses,
            "tables": [table.to_dict() for table in self.tables],
        }
