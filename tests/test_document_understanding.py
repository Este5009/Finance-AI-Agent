"""Tests for worksheet geometry and logical-table detection."""

from finance_agent.understanding.document_understanding import understand_sheet
from finance_agent.understanding.models import RawSheetData


def _raw_sheet(
    rows: list[list[object]],
    *,
    name: str = "Sheet1",
    merged_ranges: tuple[str, ...] = (),
) -> RawSheetData:
    """Build a rectangular RawSheetData fixture.

    Inputs: row values, optional sheet name, and merged ranges.
    Outputs: raw worksheet fixture.
    Assumptions: missing trailing cells are empty.
    """

    max_column = max((len(row) for row in rows), default=0)
    rectangular = tuple(
        tuple(row + [None] * (max_column - len(row)))
        for row in rows
    )
    return RawSheetData(
        sheet_name=name,
        values=rectangular,
        max_row=len(rows),
        max_column=max_column,
        merged_ranges=merged_ranges,
    )


def test_detects_flexible_header_and_merged_title() -> None:
    """Verify a table is found below a merged title and metadata rows."""

    sheet = _raw_sheet(
        [
            ["Financial Report", None, None],
            ["Currency: USD"],
            [None, None, None],
            ["Departamento", "Mes", "Monto"],
            ["Engineering", "June", 1000],
            ["Business", "June", 800],
        ],
        name="Datos",
        merged_ranges=("A1:C1",),
    )

    understanding = understand_sheet(sheet)

    assert len(understanding.tables) == 1
    table = understanding.tables[0]
    assert table.header_row == 4
    assert table.start_column == 1
    assert table.end_column == 3
    assert table.original_columns == ["Departamento", "Mes", "Monto"]
    assert table.dataframe.shape == (2, 3)
    assert understanding.empty_separator_rows == [3]
    assert understanding.merged_title_regions[0].text == "Financial Report"


def test_detects_multiple_tables_in_one_sheet() -> None:
    """Verify vertically separated logical tables remain independent."""

    sheet = _raw_sheet(
        [
            ["Department", "Revenue"],
            ["Engineering", 1000],
            ["Business", 900],
            [None, None, None],
            ["Vendor", "Invoice", "Amount"],
            ["Supplier A", "A-100", 400],
            ["Supplier B", "B-200", 500],
            [None, None, None],
            ["Note: preliminary values"],
        ],
        name="Mixed Data",
    )

    understanding = understand_sheet(sheet)

    assert len(understanding.tables) == 2
    assert [table.header_row for table in understanding.tables] == [1, 5]
    assert understanding.tables[0].dataframe.shape == (2, 2)
    assert understanding.tables[1].dataframe.shape == (2, 3)
    assert understanding.context_regions[-1].region_type == "notes_or_footer"


def test_rejects_title_only_region_as_table() -> None:
    """Verify sparse presentation text is preserved as context, not a table."""

    sheet = _raw_sheet(
        [
            ["Annual Finance Overview", None, None],
            ["Prepared for management"],
        ],
        merged_ranges=("A1:C1",),
    )

    understanding = understand_sheet(sheet)

    assert understanding.tables == []
    assert understanding.context_regions[0].region_type == "title_or_context"
