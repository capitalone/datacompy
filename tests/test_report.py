"""Unit tests for datacompy.report dataclasses and Report class."""

import dataclasses
import json
import os

import pandas as pd
import polars as pl
import pytest
from datacompy import (
    ColumnComparison,
    ColumnSummary,
    MismatchStat,
    MismatchStats,
    PandasCompare,
    PolarsCompare,
    ReportData,
    RowSummary,
    UniqueRowsData,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_report_data(
    df1_name: str = "df1",
    df2_name: str = "df2",
    unequal_rows: int = 0,
    mismatch: bool = False,
) -> ReportData:
    stat = (
        (
            MismatchStat(
                column="val",
                dtype1="int64",
                dtype2="int64",
                unequal_cnt=1,
                max_diff=1.0,
                null_diff=0,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
        )
        if mismatch
        else ()
    )

    return ReportData(
        df1_name=df1_name,
        df2_name=df2_name,
        df1_shape=(3, 2),
        df2_shape=(3, 2),
        column_count=10,
        column_summary=ColumnSummary(
            common_columns=2,
            df1_unique=0,
            df1_unique_columns=(),
            df2_unique=0,
            df2_unique_columns=(),
            df1_name=df1_name,
            df2_name=df2_name,
        ),
        row_summary=RowSummary(
            match_columns=("id",),
            on_index=False,
            has_duplicates=False,
            abs_tol=0,
            rel_tol=0,
            common_rows=3,
            df1_unique=0,
            df2_unique=0,
            unequal_rows=unequal_rows,
            equal_rows=3 - unequal_rows,
            df1_name=df1_name,
            df2_name=df2_name,
        ),
        column_comparison=ColumnComparison(
            unequal_columns=1 if mismatch else 0,
            equal_columns=1 if not mismatch else 0,
            unequal_values=1 if mismatch else 0,
        ),
        mismatch_stats=MismatchStats(
            has_mismatches=mismatch,
            has_samples=mismatch,
            stats=stat,
            samples=("col  df1  df2\n  0    1    2",) if mismatch else (),
            df1_name=df1_name,
            df2_name=df2_name,
        ),
        df1_unique_rows=UniqueRowsData(has_rows=False),
        df2_unique_rows=UniqueRowsData(has_rows=False),
    )


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


def test_report_data_is_frozen():
    data = _minimal_report_data()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        data.df1_name = "changed"  # type: ignore[misc]


def test_mismatch_stat_frozen():
    stat = MismatchStat("col", "int64", "int64", 1, 1.0, 0, 0.0, 0.0)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        stat.column = "other"  # type: ignore[misc]


def test_mismatch_stats_defaults():
    ms = MismatchStats(has_mismatches=False, has_samples=False)
    assert ms.stats == ()
    assert ms.samples == ()
    assert ms.df1_name == ""


def test_unique_rows_data_defaults():
    u = UniqueRowsData(has_rows=False)
    assert u.rows == ""


# ---------------------------------------------------------------------------
# to_template_context tests
# ---------------------------------------------------------------------------


def test_template_context_basic_keys_present():
    ctx = _minimal_report_data().to_template_context()
    for key in [
        "df1_name",
        "df2_name",
        "df1_shape",
        "df2_shape",
        "column_count",
        "column_summary",
        "row_summary",
        "column_comparison",
        "mismatch_stats",
        "df1_unique_rows",
        "df2_unique_rows",
    ]:
        assert key in ctx


def test_template_context_row_summary_match_columns_joined():
    ctx = _minimal_report_data().to_template_context()
    assert ctx["row_summary"]["match_columns"] == "id"


def test_template_context_row_summary_on_index():
    data = dataclasses.replace(
        _minimal_report_data(),
        row_summary=dataclasses.replace(
            _minimal_report_data().row_summary, on_index=True
        ),
    )
    ctx = data.to_template_context()
    assert ctx["row_summary"]["match_columns"] == "index"


def test_template_context_has_duplicates_string():
    ctx = _minimal_report_data().to_template_context()
    assert ctx["row_summary"]["has_duplicates"] in ("Yes", "No")


def test_template_context_df1_unique_formatted_with_cols():
    data = dataclasses.replace(
        _minimal_report_data(),
        column_summary=dataclasses.replace(
            _minimal_report_data().column_summary,
            df1_unique=2,
            df1_unique_columns=("col_a", "col_b"),
        ),
    )
    ctx = data.to_template_context()
    assert ctx["column_summary"]["df1_unique"] == "2 ['col_a', 'col_b']"


def test_template_context_df1_unique_plain_int_when_no_unique_cols():
    ctx = _minimal_report_data().to_template_context()
    assert ctx["column_summary"]["df1_unique"] == 0


def test_template_context_mismatch_stats_is_list_of_dicts():
    ctx = _minimal_report_data(mismatch=True).to_template_context()
    assert isinstance(ctx["mismatch_stats"]["stats"], list)
    assert isinstance(ctx["mismatch_stats"]["stats"][0], dict)
    assert "column" in ctx["mismatch_stats"]["stats"][0]


# ---------------------------------------------------------------------------
# ReportData rendering / export tests
# ---------------------------------------------------------------------------


def test_render_returns_string():
    text = _minimal_report_data().render()
    assert isinstance(text, str)
    assert "DataComPy" in text


def test_str_equals_render():
    data = _minimal_report_data()
    assert str(data) == data.render()


def test_repr():
    r = repr(_minimal_report_data("left", "right"))
    assert "left" in r
    assert "right" in r


def test_to_html_contains_pre():
    html = _minimal_report_data().to_html()
    assert "<pre>" in html
    assert "<html>" in html.lower()


def test_save_writes_html_file(tmp_path):
    dest = tmp_path / "report.html"
    _minimal_report_data().save(dest)
    assert dest.exists()
    assert "<pre>" in dest.read_text()


def test_to_dict_json_serializable():
    d = _minimal_report_data(mismatch=True).to_dict()
    assert "df1_name" in json.dumps(d)


def test_to_dict_roundtrip_fields():
    d = _minimal_report_data(mismatch=True).to_dict()
    assert d["row_summary"]["common_rows"] == 3
    assert d["mismatch_stats"]["stats"][0]["column"] == "val"


def test_render_with_custom_template(tmp_path):
    tmpl = tmp_path / "custom.j2"
    tmpl.write_text("{{ df1_name }} vs {{ df2_name }}")
    assert (
        _minimal_report_data("left", "right").render(template_path=str(tmpl))
        == "left vs right"
    )


def test_render_with_mismatches():
    text = _minimal_report_data(mismatch=True).render()
    assert "Columns with Unequal" in text
    assert "val" in text


def test_render_no_mismatches():
    assert "Columns with Unequal" not in _minimal_report_data(mismatch=False).render()


# ---------------------------------------------------------------------------
# Integration: build_report_data roundtrip via pandas and polars
# ---------------------------------------------------------------------------


def test_pandas_returns_report_data():
    df1 = pd.DataFrame({"id": [1, 2, 3], "val": [1, 2, 3]})
    df2 = pd.DataFrame({"id": [1, 2, 3], "val": [1, 9, 3]})
    data = PandasCompare(df1, df2, "id").build_report_data()
    assert isinstance(data, ReportData)
    assert data.df1_shape == (3, 2)
    assert data.row_summary.common_rows == 3
    assert data.row_summary.unequal_rows == 1
    assert data.mismatch_stats.has_mismatches is True
    assert data.mismatch_stats.stats[0].column == "val"


def test_pandas_mismatch_stats_sorted():
    df1 = pd.DataFrame({"id": [1, 2], "b": [1, 2], "a": [10, 20]})
    df2 = pd.DataFrame({"id": [1, 2], "b": [9, 9], "a": [99, 99]})
    data = PandasCompare(df1, df2, "id").build_report_data()
    cols = [s.column for s in data.mismatch_stats.stats]
    assert cols == sorted(cols)


def test_pandas_no_mismatches():
    df1 = pd.DataFrame({"id": [1, 2], "val": [1, 2]})
    df2 = pd.DataFrame({"id": [1, 2], "val": [1, 2]})
    data = PandasCompare(df1, df2, "id").build_report_data()
    assert data.mismatch_stats.has_mismatches is False
    assert data.mismatch_stats.stats == ()


def test_pandas_unique_columns():
    df1 = pd.DataFrame({"id": [1], "only_in_1": [9]})
    df2 = pd.DataFrame({"id": [1], "only_in_2": [9]})
    data = PandasCompare(df1, df2, "id").build_report_data()
    assert data.column_summary.df1_unique_columns == ("only_in_1",)
    assert data.column_summary.df2_unique_columns == ("only_in_2",)


def test_pandas_unique_rows():
    df1 = pd.DataFrame({"id": [1, 2, 3], "val": [1, 2, 3]})
    df2 = pd.DataFrame({"id": [1, 2], "val": [1, 2]})
    data = PandasCompare(df1, df2, "id").build_report_data()
    assert data.df1_unique_rows.has_rows is True
    assert "3" in data.df1_unique_rows.rows
    assert data.df2_unique_rows.has_rows is False


def test_pandas_on_index():
    df1 = pd.DataFrame({"val": [1, 2]}, index=[0, 1])
    df2 = pd.DataFrame({"val": [1, 9]}, index=[0, 1])
    data = PandasCompare(df1, df2, on_index=True).build_report_data()
    assert data.row_summary.on_index is True
    assert data.to_template_context()["row_summary"]["match_columns"] == "index"


def test_pandas_html_file(tmp_path):
    df1 = pd.DataFrame({"id": [1], "val": [1]})
    df2 = pd.DataFrame({"id": [1], "val": [1]})
    html_path = str(tmp_path / "report.html")
    text = PandasCompare(df1, df2, "id").report(html_file=html_path)
    assert isinstance(text, str)
    assert os.path.exists(html_path)


def test_polars_returns_report_data():
    df1 = pl.DataFrame({"id": [1, 2], "val": [10, 20]})
    df2 = pl.DataFrame({"id": [1, 2], "val": [10, 99]})
    data = PolarsCompare(df1, df2, "id").build_report_data()
    assert isinstance(data, ReportData)
    assert data.mismatch_stats.has_mismatches is True


def test_polars_mismatch_stats_sorted():
    df1 = pl.DataFrame({"id": [1], "z_col": [1], "a_col": [10]})
    df2 = pl.DataFrame({"id": [1], "z_col": [9], "a_col": [99]})
    data = PolarsCompare(df1, df2, "id").build_report_data()
    cols = [s.column for s in data.mismatch_stats.stats]
    assert cols == sorted(cols)


def test_report_method_returns_string():
    df1 = pd.DataFrame({"id": [1, 2], "val": [1, 2]})
    df2 = pd.DataFrame({"id": [1, 2], "val": [1, 9]})
    rpt = PandasCompare(df1, df2, "id").report()
    assert isinstance(rpt, str)
    assert "DataComPy" in rpt
