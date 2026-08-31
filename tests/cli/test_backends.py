#
# Copyright 2026 Capital One Services, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for delimiter inference and misparse detection.

``tests/cli/test_compare.py`` drives both helpers through ``main()`` on real
files, which is the right level for the paths a user actually walks. The cases
here are the ones an end to end run cannot reach or does not pin down: a
Snowflake table reference, a frame that exposes no columns, a quoted header,
and which delimiter the warning names.
"""

import argparse

import pandas as pd
import polars as pl
import pytest
from datacompy.cli.backends import infer_delimiter, suspect_delimiter

TAB = "\t"


def _namespace(
    *, input_format: str | None = None, csv_delimiter: str | None = None
) -> argparse.Namespace:
    """Return the only two attributes ``suspect_delimiter`` reads."""
    return argparse.Namespace(input_format=input_format, csv_delimiter=csv_delimiter)


@pytest.fixture(params=["pandas", "polars"])
def frame_of(request):
    """Return a builder for a frame with the given column names.

    Parametrised over both in memory backends because they disagree on the type
    of ``columns``: pandas returns an ``Index``, whose truth value raises rather
    than answering, and polars returns a plain list. ``suspect_delimiter`` has
    to accept both without testing either for truthiness.
    """

    def _build(*names: str):
        data = {name: ["x", "y"] for name in names}
        return pd.DataFrame(data) if request.param == "pandas" else pl.DataFrame(data)

    return _build


# ---------------------------------------------------------------------------
# infer_delimiter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("data.csv", ","),
        ("data.tsv", TAB),
        ("s3://bucket/nightly/data.tsv", TAB),
        ("extract.dat", ","),
        ("extract", ","),
    ],
)
def test_the_delimiter_comes_from_the_extension(ref, expected):
    assert infer_delimiter(ref, None) == expected


def test_extension_matching_is_case_insensitive():
    """Nothing downstream lowercases the path, so this helper has to."""
    assert infer_delimiter("DATA.TSV", None) == TAB


@pytest.mark.parametrize("ref", ["data.json", "data.parquet", "data.ndjson"])
def test_a_non_csv_extension_read_as_csv_gets_a_comma(ref):
    """``--input-format csv`` can point the CSV reader at any extension.

    Those rows carry ``None`` in the extension table because their format has
    no delimiter, which must resolve to the default rather than reaching the
    reader as ``None``.
    """
    assert infer_delimiter(ref, None) == ","


@pytest.mark.parametrize("ref", ["data.csv", "data.tsv", "data.parquet", "extract"])
def test_an_explicit_delimiter_wins_over_every_extension(ref):
    assert infer_delimiter(ref, ";") == ";"


# ---------------------------------------------------------------------------
# suspect_delimiter
# ---------------------------------------------------------------------------


def test_a_snowflake_table_reference_is_not_diagnosed(frame_of):
    """The warning loop runs for every backend, including Snowflake.

    ``PROD.ANALYTICS.SALES`` has an extension as far as ``Path`` is concerned,
    and it is not one this module knows, so format inference raises. That has
    to stay contained here: a diagnostic helper must not turn a working
    Snowflake comparison into an argument error.
    """
    frame = frame_of("id,name,amount")
    assert suspect_delimiter("PROD.ANALYTICS.SALES", _namespace(), frame) is None


def test_a_non_csv_input_is_not_diagnosed(frame_of):
    """A one column Parquet file is a fact about the file, not a parse failure."""
    frame = frame_of("id,name,amount")
    assert suspect_delimiter("data.parquet", _namespace(), frame) is None


def test_a_frame_without_columns_is_not_diagnosed():
    """The helper is best effort, so an unfamiliar object is skipped, not read."""

    class Opaque:
        pass

    assert suspect_delimiter("data.csv", _namespace(), Opaque()) is None


@pytest.mark.parametrize(
    ("ref", "csv_delimiter", "column_name", "warns"),
    [
        # Read with a comma (.csv): a lone foreign delimiter is an ordinary
        # name...
        ("data.csv", None, "a;b", False),
        ("data.csv", None, "a|b", False),
        ("data.csv", None, "a\tb", False),
        # ...but a repeated one is a collapsed multi-column header.
        ("data.csv", None, "a;b;c", True),
        ("data.csv", None, "a|b|c", True),
        ("data.csv", None, "a\tb\tc", True),
        # The delimiter actually used never implicates itself, however often it
        # appears (a quoted "a,b,c" survives a comma read as one column).
        ("data.csv", None, "a,b", False),
        ("data.csv", None, "a,b,c", False),
        # Two different foreign delimiters, one each, stays ambiguous -> quiet.
        ("data.csv", None, "a;b|c", False),
        # Nothing plausible in the name.
        ("data.csv", None, "id", False),
        ("data.csv", None, "amount_usd", False),
        # Read with a tab (.tsv): the mirror image.
        ("data.tsv", None, "first,last", False),
        ("data.tsv", None, "id,name,amount", True),
        ("data.tsv", None, "a;b", False),
        ("data.tsv", None, "a;b;c", True),
        ("data.tsv", None, "a\tb\tc", False),  # tab is what it was read with
        # An explicit --csv-delimiter is what "used" means, not the extension.
        ("data.csv", ";", "id,name,amount", True),
        ("data.csv", ";", "a;b;c", False),  # ';' is what it was read with
        ("data.csv", ";", "a|b", False),
        ("data.csv", ";", "a|b|c", True),
    ],
)
def test_suspect_delimiter_decision_matrix(
    frame_of, ref, csv_delimiter, column_name, warns
):
    """A single column is a misparse only when a foreign delimiter repeats.

    A plausible delimiter that appears more than once is what a collapsed
    multi-column header leaves behind. A lone occurrence is just as easily a
    genuine column name, and the delimiter the file was read with never
    implicates itself. Parametrised over both in memory backends via
    ``frame_of``.
    """
    namespace = _namespace(csv_delimiter=csv_delimiter)
    message = suspect_delimiter(ref, namespace, frame_of(column_name))
    assert (message is not None) == warns


def test_a_genuine_single_column_frame_is_not_reported(frame_of):
    assert suspect_delimiter("data.csv", _namespace(), frame_of("id")) is None


def test_a_multi_column_frame_is_not_reported(frame_of):
    frame = frame_of("id", "name", "amount")
    assert suspect_delimiter("data.csv", _namespace(), frame) is None


def test_the_warning_names_the_delimiter_the_file_was_read_with(frame_of):
    """A comma delimited file named ``.tsv`` is read with a tab and collapses."""
    message = suspect_delimiter("data.tsv", _namespace(), frame_of("id,name,amount"))
    assert message is not None
    assert "data.tsv" in message
    assert repr(TAB) in message
    assert "--csv-delimiter" in message


def test_the_warning_names_an_overridden_delimiter(frame_of):
    """The delimiter reported is the one actually used, not the inferred one."""
    message = suspect_delimiter(
        "data.csv", _namespace(csv_delimiter=";"), frame_of("id,name,amount")
    )
    assert message is not None
    assert repr(";") in message


def test_only_the_misparsed_input_is_reported(frame_of):
    """Inference is per file, so one bad input does not implicate the other."""
    namespace = _namespace()
    good = suspect_delimiter("left.csv", namespace, frame_of("id", "name", "amount"))
    bad = suspect_delimiter("right.tsv", namespace, frame_of("id,name,amount"))

    assert good is None
    assert bad is not None
    assert "right.tsv" in bad
