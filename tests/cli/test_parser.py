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

"""Tests for the argument specification and parser construction."""

import argparse
import inspect

import pytest
from datacompy.cli.backends import BACKENDS, compare_kwargs
from datacompy.cli.errors import BadArgsError, MissingExtraError
from datacompy.cli.parser import (
    ALL_BACKENDS,
    OPTIONS,
    OPTIONS_BY_FLAG,
    build_parser,
    fill_defaults,
    join_column_group,
    non_negative_int,
    single_char,
    tolerance,
)


def _parse(*args: str) -> argparse.Namespace:
    """Parse a ``compare`` invocation without running it."""
    return build_parser().parse_args(["compare", *args])


def _minimal(*extra: str) -> argparse.Namespace:
    """Parse a valid minimal invocation plus *extra*, with defaults filled in."""
    namespace = _parse("--left", "a.csv", "--right", "b.csv", "--on", "id", *extra)
    fill_defaults(namespace)
    return namespace


# ---------------------------------------------------------------------------
# Specification integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend_name", sorted(ALL_BACKENDS))
def test_every_option_kwarg_exists_on_its_backend_constructor(backend_name):
    """Guard against the parser drifting from the library constructor signatures.

    This is the reason the option table exists. If someone renames a keyword on
    a ``*Compare`` class, or adds a CLI row with a typo, this fails immediately
    instead of at runtime for a user.
    """
    backend = BACKENDS[backend_name]
    try:
        compare_cls = backend.compare_cls
    except MissingExtraError:
        pytest.skip(f"datacompy[{backend.extra}] is not installed")

    parameters = inspect.signature(compare_cls.__init__).parameters
    for opt in OPTIONS:
        if opt.kwarg is None or backend_name not in opt.backends:
            continue
        assert opt.kwarg in parameters, (
            f"{opt.flags[0]} maps to {opt.kwarg!r}, which "
            f"{compare_cls.__name__}.__init__ does not accept"
        )


def test_option_flags_and_destinations_are_unique():
    flags = [flag for opt in OPTIONS for flag in opt.flags]
    assert len(flags) == len(set(flags))
    dests = [opt.dest for opt in OPTIONS]
    assert len(dests) == len(set(dests))


def test_option_backends_are_known_backend_names():
    for opt in OPTIONS:
        assert opt.backends <= ALL_BACKENDS, opt.flags
        assert opt.backends, f"{opt.flags[0]} applies to no backend"


def test_every_backend_name_has_a_backend_implementation():
    assert set(BACKENDS) == ALL_BACKENDS


def test_compare_kwargs_omits_options_the_backend_does_not_accept():
    namespace = _minimal()
    assert "cast_column_names_lower" in compare_kwargs(namespace, "polars")
    assert "cast_column_names_lower" not in compare_kwargs(namespace, "snowflake")
    assert "on_index" in compare_kwargs(namespace, "pandas")
    assert "on_index" not in compare_kwargs(namespace, "polars")


def test_compare_kwargs_omits_unset_values_so_library_defaults_apply():
    namespace = _minimal()
    kwargs = compare_kwargs(namespace, "polars")
    assert "abs_tol" not in kwargs
    assert "rel_tol" not in kwargs
    assert kwargs["join_columns"] == ["id"]


# ---------------------------------------------------------------------------
# SUPPRESS defaults
# ---------------------------------------------------------------------------


def test_absent_flags_leave_no_attribute_until_defaults_are_filled():
    namespace = _parse("--left", "a.csv", "--right", "b.csv", "--on", "id")
    assert not OPTIONS_BY_FLAG["--ignore-case"].was_given(namespace)
    assert not hasattr(namespace, "ignore_case")

    fill_defaults(namespace)
    assert namespace.ignore_case is False
    assert namespace.backend == "polars"
    assert namespace.debug is False


def test_explicitly_passed_flag_is_distinguishable_from_its_default():
    namespace = _parse(
        "--left", "a.csv", "--right", "b.csv", "--on", "id", "--cast-column-names-lower"
    )
    opt = OPTIONS_BY_FLAG["--cast-column-names-lower"]
    assert opt.was_given(namespace)
    assert opt.value(namespace) is True

    namespace = _parse("--left", "a.csv", "--right", "b.csv", "--on", "id")
    assert not opt.was_given(namespace)
    assert opt.value(namespace) is True


# ---------------------------------------------------------------------------
# ``--on``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--on", "id"], ["id"]),
        (["--on", "id,date"], ["id", "date"]),
        (["--on", "id", "--on", "date"], ["id", "date"]),
        (["--on", "id,date", "--on", "region"], ["id", "date", "region"]),
        (["--on", " id , date "], ["id", "date"]),
    ],
)
def test_on_accepts_comma_separated_and_repeated_forms(args, expected):
    namespace = _parse("--left", "a.csv", "--right", "b.csv", *args)
    fill_defaults(namespace)
    assert OPTIONS_BY_FLAG["--on"].resolved(namespace) == expected


def test_on_rejects_an_empty_value():
    with pytest.raises(argparse.ArgumentTypeError):
        join_column_group(",,")


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


def test_tolerance_parses_a_bare_number_and_a_column_pair():
    assert tolerance("0.01") == pytest.approx(0.01)
    assert tolerance("price=0.01") == ("price", pytest.approx(0.01))
    assert tolerance(" price =0.01") == ("price", pytest.approx(0.01))


@pytest.mark.parametrize("value", ["nope", "price=nope", "=0.1", "-0.5"])
def test_tolerance_rejects_bad_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        tolerance(value)


def test_tolerance_resolves_to_a_float_or_a_per_column_dict():
    namespace = _minimal("--abs-tol", "0.01")
    assert OPTIONS_BY_FLAG["--abs-tol"].resolved(namespace) == pytest.approx(0.01)

    namespace = _minimal("--abs-tol", "price=0.01", "--abs-tol", "qty=0")
    assert OPTIONS_BY_FLAG["--abs-tol"].resolved(namespace) == pytest.approx(
        {"price": 0.01, "qty": 0.0}
    )


def test_mixing_a_bare_tolerance_with_column_pairs_is_rejected():
    namespace = _minimal("--abs-tol", "0.01", "--abs-tol", "price=0.02")
    with pytest.raises(BadArgsError, match="not both"):
        OPTIONS_BY_FLAG["--abs-tol"].resolved(namespace)


def test_repeating_a_bare_tolerance_is_rejected():
    namespace = _minimal("--rel-tol", "0.01", "--rel-tol", "0.02")
    with pytest.raises(BadArgsError, match="more than once"):
        OPTIONS_BY_FLAG["--rel-tol"].resolved(namespace)


# ---------------------------------------------------------------------------
# Other type callables
# ---------------------------------------------------------------------------


def test_single_char_translates_an_escaped_tab():
    assert single_char(",") == ","
    assert single_char("\\t") == "\t"
    assert single_char("\t") == "\t"


@pytest.mark.parametrize("value", ["", ";;", "ab"])
def test_single_char_rejects_anything_but_one_character(value):
    with pytest.raises(argparse.ArgumentTypeError):
        single_char(value)


def test_non_negative_int():
    assert non_negative_int("0") == 0
    assert non_negative_int("42") == 42
    for value in ("-1", "1.5", "many"):
        with pytest.raises(argparse.ArgumentTypeError):
            non_negative_int(value)


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def test_dataset_names_default_to_the_input_reference():
    namespace = _minimal("--left", "/data/sales_2024.parquet")
    # The later --left wins, so the default label comes from that file.
    assert OPTIONS_BY_FLAG["--df1-name"].resolved(namespace) == "sales_2024"
    assert OPTIONS_BY_FLAG["--df2-name"].resolved(namespace) == "b"


def test_snowflake_dataset_names_use_the_table_not_the_schema():
    namespace = _parse(
        "--left",
        "PROD.ANALYTICS.SALES",
        "--right",
        "STAGE.ANALYTICS.ORDERS",
        "--on",
        "id",
        "--backend",
        "snowflake",
    )
    fill_defaults(namespace)
    assert OPTIONS_BY_FLAG["--df1-name"].resolved(namespace) == "SALES"
    assert OPTIONS_BY_FLAG["--df2-name"].resolved(namespace) == "ORDERS"


def test_colliding_default_dataset_names_are_disambiguated():
    """Two sides sharing a label would make the report ambiguous.

    Comparing the same table across environments is the common Snowflake case,
    and comparing a file against itself is the common file case. Both derive the
    same label, so the sides are numbered instead.
    """
    namespace = _parse(
        "--left",
        "PROD.ANALYTICS.SALES",
        "--right",
        "STAGE.ANALYTICS.SALES",
        "--on",
        "id",
        "--backend",
        "snowflake",
    )
    fill_defaults(namespace)
    assert OPTIONS_BY_FLAG["--df1-name"].resolved(namespace) == "SALES_1"
    assert OPTIONS_BY_FLAG["--df2-name"].resolved(namespace) == "SALES_2"


def test_explicit_dataset_names_win():
    namespace = _minimal("--df1-name", "before", "--df2-name", "after")
    assert OPTIONS_BY_FLAG["--df1-name"].resolved(namespace) == "before"
    assert OPTIONS_BY_FLAG["--df2-name"].resolved(namespace) == "after"


@pytest.mark.parametrize(
    "argv",
    [
        ["--debug", "compare", "--left", "a.csv", "--right", "b.csv", "--on", "id"],
        ["compare", "--left", "a.csv", "--right", "b.csv", "--on", "id", "--debug"],
    ],
)
def test_debug_is_accepted_on_either_side_of_the_subcommand(argv):
    assert build_parser().parse_args(argv).debug is True


def test_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_version_exits_cleanly(capsys):
    from datacompy import __version__

    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out
