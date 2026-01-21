"""Native arrow based comparison."""


from copy import deepcopy
import logging
from typing import Any, Callable, Dict, Iterable, List, Tuple, cast

from ordered_set import OrderedSet
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as pat
import numpy as np
from tomlkit import value
from datacompy._typing import ArrowStreamable, ArrowArrayLike

from datacompy.base import (
    LOG,
    BaseCompare,
    df_to_str,
    get_column_tolerance,
    render,
    save_html_report,
    validate_tolerance_parameter
)
from datacompy.comparator import (
    PyArrowArrayLikeComparator,
    PyArrowNumericComparator,
    PyArrowStringComparator,
)
from datacompy.comparator.base import BaseComparator
from datacompy.comparator.string import pyarrow_normalize_string_column
from datacompy.comparator.string import DEFAULT_VALUE

LOG = logging.getLogger(__name__)

_ARROW_DEFAULT_COMPARATORS = [
    PyArrowArrayLikeComparator(),
    PyArrowNumericComparator(),
    PyArrowStringComparator(),
]

class PyArrowCompare(BaseCompare):
    """Class to compare two pyarrow Tables."""

    def __init__(
        self,
        df1: ArrowStreamable,
        df2: ArrowStreamable,
        join_columns: List[str] | str,
        abs_tol: float | Dict[str, float] = 0,
        rel_tol: float | Dict[str, float] = 0,
        df1_name: str = "df1",
        df2_name: str = "df2",
        ignore_spaces: bool = False,
        ignore_case: bool = False,
        cast_column_names_lower: bool = True,
        custom_comparators: List[BaseComparator] | None = None,
    ) -> None:
        self.cast_column_names_lower = cast_column_names_lower
        self.custom_comparators = custom_comparators or []

        # Validate tolerance parameters first
        self._abs_tol_dict = validate_tolerance_parameter(
            abs_tol, "abs_tol", "lower" if cast_column_names_lower else "preserve"
        )
        self._rel_tol_dict = validate_tolerance_parameter(
            rel_tol, "rel_tol", "lower" if cast_column_names_lower else "preserve"
        )

        if isinstance(join_columns, str):
            self.join_columns = [
                str(join_columns).lower()
                if self.cast_column_names_lower
                else str(join_columns)
            ]
        elif isinstance(join_columns, list):
            self.join_columns = [
                str(col).lower() if self.cast_column_names_lower else str(col)
                for col in join_columns
            ]
        else:
            raise TypeError(f"{join_columns} must be a string or list of string(s)")

        self._any_dupes: bool = False
        self.df1 = df1
        self.df2 = df2
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.ignore_spaces = ignore_spaces
        self.ignore_case = ignore_case
        self.df1_unq_rows: pa.Table
        self.df2_unq_rows: pa.Table
        self.intersect_rows: pa.Table
        self.column_stats: List[Dict[str, Any]] = []
        self._compare(ignore_spaces=ignore_spaces, ignore_case=ignore_case)

    @property
    def df1(self) -> ArrowStreamable:
        return self._df1

    @df1.setter
    def df1(self, df1: ArrowStreamable) -> None:
        self._df1 = convert_to_arrow(df1) if not isinstance(df1, pa.Table) else df1
        self._validate_dataframe("df1", cast_column_names_lower=self.cast_column_names_lower)

    @property
    def df2(self) -> ArrowStreamable:
        return self._df2

    @df2.setter
    def df2(self, df2: ArrowStreamable) -> None:
        self._df2 = convert_to_arrow(df2) if not isinstance(df2, pa.Table) else df2
        self._validate_dataframe("df2", cast_column_names_lower=self.cast_column_names_lower)

    def _get_comparators(self) -> List[BaseComparator]:
        """Build and return the list of comparators to be used.

        Custom comparators are placed first, followed by the default ones.
        """
        return self.custom_comparators + _ARROW_DEFAULT_COMPARATORS

    def _compare(self, ignore_spaces: bool = False, ignore_case: bool = False) -> None:
        """Compare two pyarrow Tables."""
        if self.df1.equals(self.df2):
            LOG.info("df1 pyarrow.Table.equals df2")
        else:
            LOG.info("df1 pyarrow.Table does not equal df2")
        
        LOG.info(f"Number of columns in common: {len(self.intersect_columns())}")

        LOG.info("Comparing common columns...")
        LOG.info(f"Columns in df1 and not in df2: {self.df1_unq_columns()}")
        LOG.info(f"Number of columns in df1 and not in df2: {len(self.df1_unq_columns())}")
        LOG.info(f"Columns in df2 and not in df1: {self.df2_unq_columns()}")
        LOG.info(f"Number of columns in df2 and not in df1: {len(self.df2_unq_columns())}")

        LOG.debug("Merging dataframes")
        self._dataframe_merge(ignore_spaces)
        self._intersect_compare(ignore_spaces, ignore_case)
        if self.matches():
            LOG.info("df1 matches df2")
        else:
            LOG.info("df1 does not match df2")

    def df1_unq_columns(self) -> OrderedSet[str]:
        """Get columns unique to df1."""
        unq_cols = OrderedSet(self.df1.schema.names) - OrderedSet(self.df2.schema.names)
        return cast(OrderedSet[str], unq_cols)

    def df2_unq_columns(self) -> OrderedSet[str]:
        """Get columns unique to df2."""
        unq_cols = OrderedSet(self.df2.schema.names) - OrderedSet(self.df1.schema.names)
        return cast(OrderedSet[str], unq_cols)
    
    def intersect_columns(self) -> OrderedSet[str]:
        """Get columns common to both df1 and df2."""
        return OrderedSet(self.df1.schema.names) & OrderedSet(self.df2.schema.names)
    
    def _dataframe_merge(self, ignore_spaces: bool = False) -> None:
        """Merge df1 and df2 on join_columns."""
        LOG.debug("Outer joining dataframes")

        df1 = self.df1.slice(0)
        df2 = self.df2.slice(0)
        temp_join_columns = deepcopy(self.join_columns)

        if self._any_dupes:
            LOG.debug("Duplicate rows found, deduping by order of remaining fields")
            order_column = temp_column_name(df1, df2)
            df1 = df1.append_column(
                order_column,
                generate_id_within_group(df1, temp_join_columns)
            )
            df2 = df2.append_column(
                order_column,
                generate_id_within_group(df2, temp_join_columns)
            )
            temp_join_columns.append(order_column)

        if ignore_spaces:
            for column in self.join_columns:
                df1 = df1.set_column(
                    pyarrow_normalize_string_column(df1[column], ignore_space=True, ignore_case=False)
                )
                df2 = df2.set_column(
                    pyarrow_normalize_string_column(df2[column], ignore_space=True, ignore_case=False)
                )

        # merge indicator
        df1 = df1.append_column(
            "_merge_left",
            pa.array([True] * len(df1))
        )
        df2 = df2.append_column(
            "_merge_right",
            pa.array([True] * len(df2))
        )

        # Perform outer join
        outer_joined = df1.join(
            df2, 
            keys=temp_join_columns,
            join_type="full outer",
            left_suffix="_" + self.df1_name,
            right_suffix="_" + self.df2_name,
            coalesce_keys=True
        )

        # Setting up conditions
        cond_both = pc.and_(
            pc.equal(outer_joined["_merge_left"], True),
            pc.equal(outer_joined["_merge_right"], True)
        )
        cond_left = pc.and_(
            pc.equal(outer_joined["_merge_left"], True),
            pc.is_null(outer_joined["_merge_right"])
        )
        cond_right = pc.and_(
            pc.is_null(outer_joined["_merge_left"]),
            pc.equal(outer_joined["_merge_right"], True)
        )
        # Flatten conditions to pass to selector
        cond_flat = [c.combine_chunks() for c in [cond_both, cond_left, cond_right]]

        # Processing merge indicators
        selector = pa.StructArray.from_arrays(
            cond_flat,
            names=["is_both", "is_left", "is_right"]
        )
        result = pc.case_when(
            selector,
            "both",
            "left_only",
            "right_only",
        )
        outer_joined = outer_joined.append_column("_merge", result)

        # Clean up from duplicate rows 
        if self._any_dupes:
            outer_joined = outer_joined.drop_columns([order_column])

        df1_cols = get_merged_columns(self.df1, outer_joined, self.df1_name)
        df2_cols = get_merged_columns(self.df2, outer_joined, self.df2_name)

        LOG.debug("Selecting df1 unique rows")
        self.df1_unq_rows = outer_joined.filter(
            outer_joined["_merge"] == "left_only"
        ).select(df1_cols)

        LOG.debug("Selecting df2 unique rows")
        self.df2_unq_rows = outer_joined.filter(
            outer_joined["_merge"] == "right_only"
        ).select(df2_cols)

        LOG.info(f"Number of unique rows in df1: {len(self.df1_unq_rows)}")
        LOG.info(f"Number of unique rows in df2: {len(self.df2_unq_rows)}")

        LOG.debug("Selecting intersect rows")
        self.intersect_rows = outer_joined.filter(outer_joined["_merge"] == "both")
        LOG.info(f"Number of intersecting rows (not necessarily equal): {len(self.intersect_rows)}")

    def _intersect_compare(self, ignore_spaces: bool = False, ignore_case: bool = False) -> None:
        """Run the comparison on merged dataframe"""
        match_cnt: int | float
        null_diff: int | float

        LOG.debug("Comparing intersection")
        for column in self.intersect_columns():
            if column in self.join_columns:
                col_match = column + "_match"
                match_cnt = len(self.intersect_rows)
                if not self.only_join_columns():
                    row_cnt = len(self.intersect_rows)
                else:
                    row_cnt = (
                        len(self.intersect_rows)
                        + len(self.df1_unq_rows)
                        + len(self.df2_unq_rows)
                    )
                max_diff = 0.0
                null_diff = 0
            else:
                row_cnt = len(self.intersect_rows)
                col_1 = column + "_" + self.df1_name
                col_2 = column + "_" + self.df2_name
                col_match = column + "_match"
                self.intersect_rows = self.intersect_rows.append_column(
                    col_match,
                    columns_equal(
                        self.intersect_rows[col_1],
                        self.intersect_rows[col_2],
                        rel_tol=get_column_tolerance(column, self._rel_tol_dict),
                        abs_tol=get_column_tolerance(column, self._abs_tol_dict),
                        ignore_spaces=ignore_spaces,
                        ignore_case=ignore_case,
                        comparators=self._get_comparators()
                    ),
                )
                match_cnt = pc.sum(self.intersect_rows[col_match]).as_py()
                max_diff = calculate_max_diff(
                    self.intersect_rows[col_1], self.intersect_rows[col_2]
                )
                null_diff = (
                    (self.intersect_rows[col_1].is_null()) ^ (self.intersect_rows[col_2].is_null())
                ).sum().as_py()

            if row_cnt > 0:
                match_rate = float(match_cnt) / row_cnt
            else:
                match_rate = 0.0
            LOG.info(f"{column}: {match_cnt} / {row_cnt} ({match_rate:.2%}) match")

            self.column_stats.append(
                {
                    "column": column,
                    "match_column": col_match,
                    "match_cnt": match_cnt,
                    "unequal_cnt": row_cnt - match_cnt,
                    "dtype1": str(self.df1[column].dtype),
                    "dtype2": str(self.df2[column].dtype),
                    "all_match": all(
                        (
                            self.df1[column].dtype == self.df2[column].dtype,
                            row_cnt == match_cnt,
                        )
                    ),
                    "max_diff": max_diff,
                    "null_diff": null_diff,
                    "rel_tol": get_column_tolerance(column, self._rel_tol_dict),
                    "abs_tol": get_column_tolerance(column, self._abs_tol_dict),
                }
            )
    
    def _validate_dataframe(self, index, cast_column_names_lower = True):
        """Check if the dataframe is valid arrow table and has join columns."""
        dataframe = getattr(self, index)

        if cast_column_names_lower:
            dataframe = dataframe.rename_columns(
                [col.lower() for col in dataframe.schema.names]
            )

        if not set(self.join_columns).issubset(dataframe.schema.names):
            missing_cols = set(self.join_columns) - set(dataframe.schema.names)
            raise ValueError(
                f"{index} is missing join columns: {missing_cols}"
            )
        
        if set(len(dataframe.schema.names)) < len(dataframe.schema.names):
            raise ValueError(f"{index} must have unique column names")
        
        if len(dataframe.groupby(self.join_columns)) < len(dataframe):
            LOG.warning(f"{index} has duplicate rows based on join columns")
            self._any_dupes = True

    def _get_column_summary(self) -> Dict[str, Any]:
        """Get summary of column comparison."""
        return {
            "column_summary": {
                "common_columns": len(self.intersect_columns()),
                "df1_unique": f"{len(self.df1_unq_columns())} {self.df1_unq_columns().items}",
                "df2_unique": f"{len(self.df2_unq_columns())} {self.df2_unq_columns().items}",
                "df1_name": self.df1_name,
                "df2_name": self.df2_name,
            }
        }
    
    def _get_row_summary(self) -> Dict[str, Any]:
        """Get summary of row comparison."""
        return {
            "row_summary": {
                "match_columns": ", ".join(self.join_columns),
                "abs_tol": self.abs_tol,
                "rel_tol": self.rel_tol,
                "common_rows": self.intersect_rows.num_rows,
                "df1_unique": self.df1_unq_rows.num_rows,
                "df2_unique": self.df2_unq_rows.num_rows,
                "unequal_rows": self.intersect_rows.num_rows
                - self.count_matching_rows(),
                "equal_rows": self.count_matching_rows(),
                "df1_name": self.df1_name,
                "df2_name": self.df2_name,
                "has_duplicates": "Yes" if self._any_dupes else "No",
            }
        }
    
    def _get_column_comparison(self) -> Dict[str, Any]:
        """Get detailed column comparison statistics."""
        return {
            "column_comparison": {
                "unequal_columns": len(
                    [col for col in self.column_stats if col["unequal_cnt"] > 0]
                ),
                "equal_columns": len(
                    [col for col in self.column_stats if col["unequal_cnt"] == 0]
                ),
                "unequal_values": sum([col["unequal_cnt"] for col in self.column_stats]),
            }
        }
    
    def _get_mismatch_stats(self, sample_count: int) -> dict:
        """Generate mismatch statistics for reporting."""
        match_stats = []
        match_sample = []
        any_mismatch = False
        
        for column in self.column_stats:
            if not column["all_match"]:
                any_mismatch = True
                match_stats.append(
                    {
                        "column": column["column"],
                        "dtype1": column["dtype1"],
                        "dtype2": column["dtype2"],
                        "unequal_cnt": column["unequal_cnt"],
                        "max_diff": column["max_diff"],
                        "null_diff": column["null_diff"],
                        "rel_tol": column["rel_tol"],
                        "abs_tol": column["abs_tol"],
                    }
                )
                if column["unequal_cnt"] > 0:
                    match_sample.append(
                        self.sample_mismatch(column["column"], sample_count, for_display=True)
                    )
        
        if any_mismatch:
            return {
                "mismatch_stats": {
                    "has_mismatches": True,
                    "stats": match_stats,
                    "df1_name": self.df1_name,
                    "df2_name": self.df2_name,
                    "samples": [df_to_str(sample) for sample in match_sample],
                    "has_samples": len(match_sample) > 0 and sample_count > 0,
                }
            }
        else:
            return {
                "mismatch_stats": {
                    "has_mismatches": False,
                    "has_samples": False,
                }
            }
        
    def _get_unique_rows_data(self, sample_count: int, column_count: int) -> dict:
        """Generate data for unique rows in both dataframes."""

        min_sample_count_df1 = min(sample_count, self.df1_unq_rows.num_rows)
        min_sample_count_df2 = min(sample_count, self.df2_unq_rows.num_rows)
        min_column_count_df1 = min(column_count, self.df1_unq_rows.num_columns)
        min_column_count_df2 = min(column_count, self.df2_unq_rows.num_columns)

        return {
            "df1_unique_rows": {
                "has_rows": min_sample_count_df1 > 0,
                "rows": df_to_str(
                    self.df1_unq_rows.select(
                        self.df1_unq_rows.schema.names[:min_column_count_df1]
                    ),
                    sample_count=min_sample_count_df1
                ) if self.df1_unq_rows.num_rows > 0 else "",
                "columns": list(self.df1_unq_rows.schema.names[:min_column_count_df1])
                if self.df1_unq_rows.num_columns > 0 else "",
            },
            "df2_unique_rows": {
                "has_rows": min_sample_count_df2 > 0,
                "rows": df_to_str(
                    self.df2_unq_rows.select(
                        self.df2_unq_rows.schema.names[:min_column_count_df2]
                    ),
                    sample_count=min_sample_count_df2
                ) if self.df2_unq_rows.num_rows > 0 else "",
                "columns": list(self.df2_unq_rows.schema.names[:min_column_count_df2])
                if self.df2_unq_rows.num_columns > 0 else "",
            }
        }

    def all_columns_match(self) -> bool:
        """Whether the columns all match in the dataframes."""
        return self.df1_unq_columns() == self.df2_unq_columns() == set()
    
    def all_rows_overlap(self):
        """Whether all rows are present in both dataframes."""
        return len(self.df1_unq_rows) == len(self.df2_unq_rows) == 0
    
    def count_matching_rows(self):
        """Count matching rows."""
        match_columns = []
        for column in self.intersect_columns():
            if column not in self.join_columns:
                match_columns.append(column + "_match")

        if len(match_columns) > 0:
            all_matches = pc.and_kleene(*[self.intersect_rows[col] for col in match_columns])
            return pc.sum(all_matches).as_py()
        else:
            # corner case where it is just the join columns that make the dataframes
            if self.intersect_rows.num_rows > 0:
                return self.intersect_rows.num_rows
            else:
                return 0
            
    def intersect_rows_match(self):
        """Check whether the intersect rows all match."""
        if self.intersect_rows.num_rows == 0:
            return False
        
        actual_length = self.intersect_rows.num_rows
        return self.count_matching_rows() == actual_length
    
    def matches(self, ignore_extra_columns = False):
        """Return True or False if the dataframes match.

        Parameters
        ----------
        ignore_extra_columns : bool
            Ignores any columns in one dataframe and not in the other.

        Returns
        -------
        bool
            True or False if the dataframes match.
        """
        return (
            (ignore_extra_columns or self.all_columns_match())
            and self.all_rows_overlap()
            and self.intersect_rows_match()
        )
    
    def report(
            self,
            sample_count = 10, 
            column_count = 10, 
            html_file = None, 
            template_path = None
        ) -> str:
        """Generate a comparison report."""
        template_data = {
            **self._get_column_summary(),
            **self._get_row_summary(),
            **self._get_column_comparison(),
            **self._get_mismatch_stats(sample_count),
            **self._get_unique_rows_data(sample_count, column_count),
            "df1_name": self.df1_name,
            "df2_name": self.df2_name,
            "df1_shape": (self.df1.num_rows, self.df1.num_columns),
            "df2_shape": (self.df2.num_rows, self.df2.num_columns),
            "column_count": column_count,
        }

        # Determine which template to use
        template_name = template_path if template_path else "report_template.j2"

        # Render the main report
        report = render(template_name, **template_data)

        if html_file:
            save_html_report(report, html_file)

        return report

    def sample_mismatch(self, column, sample_count = 10, for_display = False) -> pa.Table | None:
        """Return sample mismatches.

        Get a sub-dataframe which contains the identifying
        columns, and df1 and df2 versions of the column.

        Parameters
        ----------
        column : str
            The raw column name (i.e. without ``_df1`` appended)
        sample_count : int, optional
            The number of sample records to return.  Defaults to 10.
        for_display : bool, optional
            Whether this is just going to be used for display (overwrite the
            column names)

        Returns
        -------
        PyArrow.Table
            A sample of the intersection dataframe, containing only the
            "pertinent" columns, for rows that don't match on the provided
            column.

        None
            When the column being requested is not an intersecting column between dataframes.
        """
        if not self.only_join_columns() and column not in self.join_columns:
            row_cnt = self.intersect_rows.num_rows
            col_match = self.intersect_rows[column + "_match"]
            match_cnt = pc.sum(col_match).as_py()
            sample_count = min(sample_count, row_cnt - match_cnt)
            sample_mismatch_mask = pc.invert(
                pc.coalesce(self.intersect_rows[col_match], False)
            )
            mismatch_rows = self.intersect_rows.filter(sample_mismatch_mask)
            sample = mismatch_rows.slice(0, sample_count)
            return_cols = [
                *self.join_columns,
                column + "_" + self.df1_name,
                column + "_" + self.df2_name,
            ]
            to_return = sample.select(return_cols)
            if for_display:
                to_return = to_return.rename_columns(
                    [
                        *self.join_columns,
                        f"{column} ({self.df1_name})",
                        f"{column} ({self.df2_name})",
                    ]
                )
            return to_return
        else:
            row_cnt = (
                len(self.intersect_rows)
                + len(self.df1_unq_rows)
                + len(self.df2_unq_rows)
            )
            col_match = self.intersect_rows[column]
            match_cnt = col_match.count()
            sample_count = min(sample_count, row_cnt - match_cnt)
            unique_rows = pa.concat_tables([self.df1_unq_rows[column], self.df2_unq_rows[column]])
            sample = unique_rows.slice(0, sample_count)
            return sample

    
    def all_mismatch(self, ignore_matching_cols = False) -> pa.Table:
        """Get all rows with any columns that have a mismatch.

        Returns all df1 and df2 versions of the columns and join
        columns.

        Parameters
        ----------
        ignore_matching_cols : bool, optional
            Whether showing the matching columns in the output or not. The default is False.

        Returns
        -------
        PyArrow.DataFrame
            All rows of the intersection dataframe, containing any columns, that don't match.
        """
        match_list = []
        return_list = []
        if self.only_join_columns():
            LOG.info("Only join keys in data, returning mismatches based on unq_rows")
            return pa.concat_tables([self.df1_unq_rows, self.df2_unq_rows])
        
        for col in self.intersect_rows.schema.names:
            if col.endswith("_match"):
                orig_col_name = col[:-6]

                col_comparison = columns_equal(
                    col_1=self.intersect_rows[orig_col_name + "_" + self.df1_name],
                    col_2=self.intersect_rows[orig_col_name + "_" + self.df2_name],
                    rel_tol=get_column_tolerance(orig_col_name, self._rel_tol_dict),
                    abs_tol=get_column_tolerance(orig_col_name, self._abs_tol_dict),
                    ignore_spaces=self.ignore_spaces,
                    ignore_case=self.ignore_case,
                    comparators=self._get_comparators(),
                )

                if not ignore_matching_cols or (
                    ignore_matching_cols and not col_comparison.all()
                ):
                    LOG.debug(f"Adding column {orig_col_name} to the result.")
                    match_list.append(col)
                    return_list.extend(
                        [
                            orig_col_name + "_" + self.df1_name,
                            orig_col_name + "_" + self.df2_name,
                        ]
                    )
                elif ignore_matching_cols:
                    LOG.debug(
                        f"Column {orig_col_name} is equal in df1 and df2. It will not be added to the result."
                    )
        
        if len(match_list) == 0:
            LOG.info("No match columns found, returning mismatches based on unq_rows")
            return pa.concat_tables(
                [
                    self.df1_unq_rows.select(self.join_columns),
                    self.df2_unq_rows.select(self.join_columns),
                ]
            )
        
        # Get rows where any of the match columns is False
        all_match = pc.and_kleene(*[self.intersect_rows[match_col] for match_col in match_list])
        all_table = self.intersect_rows.append_column("__all", all_match)
        mismatch_mask = pc.invert(pc.fill_null(all_table["__all"], False))
        to_return = all_table.filter(mismatch_mask).select(self.join_columns + return_list)
        return to_return

    def subset(self) -> bool:
        """Check if one dataframe is a subset of the other."""
        return (
            self.df2_unq_columns() == set()
            and len(self.df2_unq_rows) == 0
            and self.intersect_rows_match()
        )


def convert_to_arrow(data_object: ArrowStreamable) -> pa.Table:
    """Convert a streamable dataframe to pyarrow Table."""
    LOG.info(f"Validating {data_object} dataframe as PyArrow streamable")
    if hasattr(data_object, "__arrow_c_stream__"):
        return pa.Table(data_object)
    else:
        raise TypeError("Dataframe is not a pyarrow streamable object")

def columns_equal(
    col_1: pa.Array,
    col_2: pa.Array,
    rel_tol: float = 0,
    abs_tol: float = 0,
    ignore_spaces: bool = False,
    ignore_case: bool = False,
    comparators: List[BaseComparator] | None = None,
    **kwargs
) -> ArrowArrayLike:
    """Compare two columns from a dataframe.

    Returns a True/False series,
    with the same index as column 1.

    - Two nulls (np.nan) will evaluate to True.
    - A null and a non-null value will evaluate to False.
    - Numeric values will use the relative and absolute tolerances.
    - Decimal values (decimal.Decimal) will attempt to be converted to floats
      before comparing
    - Non-numeric values (i.e. where np.isclose can't be used) will just
      trigger True on two nulls or exact matches.

    Parameters
    ----------
    col_1 : Polars.Series
        The first column to look at
    col_2 : Polars.Series
        The second column
    rel_tol : float, optional
        Relative tolerance
    abs_tol : float, optional
        Absolute tolerance
    ignore_spaces : bool, optional
        Flag to strip whitespace (including newlines) from string columns
    ignore_case : bool, optional
        Flag to ignore the case of string columns
    comparators : list of ``BaseComparator``, optional
        A list of custom comparator classes to use to compare columns.
    **kwargs
        Additional keyword arguments to pass to custom comparators.

    Returns
    -------
    ArrowArrayLike
        A series of Boolean values.  True == the values match, False == the
        values don't match.
    """
    compare: ArrowArrayLike
    comparators_ = comparators
    if not comparators_:
        comparators_ = _ARROW_DEFAULT_COMPARATORS

    for comparator in comparators_:
        if isinstance(comparator, PyArrowNumericComparator):
            compare = comparator.compare(col_1, col_2, rtol=rel_tol, atol=abs_tol)
        elif isinstance(comparator, PyArrowStringComparator):
            compare = comparator.compare(
                col_1, col_2, ignore_space=ignore_spaces, ignore_case=ignore_case
            )
        elif isinstance(comparator, PyArrowArrayLikeComparator):
            compare = comparator.compare(col_1, col_2)
        else:
            # for custom comparators pass all the available parameters
            # custom comparators can ignore what they don't need.
            compare = comparator.compare(col_1, col_2, **kwargs)

        if compare is not None:
            LOG.info(
                f"Using comparator: {comparator.__class__.__name__} for column ({col_1.name}, {col_2.name}) comparison."
            )
            return compare

    compare = pa.Array([False] * len(col_1))

def get_merged_columns(
    original_df: ArrowStreamable,
    merged_df: ArrowStreamable,
    suffix: str
) -> List[str]:
    """Get columns from merged dataframe corresponding to original dataframe."""
    columns = []
    for column in original_df.schema.names:
        if column in merged_df.schema.names:
            columns.append(column)
        elif column + "_" + suffix in merged_df.schema.names:
            columns.append(column + "_" + suffix)
        else:
            raise ValueError(f"Column {column} not found in merged dataframe")

    return columns

def calculate_max_diff(col_1: pa.Array, col_2: pa.Array) -> float:
    """Calculate the maximum absolute difference between two numeric pyarrow Arrays."""
    if len(col_1) != len(col_2):
        raise ValueError("Columns must be of the same length to calculate max difference.")
    
    try:
        col_1_float = pc.cast(col_1, pa.float64())
        col_2_float = pc.cast(col_2, pa.float64())
        diff = pc.abs(pc.subtract(col_1_float, col_2_float))
        return pc.cast(pc.max(diff), pa.float64()).as_py()
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return 0.0

# Defining custom temp_column_name since base.py wouldn't be compatible with pyarrow Tables    
def temp_column_name(*dataframes) -> str:
    """Generate a temporary column name that does not exist in either dataframe."""
    counter = 1
    columns = []
    for df in dataframes:
        if df is not None:
            columns.extend(df.schema.names)
    while True:
        tmp = f"_temp_{counter}"
        if tmp not in columns:
            return tmp
        counter += 1

def generate_id_within_group(
    dataframe: pa.Table,
    join_columns: List[str]
) -> pa.Array:
    """Generate a unique ID within groups defined by join_columns."""
    has_nulls = any(pc.any(pc.is_null(dataframe.column(col))).as_py() for col in join_columns)
    if has_nulls:
        # If there are nulls, fill them with a default value
        for col in join_columns:
            col_str = pc.cast(dataframe.column(col), pa.string())
            if pc.any(pc.equal(col_str, DEFAULT_VALUE)).as_py():
                raise ValueError(f"{DEFAULT_VALUE} was found in your join columns.")

        filled_columns = [
            pc.fill_null(pc.cast(dataframe.column(col), pa.string()), DEFAULT_VALUE)
            for col in join_columns
        ]
    else:
        filled_columns = [dataframe.column(col) for col in join_columns]

    # Create composite key for grouping
    if len(filled_columns) == 1:
        group_key = filled_columns[0]
    else:
        group_key = filled_columns[0]
        for col in filled_columns[1:]:
            group_key = pc.binary_join_element_wise(
                pc.cast(group_key, pa.string()),
                pc.cast(col, pa.string()),
                separator="||"
            )

    # Get count with each group
    indices = pc.sort_indices(group_key)
    sorted_keys = pc.take(group_key, indices)
    n = len(dataframe)
    counts = [0] * n
    group_counts = dict[str, int] = {}
    sorted_keys_list = sorted_keys.to_pylist()
    indices_list = indices.to_pylist()

    for i, key in enumerate(sorted_keys_list):
        key_str = str(key)
        if key_str not in group_counts:
            group_counts[key_str] = 0
        group_counts[key_str] += 1
        counts[indices_list[i]] = group_counts[key_str]

    # Generate a unique ID within each group
    return pa.array(counts, type=pa.int64())
