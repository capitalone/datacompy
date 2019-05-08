# -*- coding: utf-8 -*-
#
# Copyright 2017 Capital One Services, LLC
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

"""
Testing out the datacompy functionality
"""
import logging
import sys
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import six
from pandas.util.testing import assert_series_equal
from pytest import raises

import datacompy

try:
    from unittest import mock
except ImportError:
    import mock

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def test_compare_df_setter_bad():
    df = pd.DataFrame([{"a": 1, "A": 2}, {"a": 2, "A": 2}])
    with raises(TypeError, match="df1 must be a pandas DataFrame"):
        compare = datacompy.Compare("a", "a", ["a"])
    with raises(ValueError, match="df1 must have all columns from join_columns"):
        compare = datacompy.Compare(df, df.copy(), ["b"])
    with raises(ValueError, match="df1 must have unique column names"):
        compare = datacompy.Compare(df, df.copy(), ["a"])
    df_dupe = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 3}])
    assert datacompy.Compare(df_dupe, df_dupe.copy(), ["a", "b"]).df1.equals(df_dupe)


def test_compare_df_setter_good():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = pd.DataFrame([{"A": 1, "B": 2}, {"A": 2, "B": 3}])
    compare = datacompy.Compare(df1, df2, ["a"])
    assert compare.df1.equals(df1)
    assert compare.df2.equals(df2)
    assert compare.join_columns == ["a"]
    compare = datacompy.Compare(df1, df2, ["A", "b"])
    assert compare.df1.equals(df1)
    assert compare.df2.equals(df2)
    assert compare.join_columns == ["a", "b"]


def test_compare_df_setter_different_cases():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = pd.DataFrame([{"A": 1, "b": 2}, {"A": 2, "b": 3}])
    compare = datacompy.Compare(df1, df2, ["a"])
    assert compare.df1.equals(df1)
    assert compare.df2.equals(df2)


def test_compare_df_setter_bad_index():
    df = pd.DataFrame([{"a": 1, "A": 2}, {"a": 2, "A": 2}])
    with raises(TypeError, match="df1 must be a pandas DataFrame"):
        compare = datacompy.Compare("a", "a", on_index=True)
    with raises(ValueError, match="df1 must have unique column names"):
        compare = datacompy.Compare(df, df.copy(), on_index=True)


def test_compare_on_index_and_join_columns():
    df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    with raises(Exception, match="Only provide on_index or join_columns"):
        compare = datacompy.Compare(df, df.copy(), on_index=True, join_columns=["a"])


def test_compare_df_setter_good_index():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    compare = datacompy.Compare(df1, df2, on_index=True)
    assert compare.df1.equals(df1)
    assert compare.df2.equals(df2)


def test_columns_overlap():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 3}])
    compare = datacompy.Compare(df1, df2, ["a"])
    assert compare.df1_unq_columns == set()
    assert compare.df2_unq_columns == set()
    assert compare.intersect_columns == set(["a", "b"])


def test_columns_no_overlap():
    df1 = pd.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    df2 = pd.DataFrame([{"a": 1, "b": 2, "d": "oh"}, {"a": 2, "b": 3, "d": "ya"}])
    compare = datacompy.Compare(df1, df2, ["a"])
    assert compare.df1_unq_columns == set(["c"])
    assert compare.df2_unq_columns == set(["d"])
    assert compare.intersect_columns == set(["a", "b"])


def test_10k_rows():
    df1 = pd.DataFrame(np.random.randint(0, 100, size=(10000, 2)), columns=["b", "c"])
    df1.reset_index(inplace=True)
    df1.columns = ["a", "b", "c"]
    df2 = df1.copy()
    df2["b"] = df2["b"] + 0.1
    compare_tol = datacompy.Compare(df1, df2, ["a"], abs_tol=0.2)
    assert compare_tol.matches()
    assert len(compare_tol.df1_unq_rows) == 0
    assert len(compare_tol.df2_unq_rows) == 0
    assert compare_tol.intersect_columns == set(["a", "b", "c"])
    assert compare_tol.all_columns_match()
    assert compare_tol.all_rows_overlap()
    assert compare_tol.intersect_rows_match()

    compare_no_tol = datacompy.Compare(df1, df2, ["a"])
    assert not compare_no_tol.matches()
    assert len(compare_no_tol.df1_unq_rows) == 0
    assert len(compare_no_tol.df2_unq_rows) == 0
    assert compare_no_tol.intersect_columns == set(["a", "b", "c"])
    assert compare_no_tol.all_columns_match()
    assert compare_no_tol.all_rows_overlap()
    assert not compare_no_tol.intersect_rows_match()


@mock.patch("datacompy.logging.debug")
def test_subset(mock_debug):
    df1 = pd.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    df2 = pd.DataFrame([{"a": 1, "c": "hi"}])
    comp = datacompy.Compare(df1, df2, ["a"])
    assert comp.subset()
    assert mock_debug.called_with("Checking equality")


@mock.patch("datacompy.logging.info")
def test_not_subset(mock_info):
    df1 = pd.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "yo"}])
    df2 = pd.DataFrame([{"a": 1, "b": 2, "c": "hi"}, {"a": 2, "b": 2, "c": "great"}])
    comp = datacompy.Compare(df1, df2, ["a"])
    assert not comp.subset()
    assert mock_info.called_with("Sample c mismatch: a: 2, df1: yo, df2: great")


def test_large_subset():
    df1 = pd.DataFrame(np.random.randint(0, 100, size=(10000, 2)), columns=["b", "c"])
    df1.reset_index(inplace=True)
    df1.columns = ["a", "b", "c"]
    df2 = df1[["a", "b"]].sample(50).copy()
    comp = datacompy.Compare(df1, df2, ["a"])
    assert not comp.matches()
    assert comp.subset()


def test_string_joiner():
    df1 = pd.DataFrame([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    df2 = pd.DataFrame([{"ab": 1, "bc": 2}, {"ab": 2, "bc": 2}])
    compare = datacompy.Compare(df1, df2, "ab")
    assert compare.matches()


@pytest.mark.xfail(
    pd.__version__ >= "0.24.0",
    reason="Pandas behaviour change http://pandas.pydata.org/pandas-docs/version/0.24.0rc1/whatsnew/v0.24.0.html",
)
def test_decimal_with_joins():
    df1 = pd.DataFrame([{"a": Decimal("1"), "b": 2}, {"a": Decimal("2"), "b": 2}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}])
    with raises(ValueError):
        compare = datacompy.Compare(df1, df2, "a")


def test_decimal_with_nulls():
    df1 = pd.DataFrame([{"a": 1, "b": Decimal("2")}, {"a": 2, "b": Decimal("2")}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 2}, {"a": 3, "b": 2}])
    compare = datacompy.Compare(df1, df2, "a")
    assert not compare.matches()
    assert compare.all_columns_match()
    assert not compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_joins():
    df1 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df2 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    compare = datacompy.Compare(df1, df2, "a")
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_index_joining():
    df1 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df2 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    compare = datacompy.Compare(df1, df2, on_index=True)
    assert compare.matches()


def test_index_joining_strings_i_guess():
    df1 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df2 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df1.index = df1["a"]
    df2.index = df2["a"]
    df1.index.name = df2.index.name = None
    compare = datacompy.Compare(df1, df2, on_index=True)
    assert compare.matches()


def test_index_joining_non_overlapping():
    df1 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}])
    df2 = pd.DataFrame([{"a": "hi", "b": 2}, {"a": "bye", "b": 2}, {"a": "back fo mo", "b": 3}])
    compare = datacompy.Compare(df1, df2, on_index=True)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.intersect_rows_match()
    assert len(compare.df1_unq_rows) == 0
    assert len(compare.df2_unq_rows) == 1
    assert list(compare.df2_unq_rows["a"]) == ["back fo mo"]


### Duplicate testing!
def test_simple_dupes_one_field():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    compare = datacompy.Compare(df1, df2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_two_fields():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 2}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 2}])
    compare = datacompy.Compare(df1, df2, join_columns=["a", "b"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_index():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}])
    df1.index = df1["a"]
    df2.index = df2["a"]
    df1.index.name = df2.index.name = None
    compare = datacompy.Compare(df1, df2, on_index=True)
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_one_field_two_vals():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    compare = datacompy.Compare(df1, df2, join_columns=["a"])
    assert compare.matches()
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_one_field_two_vals():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 2, "b": 0}])
    compare = datacompy.Compare(df1, df2, join_columns=["a"])
    assert not compare.matches()
    assert len(compare.df1_unq_rows) == 1
    assert len(compare.df2_unq_rows) == 1
    assert len(compare.intersect_rows) == 1
    # Just render the report to make sure it renders.
    t = compare.report()


def test_simple_dupes_one_field_three_to_two_vals():
    df1 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}, {"a": 1, "b": 0}])
    df2 = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 0}])
    compare = datacompy.Compare(df1, df2, join_columns=["a"])
    assert not compare.matches()
    assert len(compare.df1_unq_rows) == 1
    assert len(compare.df2_unq_rows) == 0
    assert len(compare.intersect_rows) == 2
    # Just render the report to make sure it renders.
    t = compare.report()


def test_dupes_from_real_data():
    data = """acct_id,acct_sfx_num,trxn_post_dt,trxn_post_seq_num,trxn_amt,trxn_dt,debit_cr_cd,cash_adv_trxn_comn_cntry_cd,mrch_catg_cd,mrch_pstl_cd,visa_mail_phn_cd,visa_rqstd_pmt_svc_cd,mc_pmt_facilitator_idn_num
100,0,2017-06-17,1537019,30.64,2017-06-15,D,CAN,5812,M2N5P5,,,0.0
200,0,2017-06-24,1022477,485.32,2017-06-22,D,USA,4511,7114,7.0,1,
100,0,2017-06-17,1537039,2.73,2017-06-16,D,CAN,5812,M4J 1M9,,,0.0
200,0,2017-06-29,1049223,22.41,2017-06-28,D,USA,4789,21211,,A,
100,0,2017-06-17,1537029,34.05,2017-06-16,D,CAN,5812,M4E 2C7,,,0.0
200,0,2017-06-29,1049213,9.12,2017-06-28,D,CAN,5814,0,,,
100,0,2017-06-19,1646426,165.21,2017-06-17,D,CAN,5411,M4M 3H9,,,0.0
200,0,2017-06-30,1233082,28.54,2017-06-29,D,USA,4121,94105,7.0,G,
100,0,2017-06-19,1646436,17.87,2017-06-18,D,CAN,5812,M4J 1M9,,,0.0
200,0,2017-06-30,1233092,24.39,2017-06-29,D,USA,4121,94105,7.0,G,
100,0,2017-06-19,1646446,5.27,2017-06-17,D,CAN,5200,M4M 3G6,,,0.0
200,0,2017-06-30,1233102,61.8,2017-06-30,D,CAN,4121,0,,,
100,0,2017-06-20,1607573,41.99,2017-06-19,D,CAN,5661,M4C1M9,,,0.0
200,0,2017-07-01,1009403,2.31,2017-06-29,D,USA,5814,22102,,F,
100,0,2017-06-20,1607553,86.88,2017-06-19,D,CAN,4812,H2R3A8,,,0.0
200,0,2017-07-01,1009423,5.5,2017-06-29,D,USA,5812,2903,,F,
100,0,2017-06-20,1607563,25.17,2017-06-19,D,CAN,5641,M4C 1M9,,,0.0
200,0,2017-07-01,1009433,214.12,2017-06-29,D,USA,3640,20170,,A,
100,0,2017-06-20,1607593,1.67,2017-06-19,D,CAN,5814,M2N 6L7,,,0.0
200,0,2017-07-01,1009393,2.01,2017-06-29,D,USA,5814,22102,,F,"""
    df1 = pd.read_csv(six.StringIO(data), sep=",")
    df2 = df1.copy()
    compare_acct = datacompy.Compare(df1, df2, join_columns=["acct_id"])
    assert compare_acct.matches()
    compare_unq = datacompy.Compare(
        df1, df2, join_columns=["acct_id", "acct_sfx_num", "trxn_post_dt", "trxn_post_seq_num"]
    )
    assert compare_unq.matches()
    # Just render the report to make sure it renders.
    t = compare_acct.report()
    r = compare_unq.report()


def test_strings_with_joins_with_ignore_spaces():
    df1 = pd.DataFrame([{"a": "hi", "b": " A"}, {"a": "bye", "b": "A"}])
    df2 = pd.DataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "A "}])
    compare = datacompy.Compare(df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = datacompy.Compare(df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_strings_with_joins_with_ignore_case():
    df1 = pd.DataFrame([{"a": "hi", "b": "a"}, {"a": "bye", "b": "A"}])
    df2 = pd.DataFrame([{"a": "hi", "b": "A"}, {"a": "bye", "b": "a"}])
    compare = datacompy.Compare(df1, df2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = datacompy.Compare(df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_joins_with_ignore_spaces():
    df1 = pd.DataFrame([{"a": 1, "b": " A"}, {"a": 2, "b": "A"}])
    df2 = pd.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A "}])
    compare = datacompy.Compare(df1, df2, "a", ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = datacompy.Compare(df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_decimal_with_joins_with_ignore_case():
    df1 = pd.DataFrame([{"a": 1, "b": "a"}, {"a": 2, "b": "A"}])
    df2 = pd.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "a"}])
    compare = datacompy.Compare(df1, df2, "a", ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = datacompy.Compare(df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_index_with_joins_with_ignore_spaces():
    df1 = pd.DataFrame([{"a": 1, "b": " A"}, {"a": 2, "b": "A"}])
    df2 = pd.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "A "}])
    compare = datacompy.Compare(df1, df2, on_index=True, ignore_spaces=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = datacompy.Compare(df1, df2, "a", ignore_spaces=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_index_with_joins_with_ignore_case():
    df1 = pd.DataFrame([{"a": 1, "b": "a"}, {"a": 2, "b": "A"}])
    df2 = pd.DataFrame([{"a": 1, "b": "A"}, {"a": 2, "b": "a"}])
    compare = datacompy.Compare(df1, df2, on_index=True, ignore_case=False)
    assert not compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert not compare.intersect_rows_match()

    compare = datacompy.Compare(df1, df2, "a", ignore_case=True)
    assert compare.matches()
    assert compare.all_columns_match()
    assert compare.all_rows_overlap()
    assert compare.intersect_rows_match()


def test_dupes_with_nulls():
    df1 = pd.DataFrame(
        {
            "fld_1": [1, 2, 2, 3, 3, 4, 5, 5],
            "fld_2": ["A", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )
    df2 = pd.DataFrame({"fld_1": [1, 2, 3, 4, 5], "fld_2": ["A", np.nan, np.nan, np.nan, np.nan]})
    comp = datacompy.Compare(df1, df2, join_columns=["fld_1", "fld_2"])
    assert comp.subset()
