
.. _comparator_usage:

===========================
Comparator Framework Usage
===========================

.. versionadded:: 1.0.0

Overview
========

Version 1.0.0 of DataComPy introduces a new, modular comparator framework designed for extensibility and customization.
Previously, the logic for comparing different data types was tightly coupled within each backend's main
``Compare`` class (e.g., ``PandasCompare``). This made it difficult to alter how comparisons were performed.

The new framework moves type-specific comparison logic into a series of independent comparator classes
found in the ``datacompy.comparator`` module. This allows users to create and use their own custom comparators to
handle unique data types or implement specialized comparison logic.

Core Concepts
=============

The framework is built around a few key ideas: the ``BaseComparator`` abstract class, a fallback mechanism, and a pipeline of comparators.

The BaseComparator Class
------------------------

All comparators, both built-in and custom, must inherit from ``datacompy.comparator.base.BaseComparator``.
This class defines the interface that DataComPy's comparison engine expects.

The most important part of this interface is the ``compare`` method:

.. code-block:: python

    from abc import ABC, abstractmethod
    from typing import Any

    class BaseComparator(ABC):
        @abstractmethod
        def compare(self, col1: Any, col2: Any, **kwargs) -> Any:
            """Check if two columns are equal."""
            raise NotImplementedError()

When you create a custom comparator, you must implement this method.

The Fallback Mechanism
----------------------

A crucial feature of the framework is its fallback (or "chain-of-responsibility") mechanism.
When comparing two columns, DataComPy iterates through a list of comparators and calls their ``compare`` method.

.. important::

    If a comparator is not suitable for the data types it receives, it must return ``None``.

Returning ``None`` signals to the comparison engine that the comparator could not handle the columns,
prompting the engine to try the next comparator in the pipeline. If a comparator successfully performs
a comparison, it should return a boolean Series (for Pandas/Polars) or a boolean Column expression
(for Spark/Snowflake) indicating which values are equal.

Comparator Pipeline
-------------------

When you initiate a ``Compare`` object, it creates a pipeline of comparators to use.
If you provide custom comparators via the ``custom_comparators`` parameter, they are placed
at the **beginning** of this pipeline.

The order of execution is:
1. Your list of custom comparators, in the order you provided them.
2. DataComPy's built-in comparators (for arrays, numerics, and strings).

This ensures that your custom logic is always tried first.

Creating a Custom Comparator
============================

To create a custom comparator, you need to:

1. Create a class that inherits from ``datacompy.comparator.base.BaseComparator``.
2. Implement the ``compare`` method.
3. Inside ``compare``, add logic to check if your comparator is applicable to the input columns.
   If not, return ``None``.
4. Add your custom comparison logic and return the boolean result.

Example: A Custom Phone Number Comparator
------------------------------------------

Imagine you have two dataframes with phone numbers stored as strings, but in inconsistent formats
(e.g., with or without parentheses, hyphens, or spaces). DataComPy's default string comparator would
treat ``"(123) 456-7890"`` and ``"1234567890"`` as different.

Let's create a custom comparator to handle this. It will strip all non-numeric characters before
comparing.

.. code-block:: python

    import pandas as pd
    import datacompy
    from datacompy.comparator.base import BaseComparator

    class PhoneNumberComparator(BaseComparator):
        """
        Custom comparator for US phone numbers.

        This comparator strips all non-numeric characters from strings
        before comparing them.
        """
        def compare(self, col1: pd.Series, col2: pd.Series) -> pd.Series | None:
            """
            Compare two series of phone numbers.
            """
            # 1. Check if this comparator is applicable. We only want to act on
            #    columns that are string or object type.
            if not (pd.api.types.is_string_dtype(col1) and pd.api.types.is_string_dtype(col2)):
                return None # Signal to fallback to the next comparator

            # 2. Implement the custom comparison logic.
            # Strip non-numeric characters.
            norm_col1 = col1.str.replace(r'[^0-9]', '', regex=True)
            norm_col2 = col2.str.replace(r'[^0-9]', '', regex=True)

            # 3. Return the boolean result. Handle NaNs to match default behavior.
            return (norm_col1 == norm_col2) | (col1.isnull() & col2.isnull())

Using the Custom Comparator
===========================

To use your custom comparator, pass an instance of it in a list to the ``custom_comparators``
argument of the ``Compare`` constructor. Let's see it in action with our ``PhoneNumberComparator``.

Setup
-----

First, let's create two sample DataFrames. Notice that customer ID `2` has the same phone number
but with different formatting.

.. code-block:: python

    # Sample DataFrames
    df1 = pd.DataFrame({
        'cust_id': [1, 2, 3],
        'phone': ['123-456-7890', '(987) 654-3210', '555-555-5555'],
    })

    df2 = pd.DataFrame({
        'cust_id': [1, 2, 3],
        'phone': ['123-456-7890', '9876543210', '555-123-4567'],
    })

Comparison Without the Custom Comparator
----------------------------------------

If we run the comparison without our custom logic, the phone number for customer `2` and `3`
will be marked as mismatches.

.. code-block:: python

    # Without the custom comparator
    compare_default = datacompy.PandasCompare(
        df1,
        df2,
        join_columns=['cust_id']
    )
    # This will show a mismatch for cust_id 2
    print(compare_default.report())

The report will show two unequal value for the ``phone`` column.

.. code-block:: text

   ...
   Sample Rows with Unequal Values
   -------------------------------

      cust_id     phone (df1)   phone (df2)
   0        2  (987) 654-3210    9876543210
   1        3    555-555-5555  555-123-4567
   ...

Comparison With the Custom Comparator
-------------------------------------

Now, let's pass our ``PhoneNumberComparator`` to the ``PandasCompare`` object.

.. code-block:: python

    # With the custom comparator
    compare_custom = datacompy.PandasCompare(
        df1,
        df2,
        join_columns=['cust_id'],
        custom_comparators=[PhoneNumberComparator()]
    )
    # This will now show a match for cust_id 2's phone
    print(compare_custom.report())


This time, our custom logic is applied first. It correctly identifies that the phone numbers for
customer `2` are the same after normalization. The report will now correctly show that customer `3`
is a mismatch only.

.. code-block:: text

    ...
    Sample Rows with Unequal Values
    -------------------------------

       cust_id   phone (df1)   phone (df2)
    0        3  555-555-5555  555-123-4567
    ...

This example demonstrates how you can easily extend DataComPy to handle your project's
specific data comparison needs.
