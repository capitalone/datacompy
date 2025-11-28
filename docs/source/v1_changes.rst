======================
Datacompy v1.0.0 Changes
======================

This page outlines the major changes introduced in v1.0.0 of Datacompy.

Modular Comparators
-------------------

In versions ``v0.*``, the logic for comparing different data types was embedded within the main ``Compare`` classes for each dataframe type (Pandas, Spark, etc.). This made it difficult to customize or extend the comparison logic.

With ``v1.0.0``, the comparison logic has been refactored into a modular framework. The core comparators for different data types are now located in the ``datacompy.comparator`` module.

Each comparator is a class that inherits from ``datacompy.comparator.base.BaseComparator`` and implements a ``compare`` method.

This change has several benefits:

*   **Extensibility**: Users can now create their own custom comparators for new data types or to implement custom comparison logic.
*   **Maintainability**: The comparison logic is now more organized and easier to maintain.
*   **Clarity**: It is clearer how different data types are compared.

For example, you could create a custom comparator for a specific string format:

.. code-block:: python

    from datacompy.comparator.base import BaseComparator
    import pandas as pd

    class MyCustomStringComparator(BaseComparator):
        """
        A custom comparator that ignores case and whitespace.
        """
        def compare(self, series_a: pd.Series, series_b: pd.Series) -> pd.Series:
            # Check if we should apply this logic
            if not (series_a.dtype == 'object' and series_b.dtype == 'object'):
                return None

            # Custom comparison logic here
            return (series_a.str.lower().str.strip() ==
                    series_b.str.lower().str.strip())

.. note::

    If a comparator is not suitable for a given data type or comparison, its ``compare`` method should return ``None``.
    This will signal ``datacompy`` to fall back and try the next available comparator in the pipeline.
    This "short-circuiting" mechanism allows for building flexible and robust data comparison workflows.

Using Custom Comparators
------------------------

As of ``v1.0.0``, you can pass a list of your custom comparator instances to the main ``Compare`` object during initialization. Your custom comparators will be tried first, before DataComPy's default comparators.

Here is how you would use the custom comparator from the example above:

.. code-block:: python

    import datacompy
    import pandas as pd

    # Assume df1 and df2 are your DataFrames

    compare = datacompy.PandasCompare(
        df1,
        df2,
        join_columns=['id'],
        df1_name='original',
        df2_name='new',
        custom_comparators=[MyCustomStringComparator()]  # Pass your comparator here
    )

    print(compare.report())
