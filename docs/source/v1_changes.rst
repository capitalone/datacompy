======================
Datacompy v1.0.0 Changes
======================

This page outlines the major changes introduced in v1.0.0 of Datacompy.

Modular Comparators
-------------------

In versions ``v0.*``, the logic for comparing different data types was embedded within the main ``Compare`` classes for each dataframe type (Pandas, Spark, etc.). This made it difficult to customize or extend the comparison logic.

Starting with ``v1.0.0a1``, we have begun refactoring the comparison logic into a modular framework. The core comparators for different data types are now located in the ``datacompy.comparator`` module.

Each comparator is a class that inherits from ``datacompy.comparator.base.BaseComparator`` and implements a ``compare`` method.

This change has several benefits:

*   **Extensibility**: Users can now create their own custom comparators for new data types or to implement custom comparison logic.
*   **Maintainability**: The comparison logic is now more organized and easier to maintain.
*   **Clarity**: It is clearer how different data types are compared.

For example, you could create a custom comparator for a specific string format:

.. code-block:: python

    from datacompy.comparator.base import BaseComparator

    class MyCustomStringComparator(BaseComparator):
        """
        A custom comparator that ignores case and whitespace.
        """
        def compare(self, series_a, series_b):
            # Custom comparison logic here
            return (series_a.str.lower().str.strip() ==
                    series_b.str.lower().str.strip())

.. note::

    If a comparator is not suitable for a given data type or comparison, its ``compare`` method can return ``None``.
    This will signal ``datacompy`` to fall back and try the next available comparator.
    This "short-circuiting" mechanism allows for building flexible and robust data comparison workflows.

While direct integration of custom comparators is not yet fully exposed in the public API in `1.0.0a1`, this modular foundation paves the way for future enhancements and user-driven extensions.
