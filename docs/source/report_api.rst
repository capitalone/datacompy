Report API
==========

DataComPy's ``datacompy.report`` module provides a typed data model and
rendering class that every backend uses to generate comparison reports.
All backends share the same output structure, making it easy to build
custom dashboards, export to JSON, or plug in alternative templates.

Programmatic Access
-------------------

Call ``compare.build_report_data()`` on any compare object to get a
:class:`~datacompy.report.ReportData` instance **without** triggering
any rendering:

.. code-block:: python

    import pandas as pd
    from datacompy import PandasCompare

    df1 = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df2 = pd.DataFrame({"id": [1, 2, 3], "val": [10, 99, 30]})

    compare = PandasCompare(df1, df2, join_columns="id")
    data = compare.build_report_data()

    # Inspect structured fields directly
    print(data.row_summary.unequal_rows)          # 1
    print(data.mismatch_stats.stats[0].column)   # 'val'
    print(data.column_summary.common_columns)     # 2

The same method is available on all backends (``PolarsCompare``,
``SparkSQLCompare``, ``SnowflakeCompare``).

Rendering and Export
--------------------

Wrap a :class:`~datacompy.report.ReportData` in a
:class:`~datacompy.report.Report` for rendering:

.. code-block:: python

    from datacompy import Report

    data = compare.build_report_data()
    rep = Report(data)

    # Plain-text report (same as compare.report())
    print(rep.render())

    # Save HTML file
    rep.save("comparison.html")

    # JSON-serializable dict (for dashboards / APIs)
    import json
    payload = json.dumps(rep.to_dict())

Custom Templates
~~~~~~~~~~~~~~~~

Pass a ``template_path`` to use your own Jinja2 template:

.. code-block:: python

    rep = Report(data, template_path="my_report.j2")
    print(rep.render())

The template receives the same context dict as the default
``report_template.j2``; see :doc:`template_guide` for the full variable
reference.

Data Classes
------------

.. automodule:: datacompy.report
   :members: ReportData, ColumnSummary, RowSummary, ColumnComparison,
             MismatchStat, MismatchStats, UniqueRowsData, Report
   :show-inheritance:
   :undoc-members:
