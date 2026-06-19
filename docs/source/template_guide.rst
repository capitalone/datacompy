Template Customization Guide
============================

DataComPy allows you to customize the report output by providing your own Jinja2 templates.
This guide explains how to create and use custom templates for comparison reports.

Template Basics
---------------

Custom templates are Jinja2 templates that receive comparison data and format it according to your needs.
The template context is produced by calling ``dataclasses.asdict()`` on the
:class:`~datacompy.report.ReportData` instance, so every field is passed in
with its **typed value** — no pre-formatting is applied.  All formatting
decisions belong in the template.

Available Template Variables
----------------------------

The following variables are available in the template context.  They mirror
the fields of :class:`~datacompy.report.ReportData` and its nested dataclasses.

   +------------------------+--------------------------------------------------------------------------------+
   | Variable               | Description                                                                    |
   +========================+================================================================================+
   | ``df1_name``,          | Names of the dataframes being compared (str).                                  |
   | ``df2_name``           |                                                                                |
   +------------------------+--------------------------------------------------------------------------------+
   | ``df1_shape``,         | Tuples of ``(rows, columns)``. Index ``[0]`` is row count,                     |
   | ``df2_shape``          | ``[1]`` is column count.                                                       |
   +------------------------+--------------------------------------------------------------------------------+
   | ``column_count``       | Maximum number of columns shown in unique-row sample tables (int).             |
   +------------------------+--------------------------------------------------------------------------------+
   | ``column_summary``     | Dict with column statistics including:                                         |
   |                        |                                                                                |
   |                        | - ``common_columns`` (int): count of columns in both DataFrames                |
   |                        | - ``df1_unique`` (int): number of columns only in df1                          |
   |                        | - ``df1_unique_columns`` (list of str): names of those columns                 |
   |                        | - ``df2_unique`` (int): number of columns only in df2                          |
   |                        | - ``df2_unique_columns`` (list of str): names of those columns                 |
   |                        | - ``df1_name``, ``df2_name`` (str): DataFrame labels                           |
   +------------------------+--------------------------------------------------------------------------------+
   | ``row_summary``        | Dict with row statistics including:                                            |
   |                        |                                                                                |
   |                        | - ``match_columns`` (list of str): join column names; empty                    |
   |                        |   list when matching on index                                                  |
   |                        | - ``on_index`` (bool): true when the comparison was joined on index            |
   |                        | - ``has_duplicates`` (bool): true when duplicate join-key values exist         |
   |                        | - ``abs_tol``, ``rel_tol``: float (global) or                                  |
   |                        |   ``dict[str, float]`` per-column tolerances                                   |
   |                        | - ``common_rows`` (int): rows present in both DataFrames                       |
   |                        | - ``df1_unique``, ``df2_unique`` (int): row counts unique to                   |
   |                        |   each side                                                                    |
   |                        | - ``unequal_rows`` (int): common rows where at least one                       |
   |                        |   compared column differs                                                      |
   |                        | - ``equal_rows`` (int): common rows where all compared columns                 |
   |                        |   match                                                                        |
   |                        | - ``df1_name``, ``df2_name`` (str): DataFrame labels                           |
   +------------------------+--------------------------------------------------------------------------------+
   | ``column_comparison``  | Dict with column comparison stats:                                             |
   |                        |                                                                                |
   |                        | - ``unequal_columns`` (int): columns with at least one                         |
   |                        |   unequal value                                                                |
   |                        | - ``equal_columns`` (int): columns where all values match                      |
   |                        | - ``unequal_values`` (int): total individual cell mismatches                   |
   +------------------------+--------------------------------------------------------------------------------+
   | ``mismatch_stats``     | Dict containing:                                                               |
   |                        |                                                                                |
   |                        | - ``has_mismatches`` (bool): true when at least one column                     |
   |                        |   has unequal values or types                                                  |
   |                        | - ``has_samples`` (bool): true when sample rows are available                  |
   |                        | - ``stats`` (list of dict): one entry per mismatched column,                   |
   |                        |   sorted by column name. Each dict has ``column`` (str),                       |
   |                        |   ``dtype1`` (str), ``dtype2`` (str), ``unequal_cnt`` (int),                   |
   |                        |   ``max_diff`` (float), ``null_diff`` (int),                                   |
   |                        |   ``rel_tol`` (float), ``abs_tol`` (float)                                     |
   |                        | - ``samples`` (list of str): pre-rendered ASCII tables of                      |
   |                        |   sample mismatched rows, one per column with mismatches                       |
   |                        | - ``df1_name``, ``df2_name`` (str): DataFrame labels                           |
   +------------------------+--------------------------------------------------------------------------------+
   | ``df1_unique_rows``,   | Dict with sample rows present in only one DataFrame:                           |
   | ``df2_unique_rows``    |                                                                                |
   |                        | - ``has_rows`` (bool): true when at least one unique row exists                |
   |                        | - ``rows`` (str): pre-rendered ASCII table; empty string when                  |
   |                        |   there are no unique rows                                                     |
   +------------------------+--------------------------------------------------------------------------------+

Formatting typed values in templates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because field values are typed (not pre-formatted), use Jinja2 expressions
to produce display strings where needed.  Three common patterns:

.. code-block:: jinja

    {# match_columns is a list; on_index is a bool #}
    Matched on: {{ "index" if row_summary.on_index else row_summary.match_columns | join(", ") }}

    {# has_duplicates is a bool #}
    Duplicates: {{ "Yes" if row_summary.has_duplicates else "No" }}

    {# df1_unique is an int; df1_unique_columns is a list #}
    Unique to df1: {{ column_summary.df1_unique ~ " " ~ column_summary.df1_unique_columns if column_summary.df1_unique_columns else column_summary.df1_unique }}


Creating a Custom Template
--------------------------

To create a custom template:

1. Create a new file with a ``.j2`` extension
2. Use Jinja2 syntax to format the output
3. Reference the available template variables as needed

Example Template
----------------

Here's a simple example template that shows basic comparison metrics:

.. code-block:: jinja

    # Data Comparison Report
    =======================

    ## DataFrames
    - {{ df1_name }}: {{ df1_shape[0] }} rows x {{ df1_shape[1] }} columns
    - {{ df2_name }}: {{ df2_shape[0] }} rows x {{ df2_shape[1] }} columns

    ## Column Summary
    - Common columns: {{ column_summary.common_columns }}
    - Columns only in {{ df1_name }}: {{ column_summary.df1_unique ~ " " ~ column_summary.df1_unique_columns if column_summary.df1_unique_columns else column_summary.df1_unique }}
    - Columns only in {{ df2_name }}: {{ column_summary.df2_unique ~ " " ~ column_summary.df2_unique_columns if column_summary.df2_unique_columns else column_summary.df2_unique }}

    ## Row Summary
    - Matched on: {{ "index" if row_summary.on_index else row_summary.match_columns | join(", ") }}
    - Rows in common: {{ row_summary.common_rows }}
    - Rows with all columns equal: {{ row_summary.equal_rows }}
    - Rows with some columns unequal: {{ row_summary.unequal_rows }}

    {% if mismatch_stats.has_mismatches %}
    ## Mismatched Columns
    {% for col in mismatch_stats.stats %}
    - {{ col.column }}: {{ col.unequal_cnt }} unequal value(s), max diff {{ "%.4f"|format(col.max_diff) }}
    {% endfor %}
    {% endif %}

Using a Custom Template
-----------------------

To use your custom template, pass its path to the ``report()`` method:

.. code-block:: python

    from datacompy import PandasCompare

    compare = PandasCompare(df1, df2, join_columns=['id'])

    # Generate report with custom template
    report = compare.report(template_path='path/to/your/template.j2')
    print(report)

Template Path Resolution
------------------------

The template path can be:

1. An absolute path to a template file
2. A path relative to the current working directory
3. A filename in the default templates directory (``datacompy/templates/``)

Jinja2 Template Features
------------------------

You can use all standard Jinja2 features in your templates, including:

- Control structures (``{% if %}``, ``{% for %}``, etc.)
- Filters (``{{ value|upper }}``, ``{{ value|default('N/A') }}``, etc.)
- Macros for reusable components
- Template inheritance

For more information on Jinja2 templating, see the `Jinja2 documentation <https://jinja.palletsprojects.com/en/3.1.x/templates/>`_.

Default Template Reference
--------------------------

The default template used by DataComPy is available in the source code at ``datacompy/templates/report_template.j2``.
You can use this as a reference when creating your own templates.
