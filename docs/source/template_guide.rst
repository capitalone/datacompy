Template Customization Guide
============================

DataComPy allows you to customize the report output by providing your own Jinja2 templates.
This guide explains how to create and use custom templates for comparison reports.

Template Basics
---------------

Custom templates are Jinja2 templates that receive comparison data and format it according to your needs.
The template receives a context dictionary produced by ``ReportData.to_template_context()``.

Available Template Variables
----------------------------

The following variables are available in the template context. A few fields
(``row_summary.match_columns``, ``row_summary.has_duplicates``,
``column_summary.df1_unique``, ``column_summary.df2_unique``) are
pre-formatted as display strings rather than raw values so the default
template can interpolate them directly.

   +------------------------+--------------------------------------------------------------------------------+
   | Variable               | Description                                                                    |
   +========================+================================================================================+
   | ``df1_name``,          | Names of the dataframes being compared.                                        |
   | ``df2_name`` (str)     |                                                                                |
   +------------------------+--------------------------------------------------------------------------------+
   | ``df1_shape``,         | Tuples of ``(rows, columns)``. Index ``[0]`` is row count,                     |
   | ``df2_shape`` (tuple)  | ``[1]`` is column count.                                                       |
   +------------------------+--------------------------------------------------------------------------------+
   | ``column_count``       | Maximum number of columns shown in unique-row sample tables.                   |
   | (int)                  |                                                                                |
   +------------------------+--------------------------------------------------------------------------------+
   | ``column_summary``     | Dict with column statistics including:                                         |
   | (dict)                 |                                                                                |
   |                        | - ``common_columns`` (int): count of columns in both DataFrames                |
   |                        | - ``df1_unique``: pre-formatted string like                                    |
   |                        |   ``"3 ['col_a','col_b']"`` when df1 has unique columns; bare                  |
   |                        |   integer ``0`` when there are none                                            |
   |                        | - ``df2_unique``: same shape as ``df1_unique`` for df2                         |
   |                        | - ``df1_name``, ``df2_name`` (str): DataFrame labels                           |
   +------------------------+--------------------------------------------------------------------------------+
   | ``row_summary``        | Dict with row statistics including:                                            |
   | (dict)                 |                                                                                |
   |                        | - ``match_columns`` (str): comma-joined join columns (e.g.                     |
   |                        |   ``"id, date"``), or ``"index"`` when matching on index                       |
   |                        | - ``has_duplicates`` (str): ``"Yes"`` or ``"No"``                              |
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
   | (dict)                 |                                                                                |
   |                        | - ``unequal_columns`` (int): columns with at least one                         |
   |                        |   unequal value                                                                |
   |                        | - ``equal_columns`` (int): columns where all values match                      |
   |                        | - ``unequal_values`` (int): total individual cell mismatches                   |
   +------------------------+--------------------------------------------------------------------------------+
   | ``mismatch_stats``     | Dict containing:                                                               |
   | (dict)                 |                                                                                |
   |                        | - ``has_mismatches`` (bool): true when at least one column                     |
   |                        |   has unequal values or types                                                  |
   |                        | - ``has_samples`` (bool): true when sample rows are available                  |
   |                        | - ``stats`` (list of dict): one entry per mismatched column,                   |
   |                        |   sorted by column name; keys are ``column``, ``dtype1``,                      |
   |                        |   ``dtype2``, ``unequal_cnt``, ``max_diff``, ``null_diff``,                    |
   |                        |   ``rel_tol``, ``abs_tol``                                                     |
   |                        | - ``samples`` (list of str): pre-rendered ASCII tables of                      |
   |                        |   sample mismatched rows, one per column with mismatches                       |
   |                        | - ``df1_name``, ``df2_name`` (str): DataFrame labels                           |
   +------------------------+--------------------------------------------------------------------------------+
   | ``df1_unique_rows``,   | Dict with sample rows present in only one DataFrame:                           |
   | ``df2_unique_rows``    |                                                                                |
   | (dict)                 | - ``has_rows`` (bool): true when at least one unique row exists                |
   |                        | - ``rows`` (str): pre-rendered ASCII table; empty string when                  |
   |                        |   there are no unique rows                                                     |
   +------------------------+--------------------------------------------------------------------------------+


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
    - Columns only in {{ df1_name }}: {{ column_summary.df1_unique }}
    - Columns only in {{ df2_name }}: {{ column_summary.df2_unique }}

    ## Row Summary
    - Matched on: {{ row_summary.match_columns }}
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
