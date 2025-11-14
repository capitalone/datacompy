Template Customization Guide
============================

DataComPy allows you to customize the report output by providing your own Jinja2 templates.
This guide explains how to create and use custom templates for comparison reports.

Template Basics
---------------

Custom templates are Jinja2 templates that receive comparison data and format it according to your needs.
The template receives a context dictionary with various comparison metrics and data samples.

Available Template Variables
----------------------------

The following variables are available in the template context:

   +------------------------+--------------------------------------------------------------------------------+
   | Variable               | Description                                                                    |
   +========================+================================================================================+
   | ``column_summary``     | Dict with column statistics including:                                         |
   |                        |                                                                                |
   |                        | - ``common_columns``: List of columns common to both dataframes                |
   |                        | - ``df1_unique``: List of columns unique to df1                                |
   |                        | - ``df2_unique``: List of columns unique to df2                                |
   |                        | - ``df1_name``: Name of the first dataframe                                    |
   |                        | - ``df2_name``: Name of the second dataframe                                   |
   +------------------------+--------------------------------------------------------------------------------+
   | ``row_summary``        | Dict with row statistics including:                                            |
   |                        |                                                                                |
   |                        | - ``match_columns``: List of columns used for matching                         |
   |                        | - ``abs_tol``: Absolute tolerance between two values                           |
   |                        | - ``rel_tol``: Relative tolerance between two values                           |
   |                        | - ``common_rows``: Number of rows in common                                    |
   |                        | - ``df1_unique``: Number of rows unique to df1                                 |
   |                        | - ``df2_unique``: Number of rows unique to df2                                 |
   |                        | - ``unequal_rows``: Number of rows with differences                            |
   |                        | - ``df1_name``: Name of the first dataframe                                    |
   |                        | - ``df2_name``: Name of the second dataframe                                   |
   +------------------------+--------------------------------------------------------------------------------+
   | ``column_comparison``  | Dict with column comparison stats:                                             |
   |                        |                                                                                |
   |                        | - ``unequal_columns``: List of columns with mismatches                         |
   |                        | - ``equal_columns``: List of columns that match exactly                        |
   |                        | - ``unequal_values``: Count of unequal values across all columns               |
   +------------------------+--------------------------------------------------------------------------------+
   | ``mismatch_stats``     | Dict containing:                                                               |
   |                        |                                                                                |
   |                        | - ``stats``: List of dicts with per-column mismatch statistics:                |
   |                        |   - ``column``: Column name                                                    |
   |                        |   - ``match``: Number of matching values                                       |
   |                        |   - ``mismatch``: Number of mismatched values                                  |
   |                        |   - ``null_diff``: Number of null value differences                            |
   |                        |   - ``total``: Total number of comparisons                                     |
   |                        | - ``samples``: Sample rows with mismatched values                              |
   |                        | - ``has_samples``: Boolean indicating if there are any samples                 |
   |                        | - ``has_mismatches``: Boolean indicating if there are any mismatches           |
   +------------------------+--------------------------------------------------------------------------------+
   | ``df1_unique_rows``    | Dict with unique rows in df1:                                                  |
   |                        |                                                                                |
   |                        | - ``has_rows``: Boolean indicating if there are unique rows                    |
   |                        | - ``rows``: Sample of unique rows (as strings)                                 |
   |                        | - ``columns``: List of column names                                            |
   +------------------------+--------------------------------------------------------------------------------+
   | ``df2_unique_rows``    | Dict with unique rows in df2:                                                  |
   |                        |                                                                                |
   |                        | - ``has_rows``: Boolean indicating if there are unique rows                    |
   |                        | - ``rows``: Sample of unique rows (as strings)                                 |
   |                        | - ``columns``: List of column names                                            |
   +------------------------+--------------------------------------------------------------------------------+
   | ``df1_shape``,         | Tuples of (rows, columns) for each dataframe                                   |
   | ``df2_shape``          |                                                                                |
   +------------------------+--------------------------------------------------------------------------------+
   | ``df1_name``,          | Names of the dataframes being compared                                         |
   | ``df2_name``           |                                                                                |
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
    =====================

    ## DataFrames
    - {{ df1_name }}: {{ df1_shape[0] }} rows × {{ df1_shape[1] }} columns
    - {{ df2_name }}: {{ df2_shape[0] }} rows × {{ df2_shape[1] }} columns

    ## Column Summary
    - Common columns: {{ column_summary.common_columns }}
    - Columns only in {{ df1_name }}: {{ column_summary.df1_unique }}
    - Columns only in {{ df2_name }}: {{ column_summary.df2_unique }}

    ## Row Summary
    - Rows with some columns unequal: {{ row_summary.unequal_rows }}
    - Rows with all columns equal: {{ row_summary.equal_rows }}

    {% if mismatch_stats %}
    ## Mismatched Columns
    {% for col in mismatch_stats %}
    - {{ col.column }}: {{ col.unequal_cnt }} mismatches
      - Match rate: {{ "%.2f"|format(col.match_rate * 100) }}%
    {% endfor %}
    {% endif %}

Using a Custom Template
-----------------------

To use your custom template, pass its path to the ``report()`` method:

.. code-block:: python

    from datacompy.core import Compare

    compare = Compare(df1, df2, join_columns=['id'])

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
