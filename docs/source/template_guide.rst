Template Customization Guide
===========================

DataComPy allows you to customize the report output by providing your own Jinja2 templates.
This guide explains how to create and use custom templates for comparison reports.

Template Basics
---------------

Custom templates are Jinja2 templates that receive comparison data and format it according to your needs.
The template receives a context dictionary with various comparison metrics and data samples.

Available Template Variables
----------------------------

The following variables are available in the template context:

.. list-table:: Template Context Variables
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``column_summary``
     - Dict with keys ``common_columns``, ``df1_unique``, ``df2_unique``
   * - ``row_summary``
     - Dict with keys ``equal_rows``, ``unequal_rows``, ``only_df1_rows``, ``only_df2_rows``
   * - ``mismatch_stats``
     - List of dicts with column mismatch statistics
   * - ``df1_shape``, ``df2_shape``
     - Tuples of (rows, columns) for each dataframe
   * - ``df1_name``, ``df2_name``
     - Names of the dataframes being compared
   * - ``join_columns``
     - List of columns used for joining
   * - ``df1_unq_rows``, ``df2_unq_rows``
     - Sample rows unique to each dataframe
   * - ``mismatch_sample``
     - Sample of rows with mismatched values

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

    from datacompy import Compare

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
