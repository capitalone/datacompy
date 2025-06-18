import logging

LOG = logging.getLogger(__name__)

# PySpark imports
try:
    import pyspark as ps
    import pyspark.sql.functions as psf
except ImportError:
    ps = None
    psf = None
    LOG.warning(
        "Please note that you are missing the optional dependency: pyspark. "
        "If you need to use this functionality it must be installed."
    )

# Snowflake imports
try:
    import snowflake.snowpark as sp
    import snowflake.snowpark.functions as spf
    import snowflake.snowpark.types as spt

except ImportError:
    sp = None
    spf = None
    spt = None
    NUMERIC_SNOWFLAKE_TYPES = None
    LOG.warning(
        "Please note that you are missing the optional dependency: snowflake-snowpark-python. "
        "If you need to use this functionality it must be installed."
    )
