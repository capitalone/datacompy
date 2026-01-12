"""Native arrow based comparison."""

import logging
from typing import Any, Callable, Dict, Iterable, List, Tuple, cast

from orderedset import OrderedSet
import pyarrow as pa
import pyarrow.compute as pc
from datacompy._typing import ArrowStreamable

from datacompy.base import (
    LOG,
    BaseCompare,
    df_to_str,
    get_column_tolerance,
    render,
    save_html_report,
    temp_column_name,
    _validate_tolerance_parameter
)

class PyArrowCompare(BaseCompare):
    """Class to compare two pyarrow Tables."""

    #TODO: Implement mehthods
    