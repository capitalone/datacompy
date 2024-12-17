# SPDX-Copyright: Copyright (c) Capital One Services, LLC
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Capital One Services, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging Module.

Module which sets up the basic logging infrustrcuture for the application.
"""

import logging
import sys

# logger formating
BRIEF_FORMAT = "%(levelname)s %(asctime)s - %(name)s: %(message)s"
VERBOSE_FORMAT = (
    "%(levelname)s|%(asctime)s|%(name)s|%(filename)s|"
    "%(funcName)s|%(lineno)d: %(message)s"
)
FORMAT_TO_USE = VERBOSE_FORMAT

# logger levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def get_logger(name=None, log_level=logging.DEBUG):
    """Set the basic logging features for the application.

    Parameters
    ----------
    name : str, optional
        The name of the logger. Defaults to ``None``
    log_level : int, optional
        The logging level. Defaults to ``logging.INFO``

    Returns
    -------
    logging.Logger
        Returns a Logger obejct which is set with the passed in paramters.
        Please see the following for more details:
        https://docs.python.org/2/library/logging.html
    """
    logging.basicConfig(format=FORMAT_TO_USE, stream=sys.stdout, level=log_level)
    logging.captureWarnings(True)
    logger = logging.getLogger(name)
    return logger
