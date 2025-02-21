"""A Python package for analyzing skeletons."""

import logging
from importlib.metadata import PackageNotFoundError, version

from rich.logging import RichHandler

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

try:
    __version__ = version("skeleplex-v2")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Kevin Yamauchi"
__email__ = "kevin.yamauchi@gmail.com"
