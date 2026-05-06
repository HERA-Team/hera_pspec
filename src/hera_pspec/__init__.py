"""
__init__.py file for hera_pspec
"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("hera_pspec")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

from . import conversions, grouping, loss, plot, pspecbeam, pstokes, testing, utils
from . import uvpspec_utils as uvputils
from .container import PSpecContainer
from .parameter import PSpecParam
from .pspecbeam import PSpecBeamFromArray, PSpecBeamGauss, PSpecBeamUV
from .pspecdata import PSpecData
from .uvpspec import UVPSpec
from .uvwindow import FTBeam, UVWindow

del version
del PackageNotFoundError
