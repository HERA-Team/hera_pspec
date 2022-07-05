"""
__init__.py file for hera_pspec
"""
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("hera_pspec")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

from . import conversions, grouping, pspecbeam, plot, pstokes, testing, utils
from . import uvpspec_utils as uvputils

from .uvpspec import UVPSpec
from .uvwindow import UVWindow, FTBeam
from .pspecdata import PSpecData
from .container import PSpecContainer
from .parameter import PSpecParam
from .pspecbeam import PSpecBeamUV, PSpecBeamGauss, PSpecBeamFromArray



del version
del PackageNotFoundError