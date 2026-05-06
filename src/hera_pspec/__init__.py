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

from . import conversions as conversions
from . import grouping as grouping
from . import loss as loss
from . import plot as plot
from . import pspecbeam as pspecbeam
from . import pstokes as pstokes
from . import testing as testing
from . import utils as utils
from . import uvpspec_utils as uvputils
from .container import PSpecContainer as PSpecContainer
from .parameter import PSpecParam as PSpecParam
from .pspecbeam import PSpecBeamFromArray as PSpecBeamFromArray
from .pspecbeam import PSpecBeamGauss as PSpecBeamGauss
from .pspecbeam import PSpecBeamUV as PSpecBeamUV
from .pspecdata import PSpecData as PSpecData
from .uvpspec import UVPSpec as UVPSpec
from .uvwindow import FTBeam as FTBeam
from .uvwindow import UVWindow as UVWindow

__all__ = [
    "__version__",
    "FTBeam",
    "PSpecBeamFromArray",
    "PSpecBeamGauss",
    "PSpecBeamUV",
    "PSpecContainer",
    "PSpecData",
    "PSpecParam",
    "UVPSpec",
    "UVWindow",
    "conversions",
    "grouping",
    "loss",
    "plot",
    "pspecbeam",
    "pstokes",
    "testing",
    "utils",
    "uvputils",
]

del version
del PackageNotFoundError
