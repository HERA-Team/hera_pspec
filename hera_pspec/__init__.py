"""
__init__.py file for hera_pspec
"""
import .version
import .conversions
import .bootstrap
from .uvpspec import UVPSpec
from .pspecdata import PSpecData
from .parameter import PSpecParam

# XXX: This will eventually be deprecated
import legacy_pspec as legacy

__version__ = version.version
