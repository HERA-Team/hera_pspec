"""
__init__.py file for hera_pspec
"""
import version
import conversions
import bootstrap

from pspecdata import PSpecData

# XXX: This will eventually be deprecated
import pspec as legacy

__version__ = version.version
