"""
__init__.py file for hera_pspec
"""
import version
import conversions
from dataset import DataSet
import pspec as legacy # XXX: This will eventually be deprecated

__version__ = version.version
