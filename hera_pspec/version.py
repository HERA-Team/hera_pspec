import warnings
from . import utils

def history_string(notes=''):
    """
    Creates a standardized history string that all functions that write to
    disk can use. Optionally add notes.

    Deprecated: please use ``utils.history_string()`` instead.
    """
    warnings.warn("version.history_string is deprecated. Please use utils.history_string.")
    return utils.history_string(notes)