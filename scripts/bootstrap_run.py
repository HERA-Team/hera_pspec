#!/usr/bin/env python
"""Deprecated shim. Use ``pspec bootstrap`` instead."""

import sys
import warnings

from hera_pspec.cli import app

warnings.warn(
    "scripts/bootstrap_run.py is deprecated and will be removed in a future release; "
    "use `pspec bootstrap` instead.",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    sys.argv = ["pspec", "bootstrap", *sys.argv[1:]]
    app()
