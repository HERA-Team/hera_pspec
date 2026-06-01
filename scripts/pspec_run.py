#!/usr/bin/env python
"""Deprecated shim. Use ``pspec run`` instead."""

import sys
import warnings

from hera_pspec.cli import app

warnings.warn(
    "scripts/pspec_run.py is deprecated and will be removed in a future release; "
    "use `pspec run` instead.",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    sys.argv = ["pspec", "run", *sys.argv[1:]]
    app()
