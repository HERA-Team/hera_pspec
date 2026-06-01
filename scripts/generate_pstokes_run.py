#!/usr/bin/env python
"""Deprecated shim. Use ``pspec generate-pstokes`` instead."""

import sys
import warnings

from hera_pspec.cli import app

warnings.warn(
    "scripts/generate_pstokes_run.py is deprecated and will be removed in a future "
    "release; use `pspec generate-pstokes` instead.",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    sys.argv = ["pspec", "generate-pstokes", *sys.argv[1:]]
    app()
