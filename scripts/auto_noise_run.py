#!/usr/bin/env python
"""Deprecated shim. Use ``pspec auto-noise`` instead."""

import sys
import warnings

from hera_pspec.cli import app

warnings.warn(
    "scripts/auto_noise_run.py is deprecated and will be removed in a future release; "
    "use `pspec auto-noise` instead.",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    sys.argv = ["pspec", "auto-noise", *sys.argv[1:]]
    app()
