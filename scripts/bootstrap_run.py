#!/usr/bin/env python
"""
Pipeline script to load a PSpecContainer and bootstrap over redundant baseline-
pair groups to produce errorbars.
"""
import sys
import os
from hera_pspec import grouping
from hera_cal._cli_tools import parse_args, run_with_profiling

# Parse commandline args
args = grouping.get_bootstrap_run_argparser()
a = parse_args(args)
kwargs = vars(a) # dict of args

# Get arguments
filename = kwargs.pop('filename')

# Run bootstrap
run_with_profiling(
    grouping.bootstrap_run, a, filename=filename, **kwargs
)

