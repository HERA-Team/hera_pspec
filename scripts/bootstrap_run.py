#!/usr/bin/env python
"""
Pipeline script to load a PSpecContainer and bootstrap over redundant baseline-
pair groups to produce errorbars.
"""
import sys
import os
from hera_pspec import pspecdata, grouping
from hera_pspec import uvpspec_utils as uvputils

# Parse commandline args
args = grouping.get_bootstrap_run_argparser()
a = args.parse_args()
kwargs = vars(a) # dict of args

# Get arguments
filename = kwargs.pop('filename')

# Run bootstrap
grouping.bootstrap_run(filename, **kwargs)

