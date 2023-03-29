#!/usr/bin/env python
import sys
import os
from hera_pspec import pspecdata
from hera_cal._cli_tools import parse_args, run_with_profiling

# parse args
args = pspecdata.get_pspec_run_argparser()
a = parse_args(args)

# turn into dictionary
kwargs = vars(a)

# get arguments
dsets = kwargs.pop('dsets')
filename = kwargs.pop('filename')
# we want to compute cross-corr power spectra by default so feed
# the inverse of the include_autocorrs arg.
kwargs['include_crosscorrs'] = not(kwargs.pop('exclude_crosscorrs'))
# get special kwargs
history = ' '.join(sys.argv)

# run pspec
run_with_profiling(
    pspecdata.pspec_run, a, dsets=dsets, filename=filename, history=history, **kwargs
)