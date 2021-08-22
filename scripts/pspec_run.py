#!/usr/bin/env python
import sys
import os
from hera_pspec import pspecdata

# parse args
args = pspecdata.get_pspec_run_argparser()
a = args.parse_args()

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
pspecdata.pspec_run(dsets, filename, history=history, **kwargs)
