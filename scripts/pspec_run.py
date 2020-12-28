#!/usr/bin/env python
import sys
import os
from hera_pspec import pspecdata

# parse args
args = pspecdata.get_pspec_run_argparser()
a = args.parse_args()
print(a.pol_pairs)
# turn into dictionary
kwargs = vars(a)

# get arguments
dsets = kwargs.pop('dsets')
filename = kwargs.pop('filename')

# get special kwargs
history = ' '.join(sys.argv)

# run pspec
pspecdata.pspec_run(dsets, filename, history=history, **kwargs)
