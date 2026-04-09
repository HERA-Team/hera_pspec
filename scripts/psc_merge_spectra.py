#!/usr/bin/env python
"""
Command-line script for merging UVPSpec power spectra
within a PSpecContainer.
"""
from hera_pspec import container
from hera_cal._cli_tools import parse_args, run_with_profiling

# Parse commandline args
args = container.get_combine_psc_argparser()
a = parse_args(args)

run_with_profiling(
    container.combine_psc_spectra, a,
    psc=a.filename,
    uvp_split_str=a.uvp_split_str,
    ext_split_str=a.ext_split_str, 
    verbose=a.verbose
)
