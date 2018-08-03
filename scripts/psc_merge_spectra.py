#!/usr/bin/env python
"""
Command-line script for merging UVPSpec power spectra
within a PSpecContainer.
"""
from hera_pspec import container

# Parse commandline args
args = container.get_merge_spectra_argparser()
a = args.parse_args()

container.merge_spectra(a.filename, uvp_split_str=a.uvp_split_str, 
                        ext_split_str=a.ext_split_str, verbose=a.verbose)
