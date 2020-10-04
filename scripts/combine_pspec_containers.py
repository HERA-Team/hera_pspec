#!/usr/bin/env python
"""
Command-line script for merging UVPSpec power spectra
within a PSpecContainer.
"""
from hera_pspec import container

# Parse commandline args
args = container.combine_pspec_containers_argparser()
a = args.parse_args()
if a.input == a.pspec_containers[0]:
    # only run if the input file is first in the list.
    container.combine_pspec_containers(psc_list=a.pspec_containers, group=a.group,
                                       output=a.output, overwrite=a.clobber)
