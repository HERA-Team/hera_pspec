#!/usr/bin/env python
"""
Pipeline script to load a PSpecContainer and bootstrap over redundant baseline-
pair groups to produce errorbars.
"""
import sys
import os
from hera_pspec import pspecdata, grouping
from hera_pspec import uvpspec_utils as uvputils

def get_pspec_bootstrap_argparser():
    a = argparse.ArgumentParser(
           description="argument parser for grouping.bootstrap_average_group()")
    
    # Add list of arguments
    a.add_argument("filename", type=str, 
                   help="Filename of HDF5 container (PSpecContainer) containing "
                        "input power spectra.")
    a.add_argument("samples", type=int, 
                   help="Number of random bootstrap samples to generate.")
    a.add_argument("--outfile", default=None, type=str, 
                   help="Filename of HDF5 container (PSpecContainer) to store "
                        "output bootstrap-averaged power spectra. If not set, "
                        "the input container will be used.")
    a.add_argument("--seed", type=int, default=None, 
                   help="Random seed to use when generating samples.")
    a.add_argument("--groups", default=None, type=list, 
                   help="List of name(s) of the groups in the PSpecContainer to "
                        "process. If not specified, all groups will be processed.")
    a.add_argument("--blpairs", default=None, type=tuple, nargs='*', 
                   help="List of *groups* of integer/tuple pairs containing "
                        "baseline pairs to run the bootstrap resampling on.\n"
                        "  (1) If not specified, sampling will be performed "
                        "over each redundant group.\n  (2) If a list of blpair "
                        "tuples/integers is specified, each will be used as the "
                        "'prototype' identifying a redundant group, and all bls "
                        "within each group will be sampled.\n  (3) If a list "
                        "of *lists* of blpair tuples/integers is specified, "
                        "the sampling will be performed over the blpairs in "
                        "each sub-list.")
    a.add_argument("--output-group", default='avg', type=str, 
                   help="The name of the root group to output the bootstrap-"
                        "averaged power spectra into. Default: 'avg'.")
    a.add_argument("--time-avg", default=False, 
                   help="Whether to average over times as well as blpairs.")
    a.add_argument("--combine-averages", default=False, 
                   help="Whether to combine averages from each UVPSpec in the "
                        "group, or leave them as separate (per-UVPSpec) averages.")
    a.add_argument("--sample-separately", default=False, 
                   help="Whether to bootstrap-sample over blpairs in each "
                        "UVPSpec separately, or sample over all blpairs over "
                        "all UVPSpec objects within a group. Default: False.")
    a.add_argument("--overwrite", default=False, action='store_true', 
                   help="Overwrite output power spectra if they already exist.")
    a.add_argument("--verbose", default=False, action='store_true', 
                   help="Report feedback to standard output.")
    return a


# Parse commandline args
args = get_pspec_bootstrap_argparser()
a = args.parse_args()
kwargs = vars(a) # dict of args

# Get arguments
filename = kwargs.pop('filename')
nsamples = kwargs.pop('samples')
outfile = kwargs.pop('outfile')
inp_groups = kwargs.pop('groups')
out_group = kwargs.pop('output-group')
blpairs = kwargs.pop('blpairs')
seed = kwargs.pop('seed')
time_avg = kwargs.pop('time-avg')
combine_avgs = kwargs.pop('combine-averages')
sample_sep = kwargs.pop('sample-separately')
overwrite = kwargs.pop('overwrite')
verbose = kwargs.pop('verbose')

# Open PSpecContainer containing power spectra to bootstrap
psc = container.PSpecContainer(filename, mode='rw')

# Open output PSpecContainer if needed
if outfile is not None:
    psc_out = container.PSpecContainer(outfile, mode='rw')
else:
    psc_out = None

# Get requested groups
if inp_groups is not None:
    # Operate on a specific group or list of groups
    groups = inp_groups
else:
    # Get all groups from the input PSpecContainer
    groups = psc.groups()

# Loop over the number of requested samples
for i in range(nsamples):
    
    # Loop over specified groups within the input container
    for grp in groups:
        if "avg/" in grp: continue
        sample_id = "%05d" % i
        
        # Run the bootstrap averaging over the specified group
        grouping.bootstrap_average_group(psc, grp, sample_id, blpairs, 
                            combine_averages=combine_avgs, 
                            sample_separately=sample_sep, 
                            time_avg=time_avg, seed=seed, 
                            output_group=out_group, 
                            overwrite=overwrite, psc_out=psc_out, 
                            verbose=verbose)


