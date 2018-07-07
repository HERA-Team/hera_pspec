#!/usr/bin/env python2
"""
psepc_pipe.py
-----------------------------------------
Copyright (c) 2018 The HERA Collaboration

This script is used as the IDR2.1 power
spectrum pipeline.

See pspec_pipe.yaml for relevant parameter selections.
"""
import multiprocess
import numpy as np
import hera_cal as hc
import hera_pspec as hp
import hera_qm as hq
from pyuvdata import UVData
import pyuvdata.utils as uvutils
import os
import sys
import glob
import yaml
from datetime import datetime
import uvtools as uvt
import json
import itertools
import aipy
import shutil
from collections import OrderedDict as odict


#-------------------------------------------------------------------------------
# Parse YAML Configuration File
#-------------------------------------------------------------------------------
# get config and load dictionary
config = sys.argv[1]
cf = hp.utils.load_config(config)

# update globals with IO params, data and analysis params
globals().update(cf['io'])
globals().update(cf['data'])
globals().update(cf['analysis'])

# get common suffix
data_suffix = os.path.splitext(data_template)[1][1:]

# open logfile
logfile = os.path.join(out_dir, logfile)
if os.path.exists(logfile) and overwrite == False:
    raise IOError("logfile {} exists and overwrite == False, quitting pipeline...".format(logfile))
lf = open(logfile, "w")
if joinlog:
    ef = lf
else:
    ef = open(os.path.join(out_dir, errfile), "w")
time = datetime.utcnow()
hp.utils.log("Starting pspec pipeline on {}\n{}\n".format(time, '-'*60), f=lf, verbose=verbose)
hp.utils.log(json.dumps(cf, indent=1) + '\n', f=lf, verbose=verbose)

# Create multiprocesses
if multiproc:
    pool = multiprocess.Pool(nproc)
    M = pool.map
else:
    M = map

# change to working dir
os.chdir(work_dir)

#-------------------------------------------------------------------------------
# Run Jacknife Split
#-------------------------------------------------------------------------------
if run_split:
    # get algorithm parameters
    globals().update(cf['algorithm']['split'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting {} jacknife split: {}\n".format("-"*60, split_type, time), f=lf, verbose=verbose)

    raise NotImplementedError


#-------------------------------------------------------------------------------
# Run OQE Pipeline
#-------------------------------------------------------------------------------
if run_pspec:
    # get algorithm parameters
    globals().update(cf['algorithm']['pspec'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting pspec OQE: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # configure dataset groupings
    groupings = hp.utils.config_pspec_blpairs(data_template, pol_pairs, group_pairs, exclude_auto_bls=exclude_auto_bls,
                                              exclude_permutations=exclude_permutations, bl_len_range=bl_len_range,
                                              bl_deg_range=bl_deg_range, xants=xants)

    # create dictionary of individual jobs to launch
    jobs = odict()
    for pair in groupings:
        # make Nblps / Nblps_per_job jobs for this pair
        blps = groupings[pair]
        Nblps = len(blps)
        Njobs = Nblps // Nblps_per_job + 1
        job_blps = [blps[i*Nblps_per_job:(i+1)*Nblps_per_job] for i in range(Njobs)]
        for i, _blps in enumerate(job_blps):
            key = tuple(hp.utils.flatten(pair) + [i])
            jobs[key] = _blps

    # create pspec worker function
    def pspec(i, outfile=outfname, dt=data_template, st=std_template, jobs=jobs, pol_pairs=pol_pairs, p=cf['algorithm']['pspec']):
        try:
            # get key
            key = jobs.keys()[i]
            # parse dsets
            dsets = [dt.format(group=key[0], pol=key[2]), dt.format(group=key[1], pol=key[3])]
            dset_labels = [os.path.basename(d) for d in dsets]
            if st is not None and st is not '' and st is not 'None':
                dsets_std = [dt.format(group=key[0], pol=key[2]), dt.format(group=key[1], pol=key[3])]
            else:
                dsets_std = None

            # pspec_run
            hp.pspecdata.pspec_run(dsets, outfile, dsets_std=dsets_std, dset_labels=dset_labels,
                                   dset_pairs=[(0, 1)], spw_ranges=p['spw_ranges'], n_dlys=p['n_dlys'],
                                   pol_pairs=pol_pairs, blpairs=jobs[key], input_data_weight=p['input_data_weight'],
                                   norm=p['norm'], taper=p['taper'], beam=p['beam'], cosmo=p['cosmo'],
                                   rephase_to_dset=p['rephase_to_dset'], trim_dset_lsts=p['trim_dset_lsts'],
                                   broadcast_dset_flags=p['broadcast_dset_flags'], time_thresh=p['time_thresh'],
                                   Jy2mK=p['Jy2mK'], overwrite=overwrite, psname_ext="_{}".format(key[4]),
                                   verbose=verbose)
        except:
            hp.utils.log("\nPSPEC job {} errored with:".format(i), f=ef, tb=sys.exc_info(), verbose=verbose)
            return 1

        return 0

    # run function over jobs
    exit_codes = np.array(M(pspec, range(len(jobs))))

    # inspect for failures
    if np.all(exit_codes != 0):
        # everything failed, raise error
        hp.utils.log("\n{}\nAll PSPEC jobs failed w/ exit codes\n {}: {}\n".format("-"*60, exit_codes, time), f=lf, verbose=verbose)
        raise ValueError("All PSPEC jobs failed")

    # if only a few, try re-run
    failures = np.where(exit_codes != 0)[0]
    counter = 1
    while True:
        if not np.all(exit_codes == 0):
            if counter >= maxiter:
                # break after certain number of tries
                break
            # run function over jobs that failed
            exit_codes = np.array(M(pspec, failures))
            # update counter
            counter += 1
            # update failures
            failures = failures[exit_codes != 0]
        else:
            # all passed
            break

    # print failures if they exist
    if len(failures) > 0:
        hp.utils.log("\nSome PSPEC jobs failed after {} tries:\n{}".format(maxiter, '\n'.join(["job {}: {}".format(i, str(jobs.keys()[i])) for i in failures])), f=lf, verbose=verbose)

    # print to log
    time = datetime.utcnow()
    hp.utils.log("\nFinished PSPEC pipeline: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Run Bootstrap Pipeline
#-------------------------------------------------------------------------------
if run_bootstrap:
    # get algorithm parameters
    globals().update(cf['algorithm']['bootstrap'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting bootstrap resampling: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get datafiles
    datafiles, datapols = uvt.utils.search_data(input_data_template.format(group=groupname), pols, matched_pols=False, reverse_nesting=False, flatten=False)







