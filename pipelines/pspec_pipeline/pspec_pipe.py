#!/usr/bin/env python2
"""
pspec_pipe.py
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
# Run Jacknife Data Difference
#-------------------------------------------------------------------------------
if run_diff:
    # get algorithm parameters
    globals().update(cf['algorithm']['diff'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting {} visibility data difference: {}\n".format("-"*60, diff_type, time), f=lf, verbose=verbose)

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

    # launch pspec jobs
    failures = hp.utils.job_monitor(pspec, range(len(jobs)), "PSPEC", lf=lf, maxiter=maxiter, verbose=verbose)

    # print failures if they exist
    if len(failures) > 0:
        hp.utils.log("\nSome PSPEC jobs failed after {} tries:\n{}".format(maxiter, '\n'.join(["job {}: {}".format(i, str(jobs.keys()[i])) for i in failures])), f=lf, verbose=verbose)

    # print to log
    time = datetime.utcnow()
    hp.utils.log("\nFinished PSPEC pipeline: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)

    # Merge power spectrum files from separate jobs
    hp.utils.log("\nStarting power spectrum file merge: {}\n{}".format(time, '-'*60), f=lf, verbose=verbose)

    # Get all groups
    groups = psc.groups()

    # Define merge function
    def merge(i, filename=filename, groups=groups):
        try:
            psc = hp.PSpecContainer(filename, mode='rw')
            grp = groups[i]
            spectra = [os.path.join(grp, sp) for sp in psc.get_spectra(grp)]
            hp.utils.merge_pspec(psc, spectra=spectra)
        except:
            hp.utils.log("\nPSPEC MERGE job {} errored with:".format(i), f=ef, tb=sys.exc_info(), verbose=verbose)
            return 1

        return 0

    # launch pspec merge jobs
    failures = hp.utils.job_monitor(merge, range(len(groups)), "PSPEC MERGE", lf=lf, maxiter=maxiter, verbose=verbose)

    # print failures if they exist
    if len(failures) > 0:
        hp.utils.log("\nSome PSPEC MERGE jobs failed after {} tries:\n{}".format(maxiter, '\n'.join(["group {}: {}".format(i, str(groups[i])) for i in failures])), f=lf, verbose=verbose)

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
    hp.utils.log("\n{}\nStarting BOOTSTRAP resampling pipeline: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # open container
    psc = hp.PSpecContainer(filename, mode='r')

    # get groups
    groups = psc.get_groups()
    del psc

    # define bootstrap function
    def bootstrap(i, filename=filename, groups=groups, p=cf['algorithm']['bootstrap']):
        try:
            # get container
            psc = hp.PSpecContainer(filename, mode='rw')

            # get spectra
            group = groups[i]
            spectra = psc.spectra(group)

            # run bootstrap
            hp.grouping.bootstrap_run(psc, spectra=spectra, time_avg=p['time_avg'], Nsamples=p['Nsamples'],
                                      seed=p['seed'], normal_std=p['normal_std'], robust_std=p['robust_std'],
                                      conf_ints=p['conf_ints'], keep_samples=p['keep_samples'],
                                      bl_error_tol=p['bl_error_tol'], overwrite=overwrite, verbose=verbose)


        except:
            hp.utils.log("\nBOOTSTRAP job {} errored with:".format(i), f=ef, tb=sys.exc_info(), verbose=verbose)
            return 1

        return 0

    # launch bootstrap jobs
    failures = hp.utils.job_monitor(bootstrap, range(len(groups)), "BOOTSTRAP", lf=lf, maxiter=maxiter, verbose=verbose)

    # print failures if they exist
    if len(failures) > 0:
        hp.utils.log("\nSome BOOTSTRAP jobs failed after {} tries:\n{}".format(maxiter, '\n'.join(["group {}: {}".format(i, str(groups[i])) for i in failures])), f=lf, verbose=verbose)

    # print to log
    time = datetime.utcnow()
    hp.utils.log("\nFinished BOOTSTRAP pipeline: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)

