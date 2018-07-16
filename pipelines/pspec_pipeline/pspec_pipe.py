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

# consolidate IO, data and analysis paramemeter dictionaries
params = odict(cf['io'].items() + cf['data'].items() + cf['analysis'].items())
assert len(params) == len(cf['io']) + len(cf['data']) + len(cf['analysis']), ""\
       "Repeated parameters found within the scope of io, data and analysis dicts"
algs = cf['algorithm']

# open logfile
logfile = os.path.join(params['out_dir'], params['logfile'])
if os.path.exists(logfile) and params['overwrite'] == False:
    raise IOError("logfile {} exists and overwrite == False, quitting pipeline...".format(logfile))
lf = open(logfile, "w")
if params['joinlog']:
    ef = lf
else:
    ef = open(os.path.join(params['out_dir'], params['errfile']), "w")
time = datetime.utcnow()
hp.utils.log("Starting pspec pipeline on {}\n{}\n".format(time, '-'*60), f=lf, verbose=params['verbose'])
hp.utils.log(json.dumps(cf, indent=1) + '\n', f=lf, verbose=params['verbose'])

# Create multiprocesses
if params['multiproc']:
    pool = multiprocess.Pool(params['nproc'])
    M = pool.map
else:
    M = map

# change to working dir
os.chdir(params['work_dir'])

#-------------------------------------------------------------------------------
# Run Visibility Data Difference
#-------------------------------------------------------------------------------
if params['run_diff']:
    # start block
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting {} visibility data difference: {}\n".format("-"*60, algs['diff']['diff_type'], time), f=lf, verbose=params['verbose'])

    raise NotImplementedError


#-------------------------------------------------------------------------------
# Run OQE Pipeline
#-------------------------------------------------------------------------------
if params['run_pspec']:
    # start block
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting pspec OQE: {}\n".format("-"*60, time), f=lf, verbose=params['verbose'])

    # configure dataset groupings
    groupings = hp.utils.config_pspec_blpairs(params['data_template'], params['pol_pairs'], params['group_pairs'], exclude_auto_bls=algs['pspec']['exclude_auto_bls'],
                                              exclude_permutations=algs['pspec']['exclude_permutations'], bl_len_range=params['bl_len_range'],
                                              bl_deg_range=params['bl_deg_range'], xants=params['xants'])

    # create dictionary of individual jobs to launch
    jobs = odict()
    for pair in groupings:
        # make Nblps / Nblps_per_job jobs for this pair
        blps = groupings[pair]
        Nblps = len(blps)
        Njobs = Nblps // algs['pspec']['Nblps_per_job'] + 1
        job_blps = [blps[i*algs['pspec']['Nblps_per_job']:(i+1)*algs['pspec']['Nblps_per_job']] for i in range(Njobs)]
        for i, _blps in enumerate(job_blps):
            key = tuple(hp.utils.flatten(pair) + [i])
            jobs[key] = _blps

    # create pspec worker function
    def pspec(i, jobs=jobs, params=params, alg=algs['pspec'], ef=ef):
        try:
            # get key
            key = jobs.keys()[i]
            # parse dsets
            dsets = [params['data_template'].format(group=key[0], pol=key[2]), params['data_template'].format(group=key[1], pol=key[3])]
            dset_labels = [os.path.basename(d) for d in dsets]
            if params['std_template'] is not None:
                dsets_std = [params['std_template'].format(group=key[0], pol=key[2]), params['std_template'].format(group=key[1], pol=key[3])]
            else:
                dsets_std = None

            # pspec_run
            hp.pspecdata.pspec_run(dsets, alg['outfname'], dsets_std=dsets_std, dset_labels=dset_labels,
                                   dset_pairs=[(0, 1)], spw_ranges=alg['spw_ranges'], n_dlys=alg['n_dlys'],
                                   pol_pairs=params['pol_pairs'], blpairs=jobs[key], input_data_weight=alg['input_data_weight'],
                                   norm=alg['norm'], taper=alg['taper'], beam=alg['beam'], cosmo=alg['cosmo'],
                                   rephase_to_dset=alg['rephase_to_dset'], trim_dset_lsts=alg['trim_dset_lsts'],
                                   broadcast_dset_flags=alg['broadcast_dset_flags'], time_thresh=alg['time_thresh'],
                                   Jy2mK=alg['Jy2mK'], overwrite=params['overwrite'], psname_ext="_{}".format(key[4]),
                                   verbose=params['verbose'])
        except:
            hp.utils.log("\nPSPEC job {} errored with:".format(i), f=ef, tb=sys.exc_info(), verbose=params['verbose'])
            return 1

        return 0

    # launch pspec jobs
    failures = hp.utils.job_monitor(pspec, range(len(jobs)), "PSPEC", lf=lf, maxiter=algs['pspec']['maxiter'], verbose=params['verbose'])

    # print failures if they exist
    if len(failures) > 0:
        hp.utils.log("\nSome PSPEC jobs failed after {} tries:\n{}".format(algs['pspec']['maxiter'], '\n'.join(["job {}: {}".format(i, str(jobs.keys()[i])) for i in failures])), f=lf, verbose=params['verbose'])

    # print to log
    time = datetime.utcnow()
    hp.utils.log("\nFinished PSPEC pipeline: {}\n{}".format(time, "-"*60), f=lf, verbose=params['verbose'])

    # Merge power spectrum files from separate jobs
    hp.utils.log("\nStarting power spectrum file merge: {}\n{}".format(time, '-'*60), f=lf, verbose=params['verbose'])

    # Get all groups
    psc = hp.PSpecContainer(algs['pspec']['outfname'], 'r')
    groups = psc.groups()
    del psc

    # Define merge function
    def merge(i, groups=groups, filename=algs['pspec']['outfname'], ef=ef, params=params):
        try:
            psc = hp.PSpecContainer(filename, mode='rw')
            grp = groups[i]
            hp.container.combine_psc_spectra(psc, groups=[grp], overwrite=params['overwrite'])
        except:
            hp.utils.log("\nPSPEC MERGE job {} errored with:".format(i), f=ef, tb=sys.exc_info(), verbose=params['verbose'])
            return 1

        return 0

    # launch pspec merge jobs
    failures = hp.utils.job_monitor(merge, range(len(groups)), "PSPEC MERGE", lf=lf, maxiter=algs['pspec']['maxiter'], verbose=params['verbose'])

    # print failures if they exist
    if len(failures) > 0:
        hp.utils.log("\nSome PSPEC MERGE jobs failed after {} tries:\n{}".format(algs['pspec']['maxiter'], '\n'.join(["group {}: {}".format(i, str(groups[i])) for i in failures])), f=lf, verbose=params['verbose'])

    # print to log
    time = datetime.utcnow()
    hp.utils.log("\nFinished PSPEC pipeline: {}\n{}".format(time, "-"*60), f=lf, verbose=params['verbose'])


#-------------------------------------------------------------------------------
# Run Bootstrap Pipeline
#-------------------------------------------------------------------------------
if params['run_bootstrap']:
    # start block
    time = datetime.utcnow()
    hp.utils.log("\n{}\nStarting BOOTSTRAP resampling pipeline: {}\n".format("-"*60, time), f=lf, verbose=params['verbose'])

    # ensure outfname is same as psepc
    if params['run_pspec'] and (algs['pspec']['outfname'] != algs['bootstrap']['psc_name']):
        raise ValueError("bootstrap psc_name {} doesn't equal pspec outfname {}".format(algs['bootstrap']['psc_name'], algs['pspec']['outfname']))

    # open container
    psc = hp.PSpecContainer(algs['bootstrap']['psc_name'], mode='r')

    # get groups
    groups = psc.groups()
    del psc

    # define bootstrap function
    def bootstrap(i, groups=groups, ef=ef, alg=algs['bootstrap'], params=params):
        try:
            # get container
            psc = hp.PSpecContainer(alg['psc_name'], mode='rw')

            # get spectra
            group = groups[i]
            spectra = [os.path.join(group, sp) for sp in psc.spectra(group)]

            # run bootstrap
            hp.grouping.bootstrap_run(psc, spectra=spectra, time_avg=alg['time_avg'], Nsamples=alg['Nsamples'],
                                      seed=alg['seed'], normal_std=alg['normal_std'], robust_std=alg['robust_std'],
                                      cintervals=alg['cintervals'], keep_samples=alg['keep_samples'],
                                      bl_error_tol=alg['bl_error_tol'], overwrite=params['overwrite'], verbose=params['verbose'])


        except:
            hp.utils.log("\nBOOTSTRAP job {} errored with:".format(i), f=ef, tb=sys.exc_info(), verbose=params['verbose'])
            return 1

        return 0

    # launch bootstrap jobs
    failures = hp.utils.job_monitor(bootstrap, range(len(groups)), "BOOTSTRAP", lf=lf, maxiter=algs['bootstrap']['maxiter'], verbose=params['verbose'])

    # print failures if they exist
    if len(failures) > 0:
        hp.utils.log("\nSome BOOTSTRAP jobs failed after {} tries:\n{}".format(algs['bootstrap']['maxiter'], '\n'.join(["group {}: {}".format(i, str(groups[i])) for i in failures])), f=lf, verbose=params['verbose'])

    # print to log
    time = datetime.utcnow()
    hp.utils.log("\nFinished BOOTSTRAP pipeline: {}\n{}".format(time, "-"*60), f=lf, verbose=params['verbose'])

