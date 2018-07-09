#!/usr/bin/env python2
"""
preprocess_data.py
-----------------------------------------
Copyright (c) 2018 The HERA Collaboration

This script is used in the IDR2.1 power
spectrum pipeline as a pre-processing step after
calibration, RFI-flagging and LSTbinning. This
additional processing includes RFI-flagging, timeavg subtraction, 
fringe-rate filtering, pseudo-stokes visibility formation,
and foreground filtering.

See preprocess_params.yaml for relevant parameter selections.
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
data_suffix = os.path.splitext(input_data_template)[1][1:]

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
hp.utils.log("Starting preprocess pipeline on {}\n{}\n".format(time, '-'*60), f=lf, verbose=verbose)
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
# Reformat Data by Baseline Type
#-------------------------------------------------------------------------------
if reformat:
    # get algorithm parameters
    globals().update(cf['algorithm']['reformat'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting baseline reformatting: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get datafiles
    datafiles, datapols = uvt.utils.search_data(input_data_template.format(group=groupname), pols, matched_pols=False, reverse_nesting=False, flatten=False)

    # choose first file and get all red baseline groups and their info
    uvd = UVData()
    uvd.read_miriad(datafiles[0][0])
    antpos, ants = uvd.get_ENU_antpos()
    antposd = dict(zip(ants, antpos))
    reds = hc.redcal.get_pos_reds(antposd, bl_error_tol=bltol, low_hi=True)
    blvs = [(antposd[r[0][0]] - antposd[r[0][1]])[:2] for r in reds]
    lens = [np.linalg.norm(blv) for blv in blvs]
    angs = [np.arctan2(*blv[::-1]) * 180 / np.pi for blv in blvs]
    for i in range(len(angs)):
        if angs[i] < 0:
            angs[i] = (angs[i] + 180) % 360

    # put in autocorrs
    reds = [zip(uvd.antenna_numbers, uvd.antenna_numbers)] + reds
    lens = [0] + lens
    angs = [0] + angs

    # iterate over polarization group
    for i, dfs in enumerate(datafiles):

        # setup bl reformat function
        def bl_reformat(j, i=i, datapols=datapols, dfs=dfs, lens=lens, angs=angs, reds=reds, data_suffix=data_suffix, p=cf['algorithm']['reformat']):
            try:
                if not p['bl_len_range'][0] < lens[j] < p['bl_len_range'][1]:
                    return 0
                outname = p['reformat_outfile'].format(len=int(round(lens[j])), deg=int(round(angs[j])), pol=datapols[i][0], suffix=data_suffix)
                outname = os.path.join(out_dir, outname)
                if os.path.exists(outname) and overwrite == False:
                    return 1
                uvd = UVData()
                uvd.read_miriad(dfs, ant_pairs_nums=reds[j])
                uvd.write_miriad(outname, clobber=True)
            except:
                hp.utils.log("\njob {} threw exception:".format(j), f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1
            return 0

        # launch jobs
        failures = job_monitor(bl_reformat, range(len(reds)), "BL REFORMAT: pol {}".format(pol), lf=lf, maxiter=maxiter, verbose=verbose)

    # edit data template
    input_data_template = os.path.join(out_dir, new_data_template.format(pol='{pol}', suffix=data_suffix))

    time = datetime.utcnow()
    hp.utils.log("\nfinished baseline reformatting: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# RFI-Flag
#-------------------------------------------------------------------------------
if rfi_flag:
    # get algorithm parameters
    globals().update(cf['algorithm']['xrfi'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting RFI flagging: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get datafiles
    datafiles, datapols = uvt.utils.search_data(input_data_template.format(group=groupname), pols, matched_pols=False, reverse_nesting=False, flatten=True)

    # setup RFI function
    def run_xrfi(i, datafiles=datafiles, p=cf['algorithm']['xrfi']):
        try:
            # setup delay filter class as container
            df = datafiles[i]
            F = hc.delay_filter.Delay_Filter()
            # load data
            F.load_data(df)
            # RFI flag if desired
            if run_xrfi:
                for k in F.data.keys():
                    new_f = hq.xrfi.xrfi(F.data[k], f=F.flags[k], **p['xrfi_params'])
                    F.flags[k] += new_f
            # write to file
            outname = os.path.join(out_dir, os.path.basename(df) + p['file_ext'])
            hc.io.update_vis(df, outname, filetype_in='miriad', filetype_out='miriad', data=F.data, flags=F.flags,
                             add_to_history='', clobber=overwrite)

        except:
            hp.utils.log("\njob {} threw exception:".format(i), f=ef, tb=sys.exc_info(), verbose=verbose)
            return 1

        return 0

    # launch jobs
    failures = hp.utils.job_monitor(run_xrfi, range(len(datafiles)), "XRFI", lf=lf, maxiter=maxiter, verbose=verbose)

    # update template
    input_data_template = os.path.join(out_dir, os.path.basename(input_data_template) + file_ext)
    data_suffix += file_ext

    time = datetime.utcnow()
    hp.utils.log("\nfinished RFI flagging: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Time Average Subtraction
#-------------------------------------------------------------------------------
if timeavg_sub:
    # get algorithm parameters
    globals().update(cf['algorithm']['timeavg_sub'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting full time-average spectra and subtraction: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get datafiles
    datafiles, datapols = uvt.utils.search_data(input_data_template.format(group=groupname, pol='{pol}'), pols, matched_pols=False, reverse_nesting=False, flatten=False)

    # load a datafile and get antenna numbers
    uvd = UVData()
    uvd.read_miriad_metadata(datafiles[0][0])
    antpos, ants = uvd.get_ENU_antpos()
    antpos_dict = dict(zip(ants, antpos))

    # get redundant baselines
    reds = hc.redcal.get_pos_reds(antpos_dict, bl_error_tol=1.0, low_hi=True)
    lens = [np.linalg.norm(antpos_dict[r[0][0]] - antpos_dict[r[0][1]]) for r in reds]
    angs = [np.arctan2(*(antpos_dict[r[0][0]] - antpos_dict[r[0][1]])[:2][::-1]) * 180 / np.pi for r in reds]
    angs = [(a + 180) % 360 if a < 0 else a for a in angs]

    # put in autocorrs
    reds = [zip(uvd.antenna_numbers, uvd.antenna_numbers)] + reds
    lens = [0] + lens
    angs = [0] + angs

    # iterate over pols
    for i, dfs in enumerate(datafiles):
        pol = datapols[i][0]

        # write full tavg function
        def full_tavg(j, pol=pol, lens=lens, angs=angs, reds=reds, dfs=dfs, data_suffix=data_suffix, p=cf['algorithm']['timeavg_sub']):
            try:
                # load data into uvdata
                uvd = UVData()
                uvd.read_miriad(dfs, ant_pairs_nums=reds[j])
                # instantiate FRF object
                F = hc.frf.FRFilter()
                # load data
                F.load_data(uvd)
                # perform full time-average to get spectrum
                F.timeavg_data(1e10, rephase=False, verbose=verbose)
                # Delay Filter if desired
                if p['dly_filter']:
                    # RFI Flag
                    for k in F.avg_data.keys():
                        # RFI Flag
                        new_f = hq.xrfi.xrfi(F.avg_data[k], f=F.avg_flags[k], **p['rfi_params'])
                        F.avg_flags[k] += new_f
                    # Delay Filter
                    DF = hc.delay_filter.Delay_Filter()
                    DF.load_data(F.input_uvdata)
                    DF.data = F.avg_data
                    DF.flags = F.avg_flags
                    DF.run_filter(**p['dly_params'])
                    # Replace timeavg spectrum with CLEAN model
                    F.avg_data = DF.CLEAN_models
                    # unflag all frequencies
                    for k in F.avg_flags.keys():
                        # only unflag spectra from non-xants
                        if np.min(F.avg_flags[k]) == False:
                            F.avg_flags[k][:] = False
                # write timeavg specctrum
                _len = lens[j]
                _deg = angs[j]
                tavg_file = "zen.{group}.{pol}.{len:03d}_{deg:03d}.{tavg_tag}.{suffix}".format(group=groupname, pol=pol, len=int(_len), deg=int(_deg), tavg_tag=p['tavg_tag'], suffix=data_suffix)
                tavg_file = os.path.join(out_dir, tavg_file)
                F.write_data(tavg_file, write_avg=True, overwrite=overwrite)
            except:
                hp.utils.log("\njob {} threw exception:".format(j), f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1

            return 0

        # launch jobs
        failures = hp.utils.job_monitor(full_tavg, range(len(reds)), "FULL TAVG: pol {}".format(pol), lf=lf, maxiter=maxiter, verbose=verbose)

        # collate tavg spectra into a single file
        tavgfiles = sorted(glob.glob(os.path.join(out_dir, "zen.{group}.{pol}.*.{tavg_tag}.{suffix}".format(group=groupname, pol=pol, tavg_tag=tavg_tag, suffix=data_suffix))))
        uvd = UVData()
        uvd.read_miriad(tavgfiles)
        tavg_out = os.path.join(out_dir, "zen.{group}.{pol}.{tavg_tag}.{suffix}".format(group=groupname, pol=pol, tavg_tag=tavg_tag, suffix=data_suffix))
        uvd.write_miriad(tavg_out, clobber=overwrite)
        for tf in tavgfiles:
            if os.path.exists(tf):
                shutil.rmtree(tf)

        # print to log
        hp.utils.log("\npol {} time-average spectra exit codes:\n {}".format(pol, exit_codes), f=lf, verbose=verbose)

        # write tavg subtraction function
        def tavg_sub(j, dfs=dfs, pol=pol, tavg_file=tavg_out, p=cf['algorithm']['timeavg_sub']):
            try:
                # load data file
                uvd = UVData()
                df = dfs[j]
                uvd.read_miriad(df)
                # get full tavg spectra
                tavg = UVData()
                tavg.read_miriad(tavg_file)
                # subtract timeavg spectrum from data
                for bl in np.unique(uvd.baseline_array):
                    bl_inds = np.where(uvd.baseline_array == bl)[0]
                    polnum = uvutils.polstr2num(pol)
                    pol_ind = np.where(uvd.polarization_array == polnum)[0]
                    if bl in tavg.baseline_array:
                        uvd.data_array[bl_inds, :, :, pol_ind] -= tavg.get_data(bl)[None]
                    else:
                        if verbose:
                            print "baseline {} not found in time-averaged spectrum".format(bl)
                            uvd.flag_array[bl_inds, :, :, pol_ind] = True

                # put uniq_bls in if it doesn't exist
                if not uvd.extra_keywords.has_key('uniq_bls'):
                    uvd.extra_keywords['uniq_bls'] = json.dumps(np.unique(uvd.baseline_array).tolist())
                # write tavg-subtracted data
                out_df = os.path.join(out_dir, os.path.basename(df) + p['file_ext'])
                uvd.history += "\nTime-Average subtracted."
                uvd.write_miriad(out_df, clobber=overwrite)
            except:
                hp.utils.log("\njob {} threw exception:".format(i), f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1

            return 0

        # launch jobs
        failures = hp.utils.job_monitor(tavg_sub, range(len(dfs)), "TAVG SUB: pol {}".format(pol), lf=lf, maxiter=maxiter, verbose=verbose)

    time = datetime.utcnow()
    hp.utils.log("\nfinished full time-average spectra and subtraction: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)

    input_data_template = os.path.join(out_dir, os.path.basename(input_data_template) + file_ext)
    data_suffix += file_ext

#-------------------------------------------------------------------------------
# Time Averaging (i.e. Fringe Rate Filtering)
#-------------------------------------------------------------------------------
if time_avg:
    # get algorithm parameters
    globals().update(cf['algorithm']['tavg'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting time averaging: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get datafiles
    datafiles, datapols = uvt.utils.search_data(input_data_template.format(group=groupname, pol='{pol}'), pols, matched_pols=False, reverse_nesting=False, flatten=False)

    # load a datafile and get antenna numbers
    uvd = UVData()
    uvd.read_miriad_metadata(datafiles[0][0])
    antpos, ants = uvd.get_ENU_antpos()
    antpos_dict = dict(zip(ants, antpos))

    # get redundant baselines
    reds = hc.redcal.get_pos_reds(antpos_dict, bl_error_tol=1.0, low_hi=True)
    lens = [np.linalg.norm(antpos_dict[r[0][0]] - antpos_dict[r[0][1]]) for r in reds]
    angs = [np.arctan2(*(antpos_dict[r[0][0]] - antpos_dict[r[0][1]])[:2][::-1]) * 180 / np.pi for r in reds]
    angs = [(a + 180) % 360 if a < 0 else a for a in angs]

    # put in autocorrs
    reds = [zip(uvd.antenna_numbers, uvd.antenna_numbers)] + reds
    lens = [0] + lens
    angs = [0] + angs

    # iterate over pol groups
    for i, dfs in enumerate(datafiles):
        pol = datapols[i][0]

        def time_average(j, dfs=dfs, pol=pol, lens=lens, angs=angs, reds=reds, data_suffix=data_suffix, p=cf['algorithm']['tavg']):
            try:
                # load data into uvdata
                uvd = UVData()
                uvd.read_miriad(dfs, ant_pairs_nums=reds[j])
                # instantiate FRF object
                F = hc.frf.FRFilter()
                # load data
                F.load_data(uvd)
                # perform time-average
                F.timeavg_data(p['t_window'], rephase=p['tavg_rephase'], verbose=verbose)
                # write timeavg spectrum
                _len = lens[j]
                _deg = angs[j]
                tavg_file = "zen.{group}.{pol}.{len:03d}_{deg:03d}.{suffix}".format(group=groupname, pol=pol, len=int(_len), deg=int(_deg), suffix=data_suffix)
                tavg_file = os.path.join(out_dir, tavg_file + p['file_ext'])
                F.write_data(tavg_file, write_avg=True, overwrite=overwrite)
            except:
                hp.utils.log("\njob {} threw exception:".format(i), f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1

            return 0

        # launch jobs
        failures = hp.utils.job_monitor(time_average, range(len(reds)), "TIME AVERAGE: pol {}".format(pol), lf=lf, maxiter=maxiter, verbose=verbose)

        # collate averaged data into time chunks
        tavg_files = os.path.join(out_dir, "zen.{group}.{pol}.*.{suffix}".format(group=groupname, pol=pol, suffix=data_suffix + file_ext))
        tavg_files = sorted(glob.glob(tavg_files))

        # pick one file to get full time information from
        uvd = UVData()
        uvd.read_miriad(tavg_files[-1])
        times = np.unique(uvd.time_array)
        Ntimes = len(times)

        # break into subfiles
        Nfiles = int(np.ceil(Ntimes / float(file_Ntimes)))
        times = [times[i*file_Ntimes:(i+1)*file_Ntimes] for i in range(Nfiles)]

        def reformat_files(j, pol=pol, tavg_files=tavg_files, times=times, data_suffix=data_suffix, p=cf['algorithm']['tavg']):
            try:
                uvd = UVData()
                uvd.read_miriad(tavg_files, time_range=[times[j].min()-1e-8, times[j].max()+1e-8])
                lst = uvd.lst_array[0]
                outfile = os.path.join(out_dir, "zen.{group}.{pol}.LST.{LST:.5f}.{suffix}".format(group=groupname, pol=pol, LST=lst, suffix=data_suffix + p['file_ext']))
                uvd.write_miriad(outfile, clobber=overwrite)
            except:
                hp.utils.log("\njob {} threw exception:".format(i), f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1

            return 0

        # launch jobs
        failures = hp.utils.job_monitor(reformat_files, range(len(times)), "TAVG REFORMAT: pol {}".format(pol), lf=lf, maxiter=maxiter, verbose=verbose)

        # clean up time averaged files
        for f in tavg_files:
            if os.path.exists(f):
                shutil.rmtree(f)

    input_data_template = os.path.join(out_dir, os.path.basename(input_data_template) + file_ext)
    data_suffix += file_ext

    time = datetime.utcnow()
    hp.utils.log("\nfinished time averaging: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Form Pseudo-Stokes Visibilities
#-------------------------------------------------------------------------------
if form_pstokes:
    # get algorithm parameters
    globals().update(cf['algorithm']['pstokes'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting pseudo-stokes: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get datafiles with reversed nesting
    datafiles, datapols = uvt.utils.search_data(input_data_template.format(group=groupname, pol='{pol}'), pols, reverse_nesting=True)

    # write pseudo-Stokes function
    def make_pstokes(i, datafiles=datafiles, datapols=datapols, p=cf['algorithm']['pstokes']):
        try:
            # get all pol files for unique datafile
            dfs = datafiles[i]
            dps = datapols[i]
            # load data
            dsets = []
            for df in dfs:
                uvd = UVData()
                uvd.read_miriad(df)
                dsets.append(uvd)
            # iterate over outstokes
            for pstokes in p['outstokes']:
                try:
                    ds = hp.pstokes.filter_dset_on_stokes_pol(dsets, pstokes)
                    ps = hp.pstokes.construct_pstokes(ds[0], ds[1], pstokes=pstokes)
                    outfile = os.path.basename(dfs[0]).replace(".{}.".format(dps[0]), ".{}.".format(pstokes))
                    outfile = os.path.join(out_dir, outfile)
                    ps.write_miriad(outfile, clobber=overwrite)
                except AssertionError:
                    if verbose:
                        print "failed to make pstokes {} for job {}".format(pstokes, i)
        except:
            hp.utils.log("job {} threw exception:".format(i), f=ef, tb=sys.exc_info(), verbose=verbose)
            return 1

        return 0

    # launch jobs
    failures = hp.utils.job_monitor(make_pstokes, range(len(datafiles)), "PSTOKES", lf=lf, maxiter=maxiter, verbose=verbose)

    # add pstokes pols to pol list for downstream calculations
    pols += outstokes

    time = datetime.utcnow()
    hp.utils.log("\nfinished pseudo-stokes: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Foreground Filtering (and data in-painting)
#-------------------------------------------------------------------------------
if fg_filt:
    # get algorithm parameters
    globals().update(cf['algorithm']['fg_filt'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting foreground filtering: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get flattened datafiles
    datafiles, datapols = uvt.utils.search_data(input_data_template.format(group=groupname, pol='{pol}'), pols, matched_pols=False, reverse_nesting=False, flatten=True)

    # write fgfilt function
    def fg_filter(i, datafiles=datafiles, p=cf['algorithm']['fg_filt']):
        try:
            # get datafile
            df = datafiles[i]
            # start FGFilter
            DF = hc.delay_filter.Delay_Filter()
            DF.load_data(df, filetype='miriad')
            # run filter
            DF.run_filter(**p['filt_params'])
            # write filtered term
            outfile = os.path.join(out_dir, os.path.basename(df) + p['filt_file_ext'])
            DF.write_filtered_data(outfile, filetype_out='miriad', clobber=overwrite, add_to_history="Foreground Filtered with: {}".format(json.dumps(p['filt_params'])))
            # write original data with in-paint
            outfile = os.path.join(out_dir, os.path.basename(df) + p['inpaint_file_ext'])
            DF.write_filtered_data(outfile, filetype_out='miriad', clobber=overwrite, write_filled_data=True, add_to_history="FG model flag inpainted with: {}".format(json.dumps(p['filt_params'])))
        except:
            hp.utils.log("job {} threw exception:".format(i), f=ef, tb=sys.exc_info(), verbose=verbose)
            return 1
        return 0

    # launch jobs
    failures = hp.utils.job_monitor(fg_filter, range(len(datafiles)), "FG FILTER", lf=lf, maxiter=maxiter, verbose=verbose)

    time = datetime.utcnow()
    hp.utils.log("\nfinished foreground-filtering: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)


