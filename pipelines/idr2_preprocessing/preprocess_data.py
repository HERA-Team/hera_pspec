#!/usr/bin/env python2
"""
preprocess_data.py
-----------------------------------------
Copyright (c) 2018 The HERA Collaboration

This script is used in the IDR2.1 power
spectrum pipeline as a pre-processing step after
calibration, RFI-flagging and LSTbinning. This
additional processing includes RFI-flagging, crosstalk subtraction, 
fringe-rate filtering, pseudo-stokes visibility formation,
and foreground filtering.

See preprocess_params.yaml for relevant parameter selections.
"""
import numpy as np
import hera_cal as hc
import hera_pspec as hp
import hera_qm as hq
from pyuvdata import UVData
import pyuvdata.utils as uvutils
import os
import sys
import glob
import multiprocess
import yaml
from datetime import datetime
import uvtools as uvt
import json

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
    ef = open(errfile, "w")
time = datetime.utcnow()
hp.utils.log("Starting preprocess pipeline on {}\n{}\n".format(time, '-'*60), f=lf, verbose=verbose)
hp.utils.log(json.dumps(cf, indent=1) + '\n', f=lf, verbose=verbose)

# Create multiprocesses
if multiprocess:
    pool = multiprocess.Pool(nproc)
    M = pool.map
else:
    M = map

#-------------------------------------------------------------------------------
# Reformat Data by Baseline Type
#-------------------------------------------------------------------------------
if reformat:
    # get algorithm parameters
    globals().update(cf['algorithm']['reformat'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting baseline reformatting: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get datafiles
    datafiles, datapols = uvt.utils.search_data(input_data_template, pols, matched_pols=False, reverse_nesting=False, flatten=False)

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

    # setup bl reformat function
    def bl_reformat(j):
        try:
            if not bl_len_range[0] < lens[j] < bl_len_range[1]:
                return 0
            outname = reformat_outfile.format(len=int(round(lens[j])), deg=int(round(angs[j])), pol=datapols[i], suffix=data_suffix)
            outname = os.path.join(out_dir, outname)
            if os.path.exists(outname) and overwrite == False:
                return 1
            uvd = UVData()
            uvd.read_miriad(dfs, ant_pairs_nums=reds[j])
            uvd.write_miriad(outname, clobber=True)
        except:
            err, _, tb = sys.exc_info()
            hp.utils.log("\n{} threw {} Exception with traceback:".format(outname, err), f=ef, tb=tb, verbose=verbose)
            return 1
        return 0

    # iterate over polarization group
    for i, dfs in enumerate(datafiles):
        # distribute across baseline types
        exit_codes = M(bl_reformat, range(len(reds)))

        # print to log
        hp.utils.log("\nbaseline reformatting exit codes for pol {}:\n {}".format(datapols[i], exit_codes), f=lf, verbose=verbose)

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
    datafiles, datapols = uvt.utils.search_data(input_data_template, pols, matched_pols=False, reverse_nesting=False, flatten=True)

    # setup RFI function
    def run_xrfi(i):
        try:
            # setup delay filter class as container
            df = datafiles[i]
            F = hc.delay_filter.Delay_Filter()
            # load data
            F.load_data(df)
            # RFI flag if desired
            if run_xrfi:
                for k in F.data.keys():
                    new_f = hq.xrfi.xrfi(F.data[k], f=F.flags[k], **xrfi_params)
                    F.flags[k] += new_f
            # write to file
            outname = os.path.join(out_dir, os.path.basename(df) + file_ext)
            hc.io.update_vis(df, outname, filetype_in='miriad', filetype_out='miriad', data=F.data, flags=F.flags,
                             add_to_history='', clobber=overwrite)

        except:
            err, _, tb = sys.exc_info()
            hp.utils.log("\n{} threw {} Exception with traceback:".format(outname, err), f=ef, tb=tb, verbose=verbose)
            return 1
        return 0

    # run xtalk sub on each datafile
    exit_codes = M(run_xrfi, range(len(datafiles)))

    # print to log
    hp.utils.log("\nRFI flag exit codes:\n {}".format(exit_codes), f=lf, verbose=verbose)

    input_data_template += file_ext

    time = datetime.utcnow()
    hp.utils.log("\nfinished RFI flagging: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)


#-------------------------------------------------------------------------------
# Crosstalk Subtraction
#-------------------------------------------------------------------------------
if xtalk_sub:
    # get algorithm parameters
    globals().update(cf['algorithm']['xtalk'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting crosstalk subtraction: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get datafiles
    datafiles, datapols = uvt.utils.search_data(input_data_template, pols, matched_pols=False, reverse_nesting=False, flatten=True)

    # write xtalk sub function
    def xtalk_sub(j):
        try:
            # get datafile
            df = datafiles[j]
            # instantiate FRF object
            F = hc.frf.FRFilter()
            # load data
            F.load_data(df)
            # perform full time-average to get xtalk spectrum
            F.timeavg_data(1e10, rephase=False, verbose=verbose)
            # Delay Filter if desired
            if dly_filter:
                # RFI Flag
                for k in F.avg_data.keys():
                    # RFI Flag
                    new_f = hq.xrfi.xrfi(F.avg_data[k], f=F.avg_flags[k], **rfi_params)
                    F.avg_flags[k] += new_f
                # Delay Filter
                DF = hc.delay_filter.Delay_Filter()
                DF.load_data(F.inp_uvdata)
                DF.data = F.avg_data
                DF.flags = F.avg_flags
                DF.run_filter(**dly_params)
                # Replace xtalk spectrum with CLEAN model
                F.avg_data = DF.CLEAN_models
                for k in F.avg_flags.keys():
                    F.avg_flags[k][:] = False
            # subtract xtalk spectrum from data
            for k in F.data.keys():
                F.data[k] -= F.avg_data[k]
            # write xtalk specctrum
            xtalk_file = xtalk_ext.join(os.path.splittext(os.path.basenname(df)))
            xtalk_file = os.path.join(out_dir, xtalk_file)
            F.write_data(xtalk_file, write_avg=True, overwrite=overwrite)
            # write xtalk-subtracted data
            out_df = os.path.join(out_dir, os.path.basename(df) + file_ext)
            F.write_data(out_df, write_avg=False, overwrite=overwrite)
        except:
            err, _, tb = sys.exc_info()
            hp.utils.log("\n{} threw {} Exception with traceback:".format(df, err), f=ef, tb=tb, verbose=verbose)
            return 1

        return 0

    # run xtalk sub on each datafile
    exit_codes = M(xtalk_sub, range(len(datafiles)))

    # print to log
    hp.utils.log("\nxtalk subtraction exit codes:\n {}".format(exit_codes), f=lf, verbose=verbose)

    input_data_template += file_ext

    time = datetime.utcnow()
    hp.utils.log("\nfinished crosstalk subtraction: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Time Averaging (i.e. Fringe Rate Filtering)
#-------------------------------------------------------------------------------
if time_avg:
    # get algorithm parameters
    globals().update(cf['algorithm']['tavg'])
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting time averaging: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # get datafiles
    datafiles, datapols = uvt.utils.search_data(input_data_template, pols, matched_pols=False, reverse_nesting=False, flatten=True)

    # write time_avg function
    def time_average(j):
        try:
            # get datafile
            df = datafiles[j]
            # instantiate FRF object
            F = hc.frf.FRFilter()
            # load data
            F.load_data(df, filetype='miriad')
            # perform time average
            F.timeavg_data(t_window, rephase=tavg_rephase, verbose=verbose)
            # write data
            out_df = os.path.join(out_dir, os.path.basename(df) + file_ext)
            F.write_data(out_df, write_avg=True, overwrite=overwrite)
        except:
            err, _, tb = sys.exc_info()
            hp.utils.log("\n{} threw {} Exception with traceback:".format(df, err), f=ef, tb=tb, verbose=verbose)
            return 1

        return 0

    # iterate over datafiles
    exit_codes = M(time_average, range(len(datafiles)))

    # print to log
    hp.utils.log("\ntime average exit codes:\n {}".format(exit_codes), f=lf, verbose=verbose)

    input_data_template += file_ext

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
    datafiles, datapols = uvt.utils.search_data(input_data_template, pols, reverse_nesting=True)

    # write pseudo-Stokes function
    def make_pstokes(i):
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
            for pstokes in outstokes:
                try:
                    ds = hp.pstokes.choose_inp_dsets(dsets, pstokes)
                    ps = hp.pstokes.construct_pstokes(ds[0], ds[1], pstokes=pstokes)
                    outfile = os.path.basename(dfs[0]).replace(".{}.".format(dps[0]), ".{}.".format(pstokes))
                    outfile = os.path.join(out_dir, outfile)
                    ps.write_miriad(outfile, clobber=overwrite)
                except AssertionError:
                    err, _, tb = sys.exc_info()
                    hp.utils.log("datafile {} outstokes {} threw {} Exception with traceback:".format(dfs[0], pstokes, err), f=ef, tb=tb, verbose=verbose)
                    pass
        except:
            pass
            return 1
        return 0

    # iterate over unique datafiles and construct pstokes
    exit_codes = M(make_pstokes, range(len(datafiles)))

    # print to log
    hp.utils.log("\npseudo stokes exit codes:\n {}".format(exit_codes), f=lf, verbose=verbose)

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
    datafiles, datapols = uvt.utils.search_data(input_data_template, pols, matched_pols=False, reverse_nesting=False, flatten=True)

    # write fgfilt function
    def fg_filter(i):
        try:
            # get datafile
            df = datafiles[i]
            # start FGFilter
            DF = hc.delay_filter.Delay_Filter()
            DF.load_data(df, filetype='miriad')
            # run filter
            DF.run_filter(**cf['algorithm']['fg_filt']['filt_params'])
            # write filtered term
            outfile = os.path.join(out_dir, os.path.basename(df) + filt_file_ext)
            DF.write_filtered_data(outfile, filetype_out='miriad', clobber=overwrite, add_to_history="Foreground Filtered with: {}".format(json.dumps(cf['algorithm']['fg_filt']['filt_params'])))
            # write original data with in-paint
            outfile = os.path.join(out_dir, os.path.basename(df) + inpaint_file_ext)
            DF.write_filtered_data(outfile, filetype_out='miriad', clobber=overwrite, write_filled_data=True, add_to_history="FG model flag inpainted with: {}".format(json.dumps(cf['algorithm']['fg_filt']['filt_params'])))
        except:
            err, _, tb = sys.exc_info()
            hp.utils.log("datafile {} threw {} Exception with traceback:".format(df, err), f=ef, tb=tb, verbose=verbose)
            return 1
        return 0

    # iterate over datafiles and filter
    exit_codes = M(fg_filter, range(len(datafiles)))

    # print to log
    hp.utils.log("\nfg filtering exit codes:\n {}".format(exit_codes), f=lf, verbose=verbose)

    time = datetime.utcnow()
    hp.utils.log("\nfinished foreground-filtering: {}\n{}".format(time, "-"*60), f=lf, verbose=verbose)


