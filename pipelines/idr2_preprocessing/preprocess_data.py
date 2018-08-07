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
import pyuvdata
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
# Get config and load dictionary
config = sys.argv[1]
cf = hp.utils.load_config(config)

# Consolidate IO, data and analysis parameter dictionaries
params = odict(cf['io'].items() + cf['data'].items() + cf['analysis'].items())
assert len(params) == len(cf['io']) + len(cf['data']) + len(cf['analysis']), ""\
       "Repeated parameters found within the scope of io, data and analysis dicts"
algs = cf['algorithm']

# Extract certain parameters used across the script
verbose = params['verbose']
overwrite = params['overwrite']
pols = params['pols']
data_template = os.path.join(params['data_root'], params['data_template'])
data_suffix = os.path.splitext(data_template)[1][1:]

# Open logfile
logfile = os.path.join(params['out_dir'], params['logfile'])
if os.path.exists(logfile) and params['overwrite'] == False:
    raise IOError("logfile {} exists and overwrite == False, quitting pipeline...".format(logfile))
lf = open(logfile, "w")
if params['joinlog']:
    ef = lf
else:
    ef = open(os.path.join(params['out_dir'], params['errfile']), "w")

time = datetime.utcnow()
hp.utils.log("Starting preprocess pipeline on {}\n{}\n".format(time, '-'*60), 
             f=lf, verbose=verbose)
hp.utils.log(json.dumps(cf, indent=1) + '\n', f=lf, verbose=verbose)

# Change to working dir
os.chdir(params['work_dir'])

# out_dir should be cleared before each run: issue a warning if not the case
outdir = os.path.join(params['work_dir'], params['out_dir'])
oldfiles = glob.glob(outdir+'/*')
if len(oldfiles) > 0:
    hp.utils.log("\n{}\nWARNING: out_dir should be cleaned before each new run to " \
                 "ensure proper functionality.\nIt seems like some files currently " \
                 "exist in {}\n{}\n".format('-'*50, outdir, '-'*50), f=lf, verbose=verbose)

# Define history prepend function
def prepend_history(action, param_dict):
    """ create a history string to prepend to data files """
    dict_str = '\n'.join(["{} : {}".format(*_d) for _d in param_dict.items()])
    time = datetime.utcnow()
    hist = "\nRan preprocess_data.py {} step at\nUTC {} with \nhera_pspec [{}], "\
           "hera_cal [{}],\nhera_qm [{}] and pyuvdata [{}]\nwith {} algorithm "\
           "attrs:\n{}\n{}\n".format(action, time, 
                                     hp.version.git_hash[:10],
                                     hc.version.git_hash[:10], 
                                     hq.version.git_hash[:10], 
                                     pyuvdata.version.git_hash[:10], 
                                     action, '-'*50, 
                                     dict_str)
    return hist

# Assign iterator function
if params['multiproc']:
    pool = multiprocess.Pool(params['nproc'])
    M = pool.map
else:
    M = map

#-------------------------------------------------------------------------------
# Reformat Data by Baseline Type
#-------------------------------------------------------------------------------
if params['reformat']:
    
    # Start block
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting baseline reformatting: {}\n".format("-"*60, time), 
                 f=lf, verbose=verbose)

    # Get datafiles
    datafiles, datapols = uvt.utils.search_data( 
                               data_template.format(group=params['groupname']), 
                               pols, matched_pols=False, 
                               reverse_nesting=False, 
                               flatten=False )

    # Get redundant groups as a good split for parallelization
    reds, lens, angs = hp.utils.get_reds(datafiles[0][0], bl_error_tol=1.0, 
                                         add_autos=True,
                                         bl_len_range=params['bl_len_range'], 
                                         bl_deg_range=params['bl_deg_range'])

    # Iterate over polarization group
    for i, dfs in enumerate(datafiles):

        # Setup bl reformat function
        def bl_reformat(j, i=i, datapols=datapols, dfs=dfs, lens=lens, 
                        angs=angs, reds=reds, data_suffix=data_suffix, 
                        p=algs['reformat'], params=params):
            try:
                if not p['bl_len_range'][0] < lens[j] < p['bl_len_range'][1]:
                    return 0
                outname = p['reformat_outfile'].format(len=int(round(lens[j])), 
                                                       deg=int(round(angs[j])), 
                                                       pol=datapols[i][0], 
                                                       suffix=data_suffix)
                outname = os.path.join(params['out_dir'], outname)
                if os.path.exists(outname) and overwrite == False:
                    return 1
                uvd = UVData()
                uvd.read_miriad(dfs, bls=reds[j])
                uvd.write_miriad(outname, clobber=True)
                uvd.history = "{}{}".format(prepend_history("BL REFORMAT", p), 
                                            uvd.history)
                if params['plot']:
                    hp.utils.plot_uvdata_waterfalls(uvd, 
                                                    outname + ".{pol}.{bl}", 
                                                    data='data', 
                                                    plot_mode='log', 
                                                    format='png')
            except:
                hp.utils.log("\njob {} threw exception:".format(j), 
                             f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1
            return 0

        # Launch jobs
        failures = hp.utils.job_monitor(bl_reformat, range(len(reds)), \
                                        "BL REFORMAT: pol {}".format(pol), 
                                        M=M, lf=lf, maxiter=params['maxiter'], 
                                        verbose=verbose)

    # Edit data template
    data_template = os.path.join(
                        params['out_dir'], 
                        algs['reformat']['new_data_template'].format(
                                                        pol='{pol}', 
                                                        suffix=data_suffix))
    time = datetime.utcnow()
    hp.utils.log("\nfinished baseline reformatting: {}\n{}".format(time, "-"*60), 
                 f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# RFI-Flag
#-------------------------------------------------------------------------------
if params['rfi_flag']:
    
    # Start block
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting RFI flagging: {}\n".format("-"*60, time), 
                 f=lf, verbose=verbose)

    # Get datafiles
    datafiles, datapols = uvt.utils.search_data(
                                data_template.format(group=params['groupname']), 
                                pols, matched_pols=False, 
                                reverse_nesting=False, 
                                flatten=True )

    # Setup RFI function
    def run_xrfi(i, datafiles=datafiles, p=cf['algorithm']['xrfi'], params=params):
        try:
            # Setup delay filter class as container
            df = datafiles[i]
            F = hc.delay_filter.Delay_Filter()
            F.load_data(df) # load data
            
            # RFI flag if desired
            if run_xrfi:
                for k in F.data.keys():
                    new_f = hq.xrfi.xrfi(F.data[k], 
                                         f=F.flags[k], 
                                         **p['xrfi_params'])
                    F.flags[k] += new_f
            
            # Write to file
            add_to_history = prepend_history("XRFI", p)
            outname = os.path.join(params['out_dir'], 
                                   os.path.basename(df) + p['file_ext'])
            hc.io.update_vis(df, outname, 
                             filetype_in='miriad', 
                             filetype_out='miriad', 
                             data=F.data, 
                             flags=F.flags,
                             add_to_history=add_to_history, 
                             clobber=overwrite)

        except:
            hp.utils.log("\njob {} threw exception:".format(i), 
                         f=ef, tb=sys.exc_info(), verbose=verbose)
            return 1
        return 0

    # Launch jobs
    failures = hp.utils.job_monitor(run_xrfi, range(len(datafiles)), 
                                    "XRFI", M=M, lf=lf, 
                                    maxiter=params['maxiter'], 
                                    verbose=verbose)

    # Update template
    data_template = os.path.join(params['out_dir'], 
                                 os.path.basename(data_template) \
                                 + algs['rfi_flag']['file_ext'])
    data_suffix += algs['rfi_flag']['file_ext']

    time = datetime.utcnow()
    hp.utils.log("\nfinished RFI flagging: {}\n{}".format(time, "-"*60), 
                 f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Time Average Subtraction
#-------------------------------------------------------------------------------
if params['timeavg_sub']:
    
    # Start block
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting full time-average spectra and subtraction: {}\n".format("-"*60, time), 
                 f=lf, verbose=verbose)

    # get datafiles
    datafiles, datapols = uvt.utils.search_data(
                  data_template.format(group=params['groupname'], pol='{pol}'), 
                  pols, matched_pols=False, 
                  reverse_nesting=False, 
                  flatten=False)

    # get redundant groups as a good split for parallelization
    reds, lens, angs = hp.utils.get_reds(datafiles[0][0], 
                                         bl_error_tol=1.0, 
                                         add_autos=True,
                                         bl_len_range=params['bl_len_range'], 
                                         bl_deg_range=params['bl_deg_range'])

    # iterate over pols
    for i, dfs in enumerate(datafiles):
        pol = datapols[i][0]

        # write full tavg function
        def full_tavg(j, pol=pol, lens=lens, angs=angs, reds=reds, dfs=dfs, 
                      data_suffix=data_suffix, p=algs['timeavg_sub'], 
                      params=params):
            try:
                # load data into uvdata
                uvd = UVData()
                # read data, catch ValueError
                try:
                    uvd.read_miriad(dfs, bls=reds[j], polarizations=[pol])
                except ValueError:
                    hp.utils.log("job {} failed b/c no data is present given bls and/or pol selection".format(j))
                    return 0
                
                # Perform full time-average to get spectrum
                F = hc.frf.FRFilter() # instantiate FRF object
                F.load_data(uvd) # load data
                F.timeavg_data(1e10, rephase=False, verbose=verbose)
                
                # Delay Filter if desired
                if p['dly_filter']:
                    
                    # RFI Flag
                    for k in F.avg_data.keys():
                        # RFI Flag
                        new_f = hq.xrfi.xrfi(F.avg_data[k], 
                                             f=F.avg_flags[k], 
                                             **p['rfi_params'])
                        F.avg_flags[k] += new_f
                    
                    # Delay Filter
                    DF = hc.delay_filter.Delay_Filter()
                    DF.load_data(F.input_data)
                    DF.data = F.avg_data
                    DF.flags = F.avg_flags
                    DF.run_filter(**p['dly_params'])
                    
                    # Replace timeavg spectrum with CLEAN model
                    F.avg_data = DF.CLEAN_models
                    
                    # Unflag all frequencies
                    for k in F.avg_flags.keys():
                        # only unflag spectra from non-xants
                        if np.min(F.avg_flags[k]) == False:
                            F.avg_flags[k][:] = False
                
                # Write timeavg spectrum
                _len = lens[j]
                _deg = angs[j]
                _tavg = "zen.{group}.{pol}.{len:03d}_{deg:03d}.{tavg_tag}.{suffix}"
                tavg_file = _tavg.format( group=params['groupname'], 
                                          pol=pol, len=int(_len), deg=int(_deg), 
                                          tavg_tag=p['tavg_tag'], 
                                          suffix=data_suffix )
                tavg_file = os.path.join(params['out_dir'], tavg_file)
                add_to_history = prepend_history("FULL TIME AVG", p)
                F.write_data(tavg_file, write_avg=True, overwrite=overwrite, 
                             add_to_history=add_to_history)
            except:
                hp.utils.log("\njob {} threw exception:".format(j), 
                             f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1
            return 0

        # Launch jobs
        failures = hp.utils.job_monitor(full_tavg, range(len(reds)), 
                                        "FULL TAVG: pol {}".format(pol), 
                                        M=M, lf=lf, maxiter=params['maxiter'], 
                                        verbose=verbose)

        # Collate tavg spectra into a single file
        tavgfiles = sorted(glob.glob(
                         os.path.join(
                              params['out_dir'], 
                              "zen.{group}.{pol}.*.{tavg_tag}.{suffix}".format( 
                                      group=params['groupname'], 
                                      pol=pol, 
                                      tavg_tag=algs['timeavg_sub']['tavg_tag'], 
                                      suffix=data_suffix)) ))
        uvd = UVData()
        uvd.read_miriad(tavgfiles[::-1])
        tavg_out = os.path.join( params['out_dir'], 
                                 "zen.{group}.{pol}.{tavg_tag}.{suffix}".format(
                                       group=params['groupname'], 
                                       pol=pol, 
                                       tavg_tag=algs['timeavg_sub']['tavg_tag'], 
                                       suffix=data_suffix))
        
        # Write tavg data file
        uvd.write_miriad(tavg_out, clobber=overwrite)
        
        # Plot waterfalls if requested
        if params['plot']:
            hp.utils.plot_uvdata_waterfalls(uvd, 
                                            tavg_out + ".{pol}.{bl}", 
                                            data='data', 
                                            plot_mode='log', 
                                            format='png')
        for tf in tavgfiles:
            if os.path.exists(tf):
                shutil.rmtree(tf)

        # Write tavg subtraction function
        def tavg_sub(j, dfs=dfs, pol=pol, tavg_file=tavg_out, 
                     p=cf['algorithm']['timeavg_sub'], params=params):
            try:
                # Load data file
                uvd = UVData()
                df = dfs[j]
                uvd.read_miriad(df)
                
                # Get full tavg spectra
                tavg = UVData()
                tavg.read_miriad(tavg_file)
                
                # Subtract timeavg spectrum from data
                for bl in np.unique(uvd.baseline_array):
                    bl_inds = np.where(uvd.baseline_array == bl)[0]
                    polnum = uvutils.polstr2num(pol)
                    pol_ind = np.where(uvd.polarization_array == polnum)[0]
                    if bl in tavg.baseline_array:
                        uvd.data_array[bl_inds, :, :, pol_ind] -= tavg.get_data(bl)[None]
                    else:
                        uvd.flag_array[bl_inds, :, :, pol_ind] = True
                        if verbose:
                            print "baseline {} not found in time-averaged spectrum".format(bl)

                # Put uniq_bls in if it doesn't exist
                if not uvd.extra_keywords.has_key('uniq_bls'):
                    uvd.extra_keywords['uniq_bls'] = json.dumps(
                                        np.unique(uvd.baseline_array).tolist() )
                
                # Write tavg-subtracted data
                out_df = os.path.join(params['out_dir'], 
                                      os.path.basename(df) + p['file_ext'])
                uvd.history = "{}{}".format(prepend_history("TAVG SUB", p), 
                                            uvd.history)
                uvd.write_miriad(out_df, clobber=overwrite)
                
                
                # Plot waterfalls if requested
                if params['plot']:
                    hp.utils.plot_uvdata_waterfalls(uvd, 
                                                    out_df + ".{pol}.{bl}", 
                                                    data='data', 
                                                    plot_mode='log', 
                                                    format='png')
                
            except:
                hp.utils.log("\njob {} threw exception:".format(i), 
                             f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1
            return 0

        # Launch jobs
        failures = hp.utils.job_monitor( tavg_sub, 
                                         range(len(dfs)), 
                                         "TAVG SUB: pol {}".format(pol), 
                                         M=M, lf=lf, maxiter=params['maxiter'], 
                                         verbose=verbose )

    time = datetime.utcnow()
    hp.utils.log("\nfinished full time-average spectra and subtraction: {}\n{}".format(time, "-"*60), 
                 f=lf, verbose=verbose)

    data_template = os.path.join( params['out_dir'], 
                                  os.path.basename(data_template) \
                                  + algs['timeavg_sub']['file_ext'] )
    data_suffix += algs['timeavg_sub']['file_ext']

#-------------------------------------------------------------------------------
# Time Averaging (i.e. Fringe Rate Filtering)
#-------------------------------------------------------------------------------
if params['time_avg']:
    # Start block
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting time averaging: {}\n".format("-"*60, time), 
                 f=lf, verbose=verbose)

    # Get datafiles
    datafiles, datapols = uvt.utils.search_data( 
                                data_template.format(group=params['groupname'], 
                                                     pol='{pol}'), 
                                pols, matched_pols=False, 
                                reverse_nesting=False, 
                                flatten=False)

    # Get redundant groups as a good split for parallelization
    reds, lens, angs = hp.utils.get_reds( datafiles[0][0], 
                                          bl_error_tol=1.0, 
                                          add_autos=True,
                                          bl_len_range=params['bl_len_range'], 
                                          bl_deg_range=params['bl_deg_range'] )

    # Iterate over pol groups
    for i, dfs in enumerate(datafiles):
        pol = datapols[i][0]

        def time_average(j, dfs=dfs, pol=pol, lens=lens, angs=angs, reds=reds, 
                         data_suffix=data_suffix, p=cf['algorithm']['tavg'], 
                         params=params):
            try:
                # Load data into uvdata
                uvd = UVData()
                
                # Read data, catch ValueError
                try:
                    uvd.read_miriad(dfs, bls=reds[j], polarizations=[pol])
                except ValueError:
                    hp.utils.log("job {} failed w/ ValueError, probably b/c no data is present given bls and/or pol selection".format(j))
                    return 0
                
                # Perform time-average
                F = hc.frf.FRFilter() # instantiate FRF object
                F.load_data(uvd) # load data
                F.timeavg_data( p['t_window'], 
                                rephase=p['tavg_rephase'], 
                                verbose=verbose )
                
                # Write timeavg spectrum
                _len = lens[j]
                _deg = angs[j]
                tavg_file = "zen.{group}.{pol}.{len:03d}_{deg:03d}.{suffix}".format( 
                                        group=params['groupname'], 
                                        pol=pol, 
                                        len=int(_len), 
                                        deg=int(_deg), 
                                        suffix=data_suffix)
                tavg_file = os.path.join(params['out_dir'], 
                                         tavg_file + p['file_ext'])
                add_to_history = prepend_history("TIME AVERAGE", p)
                F.write_data(tavg_file, write_avg=True, overwrite=overwrite, 
                             add_to_history=add_to_history)
            except:
                hp.utils.log("\njob {} threw exception:".format(j), 
                             f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1
            return 0

        # Launch jobs
        failures = hp.utils.job_monitor( time_average, 
                                         range(len(reds)), 
                                         "TIME AVERAGE: pol {}".format(pol), 
                                         M=M, lf=lf, 
                                         maxiter=params['maxiter'], 
                                         verbose=verbose )

        # Collate averaged data into time chunks
        tavg_files = os.path.join( params['out_dir'], 
                                   "zen.{group}.{pol}.*.{suffix}".format(
                                                group=params['groupname'], 
                                                pol=pol, 
                                                suffix=data_suffix \
                                                    + algs['tavg']['file_ext']))
        tavg_files = sorted(glob.glob(tavg_files))
        assert len(tavg_files) > 0, "len(tavg_files) == 0"

        # Pick one file to get full time information from
        uvd = UVData()
        uvd.read_miriad(tavg_files[-1])
        times = np.unique(uvd.time_array)
        Ntimes = len(times)

        # break into subfiles
        Nfiles = int(np.ceil(Ntimes / float(algs['tavg']['file_Ntimes'])))
        times = [ times[ i*algs['tavg']['file_Ntimes'] : 
                        (i+1)*algs['tavg']['file_Ntimes'] ] 
                  for i in range(Nfiles) ]

        def reformat_files(j, pol=pol, tavg_files=tavg_files, times=times, 
                           data_suffix=data_suffix, p=cf['algorithm']['tavg'], 
                           params=params):
            try:
                uvd = UVData()
                uvd.read_miriad(tavg_files[::-1], 
                                time_range=[times[j].min()-1e-8, 
                                            times[j].max()+1e-8], 
                                polarizations=[pol])
                lst = uvd.lst_array[0] - np.median(uvd.integration_time) / 2. \
                    * 2 * np.pi / (3600. * 24)
                outfile = os.path.join(
                                params['out_dir'], 
                                "zen.{group}.{pol}.LST.{LST:.5f}.{suffix}".format(
                                            group=params['groupname'], 
                                            pol=pol, LST=lst, 
                                            suffix=data_suffix + p['file_ext']))
                uvd.write_miriad(outfile, clobber=overwrite)
                
                # Plot waterfalls if requested
                if params['plot']:
                    hp.utils.plot_uvdata_waterfalls(uvd, 
                                                    outfile + ".{pol}.{bl}", 
                                                    data='data', 
                                                    plot_mode='log', 
                                                    format='png')
                
            except:
                hp.utils.log("\njob {} threw exception:".format(j), 
                             f=ef, tb=sys.exc_info(), verbose=verbose)
                return 1

            return 0

        # Launch jobs
        failures = hp.utils.job_monitor(reformat_files, 
                                        range(len(times)), 
                                        "TAVG REFORMAT: pol {}".format(pol), 
                                        M=M, lf=lf, maxiter=params['maxiter'], 
                                        verbose=verbose)

        # Clean up time averaged files
        for f in tavg_files:
            if os.path.exists(f):
                shutil.rmtree(f)

    data_template = os.path.join(params['out_dir'], 
                                 os.path.basename(data_template) \
                                    + algs['tavg']['file_ext'])
    data_suffix += algs['tavg']['file_ext']

    time = datetime.utcnow()
    hp.utils.log("\nfinished time averaging: {}\n{}".format(time, "-"*60), 
                 f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Form Pseudo-Stokes Visibilities
#-------------------------------------------------------------------------------
if params['form_pstokes']:
    # Start block
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting pseudo-stokes: {}\n".format("-"*60, time), f=lf, verbose=verbose)

    # Get datafiles with reversed nesting
    datafiles, datapols = uvt.utils.search_data(
                                data_template.format(group=params['groupname'], 
                                                     pol='{pol}'), 
                                pols, reverse_nesting=True)

    # Write pseudo-Stokes function
    def make_pstokes(i, datafiles=datafiles, datapols=datapols, 
                     p=cf['algorithm']['pstokes'], params=params):
        try:
            # Get all pol files for unique datafile
            dfs = datafiles[i]
            dps = datapols[i]
            
            # Load data
            dsets = []
            for df in dfs:
                uvd = UVData()
                uvd.read_miriad(df)
                dsets.append(uvd)
            
            # Iterate over outstokes
            for pstokes in p['outstokes']:
                try:
                    ds = hp.pstokes.filter_dset_on_stokes_pol(dsets, pstokes)
                    ps = hp.pstokes.construct_pstokes(ds[0], ds[1], pstokes=pstokes)
                    outfile = os.path.basename(dfs[0]).replace(
                                                        ".{}.".format(dps[0]), 
                                                        ".{}.".format(pstokes))
                    outfile = os.path.join(params['out_dir'], outfile)
                    ps.history = "{}{}".format(prepend_history("FORM PSTOKES", p), 
                                               ps.history)
                    ps.write_miriad(outfile, clobber=overwrite)
                    
                    # Plot waterfalls if requested
                    if params['plot']:
                        hp.utils.plot_uvdata_waterfalls(ps, 
                                                        outfile + ".{pol}.{bl}", 
                                                        data='data', 
                                                        plot_mode='log', 
                                                        format='png')
                    
                except AssertionError:
                    hp.utils.log("failed to make pstokes {} for job {}:".format(pstokes, i), 
                                f=ef, tb=sys.exc_info(), verbose=verbose)

        except:
            hp.utils.log("job {} threw exception:".format(i), 
                         f=ef, tb=sys.exc_info(), verbose=verbose)
            return 1

        return 0

    # Launch jobs
    failures = hp.utils.job_monitor( make_pstokes, 
                                     range(len(datafiles)), 
                                     "PSTOKES", 
                                     M=M, lf=lf, maxiter=params['maxiter'], 
                                     verbose=verbose)

    # Add pstokes pols to pol list for downstream calculations
    pols += algs['pstokes']['outstokes']

    time = datetime.utcnow()
    hp.utils.log("\nFinished pseudo-stokes: {}\n{}".format(time, "-"*60), 
                 f=lf, verbose=verbose)

#-------------------------------------------------------------------------------
# Foreground Filtering (and data in-painting)
#-------------------------------------------------------------------------------
if params['fg_filt']:
    
    # Start block
    time = datetime.utcnow()
    hp.utils.log("\n{}\nstarting foreground filtering: {}\n".format("-"*60, time), 
                 f=lf, verbose=verbose)

    # Get flattened datafiles
    datafiles, datapols = uvt.utils.search_data( 
                                 data_template.format(group=params['groupname'], 
                                                      pol='{pol}'), 
                                 pols, matched_pols=False, 
                                 reverse_nesting=False, 
                                 flatten=True)

    # Write fgfilt function
    def fg_filter(i, datafiles=datafiles, p=cf['algorithm']['fg_filt'], 
                  params=params):
        try:
            # Get datafile and start FGFilter
            df = datafiles[i]
            DF = hc.delay_filter.Delay_Filter()
            DF.load_data(df, filetype='miriad')
            
            # Run filter
            DF.run_filter(**p['filt_params'])
            
            # Write filtered term
            outfile = os.path.join(params['out_dir'], 
                                   os.path.basename(df) + p['filt_file_ext'])
            add_to_history = prepend_history("FG FILTER", p)
            DF.write_filtered_data( outfile, 
                                    filetype='miriad', 
                                    clobber=overwrite, 
                                    add_to_history=add_to_history )
            
            # Write original data with in-paint
            outfile = os.path.join(params['out_dir'], 
                                   os.path.basename(df) + p['inpaint_file_ext'])
            add_to_history = prepend_history("DATA INPAINT", p)
            DF.write_filtered_data( outfile, 
                                    filetype='miriad', 
                                    clobber=overwrite, 
                                    write_filled_data=True, 
                                    add_to_history=add_to_history )
        except:
            hp.utils.log("job {} threw exception:".format(i), 
                         f=ef, tb=sys.exc_info(), verbose=verbose)
            return 1            
        return 0

    # Launch jobs
    failures = hp.utils.job_monitor( fg_filter, 
                                     range(len(datafiles)), 
                                     "FG FILTER", 
                                     M=M, lf=lf, maxiter=params['maxiter'], 
                                     verbose=verbose )

    time = datetime.utcnow()
    hp.utils.log("\nfinished foreground-filtering: {}\n{}".format(time, "-"*60), 
                 f=lf, verbose=verbose)

