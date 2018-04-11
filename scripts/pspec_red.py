#!/usr/bin/env python
"""
Run HERA OQE power spectrum estimation code on sets of redundant baselines.
"""
import numpy as np
import hera_pspec as hp
from hera_pspec.utils import log, load_config
from hera_cal import redcal
import pyuvdata as uv
import os, sys, glob, time

# Default settings for pspec calculation
pspec_defaults = {
    'overwrite':                    False,
    'little_h':                     True,
    'avg_group':                    False,
    'exclude_auto_bls':             False,
    'exclude_conjugated_blpairs':   False,
}

    
#-------------------------------------------------------------------------------
# Settings
#-------------------------------------------------------------------------------

# Get configuration filename from cmdline argument
if len(sys.argv) > 1:
    cfg_file = str(sys.argv[1])
else:
    print("Command takes one argument: config_file")
    sys.exit(1)

# Load configuration file
cfg = load_config(cfg_file)
data_cfg = cfg['data']
pspec_cfg = cfg['pspec']


#-------------------------------------------------------------------------------
# Prepare list of data files
#-------------------------------------------------------------------------------

files = []
for i in range(len(data_cfg['subdirs'])):
    files += glob.glob( os.path.join(data_cfg['root'], 
                                     data_cfg['subdirs'][i], 
                                     data_cfg['template']) )
for f in files:
    print(f)

log("Found %d files." % len(files))

#-------------------------------------------------------------------------------
# Load data files into memory
#-------------------------------------------------------------------------------
log("Loading data files...")
t0 = time.time()

# Load all miriad datafiles into UVData objects
dsets = []
for f in files:
    _d = uv.UVData()
    _d.read_miriad(f)
    dsets.append(_d)
log("Loaded data in %1.1f sec." % (time.time() - t0), lvl=1)

#-------------------------------------------------------------------------------
# Load flags and beam
#-------------------------------------------------------------------------------

# Load beam file
beamfile = os.path.join(data_cfg['root'], data_cfg['beam'])
beam = hp.pspecbeam.PSpecBeamUV(beamfile)
log("Loaded beam file: %s" % beamfile)

# Load weights
# FIXME: Need to load weights from a proper source
wgts = [None for f in files]

#-------------------------------------------------------------------------------
# Calculate power spectrum and package output into PSpecContainer
#-------------------------------------------------------------------------------

# Package data files into PSpecData object
ds = hp.PSpecData(dsets=dsets, wgts=wgts, beam=beam)

# Set-up which baselines to cross-correlate
antpos, ants = dsets[0].get_ENU_antpos(pick_data_ants=True)
antpos = dict(zip(ants, antpos))
red_bls = redcal.get_pos_reds(antpos, bl_error_tol=1.0, low_hi=True)

bls = red_bls[0] # FIXME
print("Baselines: %s" % bls)

# Replace default pspec settings if specified in config file
for key in pspec_defaults.keys():
    if key in pspec_cfg.keys(): pspec_defaults[key] = pspec_cfg[key]

# Open or create PSpecContainer to store output power spectra
ps_store = hp.PSpecContainer(pspec_cfg['output'], mode='rw')

# Loop over pairs of datasets
dset_idxs = range(len(ds.dsets))
for i in dset_idxs:
    for j in dset_idxs:
        if i == j: continue
        
        # Name for this set of power spectra
        pspec_name = "pspec_dset(%d,%d)" % (i,j)
        
        # Calculate power spectra for all baseline pairs (returns UVPSpec)
        pspecs = ds.pspec([bls,], [bls,], dsets=(i,j), 
                          input_data_weight=pspec_cfg['weight'], 
                          norm=pspec_cfg['norm'], 
                          taper=pspec_cfg['taper'], 
                          avg_group=pspec_defaults['avg_group'], 
                          exclude_auto_bls=pspec_defaults['exclude_auto_bls'], 
                          exclude_conjugated_blpairs=\
                              pspec_defaults['exclude_conjugated_blpairs'],
                          spw_ranges=None,
                          little_h=pspec_defaults['little_h'])
        
        # Store power spectra in container
        ps_store.set_pspec(group=pspec_cfg['groupname'], pspec=pspec_name, 
                           ps=pspecs, overwrite=pspec_defaults['overwrite'])

# Check that power spectra can be retrieved
pspectra = ps_store.get_pspec(pspec_cfg['groupname'], pspec=None)
print pspectra

# Open file for reading
#px = hp.PSpecContainer(pspec_cfg['output'], mode='r')
#print(px.data[pspec_cfg['groupname']].keys())

#-------------------------------------------------------------------------------
# Output empirical covariance matrix diagnostics
#-------------------------------------------------------------------------------
#ps.iC(key)


