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
# TODO: Need to find available redundant baseline sets
#bls = [(37, 39),]

antpos, ants = dsets[0].get_ENU_antpos(pick_data_ants=True)
antpos = dict(zip(ants, antpos))
red_bls = redcal.get_pos_reds(antpos, bl_error_tol=1.0, low_hi=True)

# FIXME
bls = red_bls[0]
print("Baselines:", bls)

# Calculate power spectra for all baseline pairs
# pspecs: (N_bls, N_freq, N_LST)
pspecs, blpairs = ds.pspec(bls, 
                           input_data_weight=pspec_cfg['weight'], 
                           norm=pspec_cfg['norm'], 
                           taper=pspec_cfg['taper'], 
                           little_h=True)

# Create new PSpecContainer to store output power spectra
ps_store = hp.PSpecContainer(pspec_cfg['output'], mode='rw')

# FIXME: Placeholder for proper UVPSpec implementation
for i in range(pspecs.shape[0]):
    pair_name = "(%s).(%s)" % (",".join(str(n) for n in blpairs[i][0]), 
                               ",".join(str(n) for n in blpairs[i][1]))
    print pair_name
    ps = hp.UVPSpec(data_dict={'pspec.%s' % pair_name : pspecs[i]}, 
                    attr_dict={'bls.%s' % pair_name : blpairs[i]})

    # Add power spectrum to container
    ps_store.set_pspec(group=pspec_cfg['groupname'], pspec="something", ps=ps)


# Open file for reading
px = hp.PSpecContainer(pspec_cfg['output'], mode='r')
print(px.data[pspec_cfg['groupname']])

#dset["FieldA"]

#-------------------------------------------------------------------------------
# Output empirical covariance matrix diagnostics
#-------------------------------------------------------------------------------
#ps.iC(key)


