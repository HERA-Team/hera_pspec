#!/usr/bin/env python
"""
Run HERA OQE power spectrum estimation code on sets of redundant baselines.
"""
import numpy as np
import hera_pspec as hp
import pyuvdata as uv
import os, sys, glob, time


def log(msg, lvl=0):
    """
    Add a message to the log (just prints to the terminal for now).
    
    Parameters
    ----------
    lvl : int, optional
        Indent level of the message. Each level adds two extra spaces. 
        Default: 0.
    """
    print("%s%s" % ("  "*lvl, msg))


#-------------------------------------------------------------------------------
# Settings
#-------------------------------------------------------------------------------

"""
DATA_DIR = "/lustre/aoc/projects/hera/H1C_IDR2/IDR2_1/"
SUBDIRS = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, 
           2458106, 2458107, 2458108, 2458109, 2458110, 2458111, 2458112, 
           2458113, 2458114, 2458115, 2458116, 2458140]
NAME_TEMPLATE = "zen.*.yy.HH.uvOCRSD"
"""
DATA_DIR = "../hera_pspec/"
SUBDIRS = ["data",]
NAME_TEMPLATE = "zen.*.xx.HH.uvXA"
BEAM_FILE = "data/NF_HERA_Beams.beamfits"
PSPEC_OUTPUT = "hera_red.hdf5"

# zen.2458098.66239.yy.HH.uv.vis.uvfits.flags.npz

#-------------------------------------------------------------------------------
# Prepare list of data files
#-------------------------------------------------------------------------------

files = []
for i in range(len(SUBDIRS)):
    files += glob.glob( os.path.join(DATA_DIR, SUBDIRS[i], NAME_TEMPLATE) )

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
beamfile = os.path.join(DATA_DIR, BEAM_FILE)
beam = hp.pspecbeam.PSpecBeamUV(beamfile)
log("Loaded beam file: %s" % beamfile)

# Load weights
# FIXME: Need to do this properly
wgts = [None for f in files]

#-------------------------------------------------------------------------------
# Package data files into PSpecData object
#-------------------------------------------------------------------------------
hp.PSpecData(dsets=dsets, wgts=wgts, beam=beam)


#-------------------------------------------------------------------------------
# Calculate power spectrum and package into PSpecContainer
#-------------------------------------------------------------------------------



pspecs = PSpecContainer(PSPEC_OUTPUT, mode='rw')



#-------------------------------------------------------------------------------
# Output empirical covariance matrix diagnostics
#-------------------------------------------------------------------------------


