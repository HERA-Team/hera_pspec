#!/usr/bin/env python
"""
Pipeline script to obtain error bars from autocorrelations.
"""

from . import utils
from .container import PSpecContainer
from pyuvdata import UVData

parser = utils.uvp_noise_error_parser()
parser.parse_args()

# compute Tsys from autocorr uvd
uvd = UVData()
uvd.read(args.auto_file)
auto_Tsys = utils.uvd_to_Tsys(uvd, beam=args.beam)
# load in pspec container.
psc = PSpecContainer(args.pspec_container, keep_open=False)

# get spectra and groups automatically if not provided.
if args.spectra is not None:
    spectra = args.spectra
else:
    spectra = psc.spectra()
if args.groups is not None:
    groups = args.groups
else:
    groups = psc.groups()

# iterate through spectra and groups,
# compute noise, and update container.
for group in groups:
    for spec in spectra:
        uvp = psc.get_pspec(group, spec)
        utils.uvp_noise_error(uvp, auto_Tsys,
                              err_type=args.err_type)
        psc.set_pspec(group, spec, overwrite=True)
