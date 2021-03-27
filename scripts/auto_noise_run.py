#!/usr/bin/env python
"""
Pipeline script to obtain error bars from autocorrelations.
"""

from hera_pspec import utils
from hera_pspec.container import PSpecContainer
from pyuvdata import UVData

parser = utils.uvp_noise_error_parser()
args = parser.parse_args()

# compute Tsys from autocorr uvd
uvd = UVData()
uvd.read(args.auto_file)
auto_Tsys = utils.uvd_to_Tsys(uvd, beam=args.beam)
# load in pspec container.
psc = PSpecContainer(args.pspec_container, keep_open=False, mode='rw', swmr=False)

# get groups automatically if not provided.
if args.groups is not None:
    groups = args.groups
else:
    groups = psc.groups()

# iterate through spectra and groups,
# compute noise, and update container.
for group in groups:
    if args.spectra is None:
        spectra = psc.spectra(group)
    for spec in spectra:
        uvp = psc.get_pspec(group, spec)
        utils.uvp_noise_error(uvp, auto_Tsys,
                              err_type=args.err_type)
        psc.set_pspec(group, spec, uvp, overwrite=True)
psc.save()
