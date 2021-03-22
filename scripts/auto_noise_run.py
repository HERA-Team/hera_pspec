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
# iterate through spectra, compute noise, and update container.
if args.spectra is not None:
    spectra = args.spectra
else:
    spectra = psc.spectra()
for spec in spectra:
    uvp = psc.get_pspec(args.group, spec)
    utils.uvp_noise_error(uvp, auto_Tsys,
                          err_type=args.err_type,
                          precomp_P_N=args.precomp_P_N)
    psc.set_pspec(group, spec, overwrite=True)
