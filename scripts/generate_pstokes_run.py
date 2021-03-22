"""
Pipeline script for generating pstokes visibility.
"""

from . import pstokes
from pyuvdata import UVData

ap = pstokes.gemerate_pstokes_argparser()
args = ap.parse_args()
uvd = UVData()
uvd.read(args.inputdata)
if args.outputdata is None:
    args.outputdata = args.inputdata
# generate pstokes and append to file.
uvd += pstokes.construct_pstokes(uvd, uvd, pstokes=args.pstokes)
# overwrite file.
uvd.write_uvh5(args.outputdata, clobber=args.clobber)
