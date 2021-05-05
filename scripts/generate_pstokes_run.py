#!/usr/bin/env python
"""
Pipeline script for generating pstokes visibility.
"""

from hera_pspec import pstokes
from pyuvdata import UVData
import pyuvdata

ap = pstokes.generate_pstokes_argparser()
args = ap.parse_args()
uvd = UVData()
uvd.read(args.inputdata)
if args.outputdata is None:
    args.outputdata = args.inputdata
# generate pstokes and append to file.
for p in args.pstokes:
    if pyuvdata.utils.polstr2num(p) not in uvd.polarization_array:
        uvd += pstokes.construct_pstokes(uvd, uvd, pstokes=p)
# overwrite file.
uvd.write_uvh5(args.outputdata, clobber=args.clobber)
