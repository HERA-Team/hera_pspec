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
if args.keep_vispols:
    # if inplace, append new pstokes onto existing file.
    uvd_output = copy.deepcopy(uvd)
else:
    # otherwise, output uvd does not contain original polarizations.
    uvd_output = pstokes.construct_pstokes(uvd, uvd, args.pstokes[0])
for p in args.pstokes:
    if pyuvdata.utils.polstr2num(p) not in uvd_output.polarization_array:
        uvd_output += pstokes.construct_pstokes(uvd, uvd, pstokes=p)

# write file.
uvd_output.write_uvh5(args.outputdata, clobber=args.clobber)
