#!/usr/bin/env python
"""
Pipeline script to extract autocorrelations from chunked files into waterfall.
"""
from hera_pspec import utils
from pyuvdata import UVData
from hera_cal._cli_tools import parse_args, run_with_profiling
import warnings

def main(args):
    def check_for_sumdiff(file):
        if not args.sumdiff in file:
            raise ValueError(f"Supposedly processing {args.sumdiff} files but "
                             f"{args.sumdiff} not in the filename.")
        return
    
    # In case there are files without autos
    found_autos = False
    for file_ind, file in enumerate(args.flist):
        check_for_sumdiff(file)
        try:
            main_uvd = UVData()
            main_uvd.read(file, ant_str="auto")
            found_autos = True
            break
        except ValueError: # There were no autos in that file
            continue
    
    if found_autos:
        start_ind = file_ind + 1
        if start_ind < len(args.flist): 
            for file in args.flist[file_ind + 1:]:
                check_for_sumdiff(file)
                try:
                    new_uvd = UVData()
                    new_uvd.read(file, ant_str="auto")
                    main_uvd.__add__(new_uvd, inplace=True)
                except ValueError:
                    continue
        else:
            warnings.warn("Only one file had autocorrelatons. Inputs are almost "
                          "certainly incorrect.")
        
        outfile = f"zen.LST.0.00000.{args.sumdiff}.{args.label}.foreground_filled.xtalk_filtered.chunked.waterfall.autos.uvh5"
        main_uvd.write_uvh5(outfile, clobber=True)
    else:
        raise ValueError("No autocorrelations found in any files. Check inputs.")
    
parser = utils.extract_autos_post_lstbin_parser()
args = parse_args(parser)
run_with_profiling(main, args, args)

