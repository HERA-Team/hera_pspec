"""
Module to construct Stokes (I,Q,U,V) visibilities from miriad files or UVData objects
"""
import numpy as np
import pyuvdata

# weights used in forming Stokes visibilities
pol_weights = {
    'I': {'XX': 1. , 'YY': 1. },
    'Q': {'XX': 1. , 'YY':-1. },
    'U': {'XY': 1. , 'YX': 1. },
    'V': {'XY':-1.j, 'YX': 1.j},
}

# polarization values for Stokes parameters
pol_stokes = {'I': 1,
              'Q': 2,
              'U': 3,
              'V': 4,
}

def miriad2pyuvdata(dset):
   """
   Converts Miriad file to UVData object

   Parameters
   ----------
   dset : Miriad file
   """
   uv = pyuvdata.UVData()
   return uv.read_miriad(dset)

def combine_pol(uvd1, uvd2, pol1, pol2, outfile, stokes='I'):
   """
   Reads in miriad file and combines visibilities to form the desired Stokes visibilities
   
   Parameters
   ---------
   uvd1 : UVData object
       First UVData object containing data that is used to
       form Stokes visibilities

   uvd2 : UVData oject
       Second UVData objects containing data that is used to
       form Stokes visibilities

   pol1 : Polarization, type: str
       Polarization of the first UVData object
  
   pol2 : Polarization, type: str
       Polarization of the second UVData object

   stokes: Stokes parameter, type: str
       Pseudo stokes parameter to form, can be 'I' or 'Q' or 'U' or 'V'. Default: I
  
   outfile: Name of output file, type: str
       Name of the file containing the Stokes visibilities
   """
   # extracting data array from the UVData objects
   data1 = uvd1.data_array
   data2 = uvd2.data_array

   # extracting flag array from the UVdata objects
   flag1 = uvd1.flag_array
   flag2 = uvd2.flag_array
   #constructing flags (boolean)
   flag = flag1 + flag2
   # constructing Stokes visibilities
   stdata = 0.5 * (pol_wgts[stokes][pol1]*data1 + pol_wgts[stokes][pol2]*data2)

   # assigning and writing data, flags and metadata to UVData object
   uvdS = copy.deepcopy(uvd1)
   uvdS.data_array = stdata # stokes data
   uvdS.flag_array = flag # flag array
   uvdS.polarization_array = np.array([pol_stokes[stokes]]) # polarization number
   uvdS.nsample_array = uvd1.nsample_array + uvd2.nsample_array # nsamples
   uvdS.history = 'merged to form stokes visibilities' + uvd1 + uvd2 # history
   uvdS.write_miriad(outfile)

   return uvdS

def construct_stokes(dset1, dset2, stokes='I', outfile=None):
   """
   Validates datasets required to construct desired visibilities and constructs desired Stokes parameters
   
   Parameters
   ----------
   dset1 : UVData object or Miriad file
       First UVData object or Miriad file containing data that is used to
       form Stokes visibilities

   dset2 : UVData oject or Miriad file
       Second UVData object or Miriad file containing data that is used to
       form Stokes visibilities

   stokes: Stokes parameter, type: str
       Pseudo stokes parameter to form, can be 'I' or 'Q' or 'U' or 'V'. Default: I

   outfile: Name of output file, type: str
       Name of the file containing the Stokes visibilities. Default: Input file name with the polarization label replaced by the specified Stokes parameter for example if input file is zen.all.xx.LST.1.06964.uvA the default output is zen.all.I.LST.1.06964.uvA for Stokes=I'
   """
   # convert dset1 and dset2 to UVData objects if they are miriad files
   if isinstance(dset1, pyuvdata.UVData) == False:
      uvd1 = miriad2pyuvdata(dset1)
   else:
      uvd1 = dset1
   if isinstance(dset2, pyuvdata.UVData) == False:
      uvd2 = miriad2pyuvdata(dset2)
   else:
      uvd2 = dset2

   # extracts polarization and ensures that the input dsets have the proper polarizations to form the desired stokes parameters
   pol1 = uvd1.get_pols()[0]
   pol2 = uvd2.get_pols()[0]

   # name of output file
   if outfile==None:
      outfile = dset1.replace(pol1,stokes)

   # combining the visibilities to form the desired Stokes visibilties
   uvdS = combine_pol(uvd1=uvd1,uvd2=uvd2,pol1=pol1,pol2=pol2,stokes=stokes,outfile=outfile)

   return uvdS
