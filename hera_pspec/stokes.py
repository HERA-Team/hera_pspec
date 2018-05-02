"""
Module to construct pseudo-Stokes (I,Q,U,V) visibilities from miriad files or UVData objects
"""
import numpy as np, os
import pyuvdata
import copy

# weights used in forming Stokes visibilities
pol_weights = {
    'I': {'XX': 1. , 'YY': 1. },
    'Q': {'XX': 1. , 'YY':-1. },
    'U': {'XY': 1. , 'YX': 1. },
    'V': {'XY':-1.j, 'YX': 1.j},
}

def miriad2pyuvdata(dset):
   """
   Converts Miriad file to UVData object

   Parameters
   ----------
   dset : str
      Miriad file to convert to UVData object containing visibilities and corresponding metadata
   """
   uv = pyuvdata.UVData()
   uv.read_miriad(dset)
   return uv

def _combine_pol(uvd1, uvd2, pol1, pol2, stokes='I'):
   """
   Reads in miriad file and combines visibilities to form the desired pseudo-stokes visibilities. It return UVData object containing the Stokes visibilities
   
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
  
   """
   if isinstance(uvd1, pyuvdata.UVData) == False:
      raise TypeError("uvd1 must be a pyuvdata.UVData instance")
   if isinstance(uvd2, pyuvdata.UVData) == False:
      raise TypeError("uvd2 must be a pyuvdata.UVData instance")
   
   assert isinstance(pol1, str)
   assert isinstance(pol2, str)

   # extracting data array from the UVData objects
   data1 = uvd1.data_array
   data2 = uvd2.data_array

   # extracting flag array from the UVdata objects
   flag1 = uvd1.flag_array
   flag2 = uvd2.flag_array
   #constructing flags (boolean)
   flag = flag1 + flag2
   # constructing Stokes visibilities
   stdata = 0.5 * (pol_weights[stokes][pol1]*data1 + pol_weights[stokes][pol2]*data2)

   # assigning and writing data, flags and metadata to UVData object
   uvdS = copy.deepcopy(uvd1)
   uvdS.data_array = stdata # stokes data
   uvdS.flag_array = flag # flag array
   uvdS.polarization_array = np.array([pyuvdata.polstr2num(stokes)]) # polarization number
   uvdS.nsample_array = uvd1.nsample_array + uvd2.nsample_array # nsamples
   uvdS.history = 'merged to form stokes visibilities. ' + uvd1.history + uvd2.history # history

   return uvdS

def construct_stokes(dset1, dset2, stokes='I', run_check=True):
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

   run_check: boolean
      Pyvdata attribute check
   """
   # convert dset1 and dset2 to UVData objects if they are miriad files
   if isinstance(dset1, pyuvdata.UVData) == False:
      assert isinstance(dset1, str)
      uvd1 = miriad2pyuvdata(dset1)
   else:
      uvd1 = dset1
   if isinstance(dset2, pyuvdata.UVData) == False:
      assert isinstance(dset2, str)
      uvd2 = miriad2pyuvdata(dset2)
   else:
      uvd2 = dset2

   # makes the Npol length==1 so that the UVData carries data for the required polarization only
   st_keys = pol_weights[stokes].keys()
   st_keys = st_keys[::-1]
   pvals = map(lambda p: pyuvdata.utils.polstr2num(p), st_keys) # polarization values corresponding to the polarization strings
   
   if uvd1.Npols==uvd2.Npols==1:
      # extracts polarization and ensures that the input dsets have the proper polarizations to form the desired stokes parameters
      pol1 = uvd1.get_pols()[0]
      pol2 = uvd2.get_pols()[0]
      assert(pol1 != pol2), "UVData objects have same polarization"
   else:
      pol1 = st_keys[0]; pol2 = st_keys[1]

   if uvd1.Npols > 1:
      assert (st_keys[0] in uvd1.get_pols()), "Polarization {} not found in UVData object".format(st_keys[0])
      uvd1.select(polarizations=pvals[0],inplace=True)
   if uvd2.Npols > 1:
      assert (st_keys[1] in uvd2.get_pols()), "Polarization {} not found in UVData object".format(st_keys[1])
      uvd2.select(polarizations=pvals[1],inplace=True)
   
   # combining visibilities to form the desired Stokes visibilties
   uvdS = _combine_pol(uvd1=uvd1,uvd2=uvd2,pol1=pol1,pol2=pol2, stokes=stokes)

   if run_check: uvdS.check()
   return uvdS
