"""
Module to construct pseudo-Stokes (I,Q,U,V) visibilities from miriad files or UVData objects
"""
import numpy as np, os
import pyuvdata
import copy
from collections import OrderedDict as odict
from collections.abc import Iterable
import argparse
import warnings

from . import version, __version__

# Weights used in forming Stokes visibilities.
# See pyuvdata.utils.polstr2num for conversion between polarization string
# and polarization integer. Ex. {'XX': -5, ...}
pol_weights = {
    1: odict([(-5, 1.), (-6, 1.)]),
    2: odict([(-5, 1.), (-6, -1.)]),
    3: odict([(-7, 1.), (-8, 1.)]),
    4: odict([(-7, -1.j), (-8, 1.j)])
}

def miriad2pyuvdata(dset, antenna_nums=None, bls=None, polarizations=None,
                    ant_str=None, time_range=None):
    """
    Reads-in a Miriad filepath to a UVData object

    Parameters
    ----------
    dset : str
        Miriad file to convert to UVData object containing visibilities and
        corresponding metadata

    antenna_nums: integer list
        The antennas numbers to read into the object.

    bls: list of tuples
        A list of antenna number tuples (e.g. [(0,1), (3,2)])
        specifying baselines to read into the object. Ordering of the
        numbers within the tuple does not matter. A single antenna iterable
        e.g. (1,) is interpreted as all visibilities with that antenna.

    ant_str: str
        A string containing information about what kinds of visibility data
        to read-in.  Can be 'auto', 'cross', 'all'. Cannot provide ant_str if
        antenna_nums and/or bls is not None.

    polarizations: integer or string list
        List of polarization integers or strings to read-in.
        Ex: ['xx', 'yy', ...]

    time_range: float list
        len-2 list containing min and max range of times (Julian Date) to read-in.
        Ex: [2458115.20, 2458115.40]

    Returns
    -------
    uvd : pyuvdata.UVData object
    """
    uvd = pyuvdata.UVData()
    uvd.read_miriad(dset, antenna_nums=antenna_nums, bls=bls,
                    polarizations=polarizations, ant_str=ant_str,
                    time_range=time_range)
    return uvd


def _combine_pol(uvd1, uvd2, pol1, pol2, pstokes='pI', x_orientation=None):
    """
    Combines UVData visibilities to form the desired pseudo-stokes visibilities.
    It returns UVData object containing the pseudo-stokes visibilities

    Parameters
    ----------
    uvd1 : UVData object
        First UVData object containing data that is used to
        form Stokes visibilities

    uvd2 : UVData oject
        Second UVData objects containing data that is used to
        form Stokes visibilities

    pol1 : Polarization, type: str
        Polarization of the first UVData object to use in constructing
        pStokes visibility.

    pol2 : Polarization, type: str
        Polarization of the second UVData object to use in constructing
        pStokes visibility.

    pstokes: Pseudo-stokes polarization to form, type: str or int
        Pseudo stokes polarization to form, can be in a number
        or a string ('pI', 'pQ', 'pU', or 'pV') , in which case pyuvdata 
        is used to make the conversion to integer.
        Default: pI

    x_orientation: str, optional
        Orientation in cardinal direction east or north of X dipole.
        Default keeps polarization in X and Y basis.

    Returns
    -------
    uvdS : UVData object
    """
    assert isinstance(uvd1, pyuvdata.UVData), \
        "uvd1 must be a pyuvdata.UVData instance"
    assert isinstance(uvd2, pyuvdata.UVData), \
        "uvd2 must be a pyuvdata.UVData instance"

    for i, uvd in enumerate([uvd1, uvd2]):
        pol_convention = getattr(uvd, 'pol_convention', None)
        if pol_convention is None: 
            warnings.warn(
                f'No polarization convention in uvd{i+1}. '
                'Considering it to be "avg".')
            setattr(uvd, 'pol_convention', 'avg')

    if uvd1.pol_convention != uvd2.pol_convention:
        raise ValueError('pol_convention of uvd1 and uvd2 are different.')

    # combine arrays according to polarization convention
    combined_data, combined_flags, combined_nsamples = _combine_pol_arrays(
        pol1, pol2, pstokes, uvd1.pol_convention,
        [uvd1.data_array, uvd2.data_array],
        [uvd1.flag_array, uvd2.flag_array],
        [uvd1.nsample_array, uvd2.nsample_array])

    # convert pStokes to polarization integer if a string
    if isinstance(pstokes, str):
        pstokes = pyuvdata.utils.polstr2num(pstokes, x_orientation=x_orientation)

    # assigning and writing data, flags and metadata to UVData object
    uvdS = copy.deepcopy(uvd1)
    uvdS.data_array = combined_data # pseudo-stokes data
    uvdS.flag_array = combined_flags # flag array
    uvdS.polarization_array = np.array([pstokes], dtype=int) # polarization number
    uvdS.nsample_array = combined_nsamples

    uvdS.history = f"Merged into pseudo-stokes vis with hera_pspec version {__version__}\n"+\
                   "-"*20+f"\ndset1 history:\n{uvd1.history}\n"+'-'*20+\
                   f"\ndset2 history:{uvd2.history}\n"
    return uvdS


def _combine_pol_arrays(
        pol1, pol2, pstokes, pol_convention='avg',
        data_list=None, flags_list=None, nsamples_list=None,
        x_orientation=None
    ):
    """
    Combines arrays to form the corresponding pseudo-stokes results
    as arrays.

    Parameters
    ----------
    pol1 : Polarization, type: str
        Polarization of the first UVData object to use in constructing
        pStokes visibility.

    pol2 : Polarization, type: str
        Polarization of the second UVData object to use in constructing
        pStokes visibility.

    pstokes: Pseudo-stokes polarization to form, type: str or int
        Pseudo stokes polarization to form, can be in a number
        or a string ('pI', 'pQ', 'pU', or 'pV') in which case pyuvdata
        is used to make the conversion to integer.
        Default: pI.

    pol_convention: str ('avg' or 'sum')
        Convention used to form polarization. For linear polarizations ``XX``
        and ``YY``, the stokes ``I`` sky emission can be mapped to
        ``I = (XX + YY)/2`` (the ``avg`` convention) or ``I = XX + YY`` 
        (the ``sum`` convention).

    data_list : any length 2 iterable of numpy arrays
        Iterable of data arrays to be combined to form their pseudo-Stokes equivalent.
        If only one is given, it is duplicated. For the ``avg`` convention, the
        data arrays are combined as ``(data1 + data2)/2``, in the the ``sum`` 
        convention, the output is ``data1 + data2``.
        Default is None.

    flags_list :  any length 2 iterable of numpy arrays
        Iterable of flag arrays to be combined to form their pseudo-Stokes equivalent.
        If only one is given, it is duplicated.
        Flags are combined with a logical OR.
        Default is None.

    nsamples_list :  any length 2 iterable of numpy arrays
        Iterable of nsamples arrays to be combined to form their pseudo-Stokes equivalent.
        nsamples are combined to preserve proper variance, see hera_pspec issue #391 and #406
        If only one is given, it is duplicated.
        Default is None.

    x_orientation: str, optional
        Orientation in cardinal direction east or north of X dipole.
        Default keeps polarization in X and Y basis.

    Returns
    -------
    combined_data : array
        Array of pseudo-Stokes values obtained from combining the arrays in data.

    combined_flags : array
        Array of samples obtained from combining the arrays in nsamples.

    combined_nsamples : array
        Array of pseudo-Stokes values obtained from combining the arrays in data.

    """
    # convert pol1 and/or pol2 to integer if fed as a string
    if isinstance(pol1, str):
        pol1 = pyuvdata.utils.polstr2num(pol1, x_orientation=x_orientation)
    if isinstance(pol2, str):
        pol2 = pyuvdata.utils.polstr2num(pol2, x_orientation=x_orientation)

    # convert pStokes to polarization integer if a string
    if isinstance(pstokes, str):
        pstokes = pyuvdata.utils.polstr2num(pstokes, x_orientation=x_orientation)

    # get string form of polarizations
    pol1_str = pyuvdata.utils.polnum2str(pol1)
    pol2_str = pyuvdata.utils.polnum2str(pol2)
    pstokes_str = pyuvdata.utils.polnum2str(pstokes)

    # assert pstokes in pol_weights, and pol1 and pol2 in pol_weights[pstokes]
    assert pstokes in pol_weights, \
        "unrecognized pstokes parameter {}".format(pstokes_str)
    assert pol1 in pol_weights[pstokes], \
        "pol1 {} not used in constructing pstokes {}".format(pol1_str, pstokes_str)
    assert pol2 in pol_weights[pstokes], \
        "pol2 {} not used in constructing pstokes {}".format(pol2_str, pstokes_str)

    # assert pol_convention makes sense
    if pol_convention not in ['avg', 'sum']:
        raise ValueError('pol_convention must be avg or sum.')

    # checks on input lists
    for a_list in [data_list, flags_list, nsamples_list]:
        if a_list is None:
            continue
        if not isinstance(a_list, Iterable):
            raise ValueError('Data, flags and nsamples inputs must be iterables')
        elif len(a_list) != 2:
            raise ValueError("Can only combine two arrays.")
        assert np.shape(a_list[0]) == np.shape(a_list[1]), \
            "Arrays in list must have identical shape."

    # constructing flags (boolean)
    if flags_list is not None:
        combined_flags = np.logical_or(flags_list[0], flags_list[1])
    else:
        combined_flags = None

    # constructing Stokes visibilities
    if data_list is not None:
        combined_data = \
            (pol_weights[pstokes][pol1]*data_list[0] \
            + pol_weights[pstokes][pol2]*data_list[1])
        if pol_convention == 'avg':
            combined_data *= 0.5
    else:
        combined_data = None

    # constructing nsamples
    if nsamples_list is not None:
        combined_nsamples = 4 * (nsamples_list[0]**-1 + nsamples_list[1]**-1)**-1
        # for an explanation of the factor of 4, and why it doesn't depend on pol_convention,
        # see https://github.com/HERA-Team/hera_pspec/issues/406
    else:
        combined_nsamples = None

    return combined_data, combined_flags, combined_nsamples


def construct_pstokes(dset1, dset2, pstokes='pI', run_check=True, antenna_nums=None,
                      bls=None, polarizations=None, ant_str=None, time_range=None,
                      history=''):
    """
    Validates datasets required to construct desired visibilities and
    constructs desired pseudo-Stokes visibilities. These are formed
    via the following expression

        ( V_pI )            ( 1  0  0  1 )   ( V_XX )
        | V_pQ |            | 1  0  0 -1 |   | V_XY |
        | V_pU |    = 0.5 * | 0  1  1  0 | * | V_YX |
        ( V_pV )            ( 0 -i  i  0 )   ( V_YY )

    In constructing a given pseudo-Stokes visibilities, the XX or XY polarization is
    taken from dset1, and the YX or YY pol is taken from dset2.

    Parameters
    ----------
    dset1 : UVData object or Miriad file
        First UVData object or Miriad file containing data that is used to
        form Stokes visibilities

    dset2 : UVData oject or Miriad file
        Second UVData object or Miriad file containing data that is used to
        form Stokes visibilities

    pstokes: Stokes polarization, type: str
        Pseudo stokes polarization to form, can be 'pI' or 'pQ' or 'pU' or 'pV'.
        Default: pI

    run_check: boolean
        Option to check for the existence and proper shapes of
        parameters after downselecting data on this object. Default is True.

    antenna_nums: integer list
        The antennas numbers to read into the object.

    bls: list of tuples
        A list of antenna number tuples (e.g. [(0,1), (3,2)])
        specifying baselines to read into the object. Ordering of the
        numbers within the tuple does not matter. A single antenna iterable
        e.g. (1,) is interpreted as all visibilities with that antenna.

    ant_str: str
        A string containing information about what kinds of visibility data
        to read-in.  Can be 'auto', 'cross', 'all'. Cannot provide ant_str if
        antenna_nums and/or bls is not None.

    polarizations: integer or string list
        List of polarization integers or strings to read-in.
        Ex: ['xx', 'yy', ...]

    time_range: float list
        len-2 list containing min and max range of times (Julian Date) to
        read-in. Ex: [2458115.20, 2458115.40]

    history : str
        Extra history string to add to concatenated pseudo-Stokes visibility.

    Returns
    -------
    uvdS : UVData object with pseudo-Stokes visibility
    """
    # convert dset1 and dset2 to UVData objects if they are miriad files
    if isinstance(dset1, pyuvdata.UVData) == False:
        assert isinstance(dset1, str), \
            "dset1 must be fed as a string or UVData object"
        uvd1 = miriad2pyuvdata(dset1, antenna_nums=antenna_nums, bls=bls,
                               polarizations=polarizations, ant_str=ant_str,
                               time_range=time_range)
    else:
        uvd1 = dset1
    if isinstance(dset2, pyuvdata.UVData) == False:
        assert isinstance(dset2, str), \
            "dset2 must be fed as a string or UVData object"
        uvd2 = miriad2pyuvdata(dset2, antenna_nums=antenna_nums, bls=bls,
                               polarizations=polarizations, ant_str=ant_str,
                               time_range=time_range)
    else:
        uvd2 = dset2

    # convert pstokes to integer if fed as a string
    if isinstance(pstokes, str):
        pstokes = pyuvdata.utils.polstr2num(pstokes, x_orientation=dset1.telescope.x_orientation)

    # check if dset1 and dset2 habe the same spectral window
    spw1 = uvd1.spw_array
    spw2 = uvd2.spw_array
    assert (spw1 == spw2), "dset1 and dset2 must have the same spectral windows."

    # check if dset1 and dset2 have the same frequencies
    freqs1 = uvd1.freq_array
    freqs2 = uvd2.freq_array
    if np.array_equal(freqs1, freqs2) == False:
        raise ValueError("dset1 and dset2 must have the same frequencies.")

    # check if dset1 and dset2 have the same timestamps
    times1 = uvd1.time_array
    times2 = uvd2.time_array
    if np.array_equal(times1, times2) == False:
        raise ValueError("dset1 and dset2 must have the same timestamps.")

    # check if dset1 and dset2 have the same baselines
    bls1 = uvd1.baseline_array
    bls2 = uvd2.baseline_array
    if np.array_equal(bls1, bls2) == False:
        raise ValueError("dset1 and dset2 must have the same baselines")

    # makes the Npol length==1 so that the UVData carries data for the
    # required polarization only
    st_keys = list(pol_weights[pstokes].keys())
    req_pol1 = st_keys[0]
    req_pol2 = st_keys[1]

    # check polarizations of UVData objects are consistent with the required
    # polarization to form the desired pseudo Stokes visibilities. If multiple
    # exist, downselect on polarization.
    assert req_pol1 in uvd1.polarization_array, \
        "Polarization {} not found in dset1 object".format(req_pol1)
    if uvd1.Npols > 1:
        uvd1 = uvd1.select(polarizations=req_pol1, inplace=False)

    assert req_pol2 in uvd2.polarization_array, \
        "Polarization {} not found in dset2 object".format(req_pol2)
    if uvd2.Npols > 1:
        uvd2 = uvd2.select(polarizations=req_pol2, inplace=False)

    # combining visibilities to form the desired Stokes visibilties
    uvdS = _combine_pol(uvd1=uvd1, uvd2=uvd2, pol1=req_pol1, pol2=req_pol2,
                        pstokes=pstokes)
    uvdS.history += history

    if run_check:
        uvdS.check()

    return uvdS


def filter_dset_on_stokes_pol(dsets, pstokes):
    """
    Given a list of UVData objects with dipole linear polarizations,
    and a desired output pstokes, return the two UVData objects from
    the input dsets that can be used in construct_pstokes to make
    the desired pseudo-Stokes visibility. If a single UVData object
    has multiple polarizations, this function only considers its first.

    Parameters
    ----------
    dsets : list
        List of UVData objects with linear dipole polarizations

    pstokes : str or int
        Pseudo-stokes polarization one wants to form out of input dsets.
        Ex. 'pI', or 'pU', or 1, ...

    Returns
    -------
    inp_dsets : list
        List of two UVData objects from the input dsets that can be fed
        to construct_pstokes to make the desired pseudo-Stokes visibility.
    """
    # type check
    assert isinstance(dsets, list), \
        "dsets must be fed as a list of UVData objects"
    assert np.all(isinstance(d, UVData) for d in dsets), \
        "dsets must be fed as a list of UVData objects"

    # get polarization of each dset
    pols = [d.polarization_array[0] for d in dsets]

    # convert pstokes to integer if a string
    if isinstance(pstokes, str):
        pstokes = pyuvdata.utils.polstr2num(pstokes, x_orientation=dsets[0].telescope.x_orientation)
    assert pstokes in [1, 2, 3, 4], \
        "pstokes must be fed as a pseudo-Stokes parameter"

    # get two necessary dipole pols given pstokes
    desired_pols = list(pol_weights[pstokes].keys())
    assert desired_pols[0] in pols and desired_pols[1] in pols, \
        "necessary input pols {} and {} not found in dsets".format(*desired_pols)

    inp_dsets = [dsets[pols.index(desired_pols[0])],
                 dsets[pols.index(desired_pols[1])]]

    return inp_dsets

def generate_pstokes_argparser():
    """
    Get argparser to generate pstokes from linpol files.

    Args:
        N/A
    Returns:
        a: argparser object with arguments used in generate_pstokes_run.py
    """
    a = argparse.ArgumentParser(description="argument parser for computing "
                                            "pstokes from linpol files.")
    a.add_argument("inputdata", type=str, help="Filename of UVData object with"
                                               "linearly polarized data to add pstokes to.")
    a.add_argument("--pstokes", type=str, help="list of pStokes you wish to calculate. Default is ['pI']",
                   nargs="+", default="pI")
    a.add_argument("--outputdata", type=str, help="Filename to write out data. Output includes original linear pols."
                                                   "if no outputdata is provided, will use inputdata, appending"
                                                   "pstokes to original linear pols.")
    a.add_argument("--clobber", action="store_true", default=False, help="Overwrite outputdata or original linpol only file.")
    a.add_argument("--keep_vispols", action="store_true", default=False, help="If inplace, keep the original linear polarizations in the input file. Default is False.")
    return a
