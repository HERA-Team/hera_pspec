
import numpy as np
import copy, operator
from collections import OrderedDict as odict

def _get_blpairs_from_bls(uvp, bls, only_pairs_in_bls=False):
    """
    Get baseline pair matches from a list of baseline antenna-pairs in a UVPSpec object.

    Parameters
    ----------
    uvp : UVPSpec object with at least meta-data in required params loaded in.
        If only meta-data is loaded in then h5file must be specified.

    bls : list of i6 baseline integers or baseline tuples, Ex. (2, 3) 
        Select all baseline-pairs whose first _or_ second baseline are in bls list.
        This changes if only_pairs_in_bls == True.

    only_pairs_in_bls : bool, if True, keep only baseline-pairs whose first _and_ second baseline
        are both found in bls list.

    Returns blp_select
    -------
    blp_select : boolean ndarray used to index into uvp.blpair_array to get relevant baseline-pairs
    """
    # get blpair baselines in integer form
    bl1 = np.floor(uvp.blpair_array / 1e6)
    blpair_bls = np.vstack([bl1, uvp.blpair_array - bl1*1e6]).astype(np.int32).T
    # ensure bls is in integer form
    if isinstance(bls, tuple):
        assert isinstance(bls[0], (int, np.integer)), "bls must be fed as a list of baseline tuples Ex: [(1, 2), ...]"
        bls = [uvp.antnums_to_bl(bls)]
    elif isinstance(bls, list):
        if isinstance(bls[0], tuple):
            bls = map(lambda b: uvp.antnums_to_bl(b), bls)
    elif isinstance(bls, (int, np.integer)):
        bls = [bls]
    # get indices
    if only_pairs_in_bls:
        blp_select = np.array(map(lambda blp: np.bool((blp[0] in bls) * (blp[1] in bls)), blpair_bls))
    else:
        blp_select = np.array(map(lambda blp: np.bool((blp[0] in bls) + (blp[1] in bls)), blpair_bls))

    return blp_select


def _select(uvp, spws=None, bls=None, only_pairs_in_bls=False, blpairs=None, times=None, pols=None, h5file=None):

    """
    Select function for selecting out certain slices of the data, as well as loading in data from HDF5 file.

    Parameters
    ----------
    uvp : UVPSpec object with at least meta-data in required params loaded in.
        If only meta-data is loaded in then h5file must be specified.

    spws : list of spectral window integers to select

    bls : list of i6 baseline integers or baseline tuples, Ex. (2, 3) 
        Select all baseline-pairs whose first _or_ second baseline are in bls list.
        This changes if only_pairs_in_bls == True.

    only_pairs_in_bls : bool, if True, keep only baseline-pairs whose first _and_ second baseline
        are both found in bls list.

    blpairs : list of baseline-pair tuples or integers to keep, if bls is also fed, this list is concatenated
        onto the baseline-pair list constructed from from the bls selection
    
    times : float ndarray of times from the time_avg_array to keep

    pols : list of polarization strings or integers to keep. See pyuvdata.utils.polstr2num for acceptable options.

    h5file : h5py file descriptor, used for loading in selection of data from HDF5 file
    """
    if spws is not None:
        # spectral window selection
        spw_select = uvp.spw_to_indices(spws)
        uvp.spw_array = uvp.spw_array[spw_select]
        uvp.freq_array = uvp.freq_array[spw_select]
        uvp.dly_array = uvp.dly_array[spw_select]
        uvp.Nspws = len(np.unique(uvp.spw_array))
        uvp.Ndlys = len(np.unique(uvp.dly_array))
        uvp.Nspwdlys = len(uvp.spw_array)
        if hasattr(uvp, 'scalar_array'):
            uvp.scalar_array = uvp.scalar_array[spws, :]

    if bls is not None:
        # get blpair baselines in integer form
        bl1 = np.floor(uvp.blpair_array / 1e6)
        blpair_bls = np.vstack([bl1, uvp.blpair_array - bl1*1e6]).astype(np.int32).T
        blp_select = _get_blpairs_from_bls(uvp, bls, only_pairs_in_bls=only_pairs_in_bls)

    if blpairs is not None:
        if bls is None:
            blp_select = np.zeros(uvp.Nblpairts, np.bool)
        # assert form
        assert isinstance(blpairs[0], (tuple, int, np.integer)), "blpairs must be fed as a list of baseline-pair tuples or baseline-pair integers"
        # if fed as list of tuples, convert to integers
        if isinstance(blpairs[0], tuple):
            blpairs = map(lambda blp: uvp.antnums_to_blpair(blp), blpairs)
        blpair_select = np.array(reduce(operator.add, map(lambda blp: uvp.blpair_array == blp, blpairs)))
        blp_select += blpair_select

    if times is not None:
        if bls is None and blpairs is None:
            blp_select = np.ones(uvp.Nblpairts, np.bool)
        time_select = np.array(reduce(operator.add, map(lambda t: np.isclose(uvp.time_avg_array, t, rtol=1e-16), times)))
        blp_select *= time_select

    if bls is None and blpairs is None and times is None:
        blp_select = slice(None)
    else:
        # assert something was selected
        assert blp_select.any(), "no selections provided matched any of the data... "

        # index arrays
        uvp.blpair_array = uvp.blpair_array[blp_select]
        uvp.time_1_array = uvp.time_1_array[blp_select]
        uvp.time_2_array = uvp.time_2_array[blp_select]
        uvp.time_avg_array = uvp.time_avg_array[blp_select]
        uvp.lst_1_array = uvp.lst_1_array[blp_select]
        uvp.lst_2_array = uvp.lst_2_array[blp_select]
        uvp.lst_avg_array = uvp.lst_avg_array[blp_select]
        uvp.Ntimes = len(np.unique(uvp.time_avg_array))
        uvp.Nblpairs = len(np.unique(uvp.blpair_array))
        uvp.Nblpairts = len(uvp.blpair_array)
        if bls is not None:
            bl_array = np.unique(blpair_bls)
            bl_select = reduce(operator.add, map(lambda b: uvp.bl_array==b, bl_array))
            uvp.bl_array = uvp.bl_array[bl_select]
            uvp.bl_vecs = uvp.bl_vecs[bl_select]
            uvp.Nbls = len(uvp.bl_array)        

    if pols is not None:
        # assert form
        assert isinstance(pols[0], (str, np.str, int, np.integer)), "pols must be fed as a list of pol strings or pol integers"

        # if fed as strings convert to integers
        if isinstance(pols[0], (np.str, str)):
            pols = map(lambda p: uvutils.polstr2num(p), pols)

        # create selection
        pol_select = np.array(reduce(operator.add, map(lambda p: uvp.pol_array == p, pols)))

        # edit metadata
        uvp.pol_array = uvp.pol_array[pol_select]
        uvp.Npols = len(uvp.pol_array)
        if hasattr(uvp, 'scalar_array'):
            uvp.scalar_array = uvp.scalar_array[:, pol_select]
    else:
        pol_select = slice(None)

    try:
        # select data arrays
        data = odict()
        wgts = odict()
        ints = odict()
        nsmp = odict()
        for s in np.unique(uvp.spw_array):
            if h5file is not None:
                data[s] = h5file['data_spw{}'.format(s)][blp_select, :, pol_select]
                wgts[s] = h5file['wgt_spw{}'.format(s)][blp_select, :, :, pol_select]
                ints[s] = h5file['integration_spw{}'.format(s)][blp_select, pol_select]
                nsmp[s] = h5file['nsample_spw{}'.format(s)][blp_select, pol_select]
            else:
                data[s] = uvp.data_array[s][blp_select, :, pol_select]
                wgts[s] = uvp.wgt_array[s][blp_select, :, :, pol_select]
                ints[s] = uvp.integration_array[s][blp_select, pol_select]
                nsmp[s] = uvp.nsample_array[s][blp_select, pol_select]
 
        uvp.data_array = data
        uvp.wgt_array = wgts
        uvp.integration_array = ints
        uvp.nsample_array = nsmp
    except AttributeError:
        # if no h5file fed and hasattr(uvp, data_array) is False then just load meta-data
        pass


def _blpair_to_antnums(blpair):
    """
    Convert baseline-pair integer to nested tuple of antenna numbers.

    Parameters
    ----------
    blpair : <i12 integer
        baseline-pair integer

    Returns
    -------
    antnums : tuple
        nested tuple containing baseline-pair antenna numbers. Ex. ((ant1, ant2), (ant3, ant4))
    """
    # get antennas
    ant1 = int(np.floor(blpair / 1e9))
    ant2 = int(np.floor(blpair / 1e6 - ant1*1e3))
    ant3 = int(np.floor(blpair / 1e3 - ant1*1e6 - ant2*1e3))
    ant4 = int(np.floor(blpair - ant1*1e9 - ant2*1e6 - ant3*1e3))

    # form antnums tuple
    antnums = ((ant1, ant2), (ant3, ant4))

    return antnums

def _antnums_to_blpair(antnums):
    """
    Convert nested tuple of antenna numbers to baseline-pair integer.

    Parameters
    ----------
    antnums : tuple
        nested tuple containing integer antenna numbers for a baseline-pair.
        Ex. ((ant1, ant2), (ant3, ant4))

    Returns
    -------
    blpair : <i12 integer
        baseline-pair integer
    """
    # get antennas
    ant1 = antnums[0][0]
    ant2 = antnums[0][1]
    ant3 = antnums[1][0]
    ant4 = antnums[1][1]

    # form blpair
    blpair = int(ant1*1e9 + ant2*1e6 + ant3*1e3 + ant4)

    return blpair

def _bl_to_antnums(bl):
    """
    Convert baseline integer to tuple of antenna numbers.

    Parameters
    ----------
    blpair : <i6 integer
        baseline integer

    Returns
    -------
    antnums : tuple
        tuple containing baseline antenna numbers. Ex. (ant1, ant2)
    """
    # get antennas
    ant1 = int(np.floor(bl / 1e3))
    ant2 = int(np.floor(bl - ant1*1e3))

    # form antnums tuple
    antnums = (ant1, ant2)

    return antnums

def _antnums_to_bl(antnums):
    """
    Convert tuple of antenna numbers to baseline integer.

    Parameters
    ----------
    antnums : tuple
        tuple containing integer antenna numbers for a baseline.
        Ex. (ant1, ant2)

    Returns
    -------
    blpair : <i6 integer
        baseline integer
    """
    # get antennas
    ant1 = antnums[0]
    ant2 = antnums[1]

    # form blpair
    blpair = int(ant1*1e3 + ant2)

    return blpair

def _blpair_to_bls(blpair):
    """
    Convert a blpair integer or nested tuple of antenna pairs
    into a tuple of baseline integers

    Parameters
    ----------
    blpair : baseline-pair integer or nested antenna-pair tuples
    """
    # convert to antnums if fed as ints
    if isinstance(blpair, (int, np.integer)):
        blpair = _blpair_to_antnums(blpair)

    # convert first and second baselines to baseline ints
    bl1 = _antnums_to_bl(blpair[0])
    bl2 = _antnums_to_bl(blpair[1])

    return bl1, bl2

def _conj_blpair_int(blpair):
    """
    Conjugate a baseline-pair integer

    Parameters
    ----------
    blpair : <12 int
        baseline-pair integer

    Returns
    --------
    conj_blpair : <12 int
        conjugated baseline-pair integer. 
        Ex: ((ant1, ant2), (ant3, ant4)) --> ((ant3, ant4), (ant1, ant2))
    """
    antnums = _blpair_to_antnums(blpair)
    conj_blpair = _antnums_to_blpair(antnums[::-1])
    return conj_blpair


def _conj_bl_int(bl):
    """
    Conjugate a baseline integer

    Parameters
    ----------
    blpair : i6 int
        baseline integer

    Returns
    --------
    conj_bl : i6 int
        conjugated baseline integer. 
        Ex: (ant1, ant2) --> (ant2, ant1)
    """
    antnums = _bl_to_antnums(bl)
    conj_bl = _antnums_to_bl(antnums[::-1])
    return conj_bl


def _conj_blpair(blpair, which='both'):
    """
    Conjugate one or both baseline(s) in a baseline-pair
    Ex. ((ant1, ant2), (ant3, ant4)) --> ((ant2, ant1), (ant4, ant3))

    Parameters
    ----------
    blpair : <12 int
        baseline-pair int

    which : str, options=['first', 'second', 'both']
        which baseline to conjugate

    Returns
    -------
    conj_blpair : <12 int
        blpair with one or both baselines conjugated
    """
    antnums = _blpair_to_antnums(blpair)
    if which == 'first':
        conj_blpair = _antnums_to_blpair((antnums[0][::-1], antnums[1]))
    elif which == 'second':
        conj_blpair = _antnums_to_blpair((antnums[0], antnums[1][::-1]))
    elif which == 'both':
        conj_blpair = _antnums_to_blpair((antnums[0][::-1], antnums[1][::-1]))
    else:
        raise ValueError("didn't recognize {}".format(which))

    return conj_blpair

def _fast_lookup_blpairts(src_blpts, query_blpts):
    """
    Helper function to allow fast lookups of array indices for large arrays of 
    blpair-time tuples.
    
    Parameters
    ----------
    src_blpts : list of tuples or array_like
        List of tuples or array of shape (N, 2), containing a list of (blpair, 
        time) couplets.
    
    query_blpts : list of tuples or array_like
        List of tuples or array of shape (M, 2), containing a list of (blpair, 
        time) couplets that you want to find the indices of in source_blpts.
    
    Returns
    -------
    blpts_idxs : array_like
        Array of integers of size (M,), which are indices in the source_blpts 
        array for each item in query_blpts.
    """
    # This function works by using a small hack -- the blpair-times are turned 
    # into complex numbers of the form (blpair + 1.j*time), allowing numpy 
    # array lookup functions to be used
    src_blpts = np.asarray(src_blpts)
    query_blpts = np.asarray(query_blpts)
    src_blpts = src_blpts[:,0] + 1.j*src_blpts[:,1]
    query_blpts = query_blpts[:,0] + 1.j*query_blpts[:,1]
    
    # Do np.where comparison for all new_blpts
    # (indices stored in second array returned by np.where)
    blpts_idxs = np.where(src_blpts == query_blpts[:,np.newaxis])[1]
    return blpts_idxs
    

