import numpy as np
import copy, operator
from collections import OrderedDict as odict
from pyuvdata.utils import polstr2num, polnum2str
import json
import warnings

from . import utils


def subtract_uvp(uvp1, uvp2, run_check=True, verbose=False):
    """
    Subtract uvp2.data_array from uvp1.data_array. Subtract matching
    spw, blpair-lst, polpair keys. For non-overlapping keys, remove from
    output uvp.

    Note: Entries in stats_array are added in quadrature. nsample_arrays,
    integration_arrays and wgt_arrays are inversely added in quadrature.
    If these arrays are identical in the two objects, this is equivalent
    to multiplying (dividing) the stats (nsamp, int & wgt) arrays(s) by
    sqrt(2).

    Parameters
    ----------
    uvp1 : UVPSpec object
        Object to subtract uvp2 data from

    uvp2 : UVPSpec object
        Object with data to subtract

    run_check : bool, optional
        If True, run uvp.check() before return.

    Returns
    -------
    uvp : UVPSpec object
        A copy of uvp1 with uvp2.data_array subtracted.
    """
    # select out common parts
    uvp1, uvp2 = select_common([uvp1, uvp2], spws=True, blpairs=True, lsts=True,
                               polpairs=True, times=False, inplace=False,
                               verbose=verbose)

    # get metadata
    spws1 = [spw for spw in uvp1.get_spw_ranges()]
    polpairs1 = uvp1.polpair_array.tolist()
    blps1 = sorted(set(uvp1.blpair_array))
    spws2 = [spw for spw in uvp2.get_spw_ranges()]

    # iterate over spws
    for i, spw in enumerate(spws1):
        # get uvp2 index
        i2 = spws2.index(spw)

        # iterate over pol
        for j, polpair in enumerate(polpairs1):

            # iterate over blp
            for k, blp in enumerate(blps1):

                # form keys
                key1 = (i, blp, polpair)
                key2 = (i2, blp, polpair)

                # subtract data
                blp1_inds = uvp1.blpair_to_indices(blp)
                uvp1.data_array[i][blp1_inds, :, j] -= uvp2.get_data(key2)

                # add nsample inversely in quadrature
                uvp1.nsample_array[i][blp1_inds, j] \
                    = np.sqrt( 1. / (1./uvp1.get_nsamples(key1)**2
                             + 1. / uvp2.get_nsamples(key2)**2) )

                # add integration inversely in quadrature
                uvp1.integration_array[i][blp1_inds, j] \
                    = np.sqrt(1. / (1./uvp1.get_integrations(key1)**2
                            + 1. / uvp2.get_integrations(key2)**2))

                # add wgts inversely in quadrature
                uvp1.wgt_array[i][blp1_inds, :, :, j] \
                    = np.sqrt(1. / (1./uvp1.get_wgts(key1)**2
                            + 1. / uvp2.get_wgts(key2)**2))
                uvp1.wgt_array[i][blp1_inds, :, :, j] /= \
                    uvp1.wgt_array[i][blp1_inds, :, :, j].max()

                # add stats in quadrature: real imag separately
                if hasattr(uvp1, "stats_array") and hasattr(uvp2, "stats_array"):
                    for s in uvp1.stats_array.keys():
                        stat1 = uvp1.get_stats(s, key1)
                        stat2 = uvp2.get_stats(s, key2)
                        uvp1.stats_array[s][i][blp1_inds, :, j] \
                            = np.sqrt(stat1.real**2 + stat2.real**2) \
                            + 1j*np.sqrt(stat1.imag**2 + stat2.imag**2)

                # add cov in quadrature: real and imag separately                
                if hasattr(uvp1, "cov_array_real") \
                  and hasattr(uvp2, "cov_array_real"):
                    if uvp1.cov_model == uvp2.cov_model:
                        cov1r = uvp1.get_cov(key1, component='real')
                        cov2r = uvp2.get_cov(key2, component='real')
                        uvp1.cov_array_real[i][blp1_inds, :, :, j] \
                            = np.sqrt(cov1r.real**2 + cov2r.real**2) \
                              + 1j*np.sqrt(cov1r.imag**2 + cov2r.imag**2)
                        
                        cov1i = uvp1.get_cov(key1, component='imag')
                        cov2i = uvp2.get_cov(key2, component='imag')
                        uvp1.cov_array_imag[i][blp1_inds, :, :, j] \
                            = np.sqrt(cov1i.real**2 + cov2i.real**2) \
                              + 1j*np.sqrt(cov1i.imag**2 + cov2i.imag**2)

                # same for window function
                if (hasattr(uvp1, 'window_function_array') 
                    and hasattr(uvp2, 'window_function_array')):
                    window1 = uvp1.get_window_function(key1)
                    window2 = uvp2.get_window_function(key2)
                    uvp1.window_function_array[i][blp1_inds, :, :, j] \
                        = np.sqrt(window1.real**2 + window2.real**2) \
                        + 1j*np.sqrt(window1.imag**2 + window2.imag**2)

    # run check
    if run_check:
        uvp1.check()

    return uvp1


def compress_r_params(r_params_dict):
    """
    Convert a dictionary of r_paramsters to a compressed string format

    Parameters
    ----------
    r_params_dict: Dictionary
              dictionary with parameters for weighting matrix. Proper fields
              and formats depend on the mode of data_weighting.
              data_weighting == 'dayenu':
                            dictionary with fields
                            'filter_centers', list of floats (or float) specifying the (delay) channel numbers
                                              at which to center filtering windows. Can specify fractional channel number.
                            'filter_half_widths', list of floats (or float) specifying the width of each
                                             filter window in (delay) channel numbers. Can specify fractional channel number.
                            'filter_factors', list of floats (or float) specifying how much power within each filter window
                                              is to be suppressed.
    Returns
    -------
    string containing r_params dictionary in json format and only containing one
    copy of each unique dictionary with a list of associated baselines.
    """
    if r_params_dict == {} or r_params_dict is None:
        return ''
    else:
        r_params_unique = {}
        r_params_unique_bls = {}
        r_params_index = -1
        for rp in r_params_dict:
            #do not include data set in tuple key
            already_in = False
            for rpu in r_params_unique:
                if r_params_unique[rpu] == r_params_dict[rp]:
                    r_params_unique_bls[rpu] += [rp,]
                    already_in = True
            if not already_in:
                r_params_index += 1
                r_params_unique[r_params_index] = copy.copy(r_params_dict[rp])
                r_params_unique_bls[r_params_index] = [rp,]


        for rpi in r_params_unique:
            r_params_unique[rpi]['baselines'] = r_params_unique_bls[rpi]
        r_params_str = json.dumps(r_params_unique)
        return r_params_str


def decompress_r_params(r_params_str):
    """
    Decompress json format r_params string into an r_params dictionary.

    Parameters
    ----------
    r_params_str: String
        string with compressed r_params in json format

    Returns
    -------
    r_params: dict
        Dictionary with parameters for weighting matrix. Proper fields
        and formats depend on the mode of data_weighting.
        data_weighting == 'dayenu':
                      dictionary with fields
                      'filter_centers', list of floats (or float) specifying the (delay) channel numbers
                                        at which to center filtering windows. Can specify fractional channel number.
                      'filter_half_widths', list of floats (or float) specifying the width of each
                                       filter window in (delay) channel numbers. Can specify fractional channel number.
                      'filter_factors', list of floats (or float) specifying how much power within each filter window
                                        is to be suppressed.
    """
    decompressed_r_params = {}
    if r_params_str != '' and not r_params_str is None:
        r_params = json.loads(r_params_str)
        for rpi in r_params:
            rp_dict = {}
            for r_field in r_params[rpi]:
                if not r_field == 'baselines':
                    rp_dict[r_field] = r_params[rpi][r_field]
            for blkey in r_params[rpi]['baselines']:
                decompressed_r_params[tuple(blkey)] = rp_dict
    else:
        decompressed_r_params = {}
    return decompressed_r_params


def select_common(uvp_list, spws=True, blpairs=True, times=True, polpairs=True,
                  lsts=False, inplace=False, verbose=False):
    """
    Find spectral windows, baseline-pairs, times, and/or polarization-pairs
    that a set of UVPSpec objects have in common and return new UVPSpec objects
    that contain only those data.

    If there is no overlap, an error will be raised.

    Parameters
    ----------
    uvp_list : list of UVPSpec
        List of input UVPSpec objects.

    spws : bool, optional
        Whether to retain only the spectral windows that all UVPSpec objects
        have in common. For a spectral window to be retained, the entire set of
        delays in that window must match across all UVPSpec objects (this will
        not allow you to just pull out common delays).

        If set to False, the original spectral windows will remain in each
        UVPSpec. Default: True.

    blpairs : bool, optional
        Whether to retain only the baseline-pairs that all UVPSpec objects have
        in common. Default: True.

    times : bool, optional
        Whether to retain only the (average) times that all UVPSpec objects
        have in common. This does not check to make sure that time_1 and time_2
        (i.e. the LSTs for the left-hand and right-hand visibilities that went
        into the power spectrum calculation) are the same. See the
        UVPSpec.time_avg_array attribute for more details. Default: True.

    lsts : bool, optional
        Whether to retain only the (average) lsts that all UVPSpec objects
        have in common. Similar algorithm to times. Default: False.

    polpairs : bool, optional
        Whether to retain only the polarization pairs that all UVPSpec objects
        have in common. Default: True.

    inplace : bool, optional
        Whether the selection should be applied to the UVPSpec objects
        in-place, or new copies of the objects should be returned.

    Returns
    -------
    uvp_list : list of UVPSpec, optional
        List of UVPSpec objects with the overlap selection applied. This will
        only be returned if inplace = False.
    """
    if len(uvp_list) < 2:
        raise IndexError("uvp_list must contain two or more UVPSpec objects.")

    # Get times that are common to all UVPSpec objects in the list
    if times:
        common_times = np.unique(uvp_list[0].time_avg_array)
        has_times = [np.isin(common_times, uvp.time_avg_array)
                     for uvp in uvp_list]
        common_times = common_times[np.all(has_times, axis=0)]
        if verbose: print("common_times:", common_times)

    # Get lsts that are common to all UVPSpec objects in the list
    if lsts:
        common_lsts = np.unique(uvp_list[0].lst_avg_array)
        has_lsts = [np.isin(common_lsts, uvp.lst_avg_array)
                     for uvp in uvp_list]
        common_lsts = common_lsts[np.all(has_lsts, axis=0)]
        if verbose: print("common_lsts:", common_lsts)

    # Get baseline-pairs that are common to all
    if blpairs:
        common_blpairs = np.unique(uvp_list[0].blpair_array)
        has_blpairs = [np.isin(common_blpairs, uvp.blpair_array)
                       for uvp in uvp_list]
        common_blpairs = common_blpairs[np.all(has_blpairs, axis=0)]
        if verbose: print("common_blpairs:", common_blpairs)

    # Get polarization-pairs that are common to all
    if polpairs:
        common_polpairs = np.unique(uvp_list[0].polpair_array)
        has_polpairs = [np.isin(common_polpairs, uvp.polpair_array)
                        for uvp in uvp_list]
        common_polpairs = common_polpairs[np.all(has_polpairs, axis=0)]
        if verbose: print("common_polpairs:", common_polpairs)

    # Get common spectral windows (the entire window must match)
    # Each row of common_spws is a list of that spw's index in each UVPSpec
    if spws:
        common_spws = uvp_list[0].get_spw_ranges()
        has_spws = [[x in uvp.get_spw_ranges() for x in common_spws]
                    for uvp in uvp_list]
        common_spws = [common_spws[i] for i, f in enumerate(np.all(has_spws, axis=0)) if f]
        if verbose: print("common_spws:", common_spws)

    # Check that this won't be an empty selection
    if spws and len(common_spws) == 0:
        raise ValueError("No spectral windows were found that exist in all "
                         "spectra (the entire spectral window must match).")

    if blpairs and len(common_blpairs) == 0:
        raise ValueError("No baseline-pairs were found that exist in all spectra.")

    if times and len(common_times) == 0:
        raise ValueError("No times were found that exist in all spectra.")

    if lsts and len(common_lsts) == 0:
        raise ValueError("No lsts were found that exist in all spectra.")

    if polpairs and len(common_polpairs) == 0:
        raise ValueError("No polarization-pairs were found that exist in all spectra.")

    # Apply selections
    out_list = []
    for i, uvp in enumerate(uvp_list):
        _spws, _blpairs, _times, _lsts, _polpairs = None, None, None, None, None

        # Set indices of blpairs, times, and pols to keep
        if blpairs: _blpairs = common_blpairs
        if times: _times = common_times
        if lsts: _lsts = common_lsts
        if polpairs: _pols = common_polpairs
        if spws: _spws = [uvp.get_spw_ranges().index(j) for j in common_spws]

        _uvp = uvp.select(spws=_spws, blpairs=_blpairs, times=_times,
                          polpairs=_polpairs, lsts=_lsts, inplace=inplace)
        if not inplace: out_list.append(_uvp)

    # Return if not inplace
    if not inplace: return out_list


def polpair_int2tuple(polpair, pol_strings=False):
    """
    Convert a pol-pair integer into a tuple pair of polarization
    integers. See polpair_tuple2int for more details.

    Parameters
    ----------
    polpair : int or list of int
        Integer representation of polarization pair.

    pol_strings : bool, optional
        If True, return polarization pair tuples with polarization strings.
        Otherwise, use polarization integers. Default: False.

    Returns
    -------
    polpair : tuple, length 2
        A length-2 tuple containing a pair of polarization
        integers, e.g. (-5, -5).
    """
    # Recursive evaluation
    if isinstance(polpair, (list, np.ndarray)):
        return [polpair_int2tuple(p, pol_strings=pol_strings) for p in polpair]

    # Check for integer type
    assert isinstance(polpair, (int, np.integer)), \
        "polpair must be integer: %s" % type(polpair)

    # Split into pol1 and pol2 integers
    pol1 = int(str(polpair)[:-2]) - 20
    pol2 = int(str(polpair)[-2:]) - 20

    # Check that pol1 and pol2 are in the allowed range (-8, 4)
    if (pol1 < -8 or pol1 > 4) or (pol2 < -8 or pol2 > 4):
        raise ValueError("polpair integer evaluates to an invalid "
                         "polarization pair: (%d, %d)"
                         % (pol1, pol2))
    # Convert to strings if requested
    if pol_strings:
        return (polnum2str(pol1), polnum2str(pol2))
    else:
        return (pol1, pol2)


def polpair_tuple2int(polpair, x_orientation=None):
    """
    Convert a tuple pair of polarization strings/integers into
    an pol-pair integer.

    The polpair integer is formed by adding 20 to each standardized
    polarization integer (see polstr2num and AIPS memo 117) and
    then concatenating them. For example, polarization pair
    ('pI', 'pQ') == (1, 2) == 2122.

    Parameters
    ----------
    polpair : tuple, length 2
        A length-2 tuple containing a pair of polarization strings
        or integers, e.g. ('XX', 'YY') or (-5, -5).

    x_orientation: str, optional
        Orientation in cardinal direction east or north of X dipole.
        Default keeps polarization in X and Y basis.

    Returns
    -------
    polpair : int
        Integer representation of polarization pair.
    """
    # Recursive evaluation
    if isinstance(polpair, (list, np.ndarray)):
        return [polpair_tuple2int(p) for p in polpair]

    # Check types
    assert type(polpair) in (tuple,), "pol must be a tuple"
    assert len(polpair) == 2, "polpair tuple must have 2 elements"

    # Convert strings to ints if necessary
    pol1, pol2 = polpair
    if type(pol1) is str: pol1 = polstr2num(pol1, x_orientation=x_orientation)
    if type(pol2) is str: pol2 = polstr2num(pol2, x_orientation=x_orientation)

    # Convert to polpair integer
    ppint = (20 + pol1)*100 + (20 + pol2)
    return ppint


def _get_blpairs_from_bls(uvp, bls, only_pairs_in_bls=False):
    """
    Get baseline pair matches from a list of baseline antenna-pairs in a
    UVPSpec object.

    Parameters
    ----------
    uvp : UVPSpec object
        Must at least have meta-data in required params loaded in. If only
        meta-data is loaded in then h5file must be specified.

    bls : list of i6 baseline integers or baseline tuples
        Select all baseline-pairs whose first _or_ second baseline are in bls
        list. This changes if only_pairs_in_bls == True.

    only_pairs_in_bls : bool
        If True, keep only baseline-pairs whose first _and_ second baseline are
        both found in bls list.

    Returns
    -------
    blp_select : bool ndarray
        Used to index into uvp.blpair_array to get relevant baseline-pairs
    """
    # get blpair baselines in integer form
    bl1 = np.floor(uvp.blpair_array / 1e6)
    blpair_bls = np.vstack([bl1, uvp.blpair_array - bl1*1e6]).astype(np.int32).T

    # ensure bls is in integer form
    if isinstance(bls, tuple):
        assert isinstance(bls[0], (int, np.integer)), \
            "bls must be fed as a list of baseline tuples Ex: [(1, 2), ...]"
        bls = [uvp.antnums_to_bl(bls)]
    elif isinstance(bls, list):
        if isinstance(bls[0], tuple):
            bls = [uvp.antnums_to_bl(b) for b in bls]
    elif isinstance(bls, (int, np.integer)):
        bls = [bls]

    # get indices
    if only_pairs_in_bls:
        blp_select = np.array( [bool((blp[0] in bls) * (blp[1] in bls))
                                for blp in blpair_bls] )
    else:
        blp_select = np.array( [bool((blp[0] in bls) + (blp[1] in bls))
                                for blp in blpair_bls] )

    return blp_select


def _select(uvp, spws=None, bls=None, only_pairs_in_bls=False, blpairs=None,
            times=None, lsts=None, polpairs=None, h5file=None):
    """
    Select function for selecting out certain slices of the data, as well
    as loading in data from HDF5 file.

    Parameters
    ----------
    uvp : UVPSpec object
        A UVPSpec with at least meta-data in required params loaded in.
        If only meta-data is loaded in then h5file must be specified.

    spws : list
        A list of spectral window integers to select

    bls : list
        A list of i6 baseline integers or baseline tuples, Ex. (2, 3)
        Select all baseline-pairs whose first _or_ second baseline are in
        bls list. This changes if only_pairs_in_bls == True.

    only_pairs_in_bls : bool
        If True, keep only baseline-pairs whose first _and_ second baseline
        are both found in bls list.

    blpairs : list
        A list of baseline-pair tuples or integers to keep, if bls is also
        fed, this list is concatenated onto the baseline-pair list constructed
        from from the bls selection

    times : list
        List of times from the time_avg_array to keep. Cannot be fed if
        lsts is fed.

    lsts : list
        List of lsts from the lst_avg_array to keep. Cannot be fed if
        times is fed.

    polpairs : list
        A list of polarization-pair tuples, integers, or strs to keep.

    h5file : h5py file descriptor
        Used for loading in selection of data from HDF5 file.
    """
    spw_mapping = None
    if spws is not None:
        # Get info for each spw that will be retained
        spw_freq_select = uvp.spw_to_freq_indices(spws)
        spw_dly_select = uvp.spw_to_dly_indices(spws)
        spw_select = uvp.spw_indices(spws)
        uvp.spw_freq_array = uvp.spw_freq_array[spw_freq_select]
        uvp.spw_dly_array = uvp.spw_dly_array[spw_dly_select]

        # Ordered list of old spw indices for the new spws
        spw_mapping = uvp.spw_array[spw_select]

        # Update spw-related arrays (NB data arrays haven't been reordered yet!)
        uvp.spw_array = uvp.spw_array[spw_select]
        uvp.freq_array = uvp.freq_array[spw_freq_select]
        uvp.dly_array = uvp.dly_array[spw_dly_select]
        uvp.Ndlys = len(np.unique(uvp.dly_array))
        uvp.Nspws = len(np.unique(uvp.spw_array))
        uvp.Nspwdlys = len(uvp.spw_dly_array)
        uvp.Nspwfreqs = len(uvp.spw_freq_array)
        if hasattr(uvp, 'scalar_array'):
            uvp.scalar_array = uvp.scalar_array[spw_select, :]

        # Down-convert spw indices such that spw_array == np.arange(Nspws)
        for i in range(uvp.Nspws):
            if i in uvp.spw_array:
                continue
            spw = np.min(uvp.spw_array[uvp.spw_array > i])
            spw_freq_select = uvp.spw_to_freq_indices(spw)
            spw_dly_select = uvp.spw_to_dly_indices(spw)
            spw_select = uvp.spw_indices(spw)
            uvp.spw_freq_array[spw_freq_select] = i
            uvp.spw_dly_array[spw_dly_select] = i
            uvp.spw_array[spw_select] = i

    if bls is not None:
        # get blpair baselines in integer form
        bl1 = np.floor(uvp.blpair_array / 1e6)
        blpair_bls = np.vstack([bl1, uvp.blpair_array - bl1*1e6]).astype(np.int32).T
        blp_select = _get_blpairs_from_bls(uvp, bls, only_pairs_in_bls=only_pairs_in_bls)

    if blpairs is not None:
        if bls is None:
            blp_select = np.zeros(uvp.Nblpairts, bool)

        # assert form
        assert isinstance(blpairs[0], (tuple, int, np.integer)), \
            "blpairs must be fed as a list of baseline-pair tuples or baseline-pair integers"

        # if fed as list of tuples, convert to integers
        if isinstance(blpairs[0], tuple):
            blpairs = [uvp.antnums_to_blpair(blp) for blp in blpairs]
        blpair_select = np.logical_or.reduce(
                                   [uvp.blpair_array == blp for blp in blpairs])
        blp_select += blpair_select

    if times is not None:
        if bls is None and blpairs is None:
            blp_select = np.ones(uvp.Nblpairts, bool)
        time_select = np.logical_or.reduce(
                               [np.isclose(uvp.time_avg_array, t, rtol=1e-16)
                                for t in times])
        blp_select *= time_select

    if lsts is not None:
        assert times is None, "Cannot select on lsts and times simultaneously."
        if bls is None and blpairs is None:
            blp_select = np.ones(uvp.Nblpairts, bool)
        lst_select = np.logical_or.reduce(
                            [ np.isclose(uvp.lst_avg_array, t, rtol=1e-16)
                              for t in lsts] )
        blp_select *= lst_select

    if bls is None and blpairs is None and times is None and lsts is None:
        blp_select = slice(None)
    else:
        # assert something was selected
        assert blp_select.any(), \
            "no selections provided matched any of the data... "

        # turn blp_select into slice if possible
        blp_select = np.where(blp_select)[0]
        if len(set(np.diff(blp_select))) == 0:
            # its sliceable, turn into slice object
            blp_select = slice(blp_select[0], blp_select[-1]+1)
        elif len(set(np.diff(blp_select))) == 1:
            # its sliceable, turn into slice object
            blp_select = slice(blp_select[0], blp_select[-1]+1, np.diff(blp_select)[0])

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

        # Calculate unique baselines from new blpair_array
        new_blpairs = np.unique(uvp.blpair_array)
        bl1 = np.floor(new_blpairs / 1e6)
        new_bls = np.unique([bl1, new_blpairs - bl1*1e6]).astype(np.int32)

        # Set baseline attributes
        bl_select = [bl in new_bls for bl in uvp.bl_array]
        uvp.bl_array = uvp.bl_array[bl_select]
        uvp.bl_vecs = uvp.bl_vecs[bl_select]
        uvp.Nbls = len(uvp.bl_array)

    if polpairs is not None:
        # assert form
        assert isinstance(polpairs, (list, np.ndarray)), \
            "polpairs must be passed as a list or ndarray"
        assert isinstance(polpairs[0], (tuple, int, np.integer, str)), \
            "polpairs must be fed as a list of tuples or pol integers/strings"

        # convert str to polpair integers
        polpairs = [polpair_tuple2int((p,p)) if isinstance(p, str)
                    else p for p in polpairs]

        # convert tuples to polpair integers
        polpairs = [polpair_tuple2int(p) if isinstance(p, tuple)
                    else p for p in polpairs]

        # create selection
        polpair_select = np.logical_or.reduce( [uvp.polpair_array == p
                                                for p in polpairs] )

        # turn into slice object if possible
        polpair_select = np.where(polpair_select)[0]
        if len(set(np.diff(polpair_select))) == 0:
            # sliceable
            polpair_select = slice(polpair_select[0], polpair_select[-1] + 1)
        elif len(set(np.diff(polpair_select))) == 1:
            # sliceable
            polpair_select = slice(polpair_select[0],
                                   polpair_select[-1] + 1,
                                   np.diff(polpair_select)[0])

        # edit metadata
        uvp.polpair_array = uvp.polpair_array[polpair_select]
        uvp.Npols = len(uvp.polpair_array)
        if hasattr(uvp, 'scalar_array'):
            uvp.scalar_array = uvp.scalar_array[:, polpair_select]
    else:
        polpair_select = slice(None)

    # determine if data arrays are sliceable
    if isinstance(polpair_select, slice) or isinstance(blp_select, slice):
        sliceable = True
    else:
        sliceable = False

    # only load / select heavy data if data_array exists _or_ if h5file is passed
    if h5file is not None or hasattr(uvp, 'data_array'):
        # select data arrays
        data = odict()
        wgts = odict()
        ints = odict()
        nsmp = odict()
        cov_real = odict()
        cov_imag = odict()
        stats = odict()
        window_function = odict()
        window_function_kperp = odict()
        window_function_kpara = odict()

        # determine if certain arrays are stored
        if h5file is not None:
            store_cov = 'cov_real_spw0' in h5file
            if 'cov_array_spw0' in h5file:
                store_cov = False
                warnings.warn("uvp.cov_array is no longer supported and will not be loaded. Please update this to be uvp.cov_array_real and uvp.cov_array_imag. See hera_pspec PR #181 for details.")
            store_window = 'window_function_spw0' in h5file
            exact_windows = 'window_function_kperp_spw0' in h5file
        else:
            store_cov = hasattr(uvp, 'cov_array_real')
            store_window = hasattr(uvp, 'window_function_array')
            exact_windows = hasattr(uvp, 'window_function_kperp')

        # get stats_array keys if h5file
        if h5file is not None:
            statnames = np.unique([f[f.find("_")+1: f.rfind("_")]
                                    for f in h5file.keys()
                                    if f.startswith("stats")])
        else:
            if hasattr(uvp, "stats_array"):
                statnames = uvp.stats_array.keys()
            else:
                statnames = []

        # iterate over spws
        if spw_mapping is None: spw_mapping = uvp.spw_array
        for s, s_old in zip(uvp.spw_array, spw_mapping):
            # if h5file is passed, default to loading in data
            if h5file is not None:
                # assign data arrays
                _data = h5file['data_spw{}'.format(s_old)]
                _wgts = h5file['wgt_spw{}'.format(s_old)]
                _ints = h5file['integration_spw{}'.format(s_old)]
                _nsmp = h5file['nsample_spw{}'.format(s_old)]
                # assign non-required arrays
                if store_window:
                    _window_function = h5file['window_function_spw{}'.format(s_old)]
                    if exact_windows:
                        _window_function_kperp = h5file['window_function_kperp_spw{}'.format(s_old)]
                        _window_function_kpara = h5file['window_function_kpara_spw{}'.format(s_old)]
                if store_cov:
                     _cov_real = h5file["cov_real_spw{}".format(s_old)]
                     _cov_imag = h5file["cov_imag_spw{}".format(s_old)]
                _stat = odict()
                for statname in statnames:
                    if statname not in stats:
                        stats[statname] = odict()
                    _stat[statname] = h5file["stats_{}_{}".format(statname, s_old)]

            # if no h5file, we are performing a select, so use uvp's arrays
            else:
                # assign data arrays
                _data = uvp.data_array[s_old]
                _wgts = uvp.wgt_array[s_old]
                _ints = uvp.integration_array[s_old]
                _nsmp = uvp.nsample_array[s_old]
                # assign non-required arrays
                if store_window:
                    _window_function = uvp.window_function_array[s_old]
                    if exact_windows:
                        _window_function_kperp = uvp.window_function_kperp[s_old]
                        _window_function_kpara = uvp.window_function_kpara[s_old]
                if store_cov:
                    _cov_real = uvp.cov_array_real[s_old]
                    _cov_imag = uvp.cov_array_imag[s_old]
                # assign stats array
                _stat = odict()
                for statname in statnames:
                    if statname not in stats:
                        stats[statname] = odict()
                    _stat[statname] = uvp.stats_array[statname][s_old]

            # slice data arrays and assign to dictionaries
            if sliceable:
                # can slice in 1 step
                data[s] = _data[blp_select, :, polpair_select]
                wgts[s] = _wgts[blp_select, :, :, polpair_select]
                ints[s] = _ints[blp_select, polpair_select]
                nsmp[s] = _nsmp[blp_select, polpair_select]
                if store_window:
                    window_function[s] = _window_function[blp_select, ..., polpair_select]
                    if exact_windows:
                        window_function_kperp[s] = _window_function_kperp[:, polpair_select]
                        window_function_kpara[s] = _window_function_kpara[:, polpair_select]
                if store_cov:
                    cov_real[s] = _cov_real[blp_select, :, :, polpair_select]
                    cov_imag[s] = _cov_imag[blp_select, :, :, polpair_select]
                for statname in statnames:
                    stats[statname][s] = _stat[statname][blp_select, :, polpair_select]
            else:
                # need to slice in 2 steps
                data[s] = _data[blp_select, :, :][:, :, polpair_select]
                wgts[s] = _wgts[blp_select, :, :, :][:, :, :, polpair_select]
                ints[s] = _ints[blp_select, :][:, polpair_select]
                nsmp[s] = _nsmp[blp_select, :][:, polpair_select]
                if store_window:
                    if exact_windows:
                        window_function[s] = _window_function[blp_select, :, :, :, :][:, :, :, :, polpair_select]
                        window_function_kperp[s] = _window_function_kperp[:, polpair_select]
                        window_function_kpara[s] = _window_function_kpara[:, polpair_select]
                    else:
                        window_function[s] = _window_function[blp_select, :, :, :][:, :, :, polpair_select]
                if store_cov:
                    cov_real[s] = _cov_real[blp_select, :, :, :][:, :, :, polpair_select]
                    cov_imag[s] = _cov_imag[blp_select, :, :, :][:, :, :, polpair_select]
                for statname in statnames:
                    stats[statname][s] = _stat[statname][blp_select, :, :][:, :, polpair_select]

        # assign arrays to uvp
        uvp.data_array = data
        uvp.wgt_array = wgts
        uvp.integration_array = ints
        uvp.nsample_array = nsmp

        if store_window:
            uvp.window_function_array = window_function
            if exact_windows:
                uvp.window_function_kperp = window_function_kperp
                uvp.window_function_kpara = window_function_kpara
        if len(stats) > 0:
            uvp.stats_array = stats
        if len(cov_real) > 0:
            uvp.cov_array_real = cov_real
            uvp.cov_array_imag = cov_imag

        # downselect on other attrs
        if hasattr(uvp, 'OmegaP'):
            uvp.OmegaP = uvp.OmegaP[:, polpair_select]
            uvp.OmegaPP = uvp.OmegaPP[:, polpair_select]

        # select r_params based on new bl_array
        blp_keys = uvp.get_all_keys()
        blkeys = []
        for blpkey in blp_keys:
            key1 = blpkey[1][0] + (blpkey[2][0],)
            key2 = blpkey[1][1] + (blpkey[2][1],)
            if not key1 in blkeys:
                blkeys += [key1,]
            if not key2 in blkeys:
                blkeys += [key2,]
        new_r_params = {}
        if hasattr(uvp, 'r_params') and uvp.r_params != '':
            r_params = uvp.get_r_params()
            for rpkey in r_params:
                if rpkey in blkeys:
                    new_r_params[rpkey] = r_params[rpkey]
        else:
            new_r_params = {}
        uvp.r_params = compress_r_params(new_r_params)

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
        nested tuple containing baseline-pair antenna numbers.
        Ex. ((ant1, ant2), (ant3, ant4))
    """
    # get antennas
    ant1 = int(np.floor(blpair / 1e9))
    ant2 = int(np.floor(blpair / 1e6 - ant1*1e3))
    ant3 = int(np.floor(blpair / 1e3 - ant1*1e6 - ant2*1e3))
    ant4 = int(np.floor(blpair - ant1*1e9 - ant2*1e6 - ant3*1e3))
    ant1 -= 100
    ant2 -= 100
    ant3 -= 100
    ant4 -= 100

    # form antnums tuple
    antnums = ((ant1, ant2), (ant3, ant4))

    return antnums

def _antnums_to_blpair(antnums):
    """
    Convert nested tuple of antenna numbers to baseline-pair integer.
    A baseline-pair integer is an i12 integer that is the antenna numbers
    + 100 directly concatenated (i.e. string contatenation).
    Ex: ((1, 2), (3, 4)) --> 101 + 102 + 103 + 104 --> 101102103014.

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
    ant1 = antnums[0][0] + 100
    ant2 = antnums[0][1] + 100
    ant3 = antnums[1][0] + 100
    ant4 = antnums[1][1] + 100

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
    ant1 -= 100
    ant2 -= 100

    # form antnums tuple
    antnums = (ant1, ant2)

    return antnums


def _antnums_to_bl(antnums):
    """
    Convert tuple of antenna numbers to baseline integer.
    A baseline integer is the two antenna numbers + 100
    directly (i.e. string) concatenated. Ex: (1, 2) -->
    101 + 102 --> 101102.

    Parameters
    ----------
    antnums : tuple
        tuple containing integer antenna numbers for a baseline.
        Ex. (ant1, ant2)

    Returns
    -------
    bl : <i6 integer
        baseline integer
    """
    # get antennas
    ant1 = antnums[0] + 100
    ant2 = antnums[1] + 100

    # form bl
    bl = int(ant1*1e3 + ant2)

    return bl


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


def _fast_is_in(src_blpts, query_blpts, time_prec=8):
    """
    Helper function to rapidly check if a given blpair-time couplet is in an
    array.

    Parameters
    ----------
    src_blpts : list of tuples or array_like
        List of tuples or array of shape (N, 2), containing a list of (blpair,
        time) couplets.

    query_blpts : list of tuples or array_like
        List of tuples or array of shape (M, 2), containing a list of (blpair,
        time) which will be looked up in src_blpts

    time_prec : int, optional
        Number of decimals to round time array to when performing float
        comparision. Default: 8.

    Returns
    -------
    is_in_arr: list of bools
        A list of booleans, which indicate which query_blpts are in src_blpts.
    """
    # This function converts baseline-pair-times, a tuple (blp, time)
    # to a complex number blp + 1j * time, so that "in" function is much
    # faster.
    src_blpts = np.asarray(src_blpts)
    query_blpts = np.asarray(query_blpts)

    # Slice to create complex array
    src_blpts = src_blpts[:,0] + 1.j*np.around(src_blpts[:,1], time_prec)
    query_blpts = query_blpts[:,0] + 1.j*np.around(query_blpts[:,1], time_prec)

    # see if q complex number is in src_blpts
    return [q in src_blpts for q in query_blpts]


def _fast_lookup_blpairts(src_blpts, query_blpts, time_prec=8):
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

    time_prec : int, optional
        Number of decimals to round time array to when performing float
        comparision. Default: 8.

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

    src_blpts = src_blpts[:,0] + 1.j*np.around(src_blpts[:,1], time_prec)
    query_blpts = query_blpts[:,0] + 1.j*np.around(query_blpts[:,1], time_prec)
    # Applies rounding to time values to ensure reliable float comparisons

    # Do np.where comparison for all new_blpts
    # (indices stored in second array returned by np.where)
    blpts_idxs = np.where(src_blpts == query_blpts[:,np.newaxis])[1]

    return blpts_idxs


def _get_red_bls(uvp, bl_len_tol=1., bl_ang_tol=1.):
    """
    Get redundant baseline groups that are present in a UVPSpec object.

    Parameters
    ----------
    uvp : UVPSpec
        UVPSpec object.

    bl_len_tol : float, optional
        Maximum difference in length to use for grouping baselines.
        This does not guarantee that the maximum length difference
        between any two baselines in a group is less than bl_len_tol
        however. Default: 1.0.

    bl_ang_tol : float, optional
        Maximum separation in angle to use for grouping baselines.
        This does not guarantee that the maximum angle between any
        two baselines in a group is less than bl_ang_tol however.
        Default: 1.0.

    Returns
    -------
    grp_bls : list of array_like
        List of redundant baseline groups. Each list item contains
        an array of baseline integers corresponding to the members
        of the group.

    grp_lens : list of float
        Average length of the baselines in each group.

    grp_angs : list of float
        Average angle of the baselines in each group.
    """
    # Calculate length and angle of baseline vecs
    bl_vecs = uvp.get_ENU_bl_vecs()

    lens, angs = utils.get_bl_lens_angs(bl_vecs, bl_error_tol=bl_len_tol)

    # Baseline indices
    idxs = np.arange(len(lens)).astype(int)
    grp_bls = []; grp_len = []; grp_ang = []

    # Group baselines by length and angle
    max_loops = idxs.size
    nloops = 0
    while len(idxs) > 0 and nloops < max_loops:
        nloops += 1

        # Match bls within some tolerance in length and angle
        matches = np.where(np.logical_and(
                            np.abs(lens - lens[0]) < bl_len_tol,
                            np.abs(angs - angs[0]) < bl_ang_tol) )

        # Save info about this group
        grp_bls.append(uvp.bl_array[idxs[matches]])
        grp_len.append(np.mean(lens[matches]))
        grp_ang.append(np.mean(angs[matches]))

        # Remove bls that were matched so we don't try to group them again
        idxs = np.delete(idxs, matches)
        lens = np.delete(lens, matches)
        angs = np.delete(angs, matches)

    return grp_bls, grp_len, grp_ang


def _get_red_blpairs(uvp, bl_len_tol=1., bl_ang_tol=1.):
    """
    Group baseline-pairs from a UVPSpec object according to the
    redundant groups that their constituent baselines belong to.

    NOTE: Baseline-pairs made up of baselines from two different
    redundant groups are ignored.

    Parameters
    ----------
    uvp : UVPSpec
        UVPSpec object.

    bl_len_tol : float, optional
        Maximum difference in length to use for grouping baselines.
        This does not guarantee that the maximum length difference
        between any two baselines in a group is less than bl_len_tol
        however. Default: 1.0.

    bl_ang_tol : float, optional
        Maximum separation in angle to use for grouping baselines.
        This does not guarantee that the maximum angle between any
        two baselines in a group is less than bl_ang_tol however.
        Default: 1.0.

    Returns
    -------
    grp_bls : list of array_like
        List of redundant baseline groups. Each list item contains
        an array of baseline-pair integers corresponding to the
        members of the group.

    grp_lens : list of float
        Average length of the baselines in each group.

    grp_angs : list of float
        Average angle of the baselines in each group.
    """
    # Get redundant baseline groups
    red_bls, red_lens, red_angs = _get_red_bls(uvp=uvp,
                                               bl_len_tol=bl_len_tol,
                                               bl_ang_tol=bl_ang_tol)

    # Get all available blpairs and convert to pairs of integers
    blps = [(uvp.antnums_to_bl(blp[0]), uvp.antnums_to_bl(blp[1]))
            for blp in uvp.get_blpairs()]
    bl1, bl2 = zip(*blps)

    # Build bl -> group index dict
    group_idx = {}
    for i, grp in enumerate(red_bls):
        for bl in grp:
            group_idx[bl] = i

    # Get red. group that each bl belongs to
    bl1_grp = np.array([group_idx[bl] for bl in bl1])
    bl2_grp = np.array([group_idx[bl] for bl in bl2])

    # Convert to arrays for easier indexing
    bl1 = np.array(bl1)
    bl2 = np.array(bl2)

    # Loop over redundant groups; assign blpairs to each group
    red_grps = []
    grp_ids = np.arange(len(red_bls))
    for i in grp_ids:
        # This line only keeps blpairs where both bls belong to the same red grp!
        matches = np.where(np.logical_and(bl1_grp == i, bl2_grp == i))

        # Unpack into list of blpair integers
        blpair_ints = [int("%d%d" % _blp)
                       for _blp in zip(bl1[matches], bl2[matches])]
        red_grps.append(blpair_ints)

    return red_grps, red_lens, red_angs
