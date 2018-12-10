import numpy as np
import copy, operator
from collections import OrderedDict as odict
from pyuvdata.utils import polstr2num


def subtract_uvp(uvp1, uvp2, run_check=True, verbose=False):
    """
    Subtract uvp2.data_array from uvp1.data_array. Subtract matching
    spw, blpair-lst, pol keys. For non-overlapping keys, remove from output
    uvp.

    Note: Entries in stats_array are added in quadrature. nsample_arrays,
    integration_arrays and wgt_arrays are inversely added in quadrature.
    If these arrays are identical in the two objects, this is equivalent
    to multiplying (dividing) the stats (nsamp, int & wgt) arrays(s)
    by sqrt(2).

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
                               pols=True, times=False, inplace=False, verbose=verbose)

    # get metadata
    spws1 = [spw for spw in uvp1.get_spw_ranges()]
    pols1 = uvp1.pol_array.tolist()
    blps1 = sorted(set(uvp1.blpair_array))
    spws2 = [spw for spw in uvp2.get_spw_ranges()]

    # iterate over spws
    for i, spw in enumerate(spws1):
        # get uvp2 index
        i2 = spws2.index(spw)

        # iterate over pol
        for j, pol in enumerate(pols1):

            # iterate over blp
            for k, blp in enumerate(blps1):

                # form keys
                key1 = (i, blp, pol)
                key2 = (i2, blp, pol)

                # subtract data
                blp1_inds = uvp1.blpair_to_indices(blp)
                uvp1.data_array[i][blp1_inds, :, j] -= uvp2.get_data(key2)

                # add nsample inversely in quadrature
                uvp1.nsample_array[i][blp1_inds, j] = np.sqrt(1. / (1./uvp1.get_nsamples(key1)**2 + 1./uvp2.get_nsamples(key2)**2))

                # add integration inversely in quadrature
                uvp1.integration_array[i][blp1_inds, j] = np.sqrt(1. / (1./uvp1.get_integrations(key1)**2 + 1./uvp2.get_integrations(key2)**2))

                # add wgts inversely in quadrature
                uvp1.wgt_array[i][blp1_inds, :, :, j] = np.sqrt(1. / (1./uvp1.get_wgts(key1)**2 + 1./uvp2.get_wgts(key2)**2))
                uvp1.wgt_array[i][blp1_inds, :, :, j] /= uvp1.wgt_array[i][blp1_inds, :, :, j].max()

                # add stats in quadrature: real imag separately
                if hasattr(uvp1, "stats_array") and hasattr(uvp2, "stats_array"):
                    for s in uvp1.stats_array.keys():
                        stat1 = uvp1.get_stats(s, key1)
                        stat2 = uvp2.get_stats(s, key2)
                        uvp1.stats_array[s][i][blp1_inds, :, j] = np.sqrt(stat1.real**2 + stat2.real**2) + 1j*np.sqrt(stat1.imag**2 + stat2.imag**2)

                # add cov in quadrature: real and imag separately
                if hasattr(uvp1, "cov_array") and hasattr(uvp2, "cov_array"):
                    cov1 = uvp1.get_cov(key1)
                    cov2 = uvp2.get_cov(key2)
                    uvp1.cov_array[i][blp1_inds, :, :, j] = np.sqrt(cov1.real**2 + cov2.real**2) + 1j*np.sqrt(cov1.imag**2 + cov2.imag**2)

    # run check
    if run_check:
        uvp1.check()

    return uvp1


def select_common(uvp_list, spws=True, blpairs=True, times=True, pols=True,
                  lsts=False, inplace=False, verbose=False):
    """
    Find spectral windows, baseline-pairs, times, and/or polarizations that a
    set of UVPSpec objects have in common and return new UVPSpec objects that
    contain only those data.

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

    pols : bool, optional
        Whether to retain only the polarizations that all UVPSpec objects have
        in common. Default: True.

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
        if verbose: print "common_times:", common_times

    # Get lsts that are common to all UVPSpec objects in the list
    if lsts:
        common_lsts = np.unique(uvp_list[0].lst_avg_array)
        has_lsts = [np.isin(common_lsts, uvp.lst_avg_array)
                     for uvp in uvp_list]
        common_lsts = common_lsts[np.all(has_lsts, axis=0)]
        if verbose: print "common_lsts:", common_lsts

    # Get baseline-pairs that are common to all
    if blpairs:
        common_blpairs = np.unique(uvp_list[0].blpair_array)
        has_blpairs = [np.isin(common_blpairs, uvp.blpair_array)
                       for uvp in uvp_list]
        common_blpairs = common_blpairs[np.all(has_blpairs, axis=0)]
        if verbose: print "common_blpairs:", common_blpairs

    # Get polarizations that are common to all
    if pols:
        common_pols = np.unique(uvp_list[0].pol_array)
        has_pols = [np.isin(common_pols, uvp.pol_array) for uvp in uvp_list]
        common_pols = common_pols[np.all(has_pols, axis=0)]
        if verbose: print "common_pols:", common_pols

    # Get common spectral windows (the entire window must match)
    # Each row of common_spws is a list of that spw's index in each UVPSpec
    if spws:
        common_spws = uvp_list[0].get_spw_ranges()
        has_spws = [map(lambda x: x in uvp.get_spw_ranges(), common_spws) for uvp in uvp_list]
        common_spws = [common_spws[i] for i, f in enumerate(np.all(has_spws, axis=0)) if f]
        if verbose: print "common_spws:", common_spws

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

    if pols and len(common_pols) == 0:
        raise ValueError("No polarizations were found that exist in all spectra.")

    # Apply selections
    out_list = []
    for i, uvp in enumerate(uvp_list):
        _spws, _blpairs, _times, _lsts, _pols = None, None, None, None, None

        # Set indices of blpairs, times, and pols to keep
        if blpairs: _blpairs = common_blpairs
        if times: _times = common_times
        if lsts: _lsts = common_lsts
        if pols: _pols = common_pols
        if spws: _spws = [uvp.get_spw_ranges().index(j) for j in common_spws]

        _uvp = uvp.select(spws=_spws, blpairs=_blpairs, times=_times,
                          pols=_pols, lsts=_lsts, inplace=inplace)
        if not inplace: out_list.append(_uvp)

    # Return if not inplace
    if not inplace: return out_list


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


def _select(uvp, spws=None, bls=None, only_pairs_in_bls=False, blpairs=None,
            times=None, lsts=None, pols=None, h5file=None):
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

    pols : list
        A list of polarization strings or integers to keep.
        See pyuvdata.utils.polstr2num for acceptable options.

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

    if lsts is not None:
        assert times is None, "Cannot select on lsts and times simultaneously."
        if bls is None and blpairs is None:
            blp_select = np.ones(uvp.Nblpairts, np.bool)
        lst_select = np.array(reduce(operator.add, map(lambda t: np.isclose(uvp.lst_avg_array, t, rtol=1e-16), lsts)))
        blp_select *= lst_select

    if bls is None and blpairs is None and times is None and lsts is None:
        blp_select = slice(None)
    else:
        # assert something was selected
        assert blp_select.any(), "no selections provided matched any of the data... "

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

    if pols is not None:
        # assert form
        assert isinstance(pols[0], (str, np.str, int, np.integer)), "pols must be fed as a list of pol strings or pol integers"

        # if fed as strings convert to integers
        if isinstance(pols[0], (np.str, str)):
            pols = map(lambda p: polstr2num(p), pols)

        # create selection
        pol_select = np.array(reduce(operator.add, map(lambda p: uvp.pol_array == p, pols)))

        # turn into slice object if possible
        pol_select = np.where(pol_select)[0]
        if len(set(np.diff(pol_select))) == 0:
            # sliceable
            pol_select = slice(pol_select[0], pol_select[-1] + 1)
        elif len(set(np.diff(pol_select))) == 1:
            # sliceable
            pol_select = slice(pol_select[0], pol_select[-1] + 1, np.diff(pol_select)[0])

        # edit metadata
        uvp.pol_array = uvp.pol_array[pol_select]
        uvp.Npols = len(uvp.pol_array)
        if hasattr(uvp, 'scalar_array'):
            uvp.scalar_array = uvp.scalar_array[:, pol_select]
    else:
        pol_select = slice(None)

    # determine if data arrays are sliceable
    if isinstance(pol_select, slice) or isinstance(blp_select, slice):
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
        cov = odict()
        stats = odict()

        # determine if cov_array is stored
        if h5file is not None:
            store_cov = 'cov_spw0' in h5file
        else:
            store_cov = hasattr(uvp, 'cov_array')

        # get stats_array keys if h5file
        if h5file is not None:
            statnames = np.unique([f[f.find("_")+1: f.rfind("_")] for f in h5file.keys() 
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
                # assign cov array
                if store_cov:
                    _covs = h5file['cov_spw{}'.format(s_old)]
                # assign stats array
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
                # assign cov
                if store_cov:
                    _covs = uvp.cov_array[s_old]
                # assign stats array
                _stat = odict()
                for statname in statnames:
                    if statname not in stats:
                        stats[statname] = odict()
                    _stat[statname] = uvp.stats_array[statname][s_old]

            # slice data arrays and assign to dictionaries
            if sliceable:
                # can slice in 1 step
                data[s] = _data[blp_select, :, pol_select]
                wgts[s] = _wgts[blp_select, :, :, pol_select]
                ints[s] = _ints[blp_select, pol_select]
                nsmp[s] = _nsmp[blp_select, pol_select]
                if store_cov:
                    cov[s] = _covs[blp_select, :, :, pol_select]
                for statname in statnames:
                    stats[statname][s] = _stat[statname][blp_select, :, pol_select]
            else:
                # need to slice in 2 steps
                data[s] = _data[blp_select, :, :][:, :, pol_select]
                wgts[s] = _wgts[blp_select, :, :, :][:, :, :, pol_select]
                ints[s] = _ints[blp_select, :][:, pol_select]
                nsmp[s] = _nsmp[blp_select, :][:, pol_select]
                if store_cov:
                    cov[s] = _covs[blp_select, :, :, :][:, :, :, pol_select]
                for statname in statnames:
                    stats[statname][s] = _stat[statname][blp_select, :, :][:, :, pol_select]

        # assign arrays to uvp
        uvp.data_array = data
        uvp.wgt_array = wgts
        uvp.integration_array = ints
        uvp.nsample_array = nsmp
        if len(stats) > 0:
            uvp.stats_array = stats
        if store_cov:
            uvp.cov_array = cov


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
