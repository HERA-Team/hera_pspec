
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
    src_blpts : array_like
        Array of size (N, 2), containing a list of (blpair, time) couplets.
    
    query_blpts : array_like
        Array of size (M, 2), containing a list of (blpair, time) couplets that 
        you want to find the indices of in source_blpts.
    
    Returns
    -------
    blpts_idxs : array_like
        Array of integers of size (M,), which are indices in the source_blpts 
        array for each item in query_blpts.
    """
    # This function works by using a small hack -- the blpair-times are turned 
    # into complex numbers of the form (blpair + 1.j*time), allowing numpy 
    # array lookup functions to be used
    src_blpts = np.array(src_blpts)
    query_blpts = np.array(query_blpts)
    src_blpts = src_blpts[:,0] + 1.j*src_blpts[:,1]
    query_blpts = query_blpts[:,0] + 1.j*query_blpts[:,1]
    
    # Do np.where comparison for all new_blpts
    # (indices stored in second array returned by np.where)
    blpts_idxs = np.where(src_blpts == query_blpts[:,np.newaxis])[1]
    return blpts_idxs
    

def combine_uvpspec(uvps, verbose=True):
    """
    Combine (concatenate) multiple UVPSpec objects into a single object, 
    combining along one of either spectral window [spw], baseline-pair-times 
    [blpairts], or polarization [pol]. Certain meta-data of all of the UVPSpec 
    objs must match exactly, see get_uvp_overlap for details.
    
    In addition, one can only combine data along a single data axis, with the
    condition that all other axes match exactly.

    Parameters
    ----------
    uvps : list 
        A list of UVPSpec objects to combine.

    Returns
    -------
    u : UVPSpec object
        A UVPSpec object with the data of all the inputs combined.
    """
    # Perform type checks and get concatenation axis
    (uvps, concat_ax, new_spws, new_blpts, new_pols,
     static_meta) = get_uvp_overlap(uvps, just_meta=False, verbose=verbose)
    Nuvps = len(uvps)

    # Create a new uvp
    u = UVPSpec()

    # Sort new data axes
    new_spws = sorted(new_spws)
    new_blpts = sorted(new_blpts)
    new_pols = sorted(new_pols)
    Nspws = len(new_spws)
    Nblpairts = len(new_blpts)
    Npols = len(new_pols)

    # Create new empty data arrays and fill spw arrays
    u.data_array = odict()
    u.integration_array = odict()
    u.wgt_array = odict()
    u.nsample_array = odict()
    u.scalar_array = np.empty((Nspws, Npols), np.float)
    u.freq_array, u.spw_array, u.dly_array = [], [], []
    
    # Loop over spectral windows
    for i, spw in enumerate(new_spws):
        
        # Initialize new arrays
        u.data_array[i] = np.empty((Nblpairts, spw[2], Npols), np.complex128)
        u.integration_array[i] = np.empty((Nblpairts, Npols), np.float64)
        u.wgt_array[i] = np.empty((Nblpairts, spw[2], 2, Npols), np.float64)
        u.nsample_array[i] = np.empty((Nblpairts, Npols), np.float64)
        
        # Set frequencies and delays
        spw_Nfreqs = spw[-1]
        spw_freqs = np.linspace(*spw, endpoint=False)
        spw_dlys = np.fft.fftshift(
                        np.fft.fftfreq(spw_Nfreqs, 
                                       np.median(np.diff(spw_freqs)) ) )
        u.spw_array.extend(np.ones(spw_Nfreqs, np.int32) * i)
        u.freq_array.extend(spw_freqs)
        u.dly_array.extend(spw_dlys)
    
    # Convert to numpy arrays    
    u.spw_array = np.array(u.spw_array)
    u.freq_array = np.array(u.freq_array)
    u.dly_array = np.array(u.dly_array)
    u.pol_array = np.array(new_pols)
    
    # Number of spectral windows, delays etc.
    u.Nspws = Nspws
    u.Nblpairts = Nblpairts
    u.Npols = Npols
    u.Nfreqs = len(np.unique(u.freq_array))
    u.Nspwdlys = len(u.spw_array)
    u.Ndlys = len(np.unique(u.dly_array))
    
    # Prepare time and label arrays
    u.time_1_array, u.time_2_array = np.empty(Nblpairts, np.float), \
                                     np.empty(Nblpairts, np.float)
    u.time_avg_array, u.lst_avg_array = np.empty(Nblpairts, np.float), \
                                        np.empty(Nblpairts, np.float)
    u.lst_1_array, u.lst_2_array = np.empty(Nblpairts, np.float), \
                                   np.empty(Nblpairts, np.float)
    u.blpair_array = np.empty(Nblpairts, np.int64)
    u.labels = sorted(set(np.concatenate([uvp.labels for uvp in uvps])))
    u.label_1_array = np.empty((Nspws, Nblpairts, Npols), np.int32)
    u.label_2_array = np.empty((Nspws, Nblpairts, Npols), np.int32)

    # get each uvp's data axes
    uvp_spws = [_uvp.get_spw_ranges() for _uvp in uvps]
    uvp_blpts = [zip(_uvp.blpair_array, _uvp.time_avg_array) for _uvp in uvps]
    uvp_pols = [_uvp.pol_array.tolist() for _uvp in uvps]
    
    # Construct dict of label indices, to be used for re-ordering later
    u_lbls = {lbl: ll for ll, lbl in enumerate(u.labels)}
    
    # fill in data arrays depending on concat ax
    if concat_ax == 'spw':
        
        # Concatenate spectral windows
        for i, spw in enumerate(new_spws):
            l = [spw in _u for _u in uvp_spws].index(True)
            m = [spw == _spw for _spw in uvp_spws[l]].index(True)
            
            # Lookup indices of new_blpts in the uvp_blpts[l] array
            blpts_idxs = _fast_lookup_blpairts(uvp_blpts[l], new_blpts)
            if i == 0: blpts_idxs0 = blpts_idxs.copy()
            
            # Loop over polarizations
            for k, p in enumerate(new_pols):
                q = uvp_pols[l].index(p)
                u.scalar_array[i,k] = uvps[l].scalar_array[m,q]
                
                # Loop over blpair-times
                for j, blpt in enumerate(new_blpts):
                    n = blpts_idxs[j]
                    
                    # Data/weight/integration arrays
                    u.data_array[i][j,:,k] = uvps[l].data_array[m][n,:,q]
                    u.wgt_array[i][j,:,:,k] = uvps[l].wgt_array[m][n,:,:,q]
                    u.integration_array[i][j,k] = uvps[l].integration_array[m][n, q]
                    u.nsample_array[i][j,k] = uvps[l].integration_array[m][n,q]
                    
                    # Labels
                    lbl1 = uvps[l].label_1_array[m,n,q]
                    lbl2 = uvps[l].label_2_array[m,n,q]
                    u.label_1_array[i,j,k] = u_lbls[uvps[l].labels[lbl1]]
                    u.label_2_array[i,j,k] = u_lbls[uvps[l].labels[lbl2]]
        
        # Populate new LST, time, and blpair arrays
        for j, blpt in enumerate(new_blpts):
            n = blpts_idxs0[j]
            u.time_1_array[j] = uvps[0].time_1_array[n]
            u.time_2_array[j] = uvps[0].time_2_array[n]
            u.time_avg_array[j] = uvps[0].time_avg_array[n]
            u.lst_1_array[j] = uvps[0].lst_1_array[n]
            u.lst_2_array[j] = uvps[0].lst_2_array[n]
            u.lst_avg_array[j] = uvps[0].lst_avg_array[n]
            u.blpair_array[j] = uvps[0].blpair_array[n]

    elif concat_ax == 'blpairts':
        
        # Get mapping of blpair-time indices between old UVPSpec objects and 
        # the new one
        blpts_idxs = np.concatenate( [_fast_lookup_blpairts(_blpts, new_blpts)
                                      for _blpts in uvp_blpts] )
        
        # Concatenate blpair-times
        for j, blpt in enumerate(new_blpts):
            
            l = [blpt in _blpts for _blpts in uvp_blpts].index(True)
            n = blpts_idxs[j]
            
            # Loop over spectral windows
            for i, spw in enumerate(new_spws):
                m = [spw == _spw for _spw in uvp_spws[l]].index(True)
                
                # Loop over polarizations
                for k, p in enumerate(new_pols):
                    q = uvp_pols[l].index(p)
                    u.data_array[i][j,:,k] = uvps[l].data_array[m][n,:,q]
                    u.wgt_array[i][j,:,:,k] = uvps[l].wgt_array[m][n,:,:,q]
                    u.integration_array[i][j,k] = uvps[l].integration_array[m][n,q]
                    u.nsample_array[i][j,k] = uvps[l].integration_array[m][n,q]
                    
                    # Labels
                    lbl1 = uvps[l].label_1_array[m,n,q]
                    lbl2 = uvps[l].label_2_array[m,n,q]
                    u.label_1_array[i,j,k] = u_lbls[uvps[l].labels[lbl1]]
                    u.label_2_array[i,j,k] = u_lbls[uvps[l].labels[lbl2]]
            
            # Populate new LST, time, and blpair arrays
            u.time_1_array[j] = uvps[l].time_1_array[n]
            u.time_2_array[j] = uvps[l].time_2_array[n]
            u.time_avg_array[j] = uvps[l].time_avg_array[n]
            u.lst_1_array[j] = uvps[l].lst_1_array[n]
            u.lst_2_array[j] = uvps[l].lst_2_array[n]
            u.lst_avg_array[j] = uvps[l].lst_avg_array[n]
            u.blpair_array[j] = uvps[l].blpair_array[n]

    elif concat_ax == 'pol':
        
        # Concatenate polarizations
        for k, p in enumerate(new_pols):
            l = [p in _pols for _pols in uvp_pols].index(True)
            q = uvp_pols[l].index(p)
            
            # Get mapping of blpair-time indices between old UVPSpec objects 
            # and the new one
            blpts_idxs = _fast_lookup_blpairts(uvp_blpts[l], new_blpts)
            if k == 0: blpts_idxs0 = blpts_idxs.copy() 
            
            # Loop over spectral windows
            for i, spw in enumerate(new_spws):
                m = [spw == _spw for _spw in uvp_spws[l]].index(True)
                u.scalar_array[i,k] = uvps[l].scalar_array[m,q]
                
                # Loop over blpair-times
                for j, blpt in enumerate(new_blpts):
                    n = blpts_idxs[j]
                    u.data_array[i][j,:,k] = uvps[l].data_array[m][n,:,q]
                    u.wgt_array[i][j,:,:,k] = uvps[l].wgt_array[m][n,:,:,q]
                    u.integration_array[i][j,k] = uvps[l].integration_array[m][n,q]
                    u.nsample_array[i][j,k] = uvps[l].integration_array[m][n,q]
                    
                    # Labels
                    lbl1 = uvps[l].label_1_array[m,n,q]
                    lbl2 = uvps[l].label_2_array[m,n,q]
                    u.label_1_array[i,j,k] = u_lbls[uvps[l].labels[lbl1]]
                    u.label_2_array[i,j,k] = u_lbls[uvps[l].labels[lbl2]]

        for j, blpt in enumerate(new_blpts):
            n = blpts_idxs0[j]
            u.time_1_array[j] = uvps[0].time_1_array[n]
            u.time_2_array[j] = uvps[0].time_2_array[n]
            u.time_avg_array[j] = uvps[0].time_avg_array[n]
            u.lst_1_array[j] = uvps[0].lst_1_array[n]
            u.lst_2_array[j] = uvps[0].lst_2_array[n]
            u.lst_avg_array[j] = uvps[0].lst_avg_array[n]
            u.blpair_array[j] = uvps[0].blpair_array[n]
    
    # Set baselines
    u.Nblpairs = len(np.unique(u.blpair_array))
    uvp_bls = [uvp.bl_array for uvp in uvps]
    new_bls = sorted(reduce(operator.or_, [set(bls) for bls in uvp_bls]))
    u.bl_array = np.array(new_bls)
    u.Nbls = len(u.bl_array)
    u.bl_vecs = []
    for b, bl in enumerate(new_bls):
        l = [bl in _bls for _bls in uvp_bls].index(True)
        h = [bl == _bl for _bl in uvp_bls[l]].index(True)
        u.bl_vecs.append(uvps[l].bl_vecs[h])
    u.bl_vecs = np.array(u.bl_vecs)
    u.Ntimes = len(np.unique(u.time_avg_array))
    u.history = reduce(operator.add, [uvp.history for uvp in uvps])
    u.labels = np.array(u.labels, np.str)
    
    for k in static_meta.keys():
        setattr(u, k, static_meta[k])
    
    # Run check to make sure the new UVPSpec object is valid
    u.check()
    return u


def get_uvp_overlap(uvps, just_meta=True, verbose=True):
    """
    Given a list of UVPSpec objects or a list of paths to UVPSpec objects,
    find a single data axis within ['spw', 'blpairts', 'pol'] where *all* 
    uvpspec objects contain non-overlapping data. Overlapping data are 
    delay spectra that have identical spw, blpair-time and pol metadata 
    between each other. If two uvps are completely overlapping (i.e. there
    is not non-overlapping data axis) an error is raised. If there are
    multiple non-overlapping data axes between all uvpspec pairs in uvps,
    an error is raised.

    ALl uvpspec objects must have certain attributes that agree exactly. These include
    'channel_width', 'telescope_location', 'weighting', 'OmegaP', 'beam_freqs', 'OmegaPP', 
    'beamfile', 'norm', 'taper', 'vis_units', 'norm_units', 'folded', 'cosmo', 'scalar'

    Parameters
    ----------
    uvps : list
        List of UVPSpec objects or list of string paths to UVPSpec objects

    just_meta : boolean, optional
        If uvps is a list of strings, when loading-in each uvpspec, only
        load its metadata.

    verbose : bool, optional
        print feedback to standard output

    Returns
    -------
    uvps : list
        list of input UVPSpec objects 

    concat_ax : str
        data axis ['spw', 'blpairts', 'pols'] across data can be concatenated

    unique_spws : list
        list of unique spectral window tuples (spw_freq_start, spw_freq_end, spw_Nfreqs)
        across all input uvps

    unique_blpts : list
        list of unique baseline-pair-time tuples (blpair_integer, time_avg_array JD float)
        across all input uvps

    unique_pols : list
        list of unique polarization integers across all input uvps
    """
    # type check
    assert isinstance(uvps, (list, tuple, np.ndarray)), "uvps must be fed as a list"
    assert isinstance(uvps[0], (UVPSpec, str, np.str)), "uvps must be fed as a list of UVPSpec objects or strings"
    Nuvps = len(uvps)

    # load uvps if fed as strings
    if isinstance(uvps[0], (str, np.str)):
        _uvps = []
        for u in uvps:
            uvp = UVPSpec()
            uvp.read_hdf5(u, just_meta=just_meta)
            _uvps.append(uvp)
        uvps = _uvps

    # ensure static metadata agree between all objects
    static_meta = ['channel_width', 'telescope_location', 'weighting', 'OmegaP', 'beam_freqs', 'OmegaPP', 
                   'beamfile', 'norm', 'taper', 'vis_units', 'norm_units', 'folded', 'cosmo']
    for m in static_meta:
        for u in uvps[1:]:
            if hasattr(uvps[0], m) and hasattr(u, m):
                assert uvps[0].__eq__(u, params=[m]), "Cannot concatenate UVPSpec objs: not all agree on '{}' attribute".format(m)
            else:
                assert not hasattr(uvps[0], m) and not hasattr(u, m), "Cannot concatenate UVPSpec objs: not all agree on '{}' attribute".format(m)

    static_meta = odict([(k, getattr(uvps[0], k, None)) for k in static_meta if getattr(uvps[0], k, None) is not None])
    
    # create unique data axis lists
    unique_spws = []
    unique_blpts = []
    unique_pols = []
    data_concat_axes = odict()

    blpts_comb = [] # Combined blpair + time
    # find unique items
    for uvp1 in uvps:
        for s in uvp1.get_spw_ranges():
            if s not in unique_spws: unique_spws.append(s)
        for p in uvp1.pol_array: 
            if p not in unique_pols: unique_pols.append(p)

        uvp1_blpts = zip(uvp1.blpair_array, uvp1.time_avg_array)
        uvp1_blpts_comb = [bl[0] + 1.j*bl[1] for bl in uvp1_blpts]
        blpts_comb.extend(uvp1_blpts_comb)

    unique_blpts_comb = np.unique(blpts_comb)
    unique_blpts = [(int(blt.real), blt.imag) for blt in unique_blpts_comb]

    # iterate over uvps
    for i, uvp1 in enumerate(uvps):
        # get uvp1 sets and append to unique lists
        uvp1_spws = uvp1.get_spw_ranges()
        uvp1_blpts = zip(uvp1.blpair_array, uvp1.time_avg_array)
        uvp1_pols = uvp1.pol_array

        # iterate over uvps
        for j, uvp2 in enumerate(uvps):
            if j <= i: continue
            # get uvp2 sets
            uvp2_spws = uvp2.get_spw_ranges()
            uvp2_blpts = zip(uvp2.blpair_array, uvp2.time_avg_array)
            uvp2_pols = uvp2.pol_array

            # determine if uvp1 and uvp2 are an identical match
            spw_match = len(set(uvp1_spws) ^ set(uvp2_spws)) == 0
            blpts_match = len(set(uvp1_blpts) ^ set(uvp2_blpts)) == 0
            pol_match = len(set(uvp1_pols) ^ set(uvp2_pols)) == 0

            # ensure no partial-overlaps
            if not spw_match:
                assert len(set(uvp1_spws) & set(uvp2_spws)) == 0, "uvp {} and {} have partial overlap across spw, cannot combine".format(i, j)
            if not blpts_match:
                assert len(set(uvp1_blpts) & set(uvp2_blpts)) == 0, "uvp {} and {} have partial overlap across blpairts, cannot combine".format(i, j)
            if not pol_match:
                assert len(set(uvp1_pols) & set(uvp2_pols)) == 0, "uvp {} and {} have partial overlap across pol, cannot combine".format(i, j)

            # assert all except 1 axis overlaps
            matches = [spw_match, blpts_match, pol_match]
            assert sum(matches) != 3, "uvp {} and {} have completely overlapping data, cannot combine".format(i, j)
            assert sum(matches) > 1, "uvp {} and {} are non-overlapping across multiple data axes, cannot combine".format(i, j)
            concat_ax = ['spw', 'blpairts', 'pol'][matches.index(False)]
            data_concat_axes[(i, j)] = concat_ax
            if verbose:
                print "uvp {} and {} are concatable across {} axis".format(i, j, concat_ax)

    # assert all uvp pairs have the same (single) non-overlap (concat) axis
    err_msg = "Non-overlapping data in uvps span multiple data axes:\n{}" \
              "".format("\n".join(map(lambda i: "{} & {}: {}".format(i[0][0], i[0][1], i[1]), data_concat_axes.items())))
    assert len(set(data_concat_axes.values())) == 1, err_msg

    # perform additional checks given concat ax
    if concat_ax == 'blpairts':
        # check scalar_array
        assert np.all([np.isclose(uvps[0].scalar_array, u.scalar_array) for u in uvps[1:]]), "" \
               "scalar_array must be the same for all uvps given concatenation along blpairts."

    return uvps, concat_ax, unique_spws, unique_blpts, unique_pols, static_meta

