import numpy as np
import md5
import yaml
from conversions import Cosmo_Conversions
import traceback
import operator
from hera_cal import redcal
import itertools
import argparse

def hash(w):
    """
    Return an MD5 hash of a set of weights.
    """
    DeprecationWarning("utils.hash is deprecated.")
    return md5.md5(w.copy(order='C')).digest()

def cov(d1, w1, d2=None, w2=None):
    """
    Computes an empirical covariance matrix from data vectors. If d1 is of size 
    (M,N), then the output is M x M. In other words, the second axis is the 
    axis that is averaged over in forming the covariance (e.g. a time axis).

    If d2 is provided and d1 != d2, then this computes the cross-variance, 
    i.e. <d1 d2^dagger>

    Parameters
    ----------
    d1 : array_like
        Data vector of size (M,N), where N is the length of the "averaging axis"
    w1 : integer
        Weights for averaging d1
    d2 : array_like
        Data vector of size (M,N), where N is the length of the "averaging axis"
    w2 : integer
        Weights for averaging d1

    Returns
    -------
    cov : array_like
        Covariance (or cross-variance) matrix of size (M,M)
    """
    if d2 is None: d2,w2 = d1,w1
    if not np.isreal(w1).all(): raise TypeError("Weight matrices must be real")
    if not np.isreal(w2).all(): raise TypeError("Weight matrices must be real")
    if np.less(w1, 0.).any() or np.less(w2, 0.).any(): 
        raise ValueError("Weight matrices must be positive")
    d1sum,d1wgt = (w1*d1).sum(axis=1), w1.sum(axis=1)
    d2sum,d2wgt = (w2*d2).sum(axis=1), w2.sum(axis=1)
    x1 = d1sum / np.where(d1wgt > 0, d1wgt, 1)
    x2 = d2sum / np.where(d2wgt > 0, d2wgt, 1)
    x1.shape = (-1,1); x2.shape = (-1,1)
    
    C = np.dot(w1*d1, (w2*d2).conj().T)
    W = np.dot(w1, w2.T)
    C /= np.where(W > 0, W, 1)
    C -= np.outer(x1, x2.conj())
    return C

def construct_blpairs(bls, exclude_auto_bls=False, exclude_permutations=False, group=False, Nblps_per_group=1):
    """
    Construct a list of baseline-pairs from a baseline-group. This function can be used to easily convert a 
    single list of baselines into the input needed by PSpecData.pspec(bls1, bls2, ...).

    Parameters
    ----------
    bls : list of baseline tuples, Ex. [(1, 2), (2, 3), (3, 4)]

    exclude_auto_bls: boolean, if True, exclude all baselines crossed with itself from the final blpairs list

    exclude_permutations : boolean, if True, exclude permutations and only form combinations of the bls list.
        For example, if bls = [1, 2, 3] (note this isn't the proper form of bls, but makes this example clearer)
        and exclude_permutations = False, then blpairs = [11, 12, 13, 21, 22, 23,, 31, 32, 33].
        If however exclude_permutations = True, then blpairs = [11, 12, 13, 22, 23, 33].
        Furthermore, if exclude_auto_bls = True then 11, 22, and 33 would additionally be excluded.   

    group : boolean, optional
        if True, group each consecutive Nblps_per_group blpairs into sub-lists

    Nblps_per_group : integer, number of baseline-pairs to put into each sub-group

    Returns (bls1, bls2, blpairs)
    -------
    bls1 : list of baseline tuples from the zeroth index of the blpair

    bls2 : list of baseline tuples from the first index of the blpair

    blpairs : list of blpair tuples
    """
    # assert form
    assert isinstance(bls, list) and isinstance(bls[0], tuple), "bls must be fed as list of baseline tuples"

    # form blpairs w/o explicitly forming auto blpairs
    # however, if there are repeated bl in bls, there will be auto bls in blpairs
    if exclude_permutations:
        blpairs = list(itertools.combinations(bls, 2))
    else:
        blpairs = list(itertools.permutations(bls, 2))

    # explicitly add in auto baseline pairs
    blpairs.extend(zip(bls, bls))

    # iterate through and eliminate all autos if desired
    if exclude_auto_bls:
        new_blpairs = []
        for blp in blpairs:
            if blp[0] != blp[1]:
                new_blpairs.append(blp)
        blpairs = new_blpairs

    # create bls1 and bls2 list
    bls1 = map(lambda blp: blp[0], blpairs)
    bls2 = map(lambda blp: blp[1], blpairs)

    # group baseline pairs if desired
    if group:
        Nblps = len(blpairs)
        Ngrps = int(np.ceil(float(Nblps) / Nblps_per_group))
        new_blps = []
        new_bls1 = []
        new_bls2 = []
        for i in range(Ngrps):
            new_blps.append(blpairs[i*Nblps_per_group:(i+1)*Nblps_per_group])
            new_bls1.append(bls1[i*Nblps_per_group:(i+1)*Nblps_per_group])
            new_bls2.append(bls2[i*Nblps_per_group:(i+1)*Nblps_per_group])

        bls1 = new_bls1
        bls2 = new_bls2
        blpairs = new_blps

    return bls1, bls2, blpairs


def calc_reds(uvd1, uvd2, bl_tol=1.0, filter_blpairs=True, xant_flag_thresh=0.95, exclude_auto_bls=True, 
              exclude_permutations=True, Nblps_per_group=None, bl_len_range=(0, 1e10)):
    """
    Use hera_cal.redcal to get matching redundant baselines groups from uvd1 and uvd2
    within the specified baseline tolerance, not including flagged ants.

    Parameters
    ----------
    uvd1 : UVData instance with visibility data

    uvd2 : UVData instance with visibility data

    bl_tol : float, optional
        Baseline-vector redundancy tolerance in meters
    
    filter_blpairs : bool, optional
        if True, calculate xants and filters-out baseline pairs based on xant lists
        and baselines in the data.

    xant_flag_thresh : float, optional
        Fraction of 2D visibility (per-waterfall) needed to be flagged to 
        consider the entire visibility flagged.

    exclude_auto_bls: boolean, optional
        If True, exclude all bls crossed with itself from the blpairs list

    exclude_permutations : boolean, optional
        if True, exclude permutations and only form combinations of the bls list.
        For example, if bls = [1, 2, 3] (note this isn't the proper form of bls, 
        but makes this example clearer) and exclude_permutations = False, 
        then blpairs = [11, 12, 13, 21, 22, 23, 31, 32, 33]. If however 
        exclude_permutations = True, then blpairs = [11, 12, 13, 22, 23, 33]. 
        Furthermore, if exclude_auto_bls = True then 11, 22, and 33 are excluded.   
        
    Nblps_per_group : integer
        Number of baseline-pairs to put into each sub-group. No grouping if None.
        Default: None

    bl_len_range : tuple of float, optional
        Tuple containing minimum baseline length and maximum baseline length [meters]
        to keep in baseline type selection

    Returns
    -------
    baselines1 : list of baseline tuples
        Contains list of baseline tuples that should be fed as first argument
        to PSpecData.pspec(), corresponding to uvd1

    baselines2 : list of baseline tuples
        Contains list of baseline tuples that should be fed as second argument
        to PSpecData.pspec(), corresponding to uvd2

    blpairs : list of baseline-pair tuples
        Contains the baseline-pair tuples. i.e. zip(baselines1, baselines2)

    xants1 : list of bad antenna integers for uvd1

    xants2 : list of bad antenna integers for uvd2
    """
    # get antenna positions
    antpos1, ants1 = uvd1.get_ENU_antpos(pick_data_ants=False)
    antpos1 = dict(zip(ants1, antpos1))
    antpos2, ants2 = uvd2.get_ENU_antpos(pick_data_ants=False)
    antpos2 = dict(zip(ants2, antpos2))
    antpos = dict(antpos1.items() + antpos2.items())

    # assert antenna positions match
    for a in set(antpos1).union(set(antpos2)):
        if a in antpos1 and a in antpos2:
            msg = "antenna positions from uvd1 and uvd2 do not agree to within " \
                  "tolerance of {} m".format(bl_tol)
            assert np.linalg.norm(antpos1[a] - antpos2[a]) < bl_tol, msg

    # get xants
    xants1, xants2 = [], []
    if filter_blpairs:
        xants1, xants2 = set(ants1), set(ants2)
        baselines = sorted(set(uvd1.baseline_array).union(set(uvd2.baseline_array)))
        for bl in baselines:
            # get antenna numbers
            antnums = uvd1.baseline_to_antnums(bl)

            # continue if autocorr
            if antnums[0] == antnums[1]: continue

            # work on xants1
            if bl in uvd1.baseline_array:
                # get flags
                f1 = uvd1.get_flags(bl)
                # remove from bad list if unflagged data exists
                if np.sum(f1) < reduce(operator.mul, f1.shape) * xant_flag_thresh:
                    if antnums[0] in xants1:
                        xants1.remove(antnums[0])
                    if antnums[1] in xants2:
                        xants1.remove(antnums[1])

            # work on xants2
            if bl in uvd2.baseline_array:
                # get flags
                f2 = uvd2.get_flags(bl)
                # remove from bad list if unflagged data exists
                if np.sum(f2) < reduce(operator.mul, f2.shape) * xant_flag_thresh:
                    if antnums[0] in xants2:
                        xants2.remove(antnums[0])
                    if antnums[1] in xants2:
                        xants2.remove(antnums[1])

        xants1 = sorted(xants1)
        xants2 = sorted(xants2)

    # get reds
    reds = redcal.get_pos_reds(antpos, bl_error_tol=bl_tol, low_hi=True)

    # construct baseline pairs
    baselines1, baselines2, blpairs = [], [], []
    for r in reds:
        (_bls1, _bls2, 
         _blps) = construct_blpairs(r, exclude_auto_bls=exclude_auto_bls, group=False,
                                    exclude_permutations=exclude_permutations)

        # filter based on xants, existance in uvd1 and uvd2 and bl_len_range
        bls1, bls2 = [], []
        for bl1, bl2 in _blps:
            bl1i = uvd1.antnums_to_baseline(*bl1)
            bl2i = uvd1.antnums_to_baseline(*bl2)
            bl_len = np.mean(map(lambda bl: np.linalg.norm(antpos[bl[0]]-antpos[bl[1]]), [bl1, bl2]))
            if bl_len < bl_len_range[0] or bl_len > bl_len_range[1]:
                continue
            if filter_blpairs:
                if (bl1i not in uvd1.baseline_array or bl1[0] in xants1 or bl1[1] in xants1) \
                   or (bl2i not in uvd2.baseline_array or bl2[0] in xants2 or bl2[1] in xants2):
                   continue
            bls1.append(bl1)
            bls2.append(bl2)

        if len(bls1) < 1:
            continue

        blps = zip(bls1, bls2)

        # group if desired
        if Nblps_per_group is not None:
            Ngrps = int(np.ceil(float(len(blps)) / Nblps_per_group))
            bls1 = [bls1[Nblps_per_group*i:Nblps_per_group*(i+1)] for i in range(Ngrps)]
            bls2 = [bls2[Nblps_per_group*i:Nblps_per_group*(i+1)] for i in range(Ngrps)]
            blps = [blps[Nblps_per_group*i:Nblps_per_group*(i+1)] for i in range(Ngrps)]

        baselines1.extend(bls1)
        baselines2.extend(bls2)
        blpairs.extend(blps)

    return baselines1, baselines2, blpairs, xants1, xants2


def get_delays(freqs, n_dlys=None):
    """
    Return an array of delays, tau, corresponding to the bins of the delay 
    power spectrum given by frequency array.
    
    Parameters
    ----------
    freqs : ndarray of frequencies in Hz

    n_dlys : number of delay bins, optional
        Default: None, which then assumes that the number of bins is
        equal to the number of frequency channels.

    Returns
    -------
    delays : array_like
        Delays, tau. Units: seconds.
    """
    Delta_nu = np.median(np.diff(freqs))
    n_freqs = freqs.size

    if n_dlys == None: # assume that n_dlys = n_freqs if not specified
        n_dlys = n_freqs

    # Calculate the delays
    delay = np.fft.fftshift(np.fft.fftfreq(n_dlys, d=Delta_nu))

    return delay


def spw_range_from_freqs(data, freq_range, bounds_error=True):
    """
    Return a tuple defining the spectral window that corresponds to the 
    frequency range specified in freq_range.
    
    (Spectral windows are specified as tuples containing the first and last 
    index of a frequency range in data.freq_array.)
    
    Parameters
    ----------
    data : UVData or UVPSpec object
        Object containing data with a frequency dimension.
        
    freq_range : tuple or list of tuples
        Tuples containing the lower and upper frequency bounds for each 
        spectral window. The range is inclusive of the lower frequency bound, 
        i.e. it includes all channels in freq_range[0] <= freq < freq_range[1]. 
        Frequencies are in Hz.
    
    bounds_error : bool, optional
        Whether to raise an error if a specified lower/upper frequency is 
        outside the frequency range available in 'data'. Default: True.

    Returns
    -------
    spw_range : tuple or list of tuples
        Indices of the channels at the lower and upper bounds of the specified 
        spectral window(s). 
        
        Note: If the requested spectral window is outside the available 
        frequency range, and bounds_error is False, '(None, None)' is returned. 
    """
    # Get frequency array from input object
    try:
        freqs = data.freq_array
        if len(freqs.shape) == 2 and freqs.shape[0] == 1:
            freqs = freqs.flatten() # Support UVData 2D freq_array
        elif len(freqs.shape) > 2:
            raise ValueError("data.freq_array has unsupported shape: %s" \
                             % str(freqs.shape))
    except:
        raise AttributeError("Object 'data' does not have a freq_array attribute.")
    
    # Check for a single tuple input
    is_tuple = False
    if isinstance(freq_range, tuple):
        is_tuple = True
        freq_range = [freq_range,]
    
    # Make sure freq_range is now a list (of tuples)
    if not isinstance(freq_range, list):
        raise TypeError("freq_range must be a tuple or list of tuples.")
    
    # Loop over tuples and find spectral window indices
    spw_range = []
    for frange in freq_range:
        fmin, fmax = frange
        if fmin > fmax: 
            raise ValueError("Upper bound of spectral window is less than "
                             "the lower bound.")
        
        # Check that this doesn't go beyond the available range of freqs
        if fmin < np.min(freqs) and bounds_error:
            raise ValueError("Lower bound of spectral window is below the "
                             "available frequency range. (Note: freqs should "
                             "be in Hz)")
        if fmax > np.max(freqs) and bounds_error:
            raise ValueError("Upper bound of spectral window is above the "
                             "available frequency range. (Note: freqs should "
                             "be in Hz)")

        # Get indices within this range
        idxs = np.where(np.logical_and(freqs >= fmin, freqs < fmax))[0]
        spw = (idxs[0], idxs[-1]) if idxs.size > 0 else (None, None)
        spw_range.append(spw)
    
    # Unpack from list if only a single tuple was specified originally
    if is_tuple: return spw_range[0]
    return spw_range


def spw_range_from_redshifts(data, z_range, bounds_error=True):
    """
    Return a tuple defining the spectral window that corresponds to the 
    redshift range specified in z_range.
    
    (Spectral windows are specified as tuples containing the first and last 
    index of a frequency range in data.freq_array.)
    
    Parameters
    ----------
    data : UVData or UVPSpec object
        Object containing data with a frequency dimension.
        
    z_range : tuple or list of tuples
        Tuples containing the lower and upper fredshift bounds for each 
        spectral window. The range is inclusive of the upper redshift bound, 
        i.e. it includes all channels in z_range[0] > z >= z_range[1].
    
    bounds_error : bool, optional
        Whether to raise an error if a specified lower/upper redshift is 
        outside the frequency range available in 'data'. Default: True.

    Returns
    -------
    spw_range : tuple or list of tuples
        Indices of the channels at the lower and upper bounds of the specified 
        spectral window(s).
        
        Note: If the requested spectral window is outside the available 
        frequency range, and bounds_error is False, '(None, None)' is returned. 
    """
    # Check for a single tuple input
    is_tuple = False
    if isinstance(z_range, tuple):
        is_tuple = True
        z_range = [z_range,]
    
    # Convert redshifts to frequencies (in Hz)
    freq_range = []
    for zrange in z_range:
        zmin, zmax = zrange
        freq_range.append( (Cosmo_Conversions.z2f(zmax), 
                            Cosmo_Conversions.z2f(zmin)) )
    
    # Use freq. function to get spectral window
    spw_range = spw_range_from_freqs(data=data, freq_range=freq_range, 
                                     bounds_error=bounds_error)
    
    # Unpack from list if only a single tuple was specified originally
    if is_tuple: return spw_range[0]
    return spw_range
    

def log(msg, f=None, lvl=0, tb=None, verbose=True):
    """
    Add a message to the log (just prints to the terminal for now).
    
    Parameters
    ----------
    msg : str
        Message string to print.

    f : file descriptor
        file descriptor to write message to.

    lvl : int, optional
        Indent level of the message. Each level adds two extra spaces. 
        Default: 0.

    tb : traceback object, optional
        Traceback object to print with traceback.format_tb()

    verbose : bool, optional
        if True, print msg. Even if False, still writes to file
        if f is provided.
    """
    # catch for traceback provided
    if tb is not None:
        msg += "\n{}\n".format(traceback.format_tb(tb)[0])

    # print
    output = "%s%s" % ("  "*lvl, msg)
    if verbose:
        print(output)

    # write
    if f is not None:
        f.write(output)


def load_config(config_file):
    """
    Load configuration details from a YAML file.
    """
    # Open and read config file
    with open(config_file, 'r') as cfile:
        try:
            cfg = yaml.load(cfile)
        except yaml.YAMLError as exc:
            raise(exc)
    return cfg


def flatten(nested_list):
    """
    Flatten a list of nested lists
    """
    return [item for sublist in nested_list for item in sublist]

