import numpy as np
import time, yaml
import itertools, glob
import traceback
from hera_cal import redcal
from collections import OrderedDict as odict
from pyuvdata import UVData, utils as uvutils
from datetime import datetime
import copy
from scipy.interpolate import interp1d
import uvtools as uvt
import argparse
from .conversions import Cosmo_Conversions
import inspect
from . import __version__


def cov(d1, w1, d2=None, w2=None, conj_1=False, conj_2=True):
    """
    Computes an empirical covariance matrix from data vectors. If d1 is of size
    (M,N), then the output is M x M. In other words, the second axis is the
    axis that is averaged over in forming the covariance (e.g. a time axis).

    If d2 is provided and d1 != d2, then this computes the cross-variance,
    i.e. <d1 d2^dagger> - <d1> <d2>^dagger

    The fact that the second copy is complex conjugated is the default behaviour,
    which can be altered by the conj_1 and the conj_2 kwargs. If conj_1 = False
    and conj_2 = False, then <d1 d2^t> is computed, whereas if conj_1 = True
    and conj_2 = True, then <d1^* d2^t*> is computed. (Minus the mean terms).

    Parameters_
    ----------
    d1 : array_like
        Data vector of size (M,N), where N is the length of the "averaging axis"
    w1 : integer
        Weights for averaging d1
    d2 : array_like, optional
        Data vector of size (M,N), where N is the length of the "averaging axis"
        Default: None
    w2 : integer, optional
        Weights for averaging d1. Default: None
    conj_1 : boolean, optional
        Whether to conjugate d1 or not. Default: False
    conj_2 : boolean, optional
        Whether to conjugate d2 or not. Default: True

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

    z1 = w1*d1
    z2 = w2*d2

    if conj_1:
        z1 = z1.conj()
        x1 = x1.conj()
    if conj_2:
        z2 = z2.conj()
        x2 = x2.conj()

    C = np.dot(z1, z2.T)
    W = np.dot(w1, w2.T)
    C /= np.where(W > 0, W, 1)
    C -= np.outer(x1, x2)
    return C

def variance_from_auto_correlations(uvd, bl, spw_range, time_index):
    """
    Predict noise variance on a baseline from autocorrelation amplitudes on antennas.
    Pick a baseline $b=(alpha,beta)$ where $alpha$ and $beta$ are antennas,
    The way to estimate the covariance matrix $C$ from auto-visibility is:
    $C_{ii}(b, LST) = | V(b_alpha, LST, nu_i) V(b_beta, LST, nu_i) | / {B Delta_t},
    where $b_alpha = (alpha,alpha)$ and $b_beta = (beta,beta)$.
    With LST binned over days, we have $C_{ii}(b, LST) = |V(b_alpha,nu_i,t) V(b_beta, nu_i,t)| / {N_{samples} B Delta_t}$.

    Parameters
    ----------
    uvd : UVData

    bl : tuple
        baseline (pol) key, in the format of (ant1, ant2, pol)

    spw_range : tuple
        Length-2 tuple of the spectral window

    time_index : int

    Returns
    -------
    var : ndarray, (spw_Nfreqs,)

    """
    assert isinstance(bl, tuple) and len(bl)==3, "bl must be fed as Length-3 tuple"
    assert isinstance(spw_range, tuple) and len(spw_range)==2, "spw_range must be fed as Length-2 tuple"
    dt = np.median(uvd.integration_time)
    # Delta_t
    df = uvd.channel_width
    # B
    bl1 = (bl[0],bl[0], bl[2])
    # baseline b_alpha
    bl2 = (bl[1], bl[1], bl[2])
    # baseline b_beta
    spw = slice(spw_range[0], spw_range[1])
    x_bl1 = uvd.get_data(bl1)[time_index, spw]
    x_bl2 = uvd.get_data(bl2)[time_index, spw]
    nsample_bl = uvd.get_nsamples(bl)[time_index, spw]
    nsample_bl = np.where(nsample_bl>0, nsample_bl, np.median(uvd.nsample_array[:,:,spw,:]))
    # some impainted data have zero nsample while is not flagged, and they will be assigned the median nsample within the spectral window.
    var = np.abs(x_bl1*x_bl2.conj()) / dt / df / nsample_bl

    return var

def construct_blpairs(bls, exclude_auto_bls=False, exclude_cross_bls=False,
                      exclude_permutations=False, group=False, Nblps_per_group=1):
    """
    Construct a list of baseline-pairs from a baseline-group. This function
    can be used to easily convert a single list of baselines into the input
    needed by PSpecData.pspec(bls1, bls2, ...).

    Parameters
    ----------
    bls : list of tuple
        List of baseline tuples, Ex. [(1, 2), (2, 3), (3, 4)]. Baseline
        integers are not supported, and must first be converted to tuples
        using UVData.baseline_to_antnums().

    exclude_auto_bls: bool, optional
        If True, exclude all baselines crossed with themselves from the final
        blpairs list. Default: False.

    exclude_cross_bls : bool, optional
        If True, exclude all bls crossed with a different baseline. Note if
        this and exclude_auto_bls are True then no blpairs will exist.

    exclude_permutations : bool, optional
        If True, exclude permutations and only form combinations of the bls
        list.

        For example, if bls = [1, 2, 3] (note this isn't the proper form of
        bls, but makes the example clearer) and exclude_permutations = False,
        then blpairs = [11, 12, 13, 21, 22, 23,, 31, 32, 33]. If however
        exclude_permutations = True, then blpairs = [11, 12, 13, 22, 23, 33].

        Furthermore, if exclude_auto_bls = True then 11, 22, and 33 would
        also be excluded.

        Default: False.

    group : bool, optional
        If True, group each consecutive Nblps_per_group blpairs into sub-lists.
        Default: False.

    Nblps_per_group : int, optional
        Number of baseline-pairs to put into each sub-group if group = True.
        Default: 1.

    Returns (bls1, bls2, blpairs)
    -------
    bls1, bls2 : list of tuples
        List of baseline tuples from the zeroth/first index of the blpair.

    blpairs : list of tuple
        List of blpair tuples.
    """
    # assert form
    assert isinstance(bls, (list, np.ndarray)) and isinstance(bls[0], tuple), \
        "bls must be fed as list or ndarray of baseline antnum tuples. Use " \
        "UVData.baseline_to_antnums() to convert baseline integers to tuples."
    assert (not exclude_auto_bls) or (not exclude_cross_bls), "Can't exclude both auto and cross blpairs"

    # form blpairs w/o explicitly forming auto blpairs
    # however, if there are repeated bl in bls, there will be auto bls in blpairs
    if exclude_permutations:
        blpairs = list(itertools.combinations(bls, 2))
    else:
        blpairs = list(itertools.permutations(bls, 2))

    # explicitly add in auto baseline pairs
    blpairs.extend(list(zip(bls, bls)))

    # iterate through and eliminate all autos if desired
    if exclude_auto_bls:
        new_blpairs = []
        for blp in blpairs:
            if blp[0] != blp[1]:
                new_blpairs.append(blp)
        blpairs = new_blpairs

    # same for cross
    if exclude_cross_bls:
        new_blpairs = []
        for blp in blpairs:
            if blp[0] == blp[1]:
                new_blpairs.append(blp)
        blpairs = new_blpairs

    # create bls1 and bls2 list
    bls1 = [blp[0] for blp in blpairs]
    bls2 = [blp[1] for blp in blpairs]

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


def calc_blpair_reds(uvd1, uvd2, bl_tol=1.0, filter_blpairs=True,
                     xant_flag_thresh=0.95, exclude_auto_bls=False,
                     exclude_cross_bls=False,
                     exclude_permutations=True, Nblps_per_group=None,
                     bl_len_range=(0, 1e10), bl_deg_range=(0, 180),
                     xants=None, include_autocorrs=False,
                     include_crosscorrs=True, extra_info=False):
    """
    Use hera_cal.redcal to get matching, redundant baseline-pair groups from
    uvd1 and uvd2 within the specified baseline tolerance, not including
    flagged ants.

    Parameters
    ----------
    uvd1, uvd2 : UVData
        UVData instances with visibility data for the first/second visibilities
        in the cross-spectra that will be formed.

    bl_tol : float, optional
        Baseline-vector redundancy tolerance in meters

    filter_blpairs : bool, optional
        if True, calculate xants (based on data flags) and filter-out baseline pairs
        based on actual baselines in the data.

    xant_flag_thresh : float, optional
        Fraction of 2D visibility (per-waterfall) needed to be flagged to
        consider the entire visibility flagged.

    xants : list, optional
        Additional lilst of xants to hand flag, regardless of flags in the data.

    exclude_auto_bls: boolean, optional
        If True, exclude all bls crossed with itself from the blpairs list

    exclude_cross_bls : boolean, optional
        If True, exclude all bls crossed with a different baseline. Note if
        this and exclude_auto_bls are True then no blpairs will exist.

    exclude_permutations : boolean, optional
        If True, exclude permutations and only form combinations of the bls list.

        For example, if bls = [1, 2, 3] (note this isn't the proper form of bls,
        but makes this example clearer) and exclude_permutations = False,
        then blpairs = [11, 12, 13, 21, 22, 23, 31, 32, 33]. If however
        exclude_permutations = True, then blpairs = [11, 12, 13, 22, 23, 33].
        Furthermore, if exclude_auto_bls = True then 11, 22, and 33 are excluded.

    Nblps_per_group : integer, optional
        Number of baseline-pairs to put into each sub-group. No grouping if None.
        Default: None

    bl_len_range : tuple, optional
        len-2 tuple containing minimum baseline length and maximum baseline
        length [meters] to keep in baseline type selection

    bl_deg_range : tuple, optional
        len-2 tuple containing (minimum, maximum) baseline angle in degrees
        to keep in baseline selection

    include_autocorrs : bool, optional
        If True, include autocorrelation visibilities in their own redundant group.
        If False, dont return any autocorrelation visibilities.
        default is False.

    include_crosscorrs : bool, optional
        If True, include crosscorrelation visibilities. Set to False only if you
        want to compute power spectra for autocorrelation visibilities only!
        default is True.

    extra_info : bool, optional
        If True, return three extra arrays containing
        redundant baseline group indices, lengths and angles

    Returns
    -------
    baselines1, baselines2 : lists of baseline tuples
        Lists of baseline tuples that should be fed as first/second argument
        to PSpecData.pspec(), corresponding to uvd1/uvd2

    blpairs : list of baseline-pair tuples
        Contains the baseline-pair tuples. i.e. zip(baselines1, baselines2)

    xants1, xants2 : lists
        List of bad antenna integers for uvd1 and uvd2

    red_groups : list of integers, returned as extra_info
        Lists index of redundant groups, indexing red_lens and red_angs

    red_lens : list, returned as extra_info
        List of baseline lengths [meters] with len of unique redundant groups

    red_angs : list, returned as extra_info
        List of baseline angles [degrees] (North of East in ENU)
    """
    # get antenna positions
    antpos1, ants1 = uvd1.get_ENU_antpos(pick_data_ants=False)
    antpos1 = dict(list(zip(ants1, antpos1)))
    antpos2, ants2 = uvd2.get_ENU_antpos(pick_data_ants=False)
    antpos2 = dict(list(zip(ants2, antpos2)))
    antpos = dict(list(antpos1.items()) + list(antpos2.items()))

    # assert antenna positions match
    for a in set(antpos1).union(set(antpos2)):
        if a in antpos1 and a in antpos2:
            msg = "antenna positions from uvd1 and uvd2 do not agree to within " \
                  "tolerance of {} m".format(bl_tol)
            assert np.linalg.norm(antpos1[a] - antpos2[a]) < bl_tol, msg

    # calculate xants via flags if asked
    xants1, xants2 = [], []
    if filter_blpairs and uvd1.flag_array is not None and uvd2.flag_array is not None:
        xants1, xants2 = set(ants1), set(ants2)
        baselines = sorted(set(uvd1.baseline_array).union(set(uvd2.baseline_array)))
        for bl in baselines:
            # get antenna numbers
            antnums = uvd1.baseline_to_antnums(bl)

            # continue if autocorr and we dont want to include them
            if not include_autocorrs:
                if antnums[0] == antnums[1]: continue

            if not include_crosscorrs:
                if antnums[0] != antnums[1]: continue

            # work on xants1
            if bl in uvd1.baseline_array:
                # get flags
                f1 = uvd1.get_flags(bl)
                # remove from bad list if unflagged data exists
                if np.sum(f1) < np.prod(f1.shape) * xant_flag_thresh:
                    if antnums[0] in xants1:
                        xants1.remove(antnums[0])
                    if antnums[1] != antnums[0] and antnums[1] in xants1:
                        xants1.remove(antnums[1])

            # work on xants2
            if bl in uvd2.baseline_array:
                # get flags
                f2 = uvd2.get_flags(bl)
                # remove from bad list if unflagged data exists
                if np.sum(f2) < np.prod(f2.shape) * xant_flag_thresh:
                    if antnums[0] in xants2:
                        xants2.remove(antnums[0])
                    if antnums[1] != antnums[0] and antnums[1] in xants2:
                        xants2.remove(antnums[1])

        xants1 = sorted(xants1)
        xants2 = sorted(xants2)

    # add hand-flagged xants if fed
    if xants is not None:
        xants1 += xants
        xants2 += xants

    # construct redundant groups
    reds, lens, angs = get_reds(antpos, bl_error_tol=bl_tol, xants=xants1+xants2,
                                add_autos=include_autocorrs, autos_only=not(include_crosscorrs),
                                bl_deg_range=bl_deg_range, bl_len_range=bl_len_range)
    # construct baseline pairs
    baselines1, baselines2, blpairs, red_groups = [], [], [], []
    for j, r in enumerate(reds):
        (bls1, bls2,
         blps) = construct_blpairs(r, exclude_auto_bls=exclude_auto_bls,
                                   exclude_cross_bls=exclude_cross_bls, group=False,
                                   exclude_permutations=exclude_permutations)
        if len(bls1) < 1:
            continue

        # filter based on real baselines in data
        if filter_blpairs:
            uvd1_bls = uvd1.get_antpairs()
            uvd2_bls = uvd2.get_antpairs()
            _bls1, _bls2 = [], []
            for blp in blps:
                bl1 = blp[0]
                bl2 = blp[1]
                if ((bl1 in uvd1_bls) or (bl1[::-1] in uvd1_bls)) \
                    and ((bl2 in uvd2_bls) or (bl2[::-1] in uvd2_bls)):
                    _bls1.append(bl1)
                    _bls2.append(bl2)
            bls1, bls2 = _bls1, _bls2
            blps = list(zip(bls1, bls2))

        # populate redundant group indices
        rinds = [j] * len(blps)

        # group if desired
        if Nblps_per_group is not None:
            Ngrps = int(np.ceil(float(len(blps)) / Nblps_per_group))
            bls1 = [bls1[Nblps_per_group*i:Nblps_per_group*(i+1)]
                    for i in range(Ngrps)]
            bls2 = [bls2[Nblps_per_group*i:Nblps_per_group*(i+1)]
                    for i in range(Ngrps)]
            blps = [blps[Nblps_per_group*i:Nblps_per_group*(i+1)]
                    for i in range(Ngrps)]
            rinds = [rinds[Nblps_per_group*i:Nblps_per_group*(i+1)]
                    for i in range(Ngrps)]

        baselines1.extend(bls1)
        baselines2.extend(bls2)
        blpairs.extend(blps)
        red_groups.extend(rinds)

    if extra_info:
        return baselines1, baselines2, blpairs, xants1, xants2, red_groups, lens, angs
    else:
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
    Add a message to the log.

    Parameters
    ----------
    msg : str
        Message string to print.

    f : file descriptor
        file descriptor to write message to.

    lvl : int, optional
        Indent level of the message. Each level adds two extra spaces.
        Default: 0.

    tb : traceback tuple, optional
        Output of sys.exc_info()

    verbose : bool, optional
        if True, print msg. Even if False, still writes to file
        if f is provided.
    """
    # catch for traceback if provided
    if tb is not None:
        msg += "\n{}".format('\n'.join(traceback.format_exception(*tb)))

    # print
    output = "%s%s" % ("  "*lvl, msg)
    if verbose:
        print(output)

    # write
    if f is not None:
        f.write(output)
        f.flush()


def load_config(config_file):
    """
    Load configuration details from a YAML file.
    All entries of 'None' --> None and all lists
    of lists become lists of tuples.
    """
    # define recursive replace function
    def replace(d):
        if isinstance(d, (dict, odict)):
            for k in d.keys():
                # 'None' and '' turn into None
                if d[k] == 'None': d[k] = None
                # list of lists turn into lists of tuples
                if isinstance(d[k], list) \
                and np.all([isinstance(i, list) for i in d[k]]):
                    d[k] = [tuple(i) for i in d[k]]
                elif isinstance(d[k], (dict, odict)): replace(d[k])

    # Open and read config file
    with open(config_file, 'r') as cfile:
        try:
            cfg = yaml.load(cfile, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            raise(exc)

    # Replace entries
    replace(cfg)

    return cfg


def flatten(nested_list):
    """
    Flatten a list of nested lists
    """
    return [item for sublist in nested_list for item in sublist]


def config_pspec_blpairs(uv_templates, pol_pairs, group_pairs, exclude_auto_bls=False,
                         exclude_permutations=True, bl_len_range=(0, 1e10),
                         bl_deg_range=(0, 180), xants=None, exclude_patterns=None,
                         include_autocorrs=False,
                         file_type='miriad', verbose=True):
    """
    Given a list of glob-parseable file templates and selections for
    polarization and group labels, construct a master list of
    group-pol pairs, and also a list of blpairs for each
    group-pol pair given selections on baseline angles and lengths.

    A group is a fieldname in the visibility files that denotes the
    "type" of dataset. For example, the group field in the following files
        zen.even.LST.1.01.xx.HH.uv
        zen.odd.LST.1.01.xx.HH.uv
    are the "even" and "odd" fields, which specifies the two time-binning groups.
    To form cross spectra between these two files, one would feed a group_pair
    of: group_pairs = [('even', 'odd')] and pol_pairs = [('xx', 'xx')].

    Parameters
    ----------
    uv_templates : list
        List of glob-parseable string templates, each of which must have
        a {pol} and {group} field.

    pol_pairs : list
        List of len-2 polarization tuples to use in forming cross spectra.
        Ex: [('xx', 'xx'), ('yy', 'yy'), ...]

    group_pairs : list
        List of len-2 group tuples to use in forming cross spectra.
        See top of doc-string for an explanation of a "group" in this context.
        Ex: [('grp1', 'grp1'), ('grp2', 'grp2'), ...]

    exclude_auto_bls : bool
        If True, exclude all baselines paired with itself.

    exclude_permutations : bool
        If True, exclude baseline2_cross_baseline1 if
        baseline1_cross_baseline2 exists.

    bl_len_range : len-2 tuple
        A len-2 integer tuple specifying the range of baseline lengths
        (meters in ENU frame) to consider.

    bl_deg_range : len-2 tuple
        A len-2 integer tuple specifying the range of baseline angles
        (degrees in ENU frame) to consider.

    xants : list, optional
        A list of integer antenna numbers to exclude. Default: None.

    exclude_patterns : list, optional
        A list of patterns to exclude if found in the final list of input
        files (after the templates have been filled-in). This currently
        just takes a list of strings, and does not recognize wildcards.
        Default: None.

    include_autocorrs : bool, optional
        If True, include autocorrelation visibilities
        in the set of blpair groups calculated and returned.

    file_type : str, optional
        File type of the input files. Default: 'miriad'.

    verbose : bool, optional
        If True, print feedback to stdout. Default: True.

    Returns
    -------
    groupings : dict
        A dictionary holding pol and group pair (tuple) as keys
        and a list of baseline-pairs as values.

    Notes
    -----
    A group-pol-pair is formed by self-matching unique files in the
    glob-parsed master list, and then string-formatting-in appropriate
    pol and group selections given pol_pairs and group_pairs.
    """
    # type check
    if isinstance(uv_templates, str):
        uv_templates = [uv_templates]
    assert len(pol_pairs) == len(group_pairs), "len(pol_pairs) must equal "\
                                               "len(group_pairs)"

    # get unique pols and groups
    pols = sorted(set([item for sublist in pol_pairs for item in sublist]))
    groups = sorted(set([item for sublist in group_pairs for item in sublist]))

    # parse wildcards in uv_templates to get wildcard-unique filenames
    unique_files = []
    pol_grps = []
    for template in uv_templates:
        for pol in pols:
            for group in groups:
                # parse wildcards with pol / group selection
                files = glob.glob(template.format(pol=pol, group=group))
                # if any files were parsed, add to pol_grps
                if len(files) > 0:
                    pol_grps.append((pol, group))
                # insert into unique_files with {pol} and {group} re-inserted
                for _file in files:
                    _unique_file = _file.replace(".{pol}.".format(pol=pol),
                        ".{pol}.").replace(".{group}.".format(group=group), ".{group}.")
                    if _unique_file not in unique_files:
                        unique_files.append(_unique_file)
    unique_files = sorted(unique_files)

    # Exclude user-specified patterns
    if exclude_patterns is not None:
        to_exclude = []

        # Loop over files and patterns
        for f in unique_files:
            for pattern in exclude_patterns:

                # Add to list of files to be excluded
                if pattern in f:
                    if verbose:
                        print("File matches pattern '%s' and will be excluded: %s" \
                              % (pattern, f))
                    to_exclude.append(f)
                    continue

        # Exclude files that matched a pattern
        for f in to_exclude:
            try:
                unique_files.remove(f)
            except:
                pass

        # Test for empty list and fail if found
        if len(unique_files) == 0:
            if verbose:
                print("config_pspec_blpairs: All files were filtered out!")
            return []

    # use a single file from unique_files and a single pol-group combination to get antenna positions
    _file = unique_files[0].format(pol=pol_grps[0][0], group=pol_grps[0][1])
    uvd = UVData()
    uvd.read(_file, read_data=False, file_type=file_type)

    # get baseline pairs
    (_bls1, _bls2, _, _,
     _) = calc_blpair_reds(uvd, uvd, filter_blpairs=False, exclude_auto_bls=exclude_auto_bls,
                    exclude_permutations=exclude_permutations, bl_len_range=bl_len_range,
                    include_autocorrs=include_autocorrs, bl_deg_range=bl_deg_range)

    # take out xants if fed
    if xants is not None:
        bls1, bls2 = [], []
        for bl1, bl2 in zip(_bls1, _bls2):
            if bl1[0] not in xants \
              and bl1[1] not in xants \
              and bl2[0] not in xants \
              and bl2[1] not in xants:
                bls1.append(bl1)
                bls2.append(bl2)
    else:
        bls1, bls2 = _bls1, _bls2
    blps = list(zip(bls1, bls2))

    # iterate over pol-group pairs that exist
    groupings = odict()
    for pp, gp in zip(pol_pairs, group_pairs):
        if (pp[0], gp[0]) not in pol_grps or (pp[1], gp[1]) not in pol_grps:
            if verbose:
                print("pol_pair {} and group_pair {} not found in data files".format(pp, gp))
            continue
        groupings[(tuple(gp), tuple(pp))] = blps

    return groupings


def get_blvec_reds(blvecs, bl_error_tol=1.0, match_bl_lens=False):
    """
    Given a blvecs dictionary, form groups of baseline-pairs based on
    redundancy in ENU coordinates. Note: this only uses the East-North components
    of the baseline vectors to calculate redundancy.

    Parameters:
    -----------
    blvecs : dictionary (or UVPSpec object)
        A dictionary with len-2 or 3 ndarray baseline vectors as values.
        Alternatively, this can be a UVPSpec object.

    bl_error_tol : float, optional
        Redundancy tolerance of baseline vector in meters. Default: 1.0

    match_bl_lens : bool, optional
        Combine baseline groups of identical baseline length but
        differing angle (using bl_error_tol). Default: False

    Returns:
    --------
    red_bl_grp : list
        A list of baseline groups, ordered by ascending baseline length.

    red_bl_len : list
        A list of baseline lengths in meters for each bl group

    red_bl_ang : list
        A list of baseline angles in degrees for each bl group

    red_bl_tag : list
        A list of baseline string tags denoting bl length and angle
    """
    from hera_pspec import UVPSpec
    # type check
    assert isinstance(blvecs, (dict, odict, UVPSpec)), \
        "blpairs must be fed as a dict or UVPSpec"
    if isinstance(blvecs, UVPSpec):
        # get baseline vectors
        uvp = blvecs
        bls = uvp.bl_array
        bl_vecs = uvp.get_ENU_bl_vecs()[:, :2]
        blvecs = dict(list(zip( [uvp.bl_to_antnums(_bls) for _bls in bls],
                                bl_vecs )))
        # get baseline-pairs
        blpairs = uvp.get_blpairs()
        # form dictionary
        _blvecs = odict()
        for blp in blpairs:
            bl1 = blp[0]
            bl2 = blp[1]
            _blvecs[blp] = (blvecs[bl1] + blvecs[bl2]) / 2.
        blvecs = _blvecs

    # create empty lists
    red_bl_grp = []
    red_bl_vec = []
    red_bl_len = []
    red_bl_ang = []
    red_bl_tag = []

    # iterate over each baseline in blvecs
    for bl in blvecs.keys():
        # get bl vector and properties
        bl_vec = blvecs[bl][:2]
        bl_len = np.linalg.norm(bl_vec)
        bl_ang = np.arctan2(*bl_vec[::-1]) * 180 / np.pi
        if bl_ang < 0: bl_ang = (bl_ang + 180) % 360
        bl_tag = "{:03.0f}_{:03.0f}".format(bl_len, bl_ang)

        # append to list if unique within tolerance
        if match_bl_lens:
            # match only on bl length
            match = [np.all(np.isclose(bll, bl_len, rtol=0.0, atol=bl_error_tol)) for bll in red_bl_len]
        else:
            # match on full bl vector
            match = [np.all(np.isclose(blv, bl_vec, rtol=0.0, atol=bl_error_tol)) for blv in red_bl_vec]
        if np.any(match):
            match_id = np.where(match)[0][0]
            red_bl_grp[match_id].append(bl)

        # else create new list
        else:
            red_bl_grp.append([bl])
            red_bl_vec.append(bl_vec)
            red_bl_len.append(bl_len)
            red_bl_ang.append(bl_ang)
            red_bl_tag.append(bl_tag)

    # order based on tag
    order = np.argsort(red_bl_tag)
    red_bl_grp = [red_bl_grp[i] for i in order]
    red_bl_len = [red_bl_len[i] for i in order]
    red_bl_ang = [red_bl_ang[i] for i in order]
    red_bl_tag = [red_bl_tag[i] for i in order]

    return red_bl_grp, red_bl_len, red_bl_ang, red_bl_tag


def job_monitor(run_func, iterator, action_name, M=map, lf=None, maxiter=1,
                verbose=True):
    """
    Job monitoring function, used to send elements of iterator through calls of
    run_func. Can be parallelized if the input M function is from the
    multiprocess module.

    Parameters
    ----------
    run_func : function
        A worker function to run on each element in iterator. Should return
        0 if job is successful, otherwise a failure is any non-zero integer.

    iterator : iterable
        An iterable whose elements define an individual job launch, and are
        passed through to run_func for each individual job.

    action_name : str
        A descriptive name for the operation being performed by run_func.

    M : map function
        A map function used to send elements of iterator through calls to
        run_func. Default is built-in map function.

    lf : file descriptor
        Log-file descriptor to print message to.

    maxiter : int
        Maximum number of job re-tries for failed jobs.

    verbose : bool
        If True, print feedback to stdout and logfile.

    Returns
    -------
    failures : list
        A list of failed job indices from iterator. Failures are any output of
        run_func that aren't 0.
    """
    # Start timing
    t_start = time.time()

    # run function over jobs
    exit_codes = np.array(list(M(run_func, iterator)))
    tnow = datetime.utcnow()

    # check for len-0
    if len(exit_codes) == 0:
        raise ValueError("No output generated from run_func over iterator {}".format(iterator))

    # inspect for failures
    if np.all(exit_codes != 0):
        # everything failed, raise error
        log("\n{}\nAll {} jobs failed w/ exit codes\n {}: {}\n".format("-"*60,
                                                action_name, exit_codes, tnow),
            f=lf, verbose=verbose)
        raise ValueError("All {} jobs failed".format(action_name))

    # if not all failed, try re-run
    failures = np.where(exit_codes != 0)[0]
    counter = 1
    while True:
        if not np.all(exit_codes == 0):
            if counter >= maxiter:
                # break after certain number of tries
                break
            # re-run function over jobs that failed
            exit_codes = np.array(list(M(run_func, failures)))
            # update counter
            counter += 1
            # update failures
            failures = failures[exit_codes != 0]
        else:
            # all passed
            break

    # print failures if they exist
    if len(failures) > 0:
        log("\nSome {} jobs failed after {} tries:\n{}".format(action_name,
                                                               maxiter,
                                                               failures),
            f=lf, verbose=verbose)
    else:
        t_run = time.time() - t_start
        log("\nAll {} jobs ran through ({:1.1f} sec)".format(action_name, t_run),
            f=lf, verbose=verbose)

    return failures


def get_bl_lens_angs(blvecs, bl_error_tol=1.0):
    """
    Given a list of baseline vectors in ENU (TOPO) coords, get the
    baseline length [meter] and angle [deg] given a baseline error
    tolerance [meters]

    Parameters
    ----------
    blvecs : list
        A list of ndarray of 2D or 3D baseline vectors.

    bl_error_tol : float, optional
        A baseline vector error tolerance.

    Returns
    -------
    lens : ndarray
        Array of baseline lengths [meters]

    angs : ndarray
        Array of baseline angles [degrees]
    """
    # type check
    blvecs = np.asarray(blvecs)
    assert blvecs.shape[1] in [2, 3], "blvecs must have shape (N, 2) or (N, 3)"

    # get lengths and angles
    lens = np.array([np.linalg.norm(v) for v in blvecs])
    angs = np.array([np.arctan2(*v[:2][::-1]) * 180 / np.pi for v in blvecs])
    angs = np.array([(a + 180) % 360 if a < 0 else a for a in angs])

    # Find baseline groups with ang ~ 180 deg that have y-vec within bl_error and set to ang = 0 deg.
    flip = (blvecs[:, 1] > -bl_error_tol) & (blvecs[:, 1] < 0) & (blvecs[:, 0] > 0)
    angs[flip] = 0

    return lens, angs


def get_reds(uvd, bl_error_tol=1.0, pick_data_ants=False, bl_len_range=(0, 1e4),
             bl_deg_range=(0, 180), xants=None, add_autos=False,
             autos_only=False, min_EW_cut=0,
             file_type='miriad'):
    """
    Given a UVData object, a Miriad filepath or antenna position dictionary,
    calculate redundant baseline groups using hera_cal.redcal and optionally
    filter groups based on baseline cuts and xants.

    Parameters
    ----------
    uvd : UVData object or str or dictionary
        UVData object or filepath string or antenna position dictionary.
        An antpos dict is formed via dict(zip(ants, ant_vecs)).

        N.B. If uvd is a filepath, use the `file_type` kwarg to specify the
        file type.

    bl_error_tol : float
        Redundancy tolerance in meters

    pick_data_ants : boolean
        If True, use only antennas in the UVData to construct reds, else use all
        antennas present.

    bl_len_range : float tuple
        A len-2 float tuple specifying baseline length cut in meters

    bl_deg_range : float tuple
        A len-2 float tuple specifying baseline angle cut in degrees in ENU frame

    xants : list
        List of bad antenna numbers to exclude

    add_autos : bool
        If True, add into autocorrelation group to the redundant group list.

    autos_only : bool, optional
        If True, only include autocorrelations.
        Default is False.

    min_EW_cut : float
        Baselines with a projected East-West absolute baseline length in meters
        less than this are not included in the output.

    file_type : str, optional
        File type of the input files. Default: 'miriad'.

    Returns (reds, lens, angs)
    -------
    reds : list
        List of redundant baseline (antenna-pair) groups

    lens : list
        List of baseline lengths [meters] of each group in reds

    angs : list
        List of baseline angles [degrees ENU coords] of each group in reds
    """
    # handle string and UVData object
    if isinstance(uvd, (str, UVData)):
        # load filepath
        if isinstance(uvd, str):
            _uvd = UVData()
            _uvd.read(uvd, read_data=False, file_type=file_type)
            uvd = _uvd
        # get antenna position dictionary
        antpos, ants = uvd.get_ENU_antpos(pick_data_ants=pick_data_ants)
        antpos_dict = dict(list(zip(ants, antpos)))
    elif isinstance(uvd, (dict, odict)):
        # use antenna position dictionary
        antpos_dict = uvd
    else:
        raise TypeError("uvd must be a UVData object, filename string, or dict "
                        "of antenna positions.")
    # get redundant baselines
    reds = redcal.get_pos_reds(antpos_dict, bl_error_tol=bl_error_tol)

    # get vectors, len and ang for each baseline group
    vecs = np.array([antpos_dict[r[0][0]] - antpos_dict[r[0][1]] for r in reds])
    lens, angs = get_bl_lens_angs(vecs, bl_error_tol=bl_error_tol)

    # restrict baselines
    _reds, _lens, _angs = [], [], []
    for i, (l, a) in enumerate(zip(lens, angs)):
        if l < bl_len_range[0] or l > bl_len_range[1]: continue
        if a < bl_deg_range[0] or a > bl_deg_range[1]: continue
        if np.abs(l * np.cos(a * np.pi / 180)) < min_EW_cut: continue
        _reds.append(reds[i])
        _lens.append(lens[i])
        _angs.append(angs[i])
    reds, lens, angs = _reds, _lens, _angs

    # put in autocorrs
    if add_autos:
        ants = antpos_dict.keys()
        reds = [list(zip(ants, ants))] + reds
        lens = np.insert(lens, 0, 0)
        angs = np.insert(angs, 0, 0)
        if autos_only:
            reds = reds[:1]
            lens = lens[:1]
            angs = angs[:1]


    # filter based on xants
    if xants is not None:
        _reds, _lens, _angs = [], [], []
        for i, r in enumerate(reds):
            _r = []
            for bl in r:
                if bl[0] not in xants and bl[1] not in xants:
                    _r.append(bl)
            if len(_r) > 0:
                _reds.append(_r)
                _lens.append(lens[i])
                _angs.append(angs[i])

        reds, lens, angs = _reds, _lens, _angs

    return reds, lens, angs


def pspecdata_time_difference(ds, time_diff):
    """
    Given a PSpecData object and a time difference, give the time difference PSpecData object.

    Parameters
    ----------
    ds : PSpecData object

    time_diff : float
        The time difference in seconds.

    Returns
    -------
    ds_td : PSpecData object
    """
    from hera_pspec.pspecdata import PSpecData
    uvd1 = ds.dsets[0]
    uvd2 = ds.dsets[1]
    uvd10 = uvd_time_difference(uvd1, time_diff)
    uvd20 = uvd_time_difference(uvd2, time_diff)

    ds_td = PSpecData(dsets=[uvd10, uvd20], wgts=ds.wgts, beam=ds.primary_beam)
    return ds_td


def uvd_time_difference(uvd, time_diff):
    """
    Given a UVData object and a time difference, give the time difference UVData object.

    Parameters
    ----------
    uvd : UVData object

    time_diff : float
        The time difference in seconds.

    Returns
    -------
    uvd_td : UVData object
    """
    min_time_diff = np.mean(np.unique(uvd.time_array)[1:]-np.unique(uvd.time_array)[0:-1])
    index_diff = int(time_diff / min_time_diff) + 1
    if index_diff > len(np.unique(uvd.time_array))-2:
        index_diff = len(np.unique(uvd.time_array))-2

    uvd0 = uvd.select(times=np.unique(uvd.time_array)[0:-1:index_diff], inplace=False)
    uvd1 = uvd.select(times=np.unique(uvd.time_array)[1::index_diff], inplace=False)
    data0 = uvd0.data_array
    data1 = uvd1.data_array
    data0 -= data1
    uvd0.data_array = data0 / np.sqrt(2)

    return uvd0


AUTOVISPOLS = ['XX', 'YY', 'EE', 'NN']
STOKPOLS = ['PI', 'PQ', 'PU', 'PV']
AUTOPOLS = AUTOVISPOLS + STOKPOLS


def uvd_to_Tsys(uvd, beam, Tsys_outfile=None):
    """
    Convert auto-correlations in Jy to an estimate of Tsys in Kelvin.

    A visibility Tsys is given by the geometric mean of each auto-correlation.

    Parameters
    ----------
    uvd : UVData object
        Should contain visibility auto-correlations

    beam : str or PSpecBeamBase subclass or UVPSpec object
        beam object for converting Jy <-> Kelvin. Should match
        polarizations in uvd. If str, should be a path to a FITS
        file in UVBeam format. If fed as a UVPSpec object, will
        use its OmegaP and OmegaPP attributes to make a beam.

    Tsys_outfile : str
        If fed, write UVH5 file of Tsys estimate [Kelvin] for auto-correlations

    Returns
    -------
    UVData object
        Estimate of Tsys in Kelvin for each auto-correlation
    """
    uvd = copy.deepcopy(uvd)
    # get uvd metadata
    pols = [pol for pol in uvd.get_pols() if pol.upper() in AUTOPOLS]
    # if pseudo Stokes pol in pols, substitute for pI
    pols = sorted(set([pol if pol.upper() in AUTOVISPOLS else 'pI' for pol in pols]))
    autobls = [bl for bl in uvd.get_antpairs() if bl[0] == bl[1]]
    uvd.select(bls=autobls, polarizations=pols)

    # construct beam
    from hera_pspec import pspecbeam
    from hera_pspec import uvpspec
    if isinstance(beam, str):
        beam = pspecbeam.PSpecBeamUV(beam)
    elif isinstance(beam, pspecbeam.PSpecBeamBase):
        pass
    elif isinstance(beam, uvpspec.UVPSpec):
        uvp = beam
        if hasattr(uvp, 'OmegaP'):
            # use first pol in each polpair
            uvp_pols = [pp[0] if pp[0].upper() not in STOKPOLS else 'pI' for pp in uvp.get_polpairs()]
            Op = {uvp_pol: uvp.OmegaP[:, ii] for ii, uvp_pol in enumerate(uvp_pols)}
            Opp = {uvp_pol: uvp.OmegaPP[:, ii] for ii, uvp_pol in enumerate(uvp_pols)}
            beam = pspecbeam.PSpecBeamFromArray(Op, Opp, uvp.beam_freqs, cosmo=uvp.cosmo)
        else:
            raise ValueError("UVPSpec must have OmegaP and OmegaPP to make a beam")
    else:
        raise ValueError("beam must be a string, PSpecBeamBase subclass or UVPSpec object")

    # convert autos in Jy to Tsys in Kelvin
    J2K = {pol: beam.Jy_to_mK(uvd.freq_array[0], pol=pol)/1e3 for pol in pols}
    for blpol in uvd.get_antpairpols():
        bl, pol = blpol[:2], blpol[2]
        tinds = uvd.antpair2ind(bl)
        if pol.upper() in STOKPOLS:
            pol = 'pI'
        pind = pols.index(pol)
        uvd.data_array[tinds, 0, :, pind] *= J2K[pol]

    if Tsys_outfile is not None:
        uvd.write_uvh5(Tsys_outfile, clobber=True)

    return uvd

def uvp_noise_error(uvp, auto_Tsys=None, err_type='P_N', precomp_P_N=None, P_SN_correction=True,  num_steps_scalar=2000, little_h=True):
    """
    Calculate analytic thermal noise error for a UVPSpec object.
    Adds to uvp.stats_array inplace.

    Parameters
    ----------
    uvp : UVPSpec object
        Power spectra to calculate thermal noise errors.
        If err_type == 'P_SN', uvp should not have any
        incoherent averaging applied.

    auto_Tsys : UVData object, optional
        Holds autocorrelation Tsys estimates in Kelvin (see uvd_to_Tsys)
        for all antennas and polarizations involved in uvp power spectra.
        Needed for P_N computation, not needed if feeding precomp_P_N.

    err_type : str or list of str, options = ['P_N', 'P_SN']
        Type of thermal noise error to compute. P_N is the standard
        noise-dominated analytic error (e.g. Pober+2013, Cheng+2018)
        P_SN = sqrt[ sqrt[2] P_S * P_N + P_N^2]
        is the signal + noise analytic error for the real or imag
        component of the power spectra (e.g. Kolpanis+2019, Tan+2020),
        which uses uses Re[P(tau)] as a proxy for P_S.
        To store both, feed as err_type = ['P_N', 'P_SN']

    precomp_P_N : str, optional
        If computing P_SN and P_N is already computed, use this key
        to index stats_array for P_N rather than computing it from auto_Tsys.

    P_SN_correctoin : bool, optional
        Apply correction factor if computing P_SN to account for double
        counting of noise.

    num_steps_scalar : int, optional
        Number of frequency steps to explicitly compute (and interpolate over) for integrand in scalar.
        Default is 2000

    little_h : bool, optional
        Use little_h units in power spectrum.
        Default is True.

    """
    from hera_pspec import uvpspec_utils

    # type checks
    if isinstance(err_type, str):
        err_type = [err_type]

    # get metadata if needed
    if precomp_P_N is None:
        lst_indices = np.unique(auto_Tsys.lst_array, return_index=True)[1]
        lsts = auto_Tsys.lst_array[sorted(lst_indices)]
        freqs = auto_Tsys.freq_array[0]
    # calculate scalars for spws and polpairs.
    scalar = {}
    for spw in uvp.spw_array:
        for polpair in uvp.polpair_array:
            scalar[(spw, polpair)] = uvp.compute_scalar(spw, polpair, num_steps=num_steps_scalar,
                                        little_h=little_h, noise_scalar=True)
    # iterate over spectral window
    for spw in uvp.spw_array:
        # get spw properties
        spw_range = uvp.get_spw_ranges(spw)[0]
        spw_start = np.argmin(np.abs(auto_Tsys.freq_array[0] - spw_range[0]))
        spw_stop = spw_start + spw_range[2]
        taper = uvt.dspec.gen_window(uvp.taper, spw_range[2])
        # iterate over blpairs
        for blp in uvp.get_blpairs():
            blp_int = uvp.antnums_to_blpair(blp)
            lst_avg = uvp.lst_avg_array[uvp.blpair_to_indices(blp)]
            # iterate over polarization
            for polpair in uvp.polpair_array:
                pol = uvpspec_utils.polpair_int2tuple(polpair)[0]  # integer
                polstr = uvutils.polnum2str(pol) # TODO: use uvp.x_orientation when attr is added
                if polstr.upper() in STOKPOLS:
                    pol = 'pI'
                key = (spw, blp, polpair)

                if precomp_P_N is None:
                    # take geometric mean of four antenna autocorrs and get OR'd flags
                    Tsys = (auto_Tsys.get_data(blp[0][0], blp[0][0], pol)[:, spw_start:spw_stop].real * \
                            auto_Tsys.get_data(blp[0][1], blp[0][1], pol)[:, spw_start:spw_stop].real * \
                            auto_Tsys.get_data(blp[1][0], blp[1][0], pol)[:, spw_start:spw_stop].real * \
                            auto_Tsys.get_data(blp[1][1], blp[1][1], pol)[:, spw_start:spw_stop].real)**(1./4)
                    Tflag = auto_Tsys.get_flags(blp[0][0], blp[0][0], pol)[:, spw_start:spw_stop] + \
                            auto_Tsys.get_flags(blp[0][1], blp[0][1], pol)[:, spw_start:spw_stop] + \
                            auto_Tsys.get_flags(blp[1][0], blp[1][0], pol)[:, spw_start:spw_stop] + \
                            auto_Tsys.get_flags(blp[1][1], blp[1][1], pol)[:, spw_start:spw_stop]
                    # average over frequency
                    if np.all(Tflag):
                        # fully flagged
                        Tsys = np.inf
                    else:
                        # get weights
                        Tsys = np.sum(Tsys * ~Tflag * taper, axis=-1) / np.sum(~Tflag * taper, axis=-1).clip(1e-20, np.inf)
                        Tflag = np.all(Tflag, axis=-1)
                        # interpolate to appropriate LST grid
                        if np.count_nonzero(~Tflag) > 1:
                            Tsys = interp1d(lsts[~Tflag], Tsys[~Tflag], kind='nearest', bounds_error=False, fill_value='extrapolate')(lst_avg)
                        else:
                            Tsys = Tsys[0]

                    # calculate P_N
                    P_N = uvp.generate_noise_spectra(spw, polpair, Tsys, blpairs=[blp], form='Pk', component='real', scalar=scalar[(spw, polpair)])[blp_int]

                else:
                    P_N = uvp.get_stats(precomp_P_N, key)

                if 'P_N' in err_type:
                    # set stats
                    uvp.set_stats('P_N', key, P_N)

                if 'P_SN' in err_type:
                    # calculate P_SN: see Tan+2020 and
                    # H1C_IDR2/notebooks/validation/errorbars_with_systematics_and_noise.ipynb
                    # get signal proxy
                    P_S = uvp.get_data(key).real
                    # clip negative values
                    P_S[P_S < 0] = 0
                    P_SN = np.sqrt(np.sqrt(2) * P_S * P_N + P_N**2)
                    # catch nans, set to inf
                    P_SN[np.isnan(P_SN)] = np.inf
                    # set stats
                    uvp.set_stats('P_SN', key, P_SN)

    # P_SN correction
    if P_SN_correction and "P_SN" in err_type:
        if precomp_P_N is None:
            precomp_P_N = 'P_N'
        apply_P_SN_correction(uvp, P_SN='P_SN', P_N=precomp_P_N)

def uvp_noise_error_parser():
    """
    Get argparser to generate noise error bars using autos

    Args:
        N/A
    Returns:
        a: argparser object with arguments used in auto_noise_run.py.
    """
    a = argparse.ArgumentParser(description="argument parser for computing "
                                            "thermal noise error bars from "
                                            "autocorrelations")
    a.add_argument("pspec_container", type=str,
                   help="Filename of HDF5 container (PSpecContainer) containing "
                        "input power spectra.")
    a.add_argument("auto_file", type=str, help="Filename of UVData object containing only autocorr baselines to use"
                                                "in thermal noise error bar estimation.")
    a.add_argument("beam", type=str, help="Filename for UVBeam storing primary beam.")
    a.add_argument("--groups", type=str, help="Name of power-spectrum group to compute noise for.", default=None, nargs="+")
    a.add_argument("--spectra", default=None, type=str, nargs='+',
                   help="List of power spectra names (with group prefix) to calculate noise for.")
    a.add_argument("--err_type", default="P_N", type=str,
                    nargs="+", help="Which components of noise error"
                                    "to compute, 'P_N' or 'P_SN'")
    return a

def apply_P_SN_correction(uvp, P_SN='P_SN', P_N='P_N'):
    """
    Apply correction factor to P_SN errorbar in stats_array to account
    for double counting of noise by using data as proxy for signal.
    See Jianrong Tan et al. 2021 (Errorbar methodologies).
    Operates in place. Must have both P_SN and P_N in stats_array.

    Args:
        uvp : UVPSpec object
            With P_SN and P_N errorbar (not variance) as a key in stats_array
        P_SN : str
            Key in stats_array for P_SN errorbar
        P_N : str
            Key in stats_array for P_N errorbar
    """
    assert P_SN in uvp.stats_array
    assert P_N in uvp.stats_array
    for spw in uvp.spw_array:
        # get P_SN and P_N
        p_n = uvp.stats_array[P_N][spw]
        p_sn = uvp.stats_array[P_SN][spw]
        # derive correction
        corr = 1 - (np.sqrt(1 / np.sqrt(np.pi) + 1) - 1) * p_n.real / p_sn.real.clip(1e-40, np.inf)
        corr[np.isclose(corr, 0)] = np.inf
        corr[corr < 0] = np.inf
        corr[np.isnan(corr)] = np.inf
        # apply correction
        uvp.stats_array[P_SN][spw] *= corr


def history_string(notes=''):
    """
    Creates a standardized history string that all functions that write to
    disk can use. Optionally add notes.
    """
    notes = f"""\n\nNotes:\n{notes}""" if notes else ""

    stack = inspect.stack()[1]
    history = f"""
    ------------
    This file was produced by the function {stack[3]}() in {stack[1]} using version {__version__}
    {notes}
    ------------
    """
    return history

