import numpy as np
import md5
import yaml
from conversions import Cosmo_Conversions


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

def get_delays(freqs):
    """
    Return an array of delays, tau, corresponding to the bins of the delay 
    power spectrum given by frequency array.
    
    Parameters
    ----------
    freqs : ndarray of frequencies in Hz

    Returns
    -------
    delays : array_like
        Delays, tau. Units: seconds.
    """
    # Calculate the delays
    delay = np.fft.fftshift(np.fft.fftfreq(freqs.size, d=np.median(np.diff(freqs))))
    return delay


def spw_range_from_freqs(data, freq_range, bounds_error=True):
    """
    Return the first and last frequency array indices for a spectral window, 
    where the window is specified as a range of frequencies.
    
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
    Return the first and last frequency array indices for a spectral window, 
    where the window is specified as a range of redshifts.
    
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
    

def log(msg, lvl=0):
    """
    Add a message to the log (just prints to the terminal for now).
    
    Parameters
    ----------
    lvl : int, optional
        Indent level of the message. Each level adds two extra spaces. 
        Default: 0.
    """
    print("%s%s" % ("  "*lvl, msg))


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

