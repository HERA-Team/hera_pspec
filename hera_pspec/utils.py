import numpy as np
import md5
import yaml

def hash(w):
    """
    Return an MD5 hash of a set of weights.
    """
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
