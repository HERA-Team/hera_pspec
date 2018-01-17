import numpy as np, aipy, random
    
def noise(size,amp=1.):
    """
    Generates Gaussian random complex noise. Real and complex components each get half
    the variance so that the magnitude of the complex number has the full variance.
    
    Parameters
    ----------
    size : integer
        Length of output noise vector
    amp (optional) : float
        Amplitude (standard deviation) of the random noise

    Returns
    -------
    Random noise whose magnitude has a standard deviation of amp
    """
    sig = amp/np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size)

def get_Q(mode, n_k, window='none'): #encodes the fourier transform from freq to delay
    """
    Produces a matrix Q that performs a two-sided Fourier transform and extracts a particular
    Fourier mode. In other words, computing x^t Q y is equivalent to Fourier transforming
    x and y separately, extracting one element of the Fourier transformed vectors, and then multiplying them

    Parameters
    ----------
    mode : integer
        index of the desired Fourier mode
    n_k : integer
        length of hypothetical data vectors that Q operates on

    Returns
    -------
    Q : array_like
        matrix Q
    """
    _m = np.zeros((n_k,), dtype=np.complex)
    _m[mode] = 1. #delta function at specific delay mode
    m = np.fft.fft(np.fft.ifftshift(_m)) * aipy.dsp.gen_window(n_k, window) #FFT it to go to freq
    Q = np.einsum('i,j', m, m.conj()) #dot it with its conjugate
    return Q

def cov_original(d1, w1, d2=None, w2=None):
    """
    ACL: Incredibly confused about this right now. There are potentially a few problems:
    1) I don't understand why the overall scale of the weights matters. If I use uniform
    weights, it shouldn't matter whether I'm doing (1,1,1,...) or (3,3,3,...)
    The mean respects this property. But the covariance doesn't appear to.
    If I divide by W instead of np.where(W > 1, W-1, 1), then the covariance respects
    this property too.
    2) Are the complex conjugations done properly if d2 != d1? In other parts of the code,
    does one implicitly assume that d2 is already complex conjugated? Currently, the code
    computes <x x^t> rather than <x x^dagger>. So it has the unsatisfying (wrong?) property
    that cov(d1, w1, d1, w1) != cov(d1, w1)

    """
    if d2 is None: d2,w2 = d1.conj(),w1
    d1sum,d1wgt = (w1*d1).sum(axis=1), w1.sum(axis=1)
    d2sum,d2wgt = (w2*d2).sum(axis=1), w2.sum(axis=1)
    x1,x2 = d1sum / np.where(d1wgt > 0,d1wgt,1), d2sum / np.where(d2wgt > 0,d2wgt,1)
    x1.shape = (-1,1)
    x2.shape = (-1,1)
    d1x = d1 - x1
    d2x = d2 - x2
    C = np.dot(w1*d1x,(w2*d2x).T)
    W = np.dot(w1,w2.T)
    return C / np.where(W > 1, W-1, 1)

def cov(d1, w1, d2=None, w2=None):
    """
    Computes an empirical covariance matrix from data vectors. If d1 is of size (M,N),
    then the output is M x M. In other words, the second axis is the axis that is
    averaged over in forming the covariance. (E.g., a time axis).

    If d2 is provided and d1 != d2, then this computes the cross-variance, i.e., <d1 d2^dagger>

    Parameters
    ----------
    d1 : array_like
        data vector of size (M,N), where N is the length of the "averaging axis"
    w1 : integer
        weights for averaging d1
    d2 : array_like
        data vector of size (M,N), where N is the length of the "averaging axis"
    w2 : integer
        weights for averaging d1

    Returns
    -------
    cov : array_like
        covariance (or cross-variance) matrix of size (M,M)
    """
    if d2 is None: d2,w2 = d1,w1
    if not np.isreal(w1).all(): raise TypeError("Weight matrices must be real")
    if not np.isreal(w2).all(): raise TypeError("Weight matrices must be real")
    if np.less(w1, 0.).any() or np.less(w2, 0.).any(): raise ValueError("Weight matrices must be positive")
    d1sum,d1wgt = (w1*d1).sum(axis=1), w1.sum(axis=1)
    d2sum,d2wgt = (w2*d2).sum(axis=1), w2.sum(axis=1)
    x1,x2 = d1sum / np.where(d1wgt > 0,d1wgt,1), d2sum / np.where(d2wgt > 0,d2wgt,1)
    x1.shape = (-1,1)
    x2.shape = (-1,1)
    # d1x = d1 - x1
    # d2x = d2 - x2
    C = np.dot(w1*d1,(w2*d2).conj().T)
    #print "hey hey", C
    W = np.dot(w1,w2.T)
    C /= np.where(W > 0, W, 1)
    C -= np.outer(x1,x2.conj())
    return C

def lst_align(lsts, lstres=.001, interpolation='none'):
    """
    Aligns 
    """
    lstgrid = np.arange(0, 2*np.pi, lstres)
    lstr, order = {}, {}
    for k in lsts: #orders LSTs to find overlap
        order[k] = np.argsort(lsts[k])
        lstr[k] = np.around(lsts[k][order[k]] / lstres) * lstres
    lsts_final = None
    for i,k1 in enumerate(lstr.keys()):
        for k2 in lstr.keys()[i:]:
            if lsts_final is None: lsts_final = np.intersect1d(lstr[k1],lstr[k2]) #XXX LSTs much match exactly
            else: lsts_final = np.intersect1d(lsts_final,lstr[k2])
    inds = {}
    for k in lstr: #selects correct LSTs from data
        inds[k] = order[k].take(lstr[k].searchsorted(lsts_final))
    return inds
