
import numpy as np
import aipy
import md5

DELAY = False

def hash(w):
    """
    Return an MD5 hash of a set of weights.
    """
    return md5.md5(w.copy(order='C')).digest()

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

"""

def lst_grid(lsts, data, wgts=None, lstbins=6300, wgtfunc=lambda dt,res: np.exp(-dt**2/(2*res**2))):
    lstgrid = np.linspace(0, 2*np.pi, lstbins)
    lstres = lstgrid[1]-lstgrid[0]
    if wgts is None: wgts = np.where(np.abs(data) == 0, 0, 1.)
    sumgrid,wgtgrid = 0, 0
    for lst,d,w in zip(lsts,data,wgts):
        dt = lstgrid - lst
        wf = wgtfunc(dt,lstres); wf.shape = (-1,) + (1,)*(data.ndim-1)
        d.shape, w.shape = (1,-1), (1,-1)
        wgtgrid += w * wf
        sumgrid += d * w * wf 
    datgrid = np.where(wgtgrid > 1e-10, sumgrid/wgtgrid, 0)
    return lstgrid, datgrid, wgtgrid

def lst_grid_cheap(lsts, data, wgts=None, lstbins=6300, wgtfunc=lambda dt,res: np.exp(-dt**2/(2*res**2))):
    lstgrid = np.linspace(0, 2*np.pi, lstbins)
    lstres = lstgrid[1]-lstgrid[0]
    if wgts is None: wgts = np.where(np.abs(data) == 0, 0, 1.)
    sumgrid = np.zeros((lstbins,)+data.shape[1:], dtype=data.dtype)
    wgtgrid = np.zeros(sumgrid.shape, dtype=wgts.dtype)
    for lst,d,w in zip(lsts,data,wgts):
        i,j = int(np.floor(lst/lstres)), int(np.ceil(lst/lstres))
        wi,wj = wgtfunc(lst-lstgrid[i],lstres), wgtfunc(lst-lstgrid[j],lstres)
        sumgrid[i] += d * w * wi; wgtgrid[i] += w * wi
        sumgrid[j] += d * w * wj; wgtgrid[j] += w * wj
    datgrid = np.where(wgtgrid > 1e-10, sumgrid/wgtgrid, 0)
    return lstgrid, datgrid, wgtgrid

def lst_align(lsts, lstres=.001, interpolation='none'):
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

def lst_align_data(inds, dsets, wgts=None, lsts=None):
    for k in dsets:
        k0 = k[0]
        dsets[k] = dsets[k][inds[k0]]
        if not wgts is None: wgts[k] = wgts[k][inds[k0]]
    if not lsts is None:
        for k0 in lsts: lsts[k0] = lsts[k0][inds[k0]]
    return [d for d in [dsets,wgts,lsts] if not d is None]

def boot_waterfall(p, axis=1, nsamples=None, nboots=1000, usemedian=True, verbose=False):
    boots = []
    dim = p.shape[axis]
    if nsamples is None: nsamples = dim
    for i in xrange(nboots):
        if verbose and i % 10 == 0: print '    ', i, '/', nboots
        inds = np.random.randint(0,dim,size=nsamples)
        if usemedian: pi = np.median(p.take(inds,axis=axis), axis=1)
        else: pi = np.average(p.take(inds,axis=axis), axis=1)
        boots.append(pi)
    # XXX deal with folding
    boots = np.array(boots)
    if verbose: print 'Sorting bootstraps...'
    pk = np.average(boots, axis=0) #average over all boots
    #this is excluding imag component in noise estimate `
    boots = np.sort(boots.real, axis=0) #dropping imag component here
    up_thresh = int(np.around(0.975 * boots.shape[0])) #2 sigma, single tail
    dn_thresh = int(np.around(0.025 * boots.shape[0])) #2 sigma, single tail
    #important to only include real component in estimation of error
    err_up = (boots[up_thresh] - pk.real) / 2 #effective "1 sigma" derived from actual 2 sigma
    err_dn = -(boots[dn_thresh] - pk.real) / 2 #effective "1 sigma" derived from actual 2 sigma
    return pk, (err_up,err_dn), boots
"""
