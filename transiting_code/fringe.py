'''
Implements a Fringe Rate Filter useing and FIR filter.
'''
import aipy as a
import capo
import numpy as n
from numpy.fft import ifft, fftshift, ifftshift, fftfreq, fft
import scipy.interpolate

DEFAULT_FRBINS = n.arange(-.01+5e-5/2,.01,5e-5) # Hz
#DEFAULT_FRBINS = n.arange(-.5/42.94999+5e-5/2,.5/42.94999,5e-5) # Hz old fr_bins, new inttime
DEFAULT_WGT = lambda bm: bm**2
DEFAULT_IWGT = lambda h: n.sqrt(h)

def gen_frbins(inttime, fringe_res=5e-5):
    """
    Generate fringe-rate bins appropriate for use in fr_profile().
    inttime is in seconds, returned bins in Hz.
    """
    fmax = 0.5 / inttime
    return n.arange(-fmax, fmax+fringe_res/2, fringe_res)

def mk_fng(bl, eq):
    '''Return fringe rates given eq coords and a baseline vector (measured in wavelengths) in eq coords'''
    return -2*n.pi/a.const.sidereal_day * n.dot(n.cross(n.array([0,0,1.]),bl), eq)

#fringe used in ali et.al to degrade optimal fringe rate filter.
def mk_fng_alietal(bl, eq):
    '''Return distorted fringe rates for given eq coordinates and a baseline vector (measured in wavelengths) in eq coords. This was the version used in ali et.al'''
    ey, ex, ez = eq#yes, ex and ey are flipped.
    return 2*n.pi/a.const.sidereal_day * (bl[0]*ex + bl[1]*ey * n.sqrt(1-ez**2))

def fr_profile(bm, fng, bins=DEFAULT_FRBINS, wgt=DEFAULT_WGT, iwgt=DEFAULT_IWGT):
    '''Return the fringe-rate profile (binning the beam by fringe rate).'''
    h, _ = n.histogram(fng, bins=bins, weights=wgt(bm))
    h = iwgt(h)
    h /= h.max()
    #bins given to histogram are bin edges. Want bin centers.
    bins = 0.5 * (bins[:-1] + bins[1:])
    return h, bins

def gauss(cenwid, bins): return n.exp(-(bins-cenwid[0])**2/(2*cenwid[1]**2))
def tanh(x, p, w, C = 1.0, a=1.0): return (C/2.) * (1 + a*n.tanh( (x-p)/(2*w)))
def mdl_wrap(prms, frp, bins, maxfr, mdl): return n.sum((frp - n.where(bins > maxfr,0,mdl(prms,bins)))**2)

def fit_mdl(frp, bins, maxfr, mdl=gauss, maxfun=1000, ftol=1e-6, xtol=1e-6, startprms=(.001,.0001), verbose=False):
    """
    Fit a parametrized model to the fringe-rate profile frp.  h is weights for each fringe rate in bins,
    maxfr is the maximum fringe rate (which is treated independently of bins).
    """
    bestargs, score = a.optimize.fmin(mdl_wrap, x0=startprms, args=(frp,bins,maxfr,mdl),
        full_output=1, disp=0, maxfun=maxfun, maxiter=n.Inf, ftol=ftol, xtol=xtol)[:2]
    if verbose: print 'Final prms:', bestargs, 'Score:', score
    return bestargs

# XXX wgt and iwgt seem icky
def hmap_to_fr_profile(bm_hmap, bl, lat, bins=DEFAULT_FRBINS, wgt=DEFAULT_WGT, iwgt=DEFAULT_IWGT,alietal=False):
    '''For a healpix map of the beam (in topocentric coords, not squared), a bl (in wavelengths, eq coords),
    and a latitude (in radians), return the fringe-rate profile.'''
    eq = bm_hmap.px2crd(n.arange(bm_hmap.npix()), ncrd=3) # equatorial coordinates
    eq2zen = a.coord.eq2top_m(0., lat)
    top = n.dot(eq2zen, eq)
    _bm = bm_hmap[(top[0], top[1], top[2])]
    _bm = n.where(top[2] > 0, _bm, 0)
    bm = _bm
    if alietal:
        fng = mk_fng_alietal(bl,eq)
    else:
        fng = mk_fng(bl,eq)
    return fr_profile(bm, fng, bins=bins, wgt=wgt, iwgt=iwgt)

def aa_to_fr_profile(aa, (i,j), ch, pol='I', bins=DEFAULT_FRBINS, wgt=DEFAULT_WGT, iwgt=DEFAULT_IWGT, nside=64, bl_scale=1,alietal=False, **kwargs):
    '''For an AntennaArray, for a baseline indexed by i,j, at frequency fq, return the fringe-rate profile.'''
    fq = aa.get_afreqs()[ch]
    h = a.healpix.HealpixMap(nside=nside)
    eq = h.px2crd(n.arange(h.npix()), ncrd=3)
    top = n.dot(aa._eq2zen, eq)
    if alietal:
        fng = mk_fng_alietal(aa.get_baseline(i,j,'r')*fq*bl_scale,eq)
    else:
        # XXX computing bm at all freqs, but only taking one
        fng = mk_fng(aa.get_baseline(i,j,'r')*fq*bl_scale, eq)
    _bmx = aa[0].bm_response((top), pol='x')[ch]; _bmx = n.where(top[2] > 0, _bmx, 0)
    _bmy = aa[0].bm_response((top), pol='y')[ch]; _bmy = n.where(top[2] > 0, _bmy, 0)
    if   pol == 'xx': bm = _bmx * _bmx.conj()
    elif pol == 'yy': bm = _bmy * _bmy.conj()
    elif pol == 'xy': bm = _bmx * _bmy.conj()
    elif pol == 'yx': bm = _bmy * _bmx.conj()
    elif pol ==  'I': bm = .5 * (_bmx*_bmx.conj() + _bmy*_bmy.conj())
    elif pol ==  'Q': bm = .5 * (_bmx*_bmx.conj() - _bmy*_bmy.conj())
    elif pol ==  'U': bm = .5 * (_bmx*_bmy.conj() + _bmy*_bmx.conj())
    elif pol ==  'V': bm = .5 * (_bmx*_bmy.conj() - _bmy*_bmx.conj()) #SK - is this the correct way to treat V?
    return fr_profile(bm, fng, bins=bins, wgt=wgt, iwgt=iwgt)

# XXX write a function that generates bins from inttime and time window for fir

def fir_to_frp(fir,tbins=None):
    '''Transform a fir (time domain fr filter) to a fringe rate profile.
       fir: array of fringe rate profile.
       tbins: Corresponding time bins of filter. If None, doesnt return ffringe rates.
    '''
    fir = ifftshift(fir, axes=-1)
    frp = fft(fir, axis=-1)
    frp = fftshift(frp, axes=-1)
    if tbins is not None: return frp, fftshift(fftfreq(tbins.size, tbins[1]-tbins[0]))
    else: return frp

def frp_to_fir(frp, fbins=None):
    '''Transform a fringe rate profile to a fir filter.'''
    frp = ifftshift(frp,axes=-1)
    fir = ifft(frp, axis=-1)
    fir = fftshift(fir, axes=-1)
    if fbins is not None: return fir, fftshift(fftfreq(fbins.size, fbins[1] - fbins[0]))
    else: return fir

def normalize(fx):
    return fx / n.sqrt(n.sum(n.abs(fx)**2,axis=-1))


def frp_to_firs(frp0, bins, fqs, fq0=.150, limit_maxfr=True, limit_xtalk=True, fr_xtalk=.00035, maxfr=None,
        mdl=gauss, maxfun=1000, ftol=1e-6, xtol=1e-6, startprms=(.001,.0001), window='blackman-harris', alietal=False, verbose=False ,bl_scale=1.,fr_width_scale=1., **kwargs):
    ''' Take a fringe rate profile at one frequency, fit an analytic function and extend
        to other frequencies.
        frp0: fringe rate profile at a single frequency.
        bins: fr bins that correspind to frp0.
        fqs: Frequencies to extend fiter to.
        fq0: Frequency at which frp0 is made for.
        limit_maxfr: cut of fringe rates above maximum possible fringe rate.
        fr_xtalk: Threshold for removing crosstalk.
        mdl: a function to fit the fringe rate profile too. gaussian for default.
    '''
    #print bins
    startprms = tuple( [startprms[0]*fq0/fqs[len(fqs)/2]*bl_scale,startprms[1]])
    if maxfr is None: maxfr = bins[n.argwhere(frp0 != 0).max()] # XXX check this
    prms0 = fit_mdl(frp0, bins, maxfr, mdl=mdl,maxfun=maxfun,ftol=ftol,xtol=xtol,startprms=startprms,verbose=verbose)
    #prms0 = n.array(prms0)
    prms0[1] *= fr_width_scale ##Makes filter artificially wider by factor of fr_width_scale
    if limit_maxfr:
        def limit_maxfr(fq): return tanh(bins,maxfr/fq0*fq,1e-5,a=-1.)
    else:
        def limit_maxfr(fq): return 1
    if limit_xtalk: limit_xtalk = tanh(bins,fr_xtalk,1e-5,a=1.)
    else: limit_xtalk = 1
    frps = n.array([mdl(prms0*fq/fq0,bins) * limit_maxfr(fq) * limit_xtalk for i,fq in enumerate(fqs)])
    tbins = fftshift(fftfreq(bins.size, bins[1]-bins[0]))
    firs = frp_to_fir(frps)
    firs *= a.dsp.gen_window(bins.size, window)
    if alietal:
        firs /= n.sum(n.abs(firs),axis=1).reshape(-1,1) # normalize so that n.sum(abs(fir)) = 1
    else:
        firs /= n.sqrt(n.sum(n.abs(firs)**2,axis=1).reshape(-1,1)) # normalize so that n.sum(abs(fir)**2) = 1
    return tbins, firs

def apply_frf(aa, data, wgts, i, j, pol='I', firs=None, alietal=False,
              **kwargs):
    '''Generate & apply fringe-rate filter to data for baseline (i,j).'''
    freqs, nchan = aa.get_afreqs(), data.shape[-1]
    ch0,fq0 = nchan/2, freqs[nchan/2]
    if firs is None: firs = {}
    tbins = None
    if not firs.has_key((i,j,pol)):
        frp,bins = aa_to_fr_profile(aa, (i,j), ch0, pol=pol, alietal=alietal,
                                    **kwargs)
        del(kwargs['bins'])
        tbins, firs[(i,j,pol)] = frp_to_firs(frp, bins, freqs, fq0=fq0, **kwargs)
    datf,wgtf = n.zeros_like(data), n.zeros_like(data)
    fir = firs[(i,j,pol)]
    for ch in xrange(nchan):
        #datf[:,ch] = n.convolve(data[:,ch], fir[ch,:], mode='same')
        #wgtf[:,ch] = n.convolve(wgts[:,ch], n.abs(fir[ch,:]), mode='same')
        datf[:,ch] = n.convolve(data[:,ch]*wgts[:,ch], n.conj(fir[ch,:]), mode='same')
        wgtf[:,ch] = n.convolve(wgts[:,ch], n.abs(n.conj(fir[ch,:])), mode='same')
    datf = n.where(wgtf > 0, datf/wgtf, 0)
    return datf, wgtf, tbins, firs
