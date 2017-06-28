'''Units in mK, GHz, Mpc unless stated otherwise'''
import numpy as n, aipy as a
import pfb
from binning import LST_RES, UV_RES, uv2bin, bin2uv, rebin_log

F21 = 1.42040575177 # GHz

# Temperature conversion
PAPER_BEAM_POLY = [ -1.55740671e+09,  1.14162351e+09, -2.80887022e+08,  9.86929340e+06, 7.80672834e+06, -1.55085596e+06,  1.20087809e+05, -3.47520109e+03]

HERA_BEAM_POLY = n.array([  8.07774113e+08,  -1.02194430e+09,   
    5.59397878e+08,  -1.72970713e+08, 3.30317669e+07,  -3.98798031e+06,   
    2.97189690e+05,  -1.24980700e+04, 2.27220000e+02]) # See HERA Memo #27

HERA_OP_OPP = 2.175 # See HERA Memo #27
PAPER_OP_OPP = 2.35

# Cosmological conversions
def f2z(fq):
    '''Convert frequency (GHz) to redshift for 21cm line.'''
    return (F21 / fq - 1)
def z2f(z):
    '''Convert redshift to frequency (GHz) for 21cm line.'''
    return F21 / (z+1)
def dL_df(z, omega_m=0.266):
    '''[h^-1 Mpc]/GHz, from Furlanetto et al. (2006)'''
    return (1.7 / 0.1) * ((1+z) / 10.)**.5 * (omega_m/0.15)**-0.5 * 1e3
def dL_dth(z):
    '''[h^-1 Mpc]/radian, from Furlanetto et al. (2006)'''
    return 1.9 * (1./a.const.arcmin) * ((1+z) / 10.)**.2
def dk_deta(z):
    '''2pi * [h Mpc^-1] / [GHz^-1]'''
    return 2*n.pi / dL_df(z) 
def dk_du(z):
    '''2pi * [h Mpc^-1] / [wavelengths], valid for u >> 1.'''
    #return 2*n.pi / (dL_dth(z) * (n.pi / 2)) # this expression works only for u ~ 0.5
    return 2*n.pi / dL_dth(z) # from du = 1/dth, which derives from du = d(sin(th)) using the small-angle approx
def X2Y(z):
    '''[h^-3 Mpc^3] / [str * GHz] scalar conversion between observing and cosmological coordinates'''
    return dL_dth(z)**2 * dL_df(z)

# Reionization models
def k3pk_21cm(k):
    '''Return peak \Delta_{21}^2(k), in mK**2, approximation to Furlanetto et al. (2006)'''
    return n.where(k < 0.1, 1e4 * k**2, 1e2)
def dk3pk_21cm(k, dk3):
    '''Return peak 21cm pspec, in mK**2, integrated over dk**3 box size'''
    return k3pk_21cm(k) / (2*n.pi * k**3) * dk3
def Vhat2_21cm(umag, eta, B, fq0, bm_poly=PAPER_BEAM_POLY):
    '''Return \hat V_{21}^2, the (Jy*BW)^2 magnitude of the peak 21cm pspec.'''
    z0 = f2z(fq0)
    Omega = n.polyval(bm_poly, fq0)
    dk3 = Omega * B*1e9 / X2Y(z0)
    kmag = n.sqrt((umag*dk_du(z0))**2 + (eta*dk_deta(z0))**2)
    return dk3pk_21cm(kmag, dk3) / jy2T(fq0, bm_poly=bm_poly)**2
def V_21cm(fqs, umag, bm_poly=PAPER_BEAM_POLY):
    etas = n.fft.fftfreq(fqs.size, fqs[1]-fqs[0])
    B = fqs[-1] - fqs[0]
    fq0 = n.average(fqs)
    Vhat2 = Vhat2_21cm(umag, etas, B, fq0, bm_poly=bm_poly)
    return n.sqrt(Vhat2) * n.exp(2j*n.pi*n.random.uniform(0,2*n.pi,size=Vhat2.size))

def jy2T(f, bm_poly=PAPER_BEAM_POLY):
    '''Return [mK] / [Jy] for a beam size vs. frequency (in GHz) defined by the
    polynomial bm_poly.  Default is a poly fit to the PAPER primary beam.'''
    lam = a.const.c / (f * 1e9)
    bm = n.polyval(bm_poly, f)
    return 1e-23 * lam**2 / (2 * a.const.k * bm) * 1e3
def k3pk_from_Trms(list_of_Trms, list_of_Twgt, k=.3, fq=.150, B=.001, bm_poly=PAPER_BEAM_POLY):
    z = f2z(fq)
    bm = n.polyval(bm_poly, fq)
    scalar = X2Y(z) * bm * B * k**3 / (2*n.pi**2)
    Trms2_sum, Trms2_wgt = 0, 0
    if len(list_of_Trms) == 1: Trms2_sum,Trms2_wgt = n.abs(list_of_Trms[0])**2, n.abs(list_of_Twgt[0])**2
    else:
        for i,(Ta,Wa) in enumerate(zip(list_of_Trms, list_of_Twgt)):
            for Tb,Wb in zip(list_of_Trms[i+1:], list_of_Twgt[i+1:]):
                Trms2_sum += Ta * n.conj(Tb)
                Trms2_wgt += Wa * n.conj(Wb)
    return scalar * Trms2_sum / Trms2_wgt, Trms2_wgt
def k3pk_sense_vs_t(t, k=.3, fq=.150, B=.001, bm_poly=PAPER_BEAM_POLY, Tsys=500e3):
    Trms = Tsys / n.sqrt(2*(B*1e9)*t) # This is the correct equation for a single-pol, cross-correlation measurement
    #Trms = Tsys / n.sqrt((B*1e9)*t)
    return k3pk_from_Trms([Trms], [1.], k=k, fq=fq, B=B, bm_poly=bm_poly)[0]

# Misc helper functions
def f2eta(f):
    '''Convert an array of frequencies to an array of etas (freq^-1) 
    corresponding to the bins that an FFT generates.'''
    return n.fft.fftfreq(f.shape[-1], f[1]-f[0])

# Tools for angular power spectra
def umag2l(umag): return 2*n.pi*umag - .5
def l2umag(ell): return (ell + .5) / (2*n.pi)
def Cl_from_Trms(Trms, ell):
    return ell*(ell+1)/(2*n.pi) * Trms**2
sqrt2 = n.sqrt(2)
def circ(dim, r, thresh=.4):
    '''Generate a circle of specified radius (r) in pixel
    units.  Determines sub-pixel weighting using adaptive mesh refinement.
    Mesh refinement terminates at pixels whose side length is <= the specified
    threshold (thresh).'''
    x,y = n.indices((dim,dim), dtype=n.float)
    x -= dim/2 ; y -= dim/2
    return n.where(x**2 + y**2 > r**2, 0, 1.)
    rin,rout = int(r/sqrt2)-1, int(r)+1
    d1,d2,d3,d4 = dim/2-rout,dim/2-rin,dim/2+rin,dim/2+rout
    # If big circle, start as 1 and set a bounding box to 0.  
    # If small, start as 0 and set a bounded box to 1.
    if r > dim/2:
        rv = n.ones((dim,dim), dtype=n.float)
        rv[d1:d4,d1:d4] = 0
    else:
        rv = n.zeros((dim,dim), dtype=n.float)
        rv[d2:d3,d2:d3] = 1
    # Select 4 rects that contain boundary areas and evaluate them in detail
    for a1,a2,a3,a4 in ((d1,d2,d1,d4), (d3,d4,d1,d4),
            (d2,d3,d1,d2), (d2,d3,d3,d4)):
        x_, y_ = x[a1:a2,a3:a4], y[a1:a2,a3:a4]
        rs = n.sqrt(x_**2 + y_**2)
        # Get rough answer
        rv_ = (rs <= r).astype(n.float)
        # Fine-tune the answer
        brd = n.argwhere(n.abs(rs.flatten() - r) < 1 / sqrt2).flatten()
        rv_.flat[brd] = _circ(x_.flat[brd], y_.flat[brd], r, 1., thresh)
        # Set rectangle in the actual matrix
        rv[a1:a2,a3:a4] = rv_
    return rv
def _circ(x, y, r, p, thresh):
    # Subdivide into 9 pixels
    p /= 3.
    x0,x1,x2 = x, x+p, x-p
    y0,y1,y2 = y, y+p, y-p
    x = n.array([x0,x0,x0,x1,x1,x1,x2,x2,x2]).flatten()
    y = n.array([y0,y1,y2,y0,y1,y2,y0,y1,y2]).flatten()
    r2 = x**2 + y**2
    # Get the rough answer
    rv = (r2 <= r**2).astype(n.float) * p**2
    # Fine-tune the answer
    if p > thresh:
        brd = n.argwhere(n.abs(n.sqrt(r2) - r) < p / sqrt2).flatten()
        rv[brd] = _circ(x[brd], y[brd], r, p, thresh)
    rv.shape = (9, rv.size / 9)
    rv = rv.sum(axis=0)
    return rv
def ring(dim, r_inner, r_outer, thresh=.4):
    return circ(dim, r_outer, thresh=thresh) - circ(dim, r_inner, thresh=thresh)

def Trms_vs_fq(fqs, jy_spec, umag150=20., B=.008, cen_fqs=None, ntaps=3, 
        window='kaiser3', bm_poly=PAPER_BEAM_POLY, bm_fqs=None):
    if bm_fqs is None: bm_fqs = fqs
    dfq = fqs[1] - fqs[0]
    dCH = int(n.around(B / dfq))
    if cen_fqs is None: cen_fqs = n.arange(fqs[0]+dCH*dfq*ntaps/2, fqs[-1]-dCH*dfq, dCH*dfq)
    Tspec = jy_spec * jy2T(bm_fqs, bm_poly=bm_poly)
    Trms, ks = {}, {}
    for fq0 in cen_fqs:
        z = f2z(fq0)
        umag_fq0 = umag150 * (fq0 / .150)
        k_pr = dk_du(z) * umag_fq0
        ch0 = n.argmin(n.abs(fqs-fq0))
        ch1,ch2 = ch0-dCH/2, ch0+dCH/2
        _fqs = fqs[ch1:ch2]
        etas = f2eta(_fqs)
        k_pl = dk_deta(z) * etas
        _ks = n.sqrt(k_pr**2 + k_pl**2)
        V = Tspec[ch1-ntaps/2*(ch2-ch1):ch2+(ntaps-1)/2*(ch2-ch1)]
        if ntaps <= 1:
            w = a.dsp.gen_window(V.size, window=window)
            _Trms = n.fft.ifft(V*w)
        else:
            _Trms = pfb.pfb(V, taps=ntaps, window=window, fft=n.fft.ifft)
        # Trms has both the primary beam and bandwidth divided out, matching Trms in Parsons et al. (2012).
        Trms[fq0], ks[fq0] = _Trms, (_ks, k_pl, k_pr)
    return Trms, ks
    

def Trms2_vs_umag(uvs, bms, umag_px, uv_bm_area=2., umin=4., umax=200., logstep=.1):
    ubins = 10**n.arange(n.log10(umin), n.log10(umax), logstep)
    errs, Trms2, wgts = [], [], []
    ratio = n.abs(uvs)**2 / n.abs(bms)**2
    for u in ubins:
        r_px_inner = u * 10**(-logstep/2) / umag_px
        r_px_outer = u * 10**(logstep/2) / umag_px
        rng = ring(uvs.shape[0], r_px_inner, r_px_outer)
        uvs2_r = n.abs(uvs)**2 * rng
        bms2_r = n.abs(bms)**2 * rng
        wgts.append(bms2_r.sum())
        uvs2_r_avg = uvs2_r.sum() / wgts[-1]
        print '-'*20
        print u / umag_px, uvs2_r_avg,
        print n.sum(ratio*rng) / n.sum(rng)
        Trms2.append(uvs2_r_avg)
        # Estimate average variance of samples around the ring
        sig2_r = n.abs(uvs2_r - uvs2_r_avg * bms2_r)**2
        sig = n.sqrt(sig2_r.sum() / bms2_r.sum())
        # Estimate number of independent samples around ring
        #nsamples = rng.sum() * umag_px**2 / uv_bm_area / 2
        #err = n.sqrt(var) / n.sqrt(nsamples)
        errs.append(sig)
        print u, Trms2[-1], errs[-1]
    Trms2 = n.array(Trms2); errs = n.array(errs); wgts = n.array(wgts)
    #Cls = Cl_from_Trms(Trms, ells)
    #errs = Cl_from_Trms(errs, ells)
    return ubins, Trms2, errs, wgts
