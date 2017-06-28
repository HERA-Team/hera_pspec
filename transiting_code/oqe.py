import numpy as np, aipy, random, md5

DELAY = False

def hash(w):
    return md5.md5(w.copy(order='C')).digest()

def noise(size):
    sig = 1./np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j*np.random.normal(scale=sig, size=size)

def cov(d1, w1, d2=None, w2=None):
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

def get_Q(mode, n_k, window='none'): #encodes the fourier transform from freq to delay
    if not DELAY:
        _m = np.zeros((n_k,), dtype=np.complex)
        _m[mode] = 1. #delta function at specific delay mode
        m = np.fft.fft(np.fft.ifftshift(_m)) * aipy.dsp.gen_window(n_k, window) #FFT it to go to freq
        Q = np.einsum('i,j', m, m.conj()) #dot it with its conjugate
        return Q
    else:
        # XXX need to have this depend on window
        Q = np.zeros_like(C)
        Q[mode,mode] = 1
        return Q

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

class DataSet:
    def __init__(self, dsets=None, wgts=None, lsts=None, conj=None, npzfile=None, lmin=None, lmode=None):
        self.x, self.w = {}, {}
        self.clear_cache()
        self.lmin = lmin
        self.lmode = lmode
        if not npzfile is None: self.from_npz(npzfile)
        elif not dsets is None: self.set_data(dsets, wgts=wgts, conj=conj)
    def flatten_data(self, data):
        if data is None: return None
        d = {}
        for k in data:
            for bl in data[k]:
                for pol in data[k][bl]:
                    key = (k, bl, pol)
                    d[key] = data[k][bl][pol]
        return d
    def set_data(self, dsets, wgts=None, conj=None):
        if type(dsets.values()[0]) == dict:
            dsets,wgts = self.flatten_data(dsets), self.flatten_data(wgts)
        self.x, self.w = {}, {}
        for k in dsets:
            self.x[k] = dsets[k].T
            try: self.w[k] = wgts[k].T
            except(TypeError): self.w[k] = np.ones_like(self.x[k])
            try:
                if conj[k[1]]: self.x[k] = np.conj(self.x[k])
            except(TypeError,KeyError): pass
    def add_data(self, dsets, wgts=None, conj=None):
        if type(dsets.values()[0]) == dict:
            dsets,wgts = self.flatten_data(dsets), self.flatten_data(wgts)
        for k in dsets:
            self.x[k] = dsets[k].T
            try: self.w[k] = wgts[k].T
            except(TypeError): self.w[k] = np.ones_like(self.x[k])
            try:
                if conj[k[1]]: self.x[k] = np.conj(self.x[k])
            except(TypeError,KeyError): pass
    def clear_cache(self, keys=None):
        # XXX right now clear_cache munges I if I != ones on the diagnal.  rethink I or clear_cache of I
        if keys is None:
            self._C, self._Ctrue, self._I, self._iC = {}, {}, {}, {}
            self._iCt = {}
        else:
            for k in keys:
                try: del(self._C[k])
                except(KeyError): pass
                try: del(self._Ctrue[k])
                except(KeyError): pass
                try: del(self._I[k])
                except(KeyError): pass
                try: del(self._iC[k])
                except(KeyError): pass
    def C(self, k, t=None):
        if not self._C.has_key(k): # defaults to true covariance matrix
            self.set_C({k:cov(self.x[k], self.w[k])})
            self._Ctrue[k] = self._C[k] # save computing this later, if we are making it now
        if t is None: return self._C[k]
        # If t is provided, Calculate C for the provided time index, including flagging
        w = self.w[k][:,t:t+1]
        return self._C[k] * (w * w.T)
    def set_C(self, d):
        self.clear_cache(d.keys())
        for k in d: self._C[k] = d[k]
    def Ctrue(self, k):
        if not self._Ctrue.has_key(k): self._Ctrue[k] = cov(self.x[k], self.w[k]) # must be actual covariance, no overwriting
        return self._Ctrue[k]
    def I(self, k):
        if not self._I.has_key(k):
            nchan = self.x[k].shape[0]
            self.set_I({k:np.identity(nchan)})
        return self._I[k]
    def set_I(self, d):
        #self.clear_cache(d.keys()) #XXX commented out or else it clears C's (can't set both C's and then I's if clear_cache is called for both)
        for k in d: self._I[k] = d[k]
    def iC(self, k, t=None, rcond=1e-12):
        if not self._iC.has_key(k):
            C = self.C(k)
            U,S,V = np.linalg.svd(C.conj()) # conj in advance of next step
            if self.lmin != None: S += self.lmin # ensure invertibility
            if self.lmode != None: 
                S += S[self.lmode-1]
            self.set_iC({k:np.einsum('ij,j,jk', V.T, 1./S, U.T)})
        if t is None: return self._iC[k]
        # If t is provided, Calculate iC for the provided time index, including flagging
        # XXX this does not respect manual setting of iC with ds.set_iC
        w = self.w[k][:,t:t+1]
        m = hash(w)
        if not self._iCt.has_key(k): self._iCt[k] = {}
        if not self._iCt[k].has_key(m):
            self._iCt[k][m] = np.linalg.pinv(self.C(k,t), rcond=rcond)
        return self._iCt[k][m]
    def set_iC(self, d):
        for k in d: self._iC[k] = d[k]
    def to_npz(self, filename):
        data = {}
        for k in self.x:
            sk = str(k)
            data['x '+sk] = self.x[k]
            try: data['C '+sk] = self.C[k]
            except(AttributeError): pass
            try: data['iC '+sk] = self.iC[k]
            except(AttributeError): pass
        for k in self.lsts:
            data['lst '+k] = self.lsts[k]
        np.savez(filename, **data)
    def from_npz(self, filename):
        npz = np.load(filename)
        self.x = {}
        for k in [f for f in npz.files if f.startswith('x')]: self.x[eval(k[2:])] = npz[k]        
        self.C = {}
        for k in [f for f in npz.files if f.startswith('C')]: self.C[eval(k[2:])] = npz[k]        
        self.iC = {}
        for k in [f for f in npz.files if f.startswith('iC')]: self.iC[eval(k[3:])] = npz[k]        
        self.lsts = {}
        for k in [f for f in npz.files if f.startswith('lst')]: self.lsts[k[4:]] = npz[k]        
        dsets, bls, pols = {}, {}, {}
        for k in self.x:
            dsets[k[0]] = None
            bls[k[1]] = None
            pols[k[2]] = None
        self.dsets = dsets.keys()
        self.bls = bls.keys()
        self.pols = pols.keys()
    def gen_bl_boots(self, nboots, ngps=5):
        _bls = {}
        for k in self.x: _bls[k[1]] = None
        for boot in xrange(nboots):
            bls = _bls.keys()[:]
            random.shuffle(bls)
            gps = [bls[i::ngps] for i in range(ngps)]
            gps = [[random.choice(gp) for bl in gp] for gp in gps]
            yield gps
        return
    def gen_gps(self, bls, ngps=5):
        random.shuffle(bls)
        gps = [bls[i::ngps] for i in range(ngps)]
        gps = [[random.choice(gp) for bl in gp] for gp in gps] #sample w/replacement inside each group
        return gps
    def group_data(self, keys, gps, use_cov=True): #XXX keys have format (k,bl,POL)
        # XXX avoid duplicate code for use_cov=True vs False (i.e. no separate dsC & dsI)
        sets = np.unique([key[0] for key in keys]) 
        POL = keys[0][2]
        nchan = self.x[keys[0]].shape[0]
        newkeys = []
        dsC_data, dsI_data = {},{}
        iCsum,iCxsum,Ixsum,Isum = {},{},{},{}
        for s in sets: #summing up data for each group and making new keys
            for gp in range(len(gps)):
                newkey = (s,gp)
                newkeys.append(newkey)
                iCsum[newkey] = sum([self.iC((s,bl,POL)) for bl in gps[gp]])
                iCxsum[newkey] = sum([np.dot(self.iC((s,bl,POL)),self.x[(s,bl,POL)]) for bl in gps[gp]])
                Isum[newkey] = sum([np.identity(nchan) for bl in gps[gp]])
                Ixsum[newkey] = sum([self.x[(s,bl,POL)] for bl in gps[gp]])
                dsC_data[newkey] = np.dot(np.linalg.inv(iCsum[newkey]),iCxsum[newkey]).T #finding effective summed up x based on iCsum and iCxsum
                dsI_data[newkey] = np.dot(np.linalg.inv(Isum[newkey]),Ixsum[newkey]).T #finding effective summed up x based on Isum and Ixsum
        dsC = DataSet(); dsC.add_data(dsets=dsC_data)
        dsI = DataSet(); dsI.set_data(dsets=dsI_data) #I has to be a separate dataset because it has different x's populated into it
        dsC.set_iC(iCsum) #override since if they're computed from x, they're incorrect
        dsI.set_I(Isum)
        if use_cov: return newkeys, dsC
        else: return newkeys, dsI
    def q_hat(self, k1, k2, use_cov=True, use_fft=True, cov_flagging=True):
        try:
            if self.x.has_key(k1): k1 = [k1]
        except(TypeError): pass
        try:
            if self.x.has_key(k2): k2 = [k2]
        except(TypeError): pass
        nchan = self.x[k1[0]].shape[0]
        if use_cov:
            if not cov_flagging:
                #iC1,iC2 = self.iC(k1), self.iC(k2)
                #iC1x, iC2x = np.dot(iC1, self.x[k1]), np.dot(iC2, self.x[k2])
                iC1x, iC2x = 0, 0
                for k1i in k1: iC1x += np.dot(self.iC(k1i), self.x[k1i])
                for k2i in k2: iC2x += np.dot(self.iC(k2i), self.x[k2i])
            else:
                # XXX make this work with k1,k2 being lists
                iCx = {}
                for k in (k1,k2):
                    iCx[k] = np.empty_like(self.x[k])
                    inds = {}
                    w = self.w[k]
                    ms = [hash(w[:,i]) for i in xrange(w.shape[1])]
                    for i,m in enumerate(ms): inds[m] = inds.get(m,[]) + [i]
                    iCxs = {}
                    for m in inds:
                        x = self.x[k][:,inds[m]]
                        #x = self.x[k].take(inds[m], axis=1)
                        iC = self.iC(k,inds[m][0])
                        iCx[k][:,inds[m]] = np.dot(iC,x)
                        #iCx[k].put(inds[m], np.dot(iC,x), axis=1)
                iC1x,iC2x = iCx[k1], iCx[k2]
        else:
            # XXX make this work with k1,k2 being lists
            #iC1x, iC2x = self.x[k1].copy(), self.x[k2].copy()
            iC1x, iC2x = 0, 0
            for k1i in k1: iC1x += np.dot(self.I(k1i), self.x[k1i])
            for k2i in k2: iC2x += np.dot(self.I(k2i), self.x[k2i])
            #iC1x, iC2x = np.dot(self.I(k1), self.x[k1]), np.dot(self.I(k2), self.x[k2])
        if use_fft:
            #iC1x, iC2x = np.dot(iC1, self.x[k1]), np.dot(iC2, self.x[k2])
            _iC1x, _iC2x = np.fft.fft(iC1x.conj(), axis=0), np.fft.fft(iC2x.conj(), axis=0)
            return np.conj(np.fft.fftshift(_iC1x,axes=0).conj() * np.fft.fftshift(_iC2x,axes=0)) # added conj around the whole thing because it was inconsistent with pspec_cov_v003 by a conjugation
        else: # slow, used to explicitly cross-check fft code
            # XXX make this work with k1,k2 being lists
            q = []
            for i in xrange(nchan):
                Q = get_Q(i,nchan)
                iCQiC = np.einsum('ab,bc,cd', iC1.T.conj(), Q, iC2) # C^-1 Q C^-1
                qi = np.sum(self.x[k1].conj() * np.dot(iCQiC,self.x[k2]), axis=0)
                q.append(qi)
            return np.array(q)
    def get_F(self, k1, k2, use_cov=True, cov_flagging=True):
        try:
            if self.x.has_key(k1): k1 = [k1]
        except(TypeError): pass
        try:
            if self.x.has_key(k2): k2 = [k2]
        except(TypeError): pass
        nchan = self.x[k1[0]].shape[0]
        if use_cov:
            if not cov_flagging:
                F = np.zeros((nchan,nchan), dtype=np.complex)
                #iC1,iC2 = self.iC(k1), self.iC(k2)
                iC1, iC2 = 0, 0
                for k1i in k1: iC1 += self.iC(k1i)
                for k2i in k2: iC2 += self.iC(k2i)
                if False: Ctrue1, Ctrue2 = self.Ctrue(k1), self.Ctrue(k2)
            else:
                # XXX make this work with k1,k2 being lists
                # This is for the "effective" matrix s.t. W=MF and p=Mq
                F = {}
                w1,w2 = self.w[k1], self.w[k2]
                m1s = [hash(w1[:,i]) for i in xrange(w1.shape[1])]
                m2s = [hash(w2[:,i]) for i in xrange(w2.shape[1])]
                for m1,m2 in zip(m1s,m2s): F[(k1,m1,k2,m2)] = None
                for k1,m1,k2,m2 in F.keys():
                #for m1 in self._iCt[k1]: # XXX not all m1/m2 pairs may exist in data
                #    for m2 in self._iCt[k2]:
                        F[(k1,m1,k2,m2)] = np.zeros((nchan,nchan), dtype=np.complex)
                        iCQ1,iCQ2 = {}, {}
                        for ch in xrange(nchan): # this loop is nchan^3
                            Q = get_Q(ch,nchan) 
                            iCQ1[ch] = np.dot(self._iCt[k1][m1],Q) #C^-1 Q # If ERROR: Compute q_hat first
                            iCQ2[ch] = np.dot(self._iCt[k2][m2],Q) #C^-1 Q
                        for i in xrange(nchan): # this loop goes as nchan^4
                            for j in xrange(nchan):
                                F[(k1,m1,k2,m2)][i,j] += np.einsum('ij,ji', iCQ1[i], iCQ2[j]) #C^-1 Q C^-1 Q 
                return F
        else:
            # XXX make this work with k1,k2 being lists
            #iC1 = np.linalg.inv(self.C(k1) * np.identity(nchan))
            #iC2 = np.linalg.inv(self.C(k2) * np.identity(nchan))
            iC1, iC2 = 0, 0
            for k1i in k1: iC1 += self.I(k1i)
            for k2i in k2: iC2 += self.I(k2i)
            if False: Ctrue1, Ctrue2 = self.I(k1), self.I(k2) # do this to get the effective F (see below)
            F = np.zeros((nchan,nchan), dtype=np.complex)
        #Ctrue1, Ctrue2 = self.Ctrue(k1), self.Ctrue(k2)
        if False: # This is for the "true" Fisher matrix
            CE1, CE2 = {}, {}
            for ch in xrange(nchan):
                Q = get_Q(ch,nchan)
                CE1[ch] = np.dot(Ctrue1, np.dot(iC1, np.dot(Q, iC2))) # C1 Cbar1^-1 Q Cbar2^-1
                CE2[ch] = np.dot(Ctrue2, np.dot(iC2, np.dot(Q, iC1))) # C2 Cbar2^-1 Q Cbar1^-1
                #CE1[ch] = np.einsum('ab,bc,cd,de', self.Ctrue(k1), iC1, Q, iC2) # slow
                #CE2[ch] = np.einsum('ab,bc,cd,de', self.Ctrue(k2), iC2, Q, iC1) # slow
            for i in xrange(nchan):
                for j in xrange(nchan):
                    F[i,j] += np.einsum('ij,ji', CE1[i], CE2[j]) # C E C E
        else: # This is for the "effective" matrix s.t. W=MF and p=Mq
            iCQ1,iCQ2 = {}, {}
            for ch in xrange(nchan): # this loop is nchan^3
                Q = get_Q(ch,nchan)
                iCQ1[ch] = np.dot(iC1,Q) #C^-1 Q
                iCQ2[ch] = np.dot(iC2,Q) #C^-1 Q
            for i in xrange(nchan): # this loop goes as nchan^4
                for j in xrange(nchan):
                    F[i,j] += np.einsum('ij,ji', iCQ1[i], iCQ2[j]) #C^-1 Q C^-1 Q 
        return F
    def get_MW(self, F, mode='F^-1'):
        if type(F) is dict: # recursive case for many F's at once
            M,W = {}, {}
            for key in F: M[key],W[key] = self.get_MW(F[key], mode=mode)
            return M,W
        modes = ['F^-1', 'F^-1/2', 'I', 'L^-1']; assert(mode in modes)
        if mode == 'F^-1':
            M = np.linalg.pinv(F, rcond=1e-12)
            #U,S,V = np.linalg.svd(F)
            #M = np.einsum('ij,j,jk', V.T, 1./S, U.T)
        elif mode == 'F^-1/2':
            U,S,V = np.linalg.svd(F)
            M = np.einsum('ij,j,jk', V.T, 1./np.sqrt(S), U.T)
        elif mode == 'I':
            M = np.identity(F.shape[0], dtype=F.dtype)
        else:
            #Cholesky decomposition to get M
            order = np.array([10,11,9,12,8,20,0,13,7,14,6,15,5,16,4,17,3,18,2,19,1]) # XXX needs generalizing
            iorder = np.argsort(order)
            F_o = np.take(np.take(F,order, axis=0), order, axis=1)
            L_o = np.linalg.cholesky(F_o)
            U,S,V = np.linalg.svd(L_o.conj())
            M_o = np.dot(np.transpose(V), np.dot(np.diag(1./S), np.transpose(U)))
            M = np.take(np.take(M_o,iorder, axis=0), iorder, axis=1)
        W = np.dot(M, F)
        norm  = W.sum(axis=-1); norm.shape += (1,)
        M /= norm; W = np.dot(M, F)
        return M,W
    def p_hat(self, M, q, scalar=1.):
        if type(M) is dict: # we have different M's for different times
            (k1,m1,k2,m2) = M.keys()[0]
            w1,w2 = self.w[k1], self.w[k2]
            m1s = [hash(w1[:,i]) for i in xrange(w1.shape[1])]
            m2s = [hash(w2[:,i]) for i in xrange(w2.shape[1])]
            inds = {}
            for i,(m1,m2) in enumerate(zip(m1s,m2s)):
                inds[(k1,m1,k2,m2)] = inds.get((k1,m1,k2,m2),[]) + [i]
            p = np.zeros_like(q)
            for key in inds:
                qi = q[:,inds[key]]
                #qi = q.take(inds[key], axis=1)
                p[:,inds[key]] = np.dot(M[key], qi) * scalar
                #p.put(inds[key], np.dot(M[key], qi) * scalar, axis=1)
            return p
        else: return np.dot(M, q) * scalar

'''
def oqe(dsets, conj, chans, ):
    nchan = chans.size
    bls_master = dsets.values()[0].keys()
    nbls = len(bls_master)
    Q = [get_Q(i,nchan) for i in xrange(nchan)]
    
    I,_I,_Ix = {},{},{}
    C,_C,_Cx = {},{},{}
    for k in dsets:
        I[k],_I[k],_Ix[k] = {},{},{}
        C[k],_C[k],_Cx[k] = {},{},{}
        for bl in x[k]:
            C[k][bl] = cov(x[k][bl])
            #C[k][bl] = covs[k][str(bl+(POL,))][120:141,120:141]
            #if conj[bl]: C[k][bl] = C[k][bl].conj()
            I[k][bl] = n.identity(C[k][bl].shape[0])
            U,S,V = n.linalg.svd(C[k][bl].conj()) #singular value decomposition
            _C[k][bl] = n.einsum('ij,j,jk', V.T, 1./S, U.T)
            _I[k][bl] = n.identity(_C[k][bl].shape[0])
            _Cx[k][bl] = n.dot(_C[k][bl], x[k][bl])
            _Ix[k][bl] = x[k][bl].copy()
    for boot in xrange(opts.nboot):
        print '%d / %d' % (boot+1,opts.nboot)
        bls = bls_master[:]
        if True: #shuffle and group baselines for bootstrapping
            if not SAMPLE_WITH_REPLACEMENT:
                random.shuffle(bls)
                bls = bls[:-5] # XXX
            else: #sample with replacement
                bls = [random.choice(bls) for bl in bls]
            gps = [bls[i::NGPS] for i in range(NGPS)]
            gps = [[random.choice(gp) for bl in gp] for gp in gps]
        else: #assign each baseline its own group
            gps = [bls[i::NGPS] for i in range(NGPS)]
        bls = [bl for gp in gps for bl in gp]
        #print '\n'.join([','.join(['%d_%d'%a.miriad.bl2ij(bl) for bl in gp]) for gp in gps])    
        _Iz,_Isum,_IsumQ = {},{},{}
        _Cz,_Csum,_CsumQ = {},{},{}
        print "   Getting C"
        for k in days:
            _Iz[k],_Isum[k],_IsumQ[k] = {},{},{}
            _Cz[k],_Csum[k],_CsumQ[k] = {},{},{}
            for i,gp in enumerate(gps): #sum things up over the groups
                _Iz[k][i] = sum([_Ix[k][bl] for bl in gp])
                _Cz[k][i] = sum([_Cx[k][bl] for bl in gp])
                _Isum[k][i] = sum([_I[k][bl] for bl in gp])
                _Csum[k][i] = sum([_C[k][bl] for bl in gp])
                _IsumQ[k][i] = {}
                _CsumQ[k][i] = {}
                if DELAY: #this is much faster
                    _Iz[k][i] = n.fft.fftshift(n.fft.ifft(window*_Iz[k][i], axis=0), axes=0)
                    _Cz[k][i] = n.fft.fftshift(n.fft.ifft(window*_Cz[k][i], axis=0), axes=0)
                    #XXX need to take fft of _Csum, _Isum here
                for ch in xrange(nchan): #XXX this loop makes computation go as nchan^3
                    _IsumQ[k][i][ch] = n.dot(_Isum[k][i], Q[ch])
                    _CsumQ[k][i][ch] = n.dot(_Csum[k][i], Q[ch]) #C^-1 Q
'''
