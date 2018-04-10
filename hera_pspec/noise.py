import numpy as np
import os
from hera_pspec import conversions, pspecbeam
import copy


class Sense(object):

    def __init__(self, cosmo=None, beam=None):
        """
        """
        if cosmo is not None:
            self.add_cosmology(cosmo)

        if beam is not None:
            self.add_beam(beam)


    def add_cosmology(self, cosmo):
        """
        """


    def add_beam(self, beam):
        """
        """

    def calc_scalar(self, freqs, stokes, num_steps=5000, no_Bpp_ov_BpSq=True, little_h=True):
        """
        """


        self.subband = freqs
        self.stokes = stokes
        self.scalar = scalar

    def calc_P_N(self, k, Tsys, t_int, Ncoherent=1, Nincoherent=None, form='Pk', little_h=True, verbose=True):
        """
        Calculate the noise power spectrum via Eqn. (1) of Pober et al. 2014, ApJ 782, 66

        Parameters
        ----------
        """
        # assert scalar exists
        assert hasattr(self, 'scalar'), "self.scalar must exist before one can calculate a noise spectrum, see self.calc_scalar()"

        # assert form
        assert form in ('Pk', 'Dsq'), "form must be either 'Pk' or 'Dsq' for P(k) or Delta^2(k) respectively"

        # print feedback
        freqmin = self.subband.min() / 1e6
        freqmax = self.subband.max() / 1e6
        if verbose: print("Calculating {} noise curve given:\nTsys {}\nsubband {} -- {} MHz\nstokes {}\nlittle_h {}".format(Tsys, form, freqmin, freqmax, self.stokes, little_h))

        # construct prefactor
        P_N = self.scalar * Tsys**2

        # Multiply in effective integration time
        P_N /= (t_int * Ncoherent)

        # Mulitply in incoherent averaging
        if Nincoherent is not None:
            P_N /= np.sqrt(Nincoherent)

        # Convert to Delta Sq
        if form == 'Dsq':
            P_N *= k**3 / (2*np.pi**2)

        return np.ones_like(k, np.float) * P_N



