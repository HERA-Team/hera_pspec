import numpy as np
import os
from hera_pspec import conversions, pspecbeam
import copy
import ast
from collections import OrderedDict as odict


class Sense(object):
    """ Sense object for calculating thermal sensitivities """

    def __init__(self, cosmo=None, beam=None):
        """
        Sense object for calculating thermal sensitivities.

        Parameters
        ----------
        cosmo : hera_pspec.conversions.Cosmo_Conversions instance

        beam : hera_pspec.pspecbeam.PSpecBeam instance
        """
        if cosmo is not None:
            self.add_cosmology(cosmo)

        if beam is not None:
            self.add_beam(beam)

    def add_cosmology(self, cosmo):
        """
        Add a cosmological model to self.cosmo via an instance of hera_pspec.conversions.Cosmo_Conversions

        Parameters
        ----------
        cosmo : conversions.Cosmo_Conversions instance, or self.cosmo_params string, or dictionary
        """
        if isinstance(cosmo, (str, np.str)):
            cosmo = ast.literal_eval(cosmo)
        if isinstance(cosmo, (dict, odict)):
            cosmo = conversions.Cosmo_Conversions(**cosmo)
        self.cosmo = cosmo
        self.cosmo_params = str(self.cosmo.get_params())

    def add_beam(self, beam):
        """
        Add a pspecbeam.PSpecBeam object to self as self.beam

        Parameters
        ----------
        beam : pspecbeam.PSpecBeam instance
        """
        # ensure self.cosmo and beam.cosmo are consistent, if they both exist
        if hasattr(beam, 'cosmo'):
            if hasattr(self, 'cosmo'):
                # attach self.cosmo to beam if they both exist
                beam.cosmo = self.cosmo
            else:
                # attach beam.cosmo to self if self.cosmo doesn't exist
                self.cosmo = beam.cosmo
        else:
            if hasattr(self, 'cosmo'):
                # attach self.cosmo to beam if beam.cosmo doesn't exist.
                beam.cosmo = self.cosmo
            else:
                # neither beam nor self have cosmo, raise AssertionError
                raise AssertionError("neither self nor beam have a Cosmo_Conversions instance attached. "\
                                     "See self.add_cosmology().")

        self.beam = beam

    def calc_scalar(self, freqs, stokes, num_steps=5000, little_h=True):
        """
        Calculate noise power spectrum prefactor from Eqn. (1) of Pober et al. 2014, ApJ 782, 66,
        equal to 

        scalar = X2Y(z) * Omega_P^2 / Omega_PP

        Parameters
        ----------
        freqs : float ndarray, holds frequency bins of spectral window in Hz

        stokes : str, specification of (pseudo) Stokes polarization, or linear dipole polarization
            See pyuvdata.utils.polstr2num for options.

        num_steps : number of frequency bins to use in numerical integration of scalar

        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        Result
        ------
        self.scalar : float, cosmological and beam scalar prefactor

        self.subband : float ndarray, frequencies in spectral window used to calculate self.scalar

        self.stokes : str, stokes polarization used to calculate self.scalar
        """
        # parse stokes
        if stokes == 'I': stokes = 'pseudo_I'
        self.scalar = self.beam.compute_pspec_scalar(freqs.min(), freqs.max(), len(freqs), num_steps=num_steps, 
                                                     stokes=stokes, little_h=little_h, noise_scalar=True)
        self.subband = freqs
        self.stokes = stokes

    def calc_P_N(self, k, Tsys, t_int, Ncoherent=1, Nincoherent=None, form='Pk', little_h=True):
        """
        Calculate the noise power spectrum via Eqn. (1) of Pober et al. 2014, ApJ 782, 66


        Parameters
        ----------
        k : float ndarray, cosmological wave-vectors in h Mpc^-1

        Tsys : float, System temperature in Kelvin

        t_int : float, integration time of power spectra in seconds

        Ncoherent : int, number of coherent averages of visibility data with integration time t_int

        Nincoherent : int, number of incoherent averages of pspectra (i.e. after squaring)

        form : str, power spectra form 'Pk' for P(k) and 'Dsq' for Delta^2(k)

        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        Returns (P_N)
        -------
        P_N : estimated noise power spectrum in units of mK^2 * h^-3 Mpc^3
        if form == 'Dsq', then units include a factor of h^3 k^3 / (2pi^2)
        """
        # assert scalar exists
        assert hasattr(self, 'scalar'), "self.scalar must exist before one can calculate a noise spectrum, see self.calc_scalar()"

        # assert form
        assert form in ('Pk', 'Dsq'), "form must be either 'Pk' or 'Dsq' for P(k) or Delta^2(k) respectively"

        # convert to mK
        Tsys *= 1e3

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


