import numpy as np
import os
import copy
import ast
from collections import OrderedDict as odict

from . import conversions, pspecbeam


def calc_P_N(scalar, Tsys, t_int, Ncoherent=1, Nincoherent=None, form='Pk', k=None, component='real'):
    """
    Calculate the noise power spectrum via Eqn. (22) of Cheng et al. 2018 for a specified
    component of the power spectrum.

    The noise power spectrum is written as 

    P_N = scalar * (Tsys * 1e3)^2 / (t_int * Ncoherent) / sqrt(Nincoherent)

    where scalar is a nomalization given by the cosmological model and beam response, i.e. X2Y * Omega_eff
    Tsys is the system temp in Kelvin, t_int is the integration time of the underlying data [sec], 
    Ncoherent is the number of coherent averages before forming power spectra, and Nincoherent is the 
    number of incoherent averages after squaring. If component is 'real' or 'imag' an additional factor
    of 1/sqrt(2) is multiplied.
    For an ensemble of power spectra with the same Tsys, this estimate should equal their RMS value.

    Parameters
    ----------
    scalar : float, Power spectrum normalization factor: X2Y(z) * Omega_P^2 / Omega_PP
    Tsys : float, System temperature in Kelvin
    t_int : float, integration time of power spectra in seconds
    Ncoherent : int, number of coherent averages of visibility data with integration time t_int
        Total integration time is t_int * Ncoherent
    Nincoherent : int, number of incoherent averages of pspectra (i.e. after squaring).
    form : str, power spectra form 'Pk' for P(k) and 'DelSq' for Delta^2(k)
    k : float ndarray, cosmological wave-vectors in h Mpc^-1, only needed if form == 'DelSq'
    component : str, options=['real', 'imag', 'abs']
        If component is real or imag, divide by an extra factor of sqrt(2)

    Returns (P_N)
    -------
    P_N : estimated noise power spectrum in units of mK^2 * h^-3 Mpc^3
    if form == 'DelSq', then units include a factor of h^3 k^3 / (2pi^2)
    """
    # assert form
    assert form in ('Pk', 'DelSq'), "form must be either 'Pk' or 'DelSq' for P(k) or Delta^2(k) respectively"
    assert component in ['abs', 'real', 'imag'], "component must be one of 'real', 'imag', 'abs'"

    # construct prefactor in mK^2
    P_N = scalar * (Tsys * 1e3)**2

    # Multiply in effective integration time
    P_N /= (t_int * Ncoherent)

    # Mulitply in incoherent averaging
    if Nincoherent is not None:
        P_N /= np.sqrt(Nincoherent)

    # parse component
    if component in ['real', 'imag']:
        P_N /= np.sqrt(2)

    # Convert to Delta Sq
    if form == 'DelSq':
        assert k is not None, "if form == 'DelSq' then k must be fed"
        P_N = P_N * k**3 / (2*np.pi**2)

    return P_N


class Sensitivity(object):
    """ Power spectrum thermal sensitivity calculator """

    def __init__(self, cosmo=None, beam=None):
        """
        Object for power spectrum thermal sensitivity calculations.

        Parameters
        ----------
        cosmo : hera_pspec.conversions.Cosmo_Conversions instance

        beam : hera_pspec.pspecbeam.PSpecBeam instance
        """
        if cosmo is not None:
            self.set_cosmology(cosmo)

        if beam is not None:
            self.set_beam(beam)

    def set_cosmology(self, cosmo):
        """
        Set a cosmological model to self.cosmo via an instance of hera_pspec.conversions.Cosmo_Conversions

        Parameters
        ----------
        cosmo : conversions.Cosmo_Conversions instance, or self.cosmo_params string, or dictionary
        """
        if isinstance(cosmo, str):
            cosmo = ast.literal_eval(cosmo)
        if isinstance(cosmo, (dict, odict)):
            cosmo = conversions.Cosmo_Conversions(**cosmo)
        self.cosmo = cosmo
        self.cosmo_params = str(self.cosmo.get_params())

    def set_beam(self, beam):
        """
        Set a pspecbeam.PSpecBeam object to self as self.beam

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
                                     "See self.set_cosmology().")

        self.beam = beam

    def calc_scalar(self, freqs, pol, num_steps=5000, little_h=True):
        """
        Calculate noise power spectrum prefactor from Eqn. (1) of Pober et al. 2014, ApJ 782, 66,
        equal to 

        scalar = X2Y(z) * Omega_P^2 / Omega_PP

        Parameters
        ----------
        freqs : float ndarray, holds frequency bins of spectral window in Hz

        pol : str, specification of polarization to calculate scalar for
            See pyuvdata.utils.polstr2num for options.

        num_steps : number of frequency bins to use in numerical integration of scalar

        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        Result
        ------
        self.scalar : float, cosmological and beam scalar prefactor

        self.subband : float ndarray, frequencies in spectral window used to calculate self.scalar

        self.pol : str, polarization used to calculate self.scalar
        """
        # compute scalar
        self.scalar = self.beam.compute_pspec_scalar(freqs.min(), freqs.max(), len(freqs), num_steps=num_steps, 
                                                     pol=pol, little_h=little_h, noise_scalar=True)
        self.subband = freqs
        self.pol = pol

    def calc_P_N(self, Tsys, t_int, Ncoherent=1, Nincoherent=None, form='Pk', k=None, component='real'):
        """
        Calculate the noise power spectrum via Eqn. (22) of Cheng et al. 2018 for a specified
        component of the power spectrum.

        The noise power spectrum is written as 

        P_N = scalar * (Tsys * 1e3)^2 / (t_int * Ncoherent) / sqrt(Nincoherent)

        where scalar is a nomalization given by the cosmological model and beam response, i.e. X2Y * Omega_eff
        Tsys is the system temp in Kelvin, t_int is the integration time of the underlying data [sec], 
        Ncoherent is the number of coherent averages before forming power spectra, and Nincoherent is the 
        number of incoherent averages after squaring. If component is 'real' or 'imag' a factor of 1/sqrt(2)
        is multiplied.

        Parameters
        ----------
        scalar : float, Power spectrum normalization factor: X2Y(z) * Omega_P^2 / Omega_PP
        Tsys : float, System temperature in Kelvin
        t_int : float, integration time of power spectra in seconds
        Ncoherent : int, number of coherent averages of visibility data with integration time t_int
            Total integration time is t_int * Ncoherent
        Nincoherent : int, number of incoherent averages of pspectra (i.e. after squaring).
        form : str, power spectra form 'Pk' for P(k) and 'DelSq' for Delta^2(k)
        k : float ndarray, cosmological wave-vectors in h Mpc^-1, only needed if form == 'DelSq'
        component : str, options=['real', 'imag', 'abs']
            If component is real or imag, divide by an extra factor of sqrt(2)

        Returns (P_N)
        -------
        P_N : estimated noise power spectrum in units of mK^2 * h^-3 Mpc^3
        if form == 'DelSq', then units include a factor of h^3 k^3 / (2pi^2)
        """
        # assert scalar exists
        assert hasattr(self, 'scalar'), "self.scalar must exist before one can calculate a noise spectrum, see self.calc_scalar()"

        # assert form
        assert form in ('Pk', 'DelSq'), "form must be either 'Pk' or 'DelSq' for P(k) or Delta^2(k) respectively"

        # calculate P_N
        P_N = calc_P_N(self.scalar, Tsys, t_int, Ncoherent=Ncoherent, Nincoherent=Nincoherent, form=form, 
                       k=k, component=component)

        return P_N



