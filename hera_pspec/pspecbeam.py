import numpy as np
import pyuvdata
import os
from hera_pspec import conversions
from scipy import integrate
from scipy.interpolate import interp1d
import aipy


def _compute_pspec_scalar(cosmo, beam_freqs, omega_ratio, pspec_freqs, num_steps=5000,
                          stokes='pseudo_I', taper='none', little_h=True, noise_scalar=False):
    """
    This is not to be used by the novice user to calculate a pspec scalar.
    Instead, look at the PSpecBeamUV and PSpecBeamGauss classes.

    Computes the scalar function to convert a power spectrum estimate
    in "telescope units" to cosmological units

    See arxiv:1304.4991 and HERA memo #27 for details.

    Currently, only the "pseudo Stokes I" beam is supported.
    See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
    or arxiv:1502.05072 for details.

    Parameters
    ----------
    cosmo : hera_pspec.conversions.Cosmo_Conversions instance
            Instance of the cosmological conversion object
            conversions.Cosmo_Conversions()

    beam_freqs : array of floats
            Frequency of beam integrals in omega_ratio in units of Hz.

    omega_ratio : array of floats
            Ratio of the integrated squared-beam power over the square of the integrated beam power
            for each frequency in beam_freqs. i.e. Omega_pp(nu) / Omega_p(nu)^2

    pspec_freqs : array of floats
            Array of frequencies over which power spectrum is estimated in Hz.

    num_steps : int, optional
            Number of steps to use when interpolating primary beams for
            numerical integral

    stokes: str, optional
            Which Stokes parameter's beam to compute the scalar for.
            'pseudo_I', 'pseudo_Q', 'pseudo_U', 'pseudo_V', although currently only 'pseudo_I' is implemented
            Default: 'pseudo_I'

    taper : str, optional
            Whether a tapering function (e.g. Blackman-Harris) is being
            used in the power spectrum estimation.
            Default: none

    little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc or Mpc
            Value of h is obtained from cosmo object stored in pspecbeam
            Default: h^-1 Mpc

    noise_scalar : boolean, optional
            Whether to calculate power spectrum scalar, or noise power scalar. The noise power
            scalar only differs in that the Bpp_over_BpSq term just because 1_over_Bp.
            See Pober et al. 2014, ApJ 782, 66

    Returns
    -------
    scalar: float
            [\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
            in h^-3 Mpc^3 or Mpc^3.
    """
    # get integration freqs, redshift and cosmological scalars
    df = np.median(np.diff(pspec_freqs))
    integration_freqs = np.linspace(pspec_freqs.min(), pspec_freqs.min()+df*len(pspec_freqs), num_steps, endpoint=True, dtype=np.float)
    integration_freqs_MHz = integration_freqs / 1e6  # The interpolations are generally more stable in MHz
    redshifts = cosmo.f2z(integration_freqs).flatten()
    X2Y = np.array(map(lambda z: cosmo.X2Y(z, little_h=little_h), redshifts))

    # Use linear interpolation to interpolate the frequency-dependent quantities
    # derived from the beam model to the same frequency grid as the power spectrum
    # estimation
    beam_model_freqs_MHz = beam_freqs / 1e6
    dOpp_over_Op2_fit = interp1d(beam_model_freqs_MHz, omega_ratio, kind='quadratic', fill_value='extrapolate')
    dOpp_over_Op2 = dOpp_over_Op2_fit(integration_freqs_MHz)

    # Get B_pp = \int dnu taper^2 and Bp = \int dnu
    if taper == 'none':
        dBpp_over_BpSq = np.ones_like(integration_freqs, np.float)
    else:
        dBpp_over_BpSq = aipy.dsp.gen_window(len(pspec_freqs), taper)**2
        dBpp_over_BpSq = interp1d(pspec_freqs, dBpp_over_BpSq, kind='nearest', fill_value='extrapolate')(integration_freqs)
    dBpp_over_BpSq /= (integration_freqs[-1] - integration_freqs[0])**2

    # Keep dBpp_over_BpSq term or not
    if noise_scalar:
        dBpp_over_BpSq = 1/(integration_freqs[-1] - integration_freqs[0])

    # integrate to get scalar
    d_inv_scalar = dBpp_over_BpSq * dOpp_over_Op2 / X2Y
    scalar = 1.0 / integrate.trapz(d_inv_scalar, x=integration_freqs)

    return scalar


class PSpecBeamBase(object):

    def __init__(self, cosmo=None):
        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = conversions.Cosmo_Conversions()

    def compute_pspec_scalar(self, lower_freq, upper_freq, num_freqs, num_steps=5000, stokes='pseudo_I',
                             taper='none', little_h=True, noise_scalar=False):
        """
        Computes the scalar function to convert a power spectrum estimate
        in "telescope units" to cosmological units

        See arxiv:1304.4991 and HERA memo #27 for details.

        Currently, only the "pseudo Stokes I" beam is supported.
        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        lower_freq: float
                Bottom edge of frequency band over which power spectrum is
                being estimated. Assumed to be in Hz.

        upper_freq: float
                Top edge of frequency band over which power spectrum is
                being estimated. Assumed to be in Hz.

        num_freqs : int, optional
                Number of frequencies used in estimating power spectrum.

        num_steps : int, optional
                Number of steps to use when interpolating primary beams for
                numerical integral
                Default: 10000

        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'pseudo_I', 'pseudo_Q', 'pseudo_U', 'pseudo_V', although currently only 'pseudo_I' is implemented
                Default: 'pseudo_I'

        taper : str, optional
                Whether a tapering function (e.g. Blackman-Harris) is being
                used in the power spectrum estimation.
                Default: none

        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Value of h is obtained from cosmo object stored in pspecbeam
                Default: h^-1 Mpc

        noise_scalar : boolean, optional
                Whether to calculate power spectrum scalar, or noise power scalar. The noise power
                scalar only differs in that the Bpp_over_BpSq term just because 1_over_Bp.
                See Pober et al. 2014, ApJ 782, 66

        Returns
        -------
        scalar: float
                [\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
                in h^-3 Mpc^3 or Mpc^3.
        """
        # get pspec_Freqs
        pspec_freqs = np.linspace(lower_freq, upper_freq, num_freqs, endpoint=False)

        # Get omega_ratio
        omega_ratio = self.power_beam_sq_int(stokes) / self.power_beam_int(stokes)**2

        # Get scalar
        scalar = _compute_pspec_scalar(self.cosmo, self.beam_freqs, omega_ratio, pspec_freqs,
                                       num_steps=num_steps, stokes=stokes, taper=taper, little_h=little_h,
                                       noise_scalar=noise_scalar)

        return scalar

    def Jy_to_mK(self, freqs, stokes='pseudo_I'):
        """
        Return the multiplicative factor, M [mK / Jy], to convert a visibility from Jy -> mK,

        M = 1e3 * 1e-23 * c^2 / [2 * k_b * nu^2 * Omega_p(nu)]

        where k_b is boltzmann constant, c is speed of light, nu is frequency 
        and Omega_p is the integral of the unitless beam-response (steradians),
        and the 1e3 is the conversion from K -> mK and the 1e-23 is the conversion 
        from Jy to cgs.

        Parameters
        ----------
        freqs : float ndarray, contains frequencies to evaluate conversion factor [Hz]

        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'pseudo_I', 'pseudo_Q', 'pseudo_U', 'pseudo_V', although currently only 'pseudo_I' is implemented
                Default: 'pseudo_I'

        Returns
        -------
        M : float ndarray, contains Jy -> mK factor at each frequency
        """
        if isinstance(freqs, (np.float, float)):
            freqs = np.array([freqs])
        elif not isinstance(freqs, np.ndarray):
            raise TypeError("freqs must be fed as a float ndarray")
        elif isinstance(freqs, np.ndarray) and freqs.dtype not in (float, np.float, np.float64):
            raise TypeError("freqs must be fed as a float ndarray")
        if np.min(freqs) < self.beam_freqs.min(): print "Warning: min freq {} < self.beam_freqs.min(), extrapolating...".format(np.min(freqs))
        if np.max(freqs) > self.beam_freqs.max(): print "Warning: max freq {} > self.beam_freqs.max(), extrapolating...".format(np.max(freqs))

        Op = interp1d(self.beam_freqs/1e6, self.power_beam_int(stokes=stokes), kind='quadratic', fill_value='extrapolate')(freqs/1e6)

        return 1e-20 * conversions.cgs_units.c**2 / (2 * conversions.cgs_units.kb * freqs**2 * Op)


class PSpecBeamGauss(PSpecBeamBase):

    def __init__(self, fwhm, beam_freqs, cosmo=None):
        """
        Object to store a toy (frequency independent) Gaussian beam in a PspecBeamBase object

        Parameters
        ----------
        fwhm: float, in radians
                Full width half max of the beam

        beam_freqs: float, array-like
                Frequencies over which this Gaussian beam is to be created. Units assumed to be Hz.
        """
        self.fwhm = fwhm
        self.beam_freqs = beam_freqs
        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = conversions.Cosmo_Conversions()

    def power_beam_int(self, stokes='pseudo_I'):
        """
        Computes the integral of the beam over solid angle to give
        a beam area (in sr). Uses analytic formula that the answer
        is 2 * pi * fwhm**2 / 8 ln 2.

        Trivially this returns an array (i.e., a function of frequency),
        but the results are frequency independent.

        Currently, only the "pseudo Stokes I" beam is supported.
        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'pseudo_I', 'pseudo_Q', 'pseudo_U', 'pseudo_V', although currently only 'pseudo_I' is implemented
                Default: 'pseudo_I'
        Returns
        -------
        primary beam area: float, array-like
        """
        if stokes != 'pseudo_I':
            raise NotImplementedError("Only stokes='pseudo_I' is currently supported.")
        else:
            return np.ones_like(self.beam_freqs) * 2. * np.pi * self.fwhm**2 / (8. * np.log(2.))

    def power_beam_sq_int(self, stokes='pseudo_I'):
        """
        Computes the integral of the beam**2 over solid angle to give
        a beam area (in str). Uses analytic formula that the answer
        is pi * fwhm**2 / 8 ln 2.

        Trivially this returns an array (i.e., a function of frequency),
        but the results are frequency independent.

        Currently, only the "pseudo Stokes I" beam is supported.
        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'pseudo_I', 'pseudo_Q', 'pseudo_U', 'pseudo_V', although currently only 'pseudo_I' is implemented
                Default: 'pseudo_I'
        Returns
        -------
        primary beam area: float, array-like
        """
        if stokes != 'pseudo_I':
            raise NotImplementedError("Only stokes='pseudo_I' is currently supported.")
        else:
            return np.ones_like(self.beam_freqs) * np.pi * self.fwhm**2 / (8. * np.log(2.))


class PSpecBeamUV(PSpecBeamBase):

    def __init__(self, beam_fname, cosmo=None):
        """
        Object to store the primary beam for a pspec observation.
        This is subclassed from PSpecBeamBase to take in a pyuvdata
        UVBeam object.

        Parameters
        ----------
        beam_fname: str
                Path to a pyuvdata UVBeam file
        """
        self.primary_beam = pyuvdata.UVBeam()
        self.primary_beam.read_beamfits(beam_fname)

        self.beam_freqs = self.primary_beam.freq_array[0]
        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = conversions.Cosmo_Conversions()

    def power_beam_int(self, stokes='pseudo_I'):
        """
        Computes the integral of the beam over solid angle to give
        a beam area (in str) as a function of frequency. Uses function
        in pyuvdata.

        Currently, only the "pseudo Stokes I" beam is supported.
        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'pseudo_I', 'pseudo_Q', 'pseudo_U', 'pseudo_V', although currently only 'pseudo_I' is implemented
                Default: 'pseudo_I'
        Returns
        -------
        primary beam area: float, array-like
        """
        if hasattr(self.primary_beam, 'get_beam_area'):
            return self.primary_beam.get_beam_area(stokes)
        else:
            raise NotImplementedError("Outdated version of pyuvdata.")

    def power_beam_sq_int(self, stokes='pseudo_I'):
        """
        Computes the integral of the beam**2 over solid angle to give
        a beam**2 area (in str) as a function of frequency. Uses function
        in pyuvdata.

        Currently, only the "pseudo Stokes I" beam is supported.
        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'pseudo_I', 'pseudo_Q', 'pseudo_U', 'pseudo_V', although currently only 'pseudo_I' is implemented
                Default: 'pseudo_I'
        Returns
        -------
        primary beam area: float, array-like
        """
        if hasattr(self.primary_beam, 'get_beam_area'):
            return self.primary_beam.get_beam_sq_area(stokes)
        else:
            raise NotImplementedError("Outdated version of pyuvdata.")
