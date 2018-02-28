import numpy as np, pyuvdata, os, conversions
from scipy import integrate
from scipy.interpolate import interp1d

conversion = conversions.Cosmo_Conversions()

class PSpecBeam(object):

    def __init__(self):
        pass

    def power_beam_int(self, stokes='I'):
        pass

    def power_beam_sq_int(self, stokes='I'):
        pass

    def compute_pspec_scalar(self, lower_freq, upper_freq, num_freqs, stokes='I', taper='none', little_h=True, num_steps=10000):
        pass

class PSpecBeamGauss(PSpecBeam):

    def __init__(self, FWHM, beam_freqs):
        """
        Object to store a toy (frequency independent) Gaussian beam in a PspecBeam object

        Parameters
        ----------
        FWHM: float, in radians
                Full width half max of the beam

        beam_freqs: float, array-like
                Frequencies over which this Gaussian beam is to be created
        """
        self.FWHM = FWHM
        self.beam_freqs = beam_freqs

    def power_beam_int(self, stokes='I'):
        """
        Computes the integral of the beam over solid angle to give
        a beam area (in str). Uses analytic formula that the answer
        is 2 * pi * FWHM**2 / 8 ln 2.

        Trivially this returns an array (i.e., a function of frequency),
        but the results are frequency independent.

        Parameters
        ----------
        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'I', 'Q', 'U', 'V', although currently only 'I' is implemented
                Default: 'I'
        Returns
        -------
        primary beam area: float, array-like
        """
        if stokes != 'I':
            raise NotImplementedError()
        else:
            return np.ones_like(self.beam_freqs) * 2. * np.pi * self.FWHM**2 / (8. * np.log(2.))

    def power_beam_sq_int(self, stokes='I'):
        """
        Computes the integral of the beam**2 over solid angle to give
        a beam area (in str). Uses analytic formula that the answer
        is pi * FWHM**2 / 8 ln 2.

        Trivially this returns an array (i.e., a function of frequency),
        but the results are frequency independent.

        Parameters
        ----------
        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'I', 'Q', 'U', 'V', although currently only 'I' is implemented
                Default: 'I'
        Returns
        -------
        primary beam area: float, array-like
        """
        if stokes != 'I':
            raise NotImplementedError()
        else:
            return np.ones_like(self.beam_freqs) * np.pi * self.FWHM**2 / (8. * np.log(2.))

    def compute_pspec_scalar(self, lower_freq, upper_freq, num_freqs, stokes='I', taper='none', little_h=True, num_steps=10000):
        """
        Computes the scalar function to convert a power spectrum estimate
        in "telescope units" to cosmological units

        See arxiv:1304.4991 and HERA memo #27 for details.

        Currently this is only for Stokes I.

        Parameters
        ----------
        lower_freq: float
                Bottom edge of frequency band over which power spectrum is
                being estimated. Assumed to be in Hz.

        upper_freq: float
                Top edge of frequency band over which power spectrum is
                being estimated. Assumed to be in Hz.

        num_freqs: int
                Number of frequency channels in the data.

        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'I', 'Q', 'U', 'V', although currently only 'I' is implemented
                Default: 'I'

        taper : str, optional
                Whether a tapering function (e.g. Blackman-Harris) is being
                used in the power spectrum estimation.
                Default: none

        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        num_steps : int, optional
                Number of steps to use when interpolating primary beams for
                numerical integral
                Default: 10000

        Returns
        -------
        scalar: float
                [\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
                in h^-3 Mpc^3 or Mpc^3.
        """

        integration_freqs = np.linspace(lower_freq,upper_freq,num_steps)
        integration_freqs_MHz = integration_freqs / 10**6 # The interpolations are generally more stable in MHz
        redshifts = conversion.f2z(integration_freqs).flatten()
        X2Y = np.array(map(conversion.X2Y,redshifts))

        # Use linear interpolation to interpolate the frequency-dependent quantities
        # derived from the beam model to the same frequency grid as the power spectrum
        # estimation
        beam_model_freqs_MHz = self.beam_freqs/(1.*10**6)
        dOpp_over_Op2_fit = interp1d(beam_model_freqs_MHz, self.power_beam_sq_int(stokes)/self.power_beam_int(stokes)**2)
        dOpp_over_Op2 = dOpp_over_Op2_fit(integration_freqs_MHz)


        # Get B_pp = \int dnu taper^2 and Bp = \int dnu
        if taper == 'none':
            dBpp_over_BpSq = np.ones_like(integration_freqs)
        else:
            dBpp_over_BpSq = aipy.dsp.gen_window(num_freqs,taper)**2
        dBpp_over_BpSq /= (integration_freqs[-1] - integration_freqs[0])**2

        d_inv_scalar = dBpp_over_BpSq * dOpp_over_Op2 /  X2Y

        scalar = 1 / integrate.trapz(d_inv_scalar, integration_freqs)
        if little_h == True:
            scalar *= conversion.h**3

        return scalar



class PSpecBeamUV(PSpecBeam):

    def __init__(self, beam_fname):
        """
        Object to store the primary beam for a pspec observation.
        This is subclassed from PSpecBeam to take in a pyuvdata
        UVBeam object.

        Parameters
        ----------
        beam_fname: str
                Path to a pyuvdata UVBeam file
        """
        self.primary_beam = pyuvdata.UVBeam()
        self.primary_beam.read_beamfits(beam_fname)
        self.beam_freqs = self.primary_beam.freq_array[0]

    def power_beam_int(self, stokes='I'):
        """
        Computes the integral of the beam over solid angle to give
        a beam area (in str) as a function of frequency. Uses function
        in pyuvdata.

        Parameters
        ----------
        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'I', 'Q', 'U', 'V', although currently only 'I' is implemented
                Default: 'I'
        Returns
        -------
        primary beam area: float, array-like
        """
        return self.primary_beam.get_beam_area(stokes)

    def power_beam_sq_int(self, stokes='I'):
        """
        Computes the integral of the beam**2 over solid angle to give
        a beam**2 area (in str) as a function of frequency. Uses function
        in pyuvdata.

        Parameters
        ----------
        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'I', 'Q', 'U', 'V', although currently only 'I' is implemented
                Default: 'I'
        Returns
        -------
        primary beam area: float, array-like
        """
        return self.primary_beam.get_beam_sq_area(stokes)

    def compute_pspec_scalar(self, lower_freq, upper_freq, num_freqs, stokes='I', taper='none', little_h=True, num_steps=10000):
        """
        Computes the scalar function to convert a power spectrum estimate
        in "telescope units" to cosmological units

        See arxiv:1304.4991 and HERA memo #27 for details.

        Currently this is only for Stokes I.

        Parameters
        ----------
        lower_freq: float
                Bottom edge of frequency band over which power spectrum is
                being estimated. Assumed to be in Hz.

        upper_freq: float
                Top edge of frequency band over which power spectrum is
                being estimated. Assumed to be in Hz.

        num_freqs: int
                Number of frequency channels in the data.

        stokes: str, optional
                Which Stokes parameter's beam to compute the scalar for.
                'I', 'Q', 'U', 'V', although currently only 'I' is implemented
                Default: 'I'

        taper : str, optional
                Whether a tapering function (e.g. Blackman-Harris) is being
                used in the power spectrum estimation.
                Default: none

        little_h : boolean, optional
                Whether to have cosmological length units be h^-1 Mpc or Mpc
                Default: h^-1 Mpc

        num_steps : int, optional
                Number of steps to use when interpolating primary beams for
                numerical integral
                Default: 10000

        Returns
        -------
        scalar: float
                [\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
                in h^-3 Mpc^3 or Mpc^3.
        """

        integration_freqs = np.linspace(lower_freq,upper_freq,num_steps)
        integration_freqs_MHz = integration_freqs / 10**6 # The interpolations are generally more stable in MHz
        redshifts = conversion.f2z(integration_freqs).flatten()
        X2Y = np.array(map(conversion.X2Y,redshifts))

        # Use linear interpolation to interpolate the frequency-dependent quantities
        # derived from the beam model to the same frequency grid as the power spectrum
        # estimation
        beam_model_freqs_MHz = self.beam_freqs/(1.*10**6)
        dOpp_over_Op2_fit = interp1d(beam_model_freqs_MHz, self.power_beam_sq_int(stokes)/self.power_beam_int(stokes)**2)
        dOpp_over_Op2 = dOpp_over_Op2_fit(integration_freqs_MHz)

        # Get B_pp = \int dnu taper^2 and Bp = \int dnu
        if taper == 'none':
            dBpp_over_BpSq = np.ones_like(integration_freqs)
        else:
            dBpp_over_BpSq = aipy.dsp.gen_window(num_freqs,taper)**2
        dBpp_over_BpSq /= (integration_freqs[-1] - integration_freqs[0])**2

        d_inv_scalar = dBpp_over_BpSq * dOpp_over_Op2 /  X2Y

        scalar = 1 / integrate.trapz(d_inv_scalar, integration_freqs)
        if little_h == True:
            scalar *= conversion.h**3

        return scalar

