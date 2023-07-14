import numpy as np
import os
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from pyuvdata import UVBeam, utils as uvutils
import uvtools.dspec as dspec
from collections import OrderedDict as odict


from . import conversions as conversions, uvpspec_utils as uvputils


def _compute_pspec_scalar(cosmo, beam_freqs, omega_ratio, pspec_freqs,
                          num_steps=5000, taper='none', little_h=True,
                          noise_scalar=False, exact_norm=False):
    """
    This is not to be used by the novice user to calculate a pspec scalar.
    Instead, look at the PSpecBeamUV and PSpecBeamGauss classes.

    Computes the scalar function to convert a power spectrum estimate
    in "telescope units" to cosmological units

    See arxiv:1304.4991 and HERA memo #27 for details.

    Parameters
    ----------
    cosmo : conversions.Cosmo_Conversions instance
        Instance of the cosmological conversion object.

    beam_freqs : array of floats
        Frequency of beam integrals in omega_ratio in units of Hz.

    omega_ratio : array of floats
        Ratio of the integrated squared-beam power over the square of the
        integrated beam power for each frequency in beam_freqs.
        i.e. Omega_pp(nu) / Omega_p(nu)^2

    pspec_freqs : array of floats
        Array of frequencies over which power spectrum is estimated in Hz.

    num_steps : int, optional
        Number of steps to use when interpolating primary beams for numerical
        integral. Default: 5000.

    taper : str, optional
        Whether a tapering function (e.g. Blackman-Harris) is being used in the
        power spectrum estimation. Default: 'none'.

    little_h : boolean, optional
        Whether to have cosmological length units be h^-1 Mpc or Mpc. Value of
        h is obtained from cosmo object stored in pspecbeam. Default: h^-1 Mpc.

    noise_scalar : boolean, optional
        Whether to calculate power spectrum scalar, or noise power scalar. The
        noise power scalar only differs in that the Bpp_over_BpSq term turns
        into 1_over_Bp. See Pober et al. 2014, ApJ 782, 66, and Parsons HERA
        Memo #27. Default: False.
    exact_norm : boolean, optional
        returns only X2Y for scalar if True, else uses the existing framework
        involving antenna beam and spectral tapering factors. Default: False.

    Returns
    -------
    scalar: float
        [\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
        Units: h^-3 Mpc^3 or Mpc^3.
    """
    # Get integration freqs
    df = np.median(np.diff(pspec_freqs))
    integration_freqs = np.linspace(pspec_freqs.min(),
                                    pspec_freqs.min() + df*len(pspec_freqs),
                                    num_steps, endpoint=True, dtype=float)

    # The interpolations are generally more stable in MHz
    integration_freqs_MHz = integration_freqs / 1e6

    # Get redshifts and cosmological functions
    redshifts = cosmo.f2z(integration_freqs).flatten()
    X2Y = np.array([cosmo.X2Y(z, little_h=little_h) for z in redshifts])

    if exact_norm: #Beam and spectral tapering are already taken into account in normalization. We only use averaged X2Y
        scalar = integrate.trapz(X2Y, x=integration_freqs)/(np.abs(integration_freqs[-1]-integration_freqs[0]))
        return scalar

    # Use linear interpolation to interpolate the frequency-dependent
    # quantities derived from the beam model to the same frequency grid as the
    # power spectrum estimation
    beam_model_freqs_MHz = beam_freqs / 1e6
    dOpp_over_Op2_fit = interp1d(beam_model_freqs_MHz, omega_ratio,
                                 kind='quadratic', fill_value='extrapolate')
    dOpp_over_Op2 = dOpp_over_Op2_fit(integration_freqs_MHz)

    # Get B_pp = \int dnu taper^2 and Bp = \int dnu
    if taper == 'none':
        dBpp_over_BpSq = np.ones_like(integration_freqs, float)
    else:
        dBpp_over_BpSq = dspec.gen_window(taper, len(pspec_freqs))**2.
        dBpp_over_BpSq = interp1d(pspec_freqs, dBpp_over_BpSq, kind='nearest',
                                  fill_value='extrapolate')(integration_freqs)
    dBpp_over_BpSq /= (integration_freqs[-1] - integration_freqs[0])**2.

    # Keep dBpp_over_BpSq term or not
    if noise_scalar:
        dBpp_over_BpSq = 1. / (integration_freqs[-1] - integration_freqs[0])

    # Integrate to get scalar
    d_inv_scalar = dBpp_over_BpSq * dOpp_over_Op2 / X2Y
    scalar = 1. / integrate.trapz(d_inv_scalar, x=integration_freqs)
    return scalar


class PSpecBeamBase(object):

    def __init__(self, cosmo=None):
        """
        Base class for PSpecBeam objects. Provides compute_pspec_scalar()
        method to integrate over and interpolate beam solid angles, and
        Jy_to_mK() method to convert units.

        Parameters
        ----------
        cosmo : conversions.Cosmo_Conversions object, optional
            Cosmology object. Uses the default cosmology object if not
            specified. Default: None.
        """
        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = conversions.Cosmo_Conversions()

    def compute_pspec_scalar(self, lower_freq, upper_freq, num_freqs,
                             num_steps=5000, pol='pI', taper='none',
                             little_h=True, noise_scalar=False, exact_norm=False):
        """
        Computes the scalar function to convert a power spectrum estimate
        in "telescope units" to cosmological units

        See arxiv:1304.4991 and HERA memo #27 for details.

        Currently, only the "pI", "XX" and "YY" polarization beams are supported.
        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        lower_freq : float
            Bottom edge of frequency band over which power spectrum is being
            estimated. Assumed to be in Hz.

        upper_freq : float
            Top edge of frequency band over which power spectrum is being
            estimated. Assumed to be in Hz.

        num_freqs : int, optional
            Number of frequencies used in estimating power spectrum.

        num_steps : int, optional
            Number of steps to use when interpolating primary beams for
            numerical integral. Default: 5000.

        pol: str, optional
                Which polarization to compute the beam scalar for.
                'pI', 'pQ', 'pU', 'pV', 'XX', 'YY', 'XY', 'YX'
                Default: 'pI'

        taper : str, optional
            Whether a tapering function (e.g. Blackman-Harris) is being used in
            the power spectrum estimation. Default: none.

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc or Mpc. Value
            of h is obtained from cosmo object stored in pspecbeam.
            Default: h^-1 Mpc

        noise_scalar : boolean, optional
            Whether to calculate power spectrum scalar, or noise power scalar.
            The noise power scalar only differs in that the Bpp_over_BpSq term
            just because 1_over_Bp. See Pober et al. 2014, ApJ 782, 66.

        exact_norm : boolean, optional
            returns only X2Y for scalar if True, else uses the existing framework
            involving antenna beam and spectral tapering factors. Default: False.

        Returns
        -------
        scalar: float
            [\int dnu (\Omega_PP / \Omega_P^2) ( B_PP / B_P^2 ) / (X^2 Y)]^-1
            Units: h^-3 Mpc^3 or Mpc^3.
        """
        # Get pspec_freqs
        pspec_freqs = np.linspace(lower_freq, upper_freq, num_freqs,
                                  endpoint=False)

        # Get omega_ratio
        omega_ratio = self.power_beam_sq_int(pol) \
                      / self.power_beam_int(pol)**2

        # Get scalar
        scalar = _compute_pspec_scalar(self.cosmo, self.beam_freqs,
                                       omega_ratio, pspec_freqs,
                                       num_steps=num_steps, taper=taper,
                                       little_h=little_h,
                                       noise_scalar=noise_scalar, exact_norm=exact_norm)
        return scalar

    def Jy_to_mK(self, freqs, pol='pI'):
        """
        Return the multiplicative factor [mK / Jy], to convert a visibility
        from Jy -> mK,

        factor = 1e3 * 1e-23 * c^2 / [2 * k_b * nu^2 * Omega_p(nu)]

        where k_b is boltzmann constant, c is speed of light, nu is frequency
        and Omega_p is the integral of the unitless beam-response (steradians),
        and the 1e3 is the conversion from K -> mK and the 1e-23 is the
        conversion from Jy to cgs.

        Parameters
        ----------
        freqs : float ndarray
            Contains frequencies to evaluate conversion factor [Hz].

        pol: str, optional
                Which polarization to compute the beam scalar for.
                'pI', 'pQ', 'pU', 'pV', 'XX', 'YY', 'XY', 'YX'
                Default: 'pI'

        Returns
        -------
        factor : float ndarray
            Contains Jy -> mK factor at each frequency.
        """
        # Check input types
        if isinstance(freqs, float):
            freqs = np.array([freqs])
        elif not isinstance(freqs, np.ndarray):
            raise TypeError("freqs must be fed as a float ndarray")
        elif isinstance(freqs, np.ndarray) \
            and freqs.dtype not in (float, np.float64):
            raise TypeError("freqs must be fed as a float ndarray")

        # Check frequency bounds
        if np.min(freqs) < self.beam_freqs.min():
            print("Warning: min freq {} < self.beam_freqs.min(), extrapolating...".format(np.min(freqs)))
        if np.max(freqs) > self.beam_freqs.max():
            print("Warning: max freq {} > self.beam_freqs.max(), extrapolating...".format(np.max(freqs)))

        Op = interp1d(self.beam_freqs/1e6, self.power_beam_int(pol=pol),
                      kind='quadratic', fill_value='extrapolate')(freqs/1e6)

        return 1e-20 * conversions.cgs_units.c**2 \
               / (2 * conversions.cgs_units.kb * freqs**2 * Op)

    def get_Omegas(self, polpairs):
        """
        Get OmegaP and OmegaPP across beam_freqs for requested polarization
        pairs.

        Parameters
        ----------
        polpairs : list
            List of polarization-pair tuples or integers.

        Returns
        -------
        OmegaP : array_like
            Array containing power_beam_int, shape: (Nbeam_freqs, Npols).

        OmegaPP : array_like
            Array containing power_sq_beam_int, shape: (Nbeam_freqs, Npols).
        """
        # Unpack polpairs into tuples
        if not isinstance(polpairs, (list, np.ndarray)):
            if isinstance(polpairs, (tuple, int, np.integer)):
                polpairs = [polpairs,]
            else:
                raise TypeError("polpairs is not a list of integers or tuples")

        # Convert integers to tuples
        polpairs = [uvputils.polpair_int2tuple(p)
                        if isinstance(p, (int, np.integer, np.int32)) else p
                        for p in polpairs]

        # Calculate Omegas for each pol pair
        OmegaP, OmegaPP = [], []
        for pol1, pol2 in polpairs:
            if isinstance(pol1, (int, np.integer)):
                pol1 = uvutils.polnum2str(pol1)
            if isinstance(pol2, (int, np.integer)):
                pol2 = uvutils.polnum2str(pol2)

            # Check for cross-pol; only same-pol calculation currently supported
            if pol1 != pol2:
                raise NotImplementedError(
                        "get_Omegas does not support cross-correlation between "
                        "two different visibility polarizations yet. "
                        "Could not calculate Omegas for (%s, %s)" % (pol1, pol2))

            # Calculate Omegas
            OmegaP.append(self.power_beam_int(pol=pol1))
            OmegaPP.append(self.power_beam_sq_int(pol=pol1))

        OmegaP = np.array(OmegaP).T
        OmegaPP = np.array(OmegaPP).T
        return OmegaP, OmegaPP


class PSpecBeamGauss(PSpecBeamBase):

    def __init__(self, fwhm, beam_freqs, cosmo=None):
        """
        Object to store a simple (frequency independent) Gaussian beam in a
        PspecBeamBase object.

        Parameters
        ----------
        fwhm: float
            Full width half max of the beam, in radians.

        beam_freqs: float, array-like
            Frequencies over which this Gaussian beam is to be created. Units
            assumed to be Hz.

        cosmo : conversions.Cosmo_Conversions object, optional
            Cosmology object. Uses the default cosmology object if not
            specified. Default: None.
        """
        self.fwhm = fwhm
        self.beam_freqs = beam_freqs
        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = conversions.Cosmo_Conversions()

    def power_beam_int(self, pol='pI'):
        """
        Computes the integral of the beam over solid angle to give
        a beam area (in sr). Uses analytic formula that the answer
        is 2 * pi * fwhm**2 / 8 ln 2.

        Trivially this returns an array (i.e., a function of frequency),
        but the results are frequency independent.

        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        pol: str, optional
                Which polarization to compute the beam scalar for.
                'pI', 'pQ', 'pU', 'pV', 'XX', 'YY', 'XY', 'YX'
                Default: 'pI'

        Returns
        -------
        primary_beam_area: float, array-like
            Primary beam area.
        """
        return np.ones_like(self.beam_freqs) * 2. * np.pi * self.fwhm**2 \
               / (8. * np.log(2.))

    def power_beam_sq_int(self, pol='pI'):
        """
        Computes the integral of the beam**2 over solid angle to give
        a beam area (in str). Uses analytic formula that the answer
        is pi * fwhm**2 / 8 ln 2.

        Trivially this returns an array (i.e., a function of frequency),
        but the results are frequency independent.

        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        pol: str, optional
                Which polarization to compute the beam scalar for.
                'pI', 'pQ', 'pU', 'pV', 'XX', 'YY', 'XY', 'YX'
                Default: 'pI'

        Returns
        -------
        primary_beam_area: float, array-like
            Primary beam area.
        """
        return np.ones_like(self.beam_freqs) * np.pi * self.fwhm**2 \
               / (8. * np.log(2.))


class PSpecBeamUV(PSpecBeamBase):

    def __init__(self, uvbeam, cosmo=None):
        """
        Object to store the primary beam for a pspec observation.
        This is subclassed from PSpecBeamBase to take in a pyuvdata
        UVBeam filepath or object.

        Note: If one wants to use this object for linear dipole
        polarizations (e.g. XX, XY, YX, YY) then one can feed
        uvbeam as a dipole power beam or an efield beam. If, however,
        one wants to use this for pseudo-Stokes polarizations
        (e.g. pI, pQ, pU, pV), one must feed uvbeam as a pstokes
        power beam. See pyuvdata.UVBeam for details on forming
        pstokes power beams from an efield beam.

        Parameters
        ----------
        uvbeam: str or UVBeam object
            Path to a pyuvdata UVBeam file or a UVBeam object.

        cosmo : conversions.Cosmo_Conversions object, optional
            Cosmology object. Uses the default cosmology object if not
            specified. Default: None.
        """
        # setup uvbeam object
        if isinstance(uvbeam, str):
            uvb = UVBeam()
            uvb.read_beamfits(uvbeam)
        else:
            uvb = uvbeam

        # get frequencies and set cosmology
        self.beam_freqs = uvb.freq_array[0]
        if cosmo is not None:
            self.cosmo = cosmo
        else:
            self.cosmo = conversions.Cosmo_Conversions()

        # setup primary power beam
        self.primary_beam = uvb
        if uvb.beam_type == 'efield':
            self.primary_beam.efield_to_power(inplace=True)
            self.primary_beam.peak_normalize()

    def beam_normalized_response(self, pol='pI', freq=None, x_orientation=None):
        """
        Outputs beam response for given polarization as a function
        of pixels on the sky and input frequencies.
        The response needs to be peak normalized, and is read in from
        Healpix coordinates.
        Uses interp_freq function from uvbeam for interpolation of beam
        response over given frequency values.

        Parameters
        ----------
        pol: str, optional
            Which polarization to compute the beam response for.
            'pI', 'pQ', 'pU', 'pV', 'XX', 'YY', 'XY', 'YX'
            The output shape is (Nfreq, Npixels)
            Default: 'pI'
        freq: array, optional
            Frequencies [Hz] to interpolate onto.
        x_orientation: str, optional
            Orientation in cardinal direction east or north of X dipole.
            Default keeps polarization in X and Y basis.

        Returns
        -------
        beam_res : float, array-like
            Beam response as a function healpix indices and frequency.
        omega : float, array-like
            Beam solid angle as a function of frequency
        nside : int, scalar
            used to compute resolution
        """

        if self.primary_beam.beam_type != 'power':
            raise ValueError('beam_type must be power')
        if self.primary_beam.Naxes_vec > 1:
            raise ValueError('Expect scalar for power beam, found vector')
        if self.primary_beam._data_normalization.value != 'peak':
            raise ValueError('beam must be peak normalized')
        if self.primary_beam.pixel_coordinate_system != 'healpix':
            raise ValueError('Currently only healpix format supported')

        nside = self.primary_beam.nside
        beam_res = self.primary_beam._interp_freq(freq) # interpolate beam in frequency, based on the data frequencies
        beam_res = beam_res[0]

        if isinstance(pol, str):
            pol = uvutils.polstr2num(pol, x_orientation=x_orientation)

        pol_array = self.primary_beam.polarization_array

        if pol in pol_array:
            stokes_p_ind = np.where(np.isin(pol_array, pol))[0][0]
            beam_res = beam_res[0, 0, stokes_p_ind] # extract the beam with the correct polarization, dim (nfreq X npix)
        else:
            raise ValueError('Do not have the right polarization information')

        omega = np.sum(beam_res, axis=-1) * np.pi / (3. * nside**2) #compute beam solid angle as a function of frequency

        return beam_res, omega, nside

    def power_beam_int(self, pol='pI'):
        """
        Computes the integral of the beam over solid angle to give
        a beam area (in str) as a function of frequency. Uses function
        in pyuvdata.

        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        pol: str, optional
                Which polarization to compute the beam scalar for.
                'pI', 'pQ', 'pU', 'pV', 'XX', 'YY', 'XY', 'YX'
                Default: 'pI'

        Returns
        -------
        primary_beam_area: float, array-like
            Scalar integral over beam solid angle.
        """
        if hasattr(self.primary_beam, 'get_beam_area'):
            return np.real(self.primary_beam.get_beam_area(pol))
        else:
            raise NotImplementedError("Outdated version of pyuvdata.")

    def power_beam_sq_int(self, pol='pI'):
        """
        Computes the integral of the beam**2 over solid angle to give
        a beam**2 area (in str) as a function of frequency. Uses function
        in pyuvdata.

        See Equations 4 and 5 of Moore et al. (2017) ApJ 836, 154
        or arxiv:1502.05072 for details.

        Parameters
        ----------
        pol: str, optional
                Which polarization to compute the beam scalar for.
                'pI', 'pQ', 'pU', 'pV', 'XX', 'YY', 'XY', 'YX'
                Default: 'pI'

        Returns
        -------
        primary_beam_area: float, array-like
        """
        if hasattr(self.primary_beam, 'get_beam_area'):
            return np.real(self.primary_beam.get_beam_sq_area(pol))
        else:
            raise NotImplementedError("Outdated version of pyuvdata.")


class PSpecBeamFromArray(PSpecBeamBase):

    def __init__(self, OmegaP, OmegaPP, beam_freqs, cosmo=None, x_orientation=None):
        """
        Primary beam model built from user-defined arrays for the integrals
        over beam solid angle and beam solid angle squared.

        Allowed polarizations are:

            pI, pQ, pU, pV, XX, YY, XY, YX

        Other polarizations will be ignored.

        Parameters
        ----------
        OmegaP : array_like of float (or dict of array_like)
            Integral over beam solid angle, as a function of frequency.

            If only one array is specified, this will be assumed to be for the
            I polarization. If a dict is specified, an OmegaP array for
            several polarizations can be specified.

        OmegaPP : array_like of float (or dict of array_like)
            Integral over beam solid angle squared, as a function of frequency.

            If only one array is specified, this will be assumed to be for the
            I polarization. If a dict is specified, an OmegaP array for
            several polarizations can be specified.

        beam_freqs : array_like of float
            Frequencies at which beam solid angles OmegaP and OmegaPP are
            evaluated, in Hz. This should be specified as a single array, not
            as a dict.

        cosmo : conversions.Cosmo_Conversions object, optional
            Cosmology object. Uses the default cosmology object if not
            specified. Default: None.

        x_orientation : str, optional
            Orientation in cardinal direction east or north of X dipole.
            Default keeps polarization in X and Y basis.
        """
        self.OmegaP = {}; self.OmegaPP = {}
        self.x_orientation = x_orientation
        # these are allowed pols in AIPS polarization integer convention
        # see pyuvdata.utils.polstr2num() for details
        self.allowed_pols = [1, 2, 3, 4, -5, -6, -7, -8]

        # Set beam_freqs
        self.beam_freqs = np.asarray(beam_freqs)

        if isinstance(OmegaP, np.ndarray) and isinstance(OmegaPP, np.ndarray):
            # Only single arrays were specified; assume I
            OmegaP = {1: OmegaP}
            OmegaPP = {1: OmegaPP}

        elif isinstance(OmegaP, np.ndarray) or isinstance(OmegaPP, np.ndarray):
            # Mixed dict and array types are not allowed
            raise TypeError("OmegaP and OmegaPP must both be either dicts "
                            "or arrays. Mixing dicts and arrays is not "
                            "allowed.")
        else:
            pass

        # Should now have two dicts if everything is OK
        if not isinstance(OmegaP, (odict, dict)) or not isinstance(OmegaPP, (odict, dict)):
            raise TypeError("OmegaP and OmegaPP must both be either dicts or "
                            "arrays.")

        # Check for disallowed polarizations
        for key in list(OmegaP.keys()):
            # turn into pol integer if a pol string
            if isinstance(key, str):
                new_key = uvutils.polstr2num(key, x_orientation=self.x_orientation)
                OmegaP[new_key] = OmegaP.pop(key)
                key = new_key
            # check its an allowed pol
            if key not in self.allowed_pols:
              raise KeyError("Unrecognized polarization '%s' in OmegaP." % key)
        for key in list(OmegaPP.keys()):
            # turn into pol integer if a pol string
            if isinstance(key, str):
                new_key = uvutils.polstr2num(key, x_orientation=self.x_orientation)
                OmegaPP[new_key] = OmegaPP.pop(key)
                key = new_key
            # check its an allowed pol
            if key not in self.allowed_pols:
              raise KeyError("Unrecognized polarization '%s' in OmegaPP." % key)

        # Check for available polarizations
        for pol in self.allowed_pols:
            if pol in OmegaP.keys() or pol in OmegaPP.keys():
                if pol not in OmegaP.keys() or pol not in OmegaPP.keys():
                    raise KeyError("Polarization '%s' must be specified for"
                                   " both OmegaP and OmegaPP." % pol)

                # Add arrays for this polarization
                self.add_pol(pol, OmegaP[pol], OmegaPP[pol])

        # Set cosmology
        if cosmo is None:
            self.cosmo = conversions.Cosmo_Conversions()
        else:
            self.cosmo = cosmo


    def add_pol(self, pol, OmegaP, OmegaPP):
        """
        Add OmegaP and OmegaPP for a new polarization.

        Parameters
        ----------
        pol: str
            Which polarization to add beam solid angle arrays for. Valid
            options are:

              'pI', 'pQ', 'pU', 'pV',
              'XX', 'YY', 'XY', 'YX'

            If the arrays already exist for the specified polarization, they
            will be overwritten.

        OmegaP : array_like of float
            Integral over beam solid angle, as a function of frequency. Must
            have the same shape as self.beam_freqs.

        OmegaPP : array_like of float
            Integral over beam solid angle squared, as a function of frequency.
            Must have the same shape as self.beam_freqs.
        """
        # Type check
        if isinstance(pol, str):
            pol = uvutils.polstr2num(pol, x_orientation=self.x_orientation)

        # Check for allowed polarization
        if pol not in self.allowed_pols:
            raise KeyError("Polarization '%s' is not valid." % pol)

        # Make sure OmegaP and OmegaPP are arrays
        try:
            OmegaP = np.array(OmegaP).astype(float)
            OmegaPP = np.array(OmegaPP).astype(float)
        except:
            raise TypeError("OmegaP and OmegaPP must both be array_like.")

        # Check that array dimensions are consistent
        if OmegaP.shape != self.beam_freqs.shape \
          or OmegaPP.shape != self.beam_freqs.shape:
               raise ValueError("OmegaP and OmegaPP should both "
                                "have the same shape as beam_freqs.")
        # Store arrays
        self.OmegaP[pol] = OmegaP
        self.OmegaPP[pol] = OmegaPP

        # get available pols
        self.available_pols = ", ".join(map(uvutils.polnum2str, self.OmegaP.keys()))

    def power_beam_int(self, pol='pI'):
        """
        Computes the integral of the beam over solid angle to give
        a beam area (in str) as a function of frequency.

        Parameters
        ----------
        pol: str, optional
            Which polarization to compute the beam scalar for.
                'pI', 'pQ', 'pU', 'pV',
                'XX', 'YY', 'XY', 'YX'
            Default: pI.

        Returns
        -------
        primary_beam_area: float, array-like
            Scalar integral over beam solid angle.
        """
        # type check
        if isinstance(pol, str):
            pol = uvutils.polstr2num(pol, x_orientation=self.x_orientation)

        if pol in self.OmegaP.keys():
            return self.OmegaP[pol]
        else:
            raise KeyError("OmegaP not specified for polarization '%s'. "
                           "Available polarizations are: %s" \
                           % (pol, self.available_pols))

    def power_beam_sq_int(self, pol='pI'):
        """
        Computes the integral of the beam**2 over solid angle to give
        a beam**2 area (in str) as a function of frequency.

        Parameters
        ----------
        pol: str, optional
            Which polarization to compute the beam scalar for.
              'pI', 'pQ', 'pU', 'pV',
              'XX', 'YY', 'XY', 'YX'
            Default: pI.

        Returns
        -------
        primary_beam_area: array_like
            Array of floats containing the primary beam-squared area.
        """
        # type check
        if isinstance(pol, str):
            pol = uvutils.polstr2num(pol, x_orientation=self.x_orientation)

        if pol in self.OmegaPP.keys():
            return self.OmegaPP[pol]
        else:
            raise KeyError("OmegaPP not specified for polarization '%s'. "
                           "Available polarizations are: %s" \
                           % (pol, self.available_pols))

    def __str__(self):
        """
        Return a string with useful information about this object.
        """
        s = "PSpecBeamFromArray object\n"
        s += "\tFrequency range: Min. %4.4e Hz, Max. %4.4e Hz\n" \
              % (np.min(self.beam_freqs), np.max(self.beam_freqs))
        s += "\tAvailable pols: %s" % (self.available_pols)
        return s
