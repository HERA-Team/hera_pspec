from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from astropy import units as u
from astropy.cosmology import FLRW
from scipy.linalg import circulant
from uvtools import dspec

from . import conversions

if TYPE_CHECKING:
    from .pspecbeam import PSpecBeamBase

PowerSpectrum = float | npt.NDArray[np.floating[Any]]
CosmologyInput = FLRW | str | dict[str, Any] | conversions.Cosmo_Conversions


def calc_P_N(
    scalar: float,
    Tsys: float,
    t_int: float,
    Ncoherent: int = 1,
    Nincoherent: int | None = None,
    form: Literal["Pk", "DelSq"] = "Pk",
    k: npt.NDArray[np.floating[Any]] | None = None,
    component: Literal["abs", "real", "imag"] = "real",
) -> PowerSpectrum:
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
    scalar : float, Power spectrum normalization factor: X2Y(z) * Omega_P^2 / Omega_PP,
        in L^3 / (Hz sr) for a cosmological length unit L
    Tsys : float, System temperature in Kelvin
    t_int : float, integration time of power spectra in seconds
    Ncoherent : int, number of coherent averages of visibility data with integration time t_int
        Total integration time is t_int * Ncoherent
    Nincoherent : int, number of incoherent averages of pspectra (i.e. after squaring).
    form : str, power spectra form 'Pk' for P(k) and 'DelSq' for Delta^2(k)
    k : float ndarray, cosmological wave-vectors in 1 / L, where L is the length unit of
        scalar (h Mpc^-1 for the default h^-1 Mpc), only needed if form == 'DelSq'
    component : str, options=['real', 'imag', 'abs']
        If component is real or imag, divide by an extra factor of sqrt(2)

    Returns (P_N)
    -------
    P_N : estimated noise power spectrum (float or array) in mK^2 * L^3, where scalar is in
        L^3 / (Hz sr) (L = h^-1 Mpc for the default scalar). If form == 'DelSq', P_N is
        multiplied by k^3 / (2pi^2) and is in mK^2 (k in 1 / L).
    """
    # assert form
    assert form in ("Pk", "DelSq"), (
        "form must be either 'Pk' or 'DelSq' for P(k) or Delta^2(k) respectively"
    )
    assert component in ["abs", "real", "imag"], (
        "component must be one of 'real', 'imag', 'abs'"
    )

    # construct prefactor in mK^2
    P_N = scalar * (Tsys * 1e3) ** 2

    # Multiply in effective integration time
    P_N /= t_int * Ncoherent

    # Mulitply in incoherent averaging
    if Nincoherent is not None:
        P_N /= np.sqrt(Nincoherent)

    # parse component
    if component in ["real", "imag"]:
        P_N /= np.sqrt(2)

    # Convert to Delta Sq
    if form == "DelSq":
        assert k is not None, "if form == 'DelSq' then k must be fed"
        P_N = P_N * k**3 / (2 * np.pi**2)

    return P_N


class Sensitivity:
    """Power spectrum thermal sensitivity calculator"""

    def __init__(
        self, cosmo: CosmologyInput | None = None, beam: PSpecBeamBase | None = None
    ) -> None:
        """
        Object for power spectrum thermal sensitivity calculations.

        Parameters
        ----------
        cosmo : astropy.cosmology.FLRW, optional
            Cosmology; see :meth:`set_cosmology` for the accepted forms.

        beam : hera_pspec.pspecbeam.PSpecBeam instance
        """
        if cosmo is not None:
            self.set_cosmology(cosmo)

        if beam is not None:
            self.set_beam(beam)

    def set_cosmology(self, cosmo: CosmologyInput) -> None:
        """
        Set a cosmological model to self.cosmo, an astropy.cosmology object.

        Parameters
        ----------
        cosmo : astropy.cosmology.FLRW, self.cosmo_params string, or dictionary
            The cosmology, or its parameter dictionary (see
            ``conversions.cosmology_from_params``) or that dictionary's
            ``str()``. A ``Cosmo_Conversions`` object is still accepted but
            deprecated.
        """
        if isinstance(cosmo, str):
            cosmo = ast.literal_eval(cosmo)
        self.cosmo = conversions.as_astropy_cosmology(cosmo)
        self.cosmo_params = str(conversions.params_from_cosmology(self.cosmo))

    def set_beam(self, beam: PSpecBeamBase) -> None:
        """
        Set a pspecbeam.PSpecBeam object to self as self.beam

        Parameters
        ----------
        beam : pspecbeam.PSpecBeam instance
        """
        # ensure self.cosmo and beam.cosmo are consistent, if they both exist
        if hasattr(beam, "cosmo"):
            if hasattr(self, "cosmo"):
                # attach self.cosmo to beam if they both exist
                beam.cosmo = self.cosmo
            else:
                # attach beam.cosmo to self if self.cosmo doesn't exist
                self.cosmo = beam.cosmo
        else:
            if hasattr(self, "cosmo"):
                # attach self.cosmo to beam if beam.cosmo doesn't exist.
                beam.cosmo = self.cosmo
            else:
                # neither beam nor self have cosmo, raise AssertionError
                raise AssertionError(
                    "neither self nor beam have a cosmology attached. "
                    "See self.set_cosmology()."
                )

        self.beam = beam

    def calc_scalar(
        self,
        freqs: npt.NDArray[np.floating[Any]],
        pol: str,
        num_steps: int = 5000,
        little_h: bool | None = None,
        length_unit: u.UnitBase | str | None = None,
    ) -> None:
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
            Deprecated, use ``length_unit``. True for h^-1 Mpc, False for Mpc.
            Raises ValueError if it disagrees with ``length_unit``.

        length_unit : astropy.units.Unit or str, optional
            Cosmological length unit L of the scalar, e.g. ``u.Mpc / cu.littleh``
            (h^-1 Mpc) or ``u.Mpc``. Default: ``u.Mpc / cu.littleh``.

        Result
        ------
        self.scalar : float, cosmological and beam scalar prefactor in
            L^3 / (Hz sr), with L = ``length_unit``

        self.length_unit : astropy.units.Unit, the length unit L of self.scalar

        self.subband : float ndarray, frequencies in spectral window used to calculate self.scalar

        self.pol : str, polarization used to calculate self.scalar
        """
        # stacklevel 3: the deprecation warning points at the caller of calc_scalar
        length_unit = conversions.resolve_length_unit(
            length_unit, little_h, stacklevel=3
        )
        # compute scalar
        self.scalar = self.beam.compute_pspec_scalar(
            freqs.min(),
            freqs.max(),
            len(freqs),
            num_steps=num_steps,
            pol=pol,
            length_unit=length_unit,
            noise_scalar=True,
        )
        self.length_unit = length_unit
        self.subband = freqs
        self.pol = pol

    def calc_P_N(
        self,
        Tsys: float,
        t_int: float,
        Ncoherent: int = 1,
        Nincoherent: int | None = None,
        form: Literal["Pk", "DelSq"] = "Pk",
        k: npt.NDArray[np.floating[Any]] | None = None,
        component: Literal["abs", "real", "imag"] = "real",
    ) -> PowerSpectrum:
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
        Tsys : float, System temperature in Kelvin
        t_int : float, integration time of power spectra in seconds
        Ncoherent : int, number of coherent averages of visibility data with integration time t_int
            Total integration time is t_int * Ncoherent
        Nincoherent : int, number of incoherent averages of pspectra (i.e. after squaring).
        form : str, power spectra form 'Pk' for P(k) and 'DelSq' for Delta^2(k)
        k : float ndarray, cosmological wave-vectors in 1 / L, where L is self.length_unit,
            only needed if form == 'DelSq'
        component : str, options=['real', 'imag', 'abs']
            If component is real or imag, divide by an extra factor of sqrt(2)

        Returns (P_N)
        -------
        P_N : estimated noise power spectrum (float or array) in mK^2 * L^3, where L is
            self.length_unit, the length unit of self.scalar (see calc_scalar). If
            form == 'DelSq', P_N is multiplied by k^3 / (2pi^2) and is in mK^2.
        """
        # assert scalar exists
        assert hasattr(self, "scalar"), (
            "self.scalar must exist before one can calculate a noise spectrum, see self.calc_scalar()"
        )

        # assert form
        assert form in ("Pk", "DelSq"), (
            "form must be either 'Pk' or 'DelSq' for P(k) or Delta^2(k) respectively"
        )

        # calculate P_N
        P_N = calc_P_N(
            self.scalar,
            Tsys,
            t_int,
            Ncoherent=Ncoherent,
            Nincoherent=Nincoherent,
            form=form,
            k=k,
            component=component,
        )

        return P_N


def get_approximate_delay_delay_corr_matrix(taper: str, n_delays: int) -> np.ndarray:
    r"""Compute an approximate correlation matrix for a power spectrum.

    This correlation matrix assumes that there is no intrinsic covariance between
    channels of the visibilities, and that all delay-delay covariance comes in through
    the frequency taper. In this case, the correlation between delays i and j is given
    by

    .. math:: C_{i,j} = |\mathcal{F}(\phi^2)|(\tau_j - \tau_i)^2,

    where :math:`\phi` is the taper function.

    Parameters
    ----------
    taper
        The taper function, phi, as a string. Should be an existing window function
        in `scipy.signal.windows`.
    n_delays
        The number of channels in the power spectrum window.
    """
    taper = dspec.gen_window(taper, n_delays)

    # Get the covariance as a function of separation of delay modes.
    # We take the non-negative modes only, since the negative modes are symmetric.
    cov = np.abs(np.fft.fft(taper**2)) ** 2
    return circulant(cov / cov.max())
