"""
Response matrices of the delay-space bandpowers (#328).

The quadratic estimator forms unnormalised bandpowers
``q_a = x_1^dagger E_a x_2`` with ``E_a = (1/2) R_1^dagger Q^est_a R_2``
(``PSpecData.q_hat``). If the data covariance is ``C = sum_b p_b C_{,b} + N``,
the response of ``q`` to the bandpowers ``p`` is

    <q_a> = sum_b H_ab p_b,    H_ab = (1/2) tr[R_1^dagger Q^est_a R_2 C_{,b}]

(arXiv:1502.06016 Eqs. 13-17; HERA memo #44 Eq. 11). This module defines
``C_{,b} = dC/dp_b``:

* :class:`DelayResponse`, the built-in response of top-hat (or sampled)
  delay bins. In cosmological units (HERA memo #44 Eq. 11)

      C_{,b ij} = Delta_eta B_ij / X2Y_ij * Q^alt_{b,ij} * S_ij,

  with ``Q^alt_{b,ij} = exp[2 pi i tau_b (nu_i - nu_j)]`` the matrix of
  ``PSpecData.get_Q_alt``, ``B`` the beam term, ``X2Y`` the conversion from
  Hz sr to comoving volume and ``S`` the bin-shape sinc. Everything but
  ``Q^alt_b`` is collected in the real kernel ``K = Delta_eta B ⊙ S / X2Y``,
  so ``C_{,b} = Q^alt_b ⊙ K``. Its ``convention="legacy"`` and
  ``"legacy-exact"`` kernels reproduce ``PSpecData.get_H`` / ``get_G`` and
  their ``exact_norm=True`` versions.
* A user response: any object with a method
  ``matrices(freqs, delays, beam, pol, cosmo, length_unit)`` returning the
  ``(Ndlys, N', N')`` tensor ``C_{,b}`` as an astropy Quantity in
  ``1 / length_unit**3`` (``vis_unit**2 / power_unit``) or an equivalent unit.

``PSpecData.get_response`` turns either into the float ``H``. Units are
resolved here once per spectral window and polarization; all returned arrays
are plain floats in Hz, s, sr and ``length_unit``.
"""

from dataclasses import dataclass

import astropy.cosmology.units as cu
import numpy as np
from astropy import units
from scipy.interpolate import interp1d

from . import conversions

__all__ = [
    "DelayResponse",
    "estimator_phases",
    "healpix_integral_beam",
    "omega_ratio",
    "strip_response_unit",
    "tensor_response",
    "toeplitz_response",
]

_SAMPLING_OPTIONS = (False, True)
_BEAM_OPTIONS = ("separable", "healpix")
_X2Y_OPTIONS = ("geometric", "midpoint", "center")
_BIN_WIDTH_OPTIONS = ("ndlys", "nfreqs")
_CONVENTION_OPTIONS = ("cosmological", "legacy", "legacy-exact")


def healpix_integral_beam(beam, pol, freqs_hz):
    r"""
    HEALPix beam integral ``B_ij`` [1/sr] at the frequencies ``freqs_hz``.

    ``B_ij = (pi / (3 N_side^2)) sum_pix A_i(p) A_j(p) / (Omega_p(nu_i)
    Omega_p(nu_j))`` with ``A`` the peak-normalised power beam (HERA memo #44
    Eq. 11). This is the matrix ``PSpecData.get_integral_beam`` returns.

    Parameters
    ----------
    beam : PSpecBeamUV
        Beam with a ``beam_normalized_response`` method.
    pol : str or int
        Polarization.
    freqs_hz : array_like
        Frequencies in Hz.

    Returns
    -------
    integral_beam : ndarray
        Real ``(Nfreqs, Nfreqs)`` array.
    """
    beam_res, beam_omega, nside = beam.beam_normalized_response(pol, freqs_hz)
    prod = 1.0 / beam_omega
    beam_prod = beam_res * prod[:, np.newaxis]
    # the nside term is dtheta^2, the HEALPix pixel area
    return np.pi / (3.0 * nside * nside) * np.dot(beam_prod, beam_prod.T)


def omega_ratio(beam, pol, freqs_hz):
    r"""
    ``Omega_pp(nu) / Omega_p(nu)^2`` [1/sr] at the frequencies ``freqs_hz``.

    Interpolated exactly as the power spectrum scalar does
    (``pspecbeam._compute_pspec_scalar``: quadratic in MHz, extrapolated).
    """
    ratio = beam.power_beam_sq_int(pol) / beam.power_beam_int(pol) ** 2
    fit = interp1d(
        beam.beam_freqs / 1e6, ratio, kind="quadratic", fill_value="extrapolate"
    )
    return fit(np.asarray(freqs_hz, dtype=float) / 1e6)


def estimator_phases(n_dlys, n_freqs):
    r"""
    Fourier vectors ``m_a`` of the default estimator ``Q^alt_a = m_a^* m_a^T``.

    ``m_{a,k} = exp(-2 pi i (s + a) k / N_d)`` with ``s = -N_d/2`` (even) or
    ``-(N_d - 1)/2`` (odd), as in ``utils.get_Q_alt`` (HERA memo #44 Eq. 16).

    Returns
    -------
    m : ndarray
        Complex ``(n_dlys, n_freqs)`` array.
    """
    start = -(n_dlys // 2) if n_dlys % 2 == 0 else -(n_dlys - 1) / 2
    modes = start + np.arange(n_dlys)
    return np.exp(-2j * np.pi * np.outer(modes, np.arange(n_freqs)) / n_dlys)


def _lag_phases(n_dlys, n_freqs):
    r"""
    ``phi_b(l) = Q^alt_{b,ij}`` for ``l = i - j = -(n_freqs-1) .. n_freqs-1``.

    ``Q^alt_{b,ij} = m_{b,i}^* m_{b,j} = exp(2 pi i (s + b)(i - j) / N_d)``
    depends on ``i - j`` only (design note section 7). Shape
    ``(n_dlys, 2 n_freqs - 1)``.
    """
    lags = np.arange(-(n_freqs - 1), n_freqs)
    start = -(n_dlys // 2) if n_dlys % 2 == 0 else -(n_dlys - 1) / 2
    modes = start + np.arange(n_dlys)
    return np.exp(2j * np.pi * np.outer(modes, lags) / n_dlys)


def toeplitz_response(R1, R2, kernel, n_dlys, q_est=None):
    r"""
    ``H_ab = (1/2) tr[R_1^dagger Q^est_a R_2 (Q^alt_b ⊙ K)]`` without the
    ``(Ndlys, N', N')`` tensor of ``C_{,b}``.

    ``Q^alt_b ⊙ K`` is Toeplitz-modulated (``Q^alt_{b,ij}`` depends on
    ``i - j`` only), so (design note section 7)

        H_ab = (1/2) sum_l phi_b(l) P_a(l),
        P_a(l) = sum_{i-j=l} K_ij (R_1^dagger Q^est_a R_2)_{ji}.

    For the default ``Q^est_a = m_a^* m_a^T`` the last factor is
    ``y_{a,i} z_{a,j}`` with ``y_a = R_2^T m_a`` and ``z_a = R_1^dagger m_a^*``,
    which costs O(N_d N'^2) time and O(N_d N') memory, against
    O(N_d^2 N'^2) for the loops of ``get_H``.

    Parameters
    ----------
    R1, R2 : ndarray
        ``(N, N')`` weighting matrices (``PSpecData.R``).
    kernel : ndarray
        Real ``(N', N')`` kernel ``K``.
    n_dlys : int
        Number of delay bins.
    q_est : ndarray, optional
        ``(n_dlys, N, N)`` estimator matrices. Default: ``Q^alt``.

    Returns
    -------
    H : ndarray
        Complex ``(n_dlys, n_dlys)`` response.
    """
    n_ext = kernel.shape[0]
    n_lags = 2 * n_ext - 1
    lag_sums = np.zeros((n_dlys, n_lags), dtype=complex)
    if q_est is None:
        m = estimator_phases(n_dlys, R1.shape[0])
        y = m @ R2  # y_{a,i} = (R_2^T m_a)_i
        z = np.conj(m @ R1)  # z_{a,j} = (R_1^dagger m_a^*)_j
        for lag in range(-(n_ext - 1), n_ext):
            # the diagonal i - j = lag of K
            diag = np.diagonal(kernel, offset=-lag)
            if lag >= 0:
                prod = y[:, lag:] * z[:, : n_ext - lag]
            else:
                prod = y[:, : n_ext + lag] * z[:, -lag:]
            lag_sums[:, lag + n_ext - 1] = prod @ diag
    else:
        # A_a = R_1^dagger Q^est_a R_2; weighted[a, i, j] = A_{a,ji} K_ij
        weighted = (np.conj(R1).T @ q_est @ R2).transpose(0, 2, 1) * kernel
        for lag in range(-(n_ext - 1), n_ext):
            lag_sums[:, lag + n_ext - 1] = np.trace(
                weighted, offset=-lag, axis1=1, axis2=2
            )
    return 0.5 * lag_sums @ _lag_phases(n_dlys, n_ext).T


def tensor_response(R1, R2, c_tensor, q_est):
    r"""
    ``H_ab = (1/2) tr[R_1^dagger Q^est_a R_2 C_{,b}]`` for a dense
    ``(Ndlys, N', N')`` tensor ``C_{,b}`` (the generic path for user
    responses), as one matrix product over the flattened ``(i, j)`` axes.

    Parameters
    ----------
    R1, R2 : ndarray
        ``(N, N')`` weighting matrices.
    c_tensor : ndarray
        ``(Ndlys, N', N')`` response ``C_{,b}``.
    q_est : ndarray
        ``(Ndlys, N, N)`` estimator matrices.

    Returns
    -------
    H : ndarray
        Complex ``(Ndlys, Ndlys)`` response.
    """
    n_dlys = q_est.shape[0]
    # A_{a,ji} with the (i, j) axes flattened against C_{b,ij}
    estimator = (np.conj(R1).T @ q_est @ R2).transpose(0, 2, 1)
    return 0.5 * estimator.reshape(n_dlys, -1) @ c_tensor.reshape(n_dlys, -1).T


def strip_response_unit(c_tensor, length_unit, cosmo):
    """
    Check the unit of a user response and return it as floats.

    Parameters
    ----------
    c_tensor : astropy.units.Quantity
        User response ``C_{,b}``.
    length_unit : astropy.units.Unit
        Length unit of the power spectrum.
    cosmo : astropy.cosmology.FLRW
        Cosmology; its ``H0`` converts between littleh and plain lengths.

    Returns
    -------
    c_tensor : ndarray
        The response in ``1 / length_unit**3``.

    Raises
    ------
    TypeError
        If ``c_tensor`` is not a Quantity.
    ValueError
        If its unit is not equivalent to ``1 / length_unit**3``.
    """
    target = length_unit**-3
    if not isinstance(c_tensor, units.Quantity):
        raise TypeError(
            "A response's matrices() must return an astropy Quantity in "
            f"{target} (vis_unit**2 / power_unit); got {type(c_tensor).__name__}."
        )
    try:
        return c_tensor.to_value(target, equivalencies=cu.with_H0(cosmo.H0))
    except units.UnitConversionError as err:
        raise ValueError(
            f"The response has unit {c_tensor.unit}, which is not equivalent to "
            f"{target} (vis_unit**2 / power_unit)."
        ) from err


@dataclass(frozen=True)
class DelayResponse:
    r"""
    Response ``dC/dp_b`` of top-hat (or sampled) delay bins.

    ``C_{,b} = Q^alt_b ⊙ K`` with the real kernel (HERA memo #44 Eq. 11;
    arXiv:1502.06016 Eqs. 13-17; design note section 2.1)

        K_ij = Delta_eta * B_ij * S_ij / X2Y_ij    [1 / length_unit**3].

    Parameters
    ----------
    sampling : bool, optional
        If True, the bandpowers are samples of P(k) at the delay-bin centres
        (``S = 1``, a Riemann sum with ``Delta_eta`` the delay spacing).
        If False (default), they are averages over top-hat bins:
        ``S_ij = sinc((i - j) / N_bin)`` (numpy's sinc).
    beam : {"separable", "healpix"}, optional
        ``B_ij``. ``"separable"`` (default): ``sqrt(b_i b_j)`` with
        ``b = Omega_pp / Omega_p^2`` interpolated as for the scalar (any
        PSpecBeam). ``"healpix"``: the HEALPix integral of
        ``PSpecData.get_integral_beam`` (PSpecBeamUV only).
    x2y : {"geometric", "midpoint", "center"}, optional
        ``X2Y_ij``: ``sqrt(X2Y(nu_i) X2Y(nu_j))`` (default),
        ``X2Y((nu_i + nu_j) / 2)``, or X2Y at the centre of the band.
    bin_width : {"ndlys", "nfreqs"}, optional
        Top-hat bin width ``Delta_eta = 1 / (N_bin Delta_nu)``:
        ``N_bin = Ndlys`` (default; the bins tile eta) or ``N_bin = N'``
        (the number of channels including filter extensions, as ``get_H``).
    convention : {"cosmological", "legacy", "legacy-exact"}, optional
        ``"cosmological"`` (default): the kernel above, for one-step
        normalization. ``"legacy"``: ``K = S`` with ``N_bin = N'``
        (``sampling=False``, reproduces ``get_H``) or ``K = 1``
        (``sampling=True``, reproduces ``get_G``); dimensionless, no beam or
        cosmology. ``"legacy-exact"``: ``K = Delta_tau B [⊙ S]`` with the
        HEALPix beam and the estimator ``Q^est_a = Delta_tau Q^alt_a ⊙ B``
        (``estimator_weight``), reproducing ``get_H`` / ``get_G`` with
        ``exact_norm=True``. The legacy conventions ignore ``beam``, ``x2y``
        and ``bin_width``.
    """

    sampling: bool = False
    beam: str = "separable"
    x2y: str = "geometric"
    bin_width: str = "ndlys"
    convention: str = "cosmological"

    def __post_init__(self):
        for name, options in (
            ("sampling", _SAMPLING_OPTIONS),
            ("beam", _BEAM_OPTIONS),
            ("x2y", _X2Y_OPTIONS),
            ("bin_width", _BIN_WIDTH_OPTIONS),
            ("convention", _CONVENTION_OPTIONS),
        ):
            value = getattr(self, name)
            if name == "sampling":
                valid = isinstance(value, (bool, np.bool_))
            else:
                valid = isinstance(value, str) and value in options
            if not valid:
                raise ValueError(
                    f"DelayResponse {name} must be one of {options}; got {value!r}."
                )

    @property
    def is_legacy(self):
        """True for the dimensionless ``"legacy"`` / ``"legacy-exact"`` kernels."""
        return self.convention != "cosmological"

    @staticmethod
    def _check_beam(beam, need_healpix):
        if beam is None:
            raise ValueError(
                "This DelayResponse needs a beam (PSpecData.primary_beam is None)."
            )
        if need_healpix and not hasattr(beam, "beam_normalized_response"):
            raise ValueError(
                'beam="healpix" and convention="legacy-exact" need a beam with '
                f"a HEALPix response (PSpecBeamUV); got {type(beam).__name__}."
            )

    def kernel(self, freqs_hz, delays_s, beam, pol, cosmo, length_unit):
        r"""
        The real kernel ``K`` with ``C_{,b} = Q^alt_b ⊙ K``.

        Parameters
        ----------
        freqs_hz : array_like
            The ``N'`` channel frequencies in Hz, including filter extensions.
        delays_s : array_like
            The ``Ndlys`` delay-bin centres in s.
        beam : PSpecBeamBase or None
            Beam (unused by ``convention="legacy"``).
        pol : str or int
            Polarization of the beam.
        cosmo : astropy.cosmology.FLRW or None
            Cosmology (``"cosmological"`` only).
        length_unit : astropy.units.Unit
            Length unit of the power spectrum (``"cosmological"`` only).

        Returns
        -------
        kernel : ndarray
            Real ``(N', N')`` float array: in ``1 / length_unit**3`` for
            ``"cosmological"``, dimensionless for ``"legacy"``, in s/sr for
            ``"legacy-exact"``.
        """
        freqs = np.asarray(freqs_hz, dtype=float)
        n_ext = freqs.size
        n_dlys = len(delays_s)
        lag = np.subtract.outer(np.arange(n_ext), np.arange(n_ext)).astype(float)

        if self.convention == "legacy":
            # get_G (C_{,b} = Q^alt_b) and get_H (C_{,b} = Q^alt_b ⊙ sinc((i-j)/N'))
            if self.sampling:
                return np.ones((n_ext, n_ext))
            return np.sinc(lag / float(n_ext))

        if self.convention == "legacy-exact":
            # exact_norm get_G / get_H: C_{,b} = Delta_tau (Q^alt_b ⊙ B) [⊙ S]
            # (HERA memo #44 Eq. 11a), with Delta_tau as in q_hat
            self._check_beam(beam, need_healpix=True)
            kernel = np.median(np.diff(delays_s)) * healpix_integral_beam(
                beam, pol, freqs
            )
            if not self.sampling:
                kernel = kernel * np.sinc(lag / float(n_ext))
            return kernel

        # cosmological: K = Delta_eta B ⊙ S / X2Y (design note section 2.1)
        self._check_beam(beam, need_healpix=self.beam == "healpix")
        if cosmo is None:
            raise ValueError("A cosmological DelayResponse needs a cosmology.")
        dnu = np.median(np.diff(freqs))
        n_bin = n_dlys if self.bin_width == "ndlys" else n_ext
        if self.sampling:
            # delta-function bins: Riemann sum over the delay spacing
            delta_eta = 1.0 / (n_dlys * dnu)
            shape = np.ones((n_ext, n_ext))
        else:
            delta_eta = 1.0 / (n_bin * dnu)
            shape = np.sinc(lag / float(n_bin))

        if self.beam == "separable":
            ratio = omega_ratio(beam, pol, freqs)
            beam_term = np.sqrt(np.outer(ratio, ratio))
        else:
            beam_term = healpix_integral_beam(beam, pol, freqs)

        x2y_unit = length_unit**3 / (units.sr * units.Hz)
        if self.x2y == "geometric":
            x2y = conversions.X2Y(conversions.f2z(freqs), cosmo, length_unit)
            x2y = x2y.to_value(x2y_unit)
            x2y_term = np.sqrt(np.outer(x2y, x2y))
        elif self.x2y == "midpoint":
            mid = 0.5 * np.add.outer(freqs, freqs)
            x2y_term = conversions.X2Y(
                conversions.f2z(mid.ravel()), cosmo, length_unit
            ).to_value(x2y_unit)
            x2y_term = x2y_term.reshape(mid.shape)
        else:
            centre = 0.5 * (freqs.min() + freqs.max())
            x2y_term = float(
                conversions.X2Y(conversions.f2z(centre), cosmo, length_unit).to_value(
                    x2y_unit
                )
            )
        return delta_eta * beam_term * shape / x2y_term

    def estimator_weight(self, freqs_hz, delays_s, beam, pol):
        r"""
        Element-wise weight ``w`` of the estimator, ``Q^est_a = Q^alt_a ⊙ w``.

        ``None`` (``Q^est = Q^alt``) except for ``convention="legacy-exact"``,
        where ``w = Delta_tau B`` on the spectral window without extensions
        (the ``exact_norm`` estimator of ``q_hat``, HERA memo #44 Eq. 14).

        Parameters
        ----------
        freqs_hz : array_like
            The ``N`` channel frequencies of the spectral window in Hz.
        delays_s : array_like
            Delay-bin centres in s.
        beam : PSpecBeamUV
            Beam.
        pol : str or int
            Polarization.
        """
        if self.convention != "legacy-exact":
            return None
        self._check_beam(beam, need_healpix=True)
        return np.median(np.diff(delays_s)) * healpix_integral_beam(
            beam, pol, np.asarray(freqs_hz, dtype=float)
        )
