"""
conversions.py
--------

Cosmological and instrumental
conversion functions for hera_pspec
"""

import functools
import warnings

import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import scipy.integrate as integrate
from astropy import constants
from astropy.cosmology import FLRW, LambdaCDM


class units:
    """
    Fundamental constants and conversion constants
    in ** SI ** units, as plain floats.

    The physical constants come from ``astropy.constants`` (CODATA 2018 in
    astropy >= 4); ``f21`` and ``w21`` are hera_pspec's own 21cm constants.

    c : speed of light m s-1
    ckm : speed of light in km s-1
    G : Newton constant m3 kg-1 s-2
    kb : boltzmann constant m2 kg s-2 K-1
    hp : planck constant m2 kg s-1
    sb : stefan-boltzmann constant W m-2 K-4
    f21 : 21cm frequency in Hz
    w21 : 21cm wavelength in meters
    H0_to_SI : conversion factor from km s-1 Mpc-1 to s-1
    """

    c = float(constants.c.value)  # speed of light in m s-1
    ckm = float(constants.c.to_value(u.km / u.s))  # speed of light in km s-1
    G = float(constants.G.value)  # Newton constant m3 kg-1 s-2
    kb = float(constants.k_B.value)  # boltzmann constant m2 kg s-2 K-1
    hp = float(constants.h.value)  # planck constant m2 kg s-1
    sb = float(constants.sigma_sb.value)  # stefan boltzmann constant W m-2 K-4
    f21 = 1.420405751e9  # frequency of 21cm transition in Hz
    w21 = 0.211061140542  # 21cm wavelength in meters
    H0_to_SI = float((1 * u.km / u.s / u.Mpc).to_value(1 / u.s))  # km s-1 Mpc-1 to s-1


class cgs_units:
    """
    Fundamental constants in ** CGS ** units, as plain floats.

    Values come from ``astropy.constants`` (CODATA 2018 in astropy >= 4).

    c : speed of light cm s-1
    kb : boltzmann constant erg K-1
    """

    c = float(constants.c.cgs.value)  # cm s-1
    kb = float(constants.k_B.cgs.value)  # erg K-1


F21 = units.f21 * u.Hz
"""Rest frequency of the 21cm line, as an ``astropy.units.Quantity`` in Hz."""


DEFAULT_COSMOLOGY = LambdaCDM(
    H0=67.27, Om0=0.31353, Ode0=0.6844, Ob0=0.04911, name="hera_pspec_default"
)
"""
Default cosmology of hera_pspec, as an ``astropy.cosmology.LambdaCDM``.

Same parameters as ``Cosmo_Conversions()``: Planck 2015 TT,TE,EE+lowP
(Table 4 of https://doi.org/10.1051/0004-6361/201525830) with
H0 = 67.27 km/s/Mpc, Om0 = 0.31353, Ode0 = 0.6844 and Ob0 = 0.04911.
``Tcmb0`` is 0, so there is no radiation term, and the model is slightly
open (Ok0 = 1 - Om0 - Ode0 = 0.00207). This is not ``Planck18``.
"""

_LITTLEH_LENGTH = u.Mpc / cu.littleh

# Largest |Om_k - (1 - Om_L - Om_M)| treated as consistent.
_OM_K_ATOL = 1e-12


def cosmology_from_params(params):
    """
    Build an astropy ``LambdaCDM`` from a hera_pspec cosmology dictionary.

    The dictionary has the keys of ``Cosmo_Conversions.get_params()``, as
    stored in the ``cosmo`` attribute of HDF5 files: ``H0``, ``Om_L``,
    ``Om_M``, ``Om_b``, ``Om_c`` and ``Om_k``. The result has ``H0``,
    ``Om0 = Om_M``, ``Ode0 = Om_L``, ``Ob0 = Om_b`` and ``Tcmb0 = 0`` (no
    radiation). ``Om_M`` defaults to ``Om_b + Om_c`` when missing or None.
    ``Om_k`` is only checked, since astropy derives the curvature as
    ``1 - Om0 - Ode0``; when missing or None it is not checked.

    Parameters
    ----------
    params : dict
        Cosmological parameters, e.g. ``params_from_cosmology(DEFAULT_COSMOLOGY)``.

    Returns
    -------
    cosmo : astropy.cosmology.LambdaCDM

    Raises
    ------
    ValueError
        If ``Om_k`` differs from ``1 - Om_L - Om_M`` (by more than 1e-12), or
        ``Om_b`` is not between 0 and ``Om_M``; astropy cannot represent
        either case.
    """
    Om_L = params["Om_L"]
    Om_b = params["Om_b"]
    Om_M = params.get("Om_M")
    if Om_M is None:
        Om_M = Om_b + params["Om_c"]
    Om_k = params.get("Om_k")
    if Om_k is not None and not np.abs(Om_k - (1 - Om_L - Om_M)) <= _OM_K_ATOL:
        raise ValueError(
            f"Om_k={Om_k} differs from 1 - Om_L - Om_M = "
            f"{1 - Om_L - Om_M}; astropy derives the curvature "
            "from Om0 and Ode0, so this cosmology cannot be converted."
        )
    if not 0 <= Om_b <= Om_M:
        raise ValueError(
            f"Om_b={Om_b} must be between 0 and Om_M={Om_M} "
            "to convert to an astropy cosmology."
        )
    return LambdaCDM(H0=params["H0"], Om0=Om_M, Ode0=Om_L, Ob0=Om_b)


# Fractions of Om_M given to baryons and cold dark matter when a cosmology
# has no baryon density (the split rule of Cosmo_Conversions).
_OM_B_FRACTION = 0.156635
_OM_C_FRACTION = 0.843364


def params_from_cosmology(cosmo):
    """
    Return the hera_pspec cosmology dictionary of an astropy cosmology.

    The dictionary has the keys and values that ``Cosmo_Conversions.get_params()``
    gives for the same cosmology: ``Om_L``, ``Om_b``, ``Om_c``, ``Om_M``,
    ``Om_k`` and ``H0``, as plain floats. This is the form stored in the
    ``cosmo`` attribute of HDF5 files (readable by hera_pspec v0.x).

    Parameters
    ----------
    cosmo : astropy.cosmology.FLRW
        Cosmology. Only ``H0``, ``Om0``, ``Ode0`` and ``Ob0`` are read, so the
        dark energy equation of state and any radiation are not represented.

    Returns
    -------
    params : dict
        ``Om_L = Ode0``, ``Om_M = Om0``, ``Om_b = Ob0`` and
        ``Om_c = Om0 - Ob0``, ``Om_k = 1 - Om_L - Om_M`` and ``H0`` in
        km/s/Mpc. If ``Ob0`` is unset (None or 0, astropy's default from
        version 7), ``Om_M`` is split into ``Om_b`` and ``Om_c`` with the
        fixed fractions 0.156635 / 0.843364, as ``Cosmo_Conversions`` does.
    """
    Om_M = float(cosmo.Om0)
    Om_L = float(cosmo.Ode0)
    if not cosmo.Ob0:
        Om_b = Om_M * _OM_B_FRACTION
        Om_c = Om_M * _OM_C_FRACTION
    else:
        Om_b = float(cosmo.Ob0)
        Om_c = Om_M - Om_b
    return {
        "Om_L": Om_L,
        "Om_b": Om_b,
        "Om_c": Om_c,
        "Om_M": Om_M,
        "Om_k": 1 - Om_L - Om_M,
        "H0": float(cosmo.H0.to_value(u.km / u.s / u.Mpc)),
    }


@functools.lru_cache(maxsize=32)
def _cached_cosmology_from_params(params_items):
    """``cosmology_from_params`` cached on the sorted items of the dictionary."""
    return cosmology_from_params(dict(params_items))


def _cosmology_from_params_cached(params):
    """
    Build (or reuse) the astropy cosmology of a parameter dictionary.

    Astropy cosmologies are immutable, so the same parameters give one shared
    object. This avoids building a new ``LambdaCDM`` (several milliseconds,
    and a reference cycle) every time, e.g. on each HDF5 read.
    """
    try:
        key = tuple(sorted(params.items()))
        hash(key)
    except TypeError:
        return cosmology_from_params(params)
    return _cached_cosmology_from_params(key)


def as_astropy_cosmology(cosmo):
    """
    Return a cosmology as an ``astropy.cosmology.FLRW`` object.

    Parameters
    ----------
    cosmo : astropy.cosmology.FLRW, dict or Cosmo_Conversions
        An astropy cosmology (returned as is), a hera_pspec cosmology
        dictionary with the keys of :func:`params_from_cosmology` (see
        :func:`cosmology_from_params`), or a deprecated
        :class:`Cosmo_Conversions`, which emits a ``DeprecationWarning``.

    Returns
    -------
    cosmo : astropy.cosmology.FLRW
        Equal parameter dictionaries give the same (shared) object.

    Raises
    ------
    TypeError
        If ``cosmo`` is none of the accepted types.
    ValueError
        If the parameters cannot be represented by an astropy ``LambdaCDM``
        (see :func:`cosmology_from_params`).
    """
    if isinstance(cosmo, FLRW):
        return cosmo
    if isinstance(cosmo, Cosmo_Conversions):
        warnings.warn(
            "Passing a Cosmo_Conversions object is deprecated; pass an "
            "astropy.cosmology object instead (e.g. "
            "hera_pspec.conversions.DEFAULT_COSMOLOGY).",
            DeprecationWarning,
            stacklevel=3,
        )
        return _cosmology_from_params_cached(cosmo.get_params())
    if isinstance(cosmo, dict):
        return _cosmology_from_params_cached(cosmo)
    raise TypeError(
        "cosmo must be an astropy.cosmology.FLRW object (or a cosmology "
        f"parameter dictionary), got {type(cosmo).__name__}."
    )


def length_unit_from_little_h(little_h):
    """
    Return the cosmological length unit for a ``little_h`` flag.

    Parameters
    ----------
    little_h : bool
        Whether lengths are in h^-1 Mpc (True) or Mpc (False).

    Returns
    -------
    astropy.units.Unit
        ``u.Mpc / cu.littleh`` if ``little_h``, else ``u.Mpc``.
    """
    return _LITTLEH_LENGTH if little_h else u.Mpc


def resolve_length_unit(length_unit=None, little_h=None, stacklevel=3):
    """
    Resolve a cosmological length unit from a unit argument and a legacy flag.

    Parameters
    ----------
    length_unit : astropy.units.Unit or str, optional
        Requested length unit, ``u.Mpc / cu.littleh`` (h^-1 Mpc) or ``u.Mpc``
        (or another length, optionally per littleh). Default: None, which
        means ``u.Mpc / cu.littleh`` unless ``little_h`` is given.

    little_h : bool, optional
        Deprecated flag: True for h^-1 Mpc, False for Mpc. Emits a
        ``DeprecationWarning`` when not None.

    stacklevel : int, optional
        Stack level of the deprecation warning, counted from this function.

    Returns
    -------
    astropy.units.Unit

    Raises
    ------
    ValueError
        If ``length_unit`` is not a length (optionally per littleh), or if
        both arguments are given and disagree.
    """
    if length_unit is not None:
        if isinstance(length_unit, str):
            # strings such as "Mpc / littleh" need littleh enabled to parse
            with u.add_enabled_units([cu.littleh]):
                length_unit = u.Unit(length_unit)
        else:
            length_unit = u.Unit(length_unit)
        if not (
            length_unit.is_equivalent(u.Mpc)
            or length_unit.is_equivalent(_LITTLEH_LENGTH)
        ):
            raise ValueError(
                f"length_unit must be a length such as u.Mpc or u.Mpc / "
                f"cu.littleh, got '{length_unit}'."
            )
    if little_h is None:
        return _LITTLEH_LENGTH if length_unit is None else length_unit
    warnings.warn(
        "The little_h argument is deprecated; pass length_unit=u.Mpc / "
        "cu.littleh (little_h=True) or length_unit=u.Mpc (little_h=False) "
        "instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    from_flag = length_unit_from_little_h(little_h)
    if length_unit is not None and length_unit != from_flag:
        raise ValueError(
            f"little_h={little_h} disagrees with length_unit='{length_unit}'; "
            "pass only length_unit."
        )
    return from_flag


def _to_length_unit(quantity, unit, cosmo):
    """Convert a Quantity to ``unit``, resolving littleh with ``cosmo.H0``."""
    return quantity.to(unit, cu.with_H0(cosmo.H0))


def f2z(freq):
    """
    Convert frequency to redshift for the 21 cm line.

    Parameters
    ----------
    freq : float, array_like or astropy.units.Quantity
        Frequency. A Quantity may be in any frequency unit; plain floats or
        arrays are taken to be in Hz.

    Returns
    -------
    z : float or array_like
        Redshift (plain, dimensionless).
    """
    if isinstance(freq, u.Quantity):
        freq = freq.to_value(u.Hz)
    return units.f21 / freq - 1


def z2f(z):
    """
    Convert redshift to observed frequency for the 21 cm line.

    Parameters
    ----------
    z : float or array_like
        Redshift.

    Returns
    -------
    freq : astropy.units.Quantity
        Frequency in Hz.
    """
    return units.f21 / (np.asarray(z) + 1) * u.Hz


class _CosmologyKey:
    """
    Hashable stand-in for an astropy cosmology, for ``functools.lru_cache``.

    Astropy cosmologies are not hashable. Two cosmologies of the same class
    with the same parameters get equal keys, whatever their names.
    """

    __slots__ = ("cosmology", "_key")

    def __init__(self, cosmology):
        self.cosmology = cosmology
        values = []
        for name, value in cosmology.parameters.items():
            value = getattr(value, "value", value)
            if value is not None:
                value = tuple(np.atleast_1d(value).astype(float).tolist())
            values.append((name, value))
        self._key = (type(cosmology), tuple(values))

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return isinstance(other, _CosmologyKey) and self._key == other._key


# Largest redshift array the helpers cache (larger ones are computed directly).
_MAX_CACHED_Z_SIZE = 4096


def _cached_by_redshift(func):
    """
    Cache a ``func(z, cosmo, length_unit)`` helper by redshift.

    UVWindow and the UVPSpec getters call these helpers many times with the
    same redshift, or the same small array of redshifts (one per channel);
    each call costs ~0.2 ms. The cached Quantity is copied on return, so
    callers cannot modify it.
    """

    @functools.lru_cache(maxsize=1024)
    def cached(z_bytes, z_shape, cosmology_key, length_unit):
        z = np.frombuffer(z_bytes).reshape(z_shape)
        if z.ndim == 0:
            z = float(z)
        return func(z, cosmology_key.cosmology, length_unit)

    @functools.wraps(func)
    def wrapper(z, cosmo=DEFAULT_COSMOLOGY, length_unit=_LITTLEH_LENGTH):
        if isinstance(z, u.Quantity):
            return func(z, cosmo, length_unit)
        z_array = np.asarray(z, dtype=float)
        if z_array.size > _MAX_CACHED_Z_SIZE:
            return func(z, cosmo, length_unit)
        result = cached(
            z_array.tobytes(), z_array.shape, _CosmologyKey(cosmo), u.Unit(length_unit)
        )
        return result.copy()

    wrapper.cache_clear = cached.cache_clear
    return wrapper


@_cached_by_redshift
def dRperp_dtheta(z, cosmo=DEFAULT_COSMOLOGY, length_unit=_LITTLEH_LENGTH):
    """
    Transverse comoving distance per radian of angle at redshift ``z``.

    This is the comoving transverse distance D_M(z) (Hogg 1999 Eqn. 16),
    written per radian.

    Parameters
    ----------
    z : float or array_like
        Redshift.

    cosmo : astropy.cosmology.FLRW, optional
        Cosmology. Default: ``DEFAULT_COSMOLOGY``.

    length_unit : astropy.units.Unit, optional
        Length unit of the result. Use ``u.Mpc`` for Mpc, or the default
        ``u.Mpc / cu.littleh`` for h^-1 Mpc (converted with ``cosmo.H0``).

    Returns
    -------
    dRperp_dtheta : astropy.units.Quantity
        In ``length_unit / u.rad``. ``rad`` is dimensionless in astropy, so
        products with ``dRpara_df`` convert cleanly to ``sr``.
    """
    distance = cosmo.comoving_transverse_distance(z)
    return _to_length_unit(distance, length_unit, cosmo) / u.rad


@_cached_by_redshift
def dRpara_df(z, cosmo=DEFAULT_COSMOLOGY, length_unit=_LITTLEH_LENGTH):
    """
    Radial comoving distance per unit observed frequency at redshift ``z``.

    Equal to c (1 + z)^2 / (H(z) f21) (Furlanetto 2006).

    Parameters
    ----------
    z : float or array_like
        Redshift.

    cosmo : astropy.cosmology.FLRW, optional
        Cosmology. Default: ``DEFAULT_COSMOLOGY``.

    length_unit : astropy.units.Unit, optional
        Length unit of the result. Use ``u.Mpc`` for Mpc, or the default
        ``u.Mpc / cu.littleh`` for h^-1 Mpc (converted with ``cosmo.H0``).

    Returns
    -------
    dRpara_df : astropy.units.Quantity
        In ``length_unit / u.Hz``.
    """
    zp1 = np.asarray(z) + 1.0
    distance = cosmo.hubble_distance * zp1**2 / cosmo.efunc(z)
    return _to_length_unit(distance, length_unit, cosmo) / (units.f21 * u.Hz)


def X2Y(z, cosmo=DEFAULT_COSMOLOGY, length_unit=_LITTLEH_LENGTH):
    """
    Conversion from sr Hz to comoving volume at redshift ``z``.

    Equal to ``dRperp_dtheta(z)**2 * dRpara_df(z)``.

    Parameters
    ----------
    z : float or array_like
        Redshift.

    cosmo : astropy.cosmology.FLRW, optional
        Cosmology. Default: ``DEFAULT_COSMOLOGY``.

    length_unit : astropy.units.Unit, optional
        Length unit of the result. Use ``u.Mpc`` for Mpc, or the default
        ``u.Mpc / cu.littleh`` for h^-1 Mpc (converted with ``cosmo.H0``).

    Returns
    -------
    X2Y : astropy.units.Quantity
        In ``length_unit**3 / (u.sr * u.Hz)``.
    """
    volume = dRperp_dtheta(z, cosmo, length_unit) ** 2 * dRpara_df(
        z, cosmo, length_unit
    )
    return volume.to(length_unit**3 / (u.sr * u.Hz))


@_cached_by_redshift
def bl_to_kperp(z, cosmo=DEFAULT_COSMOLOGY, length_unit=_LITTLEH_LENGTH):
    """
    Conversion factor from baseline length to k_perp at redshift ``z``.

    Multiply by a baseline length in meters to get k_perp
    (Parsons 2012, Pober 2014, Kohn 2018).

    Parameters
    ----------
    z : float or array_like
        Redshift.

    cosmo : astropy.cosmology.FLRW, optional
        Cosmology. Default: ``DEFAULT_COSMOLOGY``.

    length_unit : astropy.units.Unit, optional
        Length unit of the result. Use ``u.Mpc`` for Mpc, or the default
        ``u.Mpc / cu.littleh`` for h^-1 Mpc (converted with ``cosmo.H0``).

    Returns
    -------
    bl_to_kperp : astropy.units.Quantity
        In ``1 / length_unit / u.m`` (littleh / Mpc / m by default).
    """
    distance = _to_length_unit(
        cosmo.comoving_transverse_distance(z), length_unit, cosmo
    )
    wavelength = constants.c / z2f(z)
    kperp = 2 * np.pi / (distance * wavelength)
    return kperp.to(1 / length_unit / u.m)


@_cached_by_redshift
def tau_to_kpara(z, cosmo=DEFAULT_COSMOLOGY, length_unit=_LITTLEH_LENGTH):
    """
    Conversion factor from delay to k_parallel at redshift ``z``.

    Multiply by a delay in seconds to get k_para
    (Parsons 2012, Pober 2014, Kohn 2018).

    Parameters
    ----------
    z : float or array_like
        Redshift.

    cosmo : astropy.cosmology.FLRW, optional
        Cosmology. Default: ``DEFAULT_COSMOLOGY``.

    length_unit : astropy.units.Unit, optional
        Length unit of the result. Use ``u.Mpc`` for Mpc, or the default
        ``u.Mpc / cu.littleh`` for h^-1 Mpc (converted with ``cosmo.H0``).

    Returns
    -------
    tau_to_kpara : astropy.units.Quantity
        In ``1 / length_unit / u.s`` (littleh / Mpc / s by default).
    """
    kpara = 2 * np.pi / dRpara_df(z, cosmo, length_unit)
    return kpara.to(1 / length_unit / u.s)


class Cosmo_Conversions:
    """
    Cosmo_Conversions class for mathematical conversion functions,
    some of which require a cosmological model, others of which
    do not require a predefined cosmology.

    .. deprecated:: 1.0
        Use an ``astropy.cosmology`` object (e.g. ``DEFAULT_COSMOLOGY``, or
        ``cosmology_from_params``) with the module functions ``f2z``,
        ``z2f``, ``dRperp_dtheta``, ``dRpara_df``, ``X2Y``, ``bl_to_kperp``
        and ``tau_to_kpara`` instead. This class is kept for the v1.x series.

    Default parameter values are Planck 2015 TT,TE,EE+lowP.
    (Table 4 of https://doi.org/10.1051/0004-6361/201525830)

    Distance measures come from:
    Hogg 1999 (astro-ph/9905116)
    Furlanetto 2006 (2006PhR...433..181F)

    The distance methods are computed by an equivalent
    ``astropy.cosmology.LambdaCDM`` (``self.lcdm``, Tcmb0 = 0). The one
    exception is an explicit ``Om_k`` that differs from
    ``1 - Om_L - Om_M``, which astropy cannot represent; such instances
    keep the original closed-form/quadrature formulas.

    """

    # Largest |Om_k - (1 - Om_L - Om_M)| treated as consistent.
    _om_k_atol = _OM_K_ATOL

    def __init__(
        self,
        Om_L=0.68440,
        Om_b=0.04911,
        Om_c=0.26442,
        H0=67.27,
        Om_M=None,
        Om_k=None,
        **kwargs,
    ):
        """
        Default parameter values are Planck 2015 TT,TE,EE+lowP.
        (Table 4 of https://doi.org/10.1051/0004-6361/201525830)

        Parameters:
        -----------
        Om_L : float, Omega Lambda naught, default=0.68440
            cosmological constant energy density fraction at z=0

        Om_b : float, Omega baryon naught, default=0.04911
            baryon energy density fraction at z=0

        Om_c : float, Omega cold dark matter naught, default=0.26442
            cdm energy density fraction at z=0

        H0 : float, Hubble constant, default=67.31 km s-1 Mpc-1

        Om_M : float, Omega matter naught, default=None [optional]
            matter energy density faction at z=0

        Om_k : float, Omega curvature naught, default=None [optional]
            curvature energy density at z=0

        """
        warnings.warn(
            "Cosmo_Conversions is deprecated and will be removed in a future "
            "version; use an astropy.cosmology object instead, e.g. "
            "hera_pspec.conversions.DEFAULT_COSMOLOGY or "
            "hera_pspec.conversions.cosmology_from_params(...), with the "
            "module functions of hera_pspec.conversions.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Setup parameters
        if Om_M is not None:
            if np.isclose(Om_b + Om_c, Om_M, atol=1e-5) is False:
                Om_b = Om_M * 0.156635
                Om_c = Om_M * 0.843364
        else:
            Om_M = Om_b + Om_c

        if Om_k is None:
            Om_k = 1 - Om_L - Om_M

        ### TODO add radiation component to class and to distance functions
        self.Om_L = Om_L
        self.Om_b = Om_b
        self.Om_c = Om_c
        self.Om_M = Om_M
        self.Om_k = Om_k
        self.H0 = H0
        self.h = self.H0 / 100.0

        self.params = ["Om_L", "Om_b", "Om_c", "Om_M", "Om_k", "H0"]

        # Equivalent astropy cosmology. When Om_k is inconsistent with
        # Om_L and Om_M, astropy cannot represent the model, so the distance
        # methods fall back to the original formulas (_use_lcdm = False).
        self._use_lcdm = self._om_k_is_consistent()
        self.lcdm = self._build_lcdm()

    def _om_k_is_consistent(self):
        """Whether Om_k equals 1 - Om_L - Om_M (as astropy assumes)."""
        return bool(np.abs(self.Om_k - (1 - self.Om_L - self.Om_M)) <= self._om_k_atol)

    def _build_lcdm(self):
        """Build the LambdaCDM object without checking Om_k."""
        Ob0 = self.Om_b if 0 <= self.Om_b <= self.Om_M else 0.0
        return LambdaCDM(H0=self.H0, Om0=self.Om_M, Ode0=self.Om_L, Ob0=Ob0)

    def to_astropy(self):
        """
        Return the equivalent ``astropy.cosmology.LambdaCDM``.

        The result has ``H0``, ``Om0 = Om_M``, ``Ode0 = Om_L``,
        ``Ob0 = Om_b`` and ``Tcmb0 = 0`` (no radiation, as in this class).
        Its curvature is derived by astropy as ``Ok0 = 1 - Om0 - Ode0``.

        Returns
        -------
        cosmo : astropy.cosmology.LambdaCDM

        Raises
        ------
        ValueError
            If ``Om_k`` differs from ``1 - Om_L - Om_M``, or ``Om_b`` is not
            between 0 and ``Om_M``; astropy cannot represent either case.
        """
        return cosmology_from_params(self.get_params())

    def _astropy_for_writing(self):
        """
        Return the astropy cosmology to store in files, or None.

        Gives the same cosmology as :meth:`to_astropy`, or None where that
        raises. It reuses ``self.lcdm`` when that still matches the
        parameters, instead of building a new ``LambdaCDM``: building one
        leaves a reference cycle that holds the calling frames (and so any
        open HDF5 objects in them) alive until the garbage collector runs.
        """
        if not self._om_k_is_consistent() or not 0 <= self.Om_b <= self.Om_M:
            return None
        lcdm = self.lcdm
        if (
            lcdm.H0.value == self.H0
            and lcdm.Om0 == self.Om_M
            and lcdm.Ode0 == self.Om_L
            and lcdm.Ob0 == self.Om_b
        ):
            return lcdm
        return self.to_astropy()

    @classmethod
    def from_astropy(cls, cosmo):
        """
        Build a ``Cosmo_Conversions`` from an astropy ``LambdaCDM``.

        Reads ``H0``, ``Om0``, ``Ode0`` and ``Ob0``. If ``Ob0`` is unset (None,
        or 0, which is astropy's default from version 7), the matter density
        is split into baryons and cold dark matter with the fixed fractions
        0.156635 / 0.843364 (the split rule in ``__init__``).

        Parameters
        ----------
        cosmo : astropy.cosmology.LambdaCDM
            Cosmology to convert (``FlatLambdaCDM`` is accepted too).

        Returns
        -------
        Cosmo_Conversions

        Raises
        ------
        TypeError
            If ``cosmo`` is not a ``LambdaCDM`` (other dark energy models
            have no equivalent here).
        ValueError
            If ``cosmo`` has a non-zero ``Tcmb0``. ``Cosmo_Conversions`` has
            no radiation term, so the distances would not match.
        """
        if not isinstance(cosmo, LambdaCDM):
            raise TypeError(
                "from_astropy needs an astropy LambdaCDM or FlatLambdaCDM, "
                f"got {type(cosmo).__name__}."
            )
        if cosmo.Tcmb0.to_value(u.K) != 0:
            raise ValueError(
                f"cosmo has Tcmb0={cosmo.Tcmb0}; Cosmo_Conversions has no "
                "radiation term, so only Tcmb0=0 cosmologies can be converted."
            )
        Om_M = float(cosmo.Om0)
        if not cosmo.Ob0:
            Om_b = Om_M * 0.156635
            Om_c = Om_M * 0.843364
        else:
            Om_b = float(cosmo.Ob0)
            Om_c = Om_M - Om_b
        return cls(
            Om_L=float(cosmo.Ode0),
            Om_b=Om_b,
            Om_c=Om_c,
            H0=float(cosmo.H0.to_value(u.km / u.s / u.Mpc)),
            Om_M=Om_M,
        )

    def _hubble_distance(self, little_h):
        """Hubble distance c / H0 as a float, in Mpc or h^-1 Mpc."""
        if little_h:
            return units.ckm / 100.0
        return units.ckm / self.H0

    def _mpc_to_output(self, value, little_h):
        """Scale a float in Mpc to h^-1 Mpc if little_h."""
        return value * self.h if little_h else value

    def get_params(self):
        """
        Return a dictionary with cosmological parameters.
        """
        return dict([(p, getattr(self, p)) for p in self.params])

    def f2z(self, freq, ghz=False):
        """
        Convert frequency to redshift for 21cm line.

        Parameters:
        -----------
        freq : float
            Frequency in Hz (or GHz if ghz=True).

        ghz : bool, optional
            If True: assume freq is GHz. Default: False.

        Output:
        -------
        z : float
            Redshift
        """
        if ghz:
            freq = freq * 1e9

        return units.f21 / freq - 1

    @staticmethod
    def z2f(z, ghz=False):
        """
        Convert redshift to frequency in Hz for 21cm line.

        Parameters:
        -----------
        z : float
            Redshift.

        ghz : bool, optional
            If True: convert to GHz. Default: False.

        Output:
        -------
        freq : float
            Frequency in Hz (or GHz if ghz=True)
        """
        freq = units.f21 / (z + 1)
        if ghz:
            freq /= 1e9
        return freq

    def E(self, z):
        """
        Ratio of hubble parameters: H(z) / H(z=0)
        (Hogg99 Eqn. 14)

        Parameters:
        -----------
        z : float
            Redshift.
        """
        if self._use_lcdm:
            return self.lcdm.efunc(z)
        return np.sqrt(self.Om_M * (1 + z) ** 3 + self.Om_k * (1 + z) ** 2 + self.Om_L)

    def DC(self, z, little_h=True):
        """
        Line-of-sight comoving distance in Mpc.
        (Hogg99 Eqn. 15)

        Parameters:
        -----------
        z : float
            Redshift.

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc (True) or
            Mpc (False). Default: True (h^-1 Mpc)
        """
        if self._use_lcdm:
            return self._mpc_to_output(
                self.lcdm.comoving_distance(z).to_value(u.Mpc), little_h
            )
        d = integrate.quad(lambda z: 1 / self.E(z), 0, z)[0]
        if little_h:
            return d * units.ckm / 100.0
        else:
            return d * units.ckm / self.H0

    def DM(self, z, little_h=True):
        """
        Transverse comoving distance in Mpc.
        (Hogg99 Eqn. 16)

        Parameters:
        -----------
        z : float
            Redshift.

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc (True) or
            Mpc (False). Default: True (h^-1 Mpc)
        """
        if self._use_lcdm:
            return self._mpc_to_output(
                self.lcdm.comoving_transverse_distance(z).to_value(u.Mpc), little_h
            )
        DH = self._hubble_distance(little_h)

        if self.Om_k > 0:
            DM = (
                DH
                * np.sinh(np.sqrt(self.Om_k) * self.DC(z, little_h=little_h) / DH)
                / np.sqrt(self.Om_k)
            )
        elif self.Om_k < 0:
            DM = (
                DH
                * np.sin(
                    np.sqrt(np.abs(self.Om_k)) * self.DC(z, little_h=little_h) / DH
                )
                / np.sqrt(np.abs(self.Om_k))
            )
        else:
            DM = self.DC(z, little_h=little_h)

        return DM

    def DA(self, z, little_h=True):
        """
        Angular diameter (proper) distance in Mpc.
        (Hogg99 Eqn. 18)

        Parameters:
        -----------
        z : float
            Redshift.

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc (True) or
            Mpc (False). Default: True (h^-1 Mpc)
        """
        if self._use_lcdm:
            return self._mpc_to_output(
                self.lcdm.angular_diameter_distance(z).to_value(u.Mpc), little_h
            )
        return self.DM(z, little_h=little_h) / (1 + z)

    def dRperp_dtheta(self, z, little_h=True):
        """
        Conversion factor from angular size (radian) to transverse
        comoving distance (Mpc) at a specific redshift: [Mpc / radians]

        Parameters:
        -----------
        z : float
            Redshift.

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc (True) or
            Mpc (False). Default: True (h^-1 Mpc)
        """
        return self.DM(z, little_h=little_h)

    def dRpara_df(self, z, ghz=False, little_h=True):
        """
        Conversion from frequency bandwidth to radial comoving distance at a
        specific redshift: [Mpc / Hz]

        Parameters:
        -----------
        z : float
            Redshift.

        ghz : bool, optional
            Whether to convert output to [Mpc / GHz] (if True). Default: False.

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc (True) or
            Mpc (False). Default: True (h^-1 Mpc)
        """
        y = (1 + z) ** 2.0 / self.E(z) * self._hubble_distance(little_h) / units.f21
        if ghz:
            return y * 1e9
        else:
            return y

    def X2Y(self, z, little_h=True):
        """
        Conversion from radians^2 Hz -> Mpc^3 at a specific redshift.

        Parameters:
        -----------
        z : float
            Redshift.

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc (True) or
            Mpc (False). Default: True (h^-1 Mpc)

        Notes:
        ------
        Calls Cosmo_Conversions.dRperp_dtheta() and Cosmo_Conversions.dRpara_df().
        """
        return self.dRperp_dtheta(z, little_h=little_h) ** 2 * self.dRpara_df(
            z, little_h=little_h
        )

    def bl_to_kperp(self, z, little_h=True):
        """
        Produce the conversion factor from baseline length [meters] to
        k_perpendicular mode [h Mpc-1] at a specified redshift.

        Multiply this conversion factor by a baseline-separation length in
        [meters] to get its corresponding k_perp mode in [h Mpc-1].

        Parameters
        ----------
        z : float
            Redshift.

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc or Mpc
            Default: h^-1 Mpc

        Return
        ------
        bl2kperp : float
            Conversion factor in units [h Mpc-1 / meters]
        """
        # Parsons 2012, Pober 2014, Kohn 2018
        bl2kpara = (
            2
            * np.pi
            / (self.dRperp_dtheta(z, little_h=little_h) * (units.c / self.z2f(z)))
        )
        return bl2kpara

    def tau_to_kpara(self, z, little_h=True):
        """
        Produce the conversion factor from delay [seconds] to k_parallel mode
        [h Mpc-1] at a specified redshift.

        Multiply this conversion factor by a delay mode in [seconds]
        to get its corresponding k_para mode in [h Mpc-1].

        Parameters
        ----------
        z : float
            Redshift.

        little_h : boolean, optional
            Whether to have cosmological length units be h^-1 Mpc or Mpc
            Default: h^-1 Mpc

        Return
        ------
        tau2kpara : float
            Conversion factor in units [h Mpc-1 / seconds]
        """
        # Parsons 2012, Pober 2014, Kohn 2018
        tau2kpara = 2 * np.pi / self.dRpara_df(z, little_h=little_h, ghz=False)

        return tau2kpara

    def __str__(self):
        message = f"Cosmo_Conversions object at <{hex(id(self))}>\n"
        message += "; ".join([f"{p:s} : {getattr(self, p):0.4f}" for p in self.params])
        return message

    def __eq__(self, other):
        """
        Check two Cosmo_Conversion objects are equivalent
        """
        return self.get_params() == other.get_params()
