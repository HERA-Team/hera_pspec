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
from astropy import constants
from astropy.cosmology import FLRW, LambdaCDM, Planck18

_F21_HZ = 1.420405751e9
"""Rest frequency of the 21cm line, as a plain float in Hz (for internal use)."""

F21 = _F21_HZ * u.Hz
"""Rest frequency of the 21cm line, as an ``astropy.units.Quantity`` in Hz."""


DEFAULT_COSMOLOGY = Planck18
"""
Default cosmology of hera_pspec: ``astropy.cosmology.Planck18``.

A flat ``FlatLambdaCDM`` with H0 = 67.66 km/s/Mpc, Om0 = 0.30966 and
Ob0 = 0.04897 (Planck 2018 TT,TE,EE+lowE+lensing+BAO, Table 2 of
https://doi.org/10.1051/0004-6361/201833910), with radiation
(Tcmb0 = 2.7255 K, Neff = 3.046) and one massive neutrino of 0.06 eV.
The default of hera_pspec v0.x is :data:`LEGACY_COSMOLOGY`.
"""

LEGACY_COSMOLOGY = LambdaCDM(
    H0=67.27, Om0=0.31353, Ode0=0.6844, Ob0=0.04911, name="hera_pspec_default"
)
"""
Default cosmology of hera_pspec v0.x, as an ``astropy.cosmology.LambdaCDM``.

Planck 2015 TT,TE,EE+lowP (Table 4 of
https://doi.org/10.1051/0004-6361/201525830) with
H0 = 67.27 km/s/Mpc, Om0 = 0.31353, Ode0 = 0.6844 and Ob0 = 0.04911.
``Tcmb0`` is 0, so there is no radiation term, and the model is slightly
open (Ok0 = 1 - Om0 - Ode0 = 0.00207). Pass it as ``cosmo`` (or
``pspec run --cosmology hera_pspec_default``) to reproduce results of
hera_pspec v0.x.
"""

_LITTLEH_LENGTH = u.Mpc / cu.littleh

# Largest |Om_k - (1 - Om_L - Om_M)| treated as consistent.
_OM_K_ATOL = 1e-12


def cosmology_from_params(params):
    """
    Build an astropy ``LambdaCDM`` from a hera_pspec cosmology dictionary.

    The dictionary has the keys stored in the ``cosmo`` attribute of HDF5 files: ``H0``, ``Om_L``,
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
# has no baryon density (the split rule of hera_pspec v0.x).
_OM_B_FRACTION = 0.156635
_OM_C_FRACTION = 0.843364


def params_from_cosmology(cosmo):
    """
    Return the hera_pspec cosmology dictionary of an astropy cosmology.

    The dictionary has the keys and values that hera_pspec v0.x stored for the
    same cosmology: ``Om_L``, ``Om_b``, ``Om_c``, ``Om_M``,
    ``Om_k`` and ``H0``, as plain floats. This is the form stored in the
    ``cosmo`` attribute of HDF5 files (readable by hera_pspec v0.x).

    Parameters
    ----------
    cosmo : astropy.cosmology.FLRW
        Cosmology. Only ``H0``, ``Om0``, ``Ode0`` and ``Ob0`` are read, so the
        dark energy equation of state and any radiation are not represented.
        For a cosmology with radiation, such as ``Planck18`` (the default),
        ``Om_k`` then holds the radiation density (0.0015 for ``Planck18``),
        and the ``LambdaCDM`` rebuilt from the dictionary
        (:func:`cosmology_from_params`, or hera_pspec v0.x) treats it as
        curvature: for ``Planck18`` its distances differ by 0.2-0.4% and
        ``X2Y`` by 0.7-1% at z = 5-25.

    Returns
    -------
    params : dict
        ``Om_L = Ode0``, ``Om_M = Om0``, ``Om_b = Ob0`` and
        ``Om_c = Om0 - Ob0``, ``Om_k = 1 - Om_L - Om_M`` and ``H0`` in
        km/s/Mpc. If ``Ob0`` is unset (None or 0, astropy's default from
        version 7), ``Om_M`` is split into ``Om_b`` and ``Om_c`` with the
        fixed fractions 0.156635 / 0.843364, as hera_pspec v0.x did.
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


def as_astropy_cosmology(cosmo):
    """
    Check that a cosmology is an ``astropy.cosmology.FLRW`` object.

    Parameters
    ----------
    cosmo : astropy.cosmology.FLRW
        An astropy cosmology, e.g. ``astropy.cosmology.Planck18`` or
        :data:`LEGACY_COSMOLOGY`.

    Returns
    -------
    cosmo : astropy.cosmology.FLRW
        ``cosmo`` itself.

    Raises
    ------
    TypeError
        If ``cosmo`` is not an astropy FLRW cosmology (for example a
        dictionary of cosmological parameters).
    """
    if isinstance(cosmo, FLRW):
        return cosmo
    raise TypeError(
        "cosmo must be an astropy.cosmology.FLRW object (e.g. "
        "astropy.cosmology.Planck18 or hera_pspec.conversions.LEGACY_COSMOLOGY), "
        f"got {type(cosmo).__name__}."
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
    return _F21_HZ / freq - 1


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
    return _F21_HZ / (np.asarray(z) + 1) * u.Hz


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
    return _to_length_unit(distance, length_unit, cosmo) / F21


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
