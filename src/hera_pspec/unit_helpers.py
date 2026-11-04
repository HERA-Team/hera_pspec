"""
Unit helpers for hera_pspec.

Parsing and canonical forms of the ``norm_units`` / ``vis_units`` strings,
the extra units hera_pspec uses in strings (``littleh`` and ``uncalib``), the
unit and cosmology attributes of HDF5 files, and the cached power-spectrum
units of a UVPSpec.
"""

import contextlib
import functools

import astropy.cosmology.units as cu
import astropy.units as u
from astropy.cosmology import Cosmology

from . import conversions

#: Unit of visibilities that have not been flux calibrated (pyuvdata's
#: ``vis_units = "uncalib"``). Power spectra of such data are in
#: ``uncalib**2`` times the normalisation unit.
uncalib = u.def_unit(
    "uncalib", format={"latex": r"\mathrm{uncalib}"}, doc="Uncalibrated visibility unit"
)


@contextlib.contextmanager
def enabled_hera_units():
    """
    Context manager that enables the extra units hera_pspec uses in strings.

    Inside it, ``astropy.cosmology.units.littleh`` and :data:`uncalib` are
    known units, so unit strings such as ``"littleh / Mpc"`` (what
    ``Unit.to_string()`` gives for ``cu.littleh / u.Mpc``) or
    ``"uncalib2 Hz sr"`` parse with ``astropy.units.Unit``. Outside it they
    raise. Note that ``"h"`` is astropy's hour either way.

    Examples
    --------
    >>> with enabled_hera_units():
    ...     unit = u.Unit("littleh / Mpc")
    """
    with u.add_enabled_units([cu.littleh, uncalib]):
        yield


@functools.lru_cache(maxsize=256)
def parse_unit_string(unit_string):
    """
    Parse a unit string as written to hera_pspec HDF5 files.

    The string is parsed by ``astropy.units.Unit`` inside
    :func:`enabled_hera_units`, so ``littleh`` is understood. Results are
    cached, since files repeat the same few strings.

    Parameters
    ----------
    unit_string : str or bytes
        Unit string, e.g. ``"Hz"`` or ``"littleh / Mpc"``.

    Returns
    -------
    unit : astropy.units.UnitBase

    Raises
    ------
    ValueError
        If the string is not a valid unit.
    """
    if isinstance(unit_string, bytes):
        unit_string = unit_string.decode()
    with enabled_hera_units():
        return u.Unit(unit_string)


@functools.lru_cache(maxsize=256)
def unit_to_string(unit):
    """
    Return ``unit.to_string()``, cached (units are immutable).

    Parameters
    ----------
    unit : astropy.units.UnitBase

    Returns
    -------
    str
    """
    return unit.to_string()


def convert_from_stored_unit(value, stored_unit, declared_unit, name):
    """
    Convert a value read from file to its declared unit.

    Parameters
    ----------
    value : array_like or float
        Value as read from file, in ``stored_unit``.
    stored_unit : str, bytes or None
        Unit string stored next to the value (a ``unit`` dataset attribute or
        a ``<name>_unit`` group attribute). None means no unit was stored,
        and the value is taken to be in ``declared_unit``.
    declared_unit : astropy.units.UnitBase
        Unit the value must be returned in.
    name : str
        Name of the parameter, used in error messages.

    Returns
    -------
    value : array_like or float
        Plain value (not a Quantity) in ``declared_unit``. Returned unchanged
        if ``stored_unit`` is None or the same unit.

    Raises
    ------
    ValueError
        If ``stored_unit`` cannot be parsed or is not equivalent to
        ``declared_unit``.
    """
    if stored_unit is None:
        return value
    if isinstance(stored_unit, bytes):
        stored_unit = stored_unit.decode()
    if stored_unit == unit_to_string(declared_unit):
        return value
    try:
        unit = parse_unit_string(stored_unit)
    except ValueError as err:
        raise ValueError(
            f"{name} is stored with unit '{stored_unit}', which is not a valid unit"
        ) from err
    if unit == declared_unit:
        return value
    if not unit.is_equivalent(declared_unit):
        raise ValueError(
            f"{name} is stored in unit '{stored_unit}', which is not equivalent "
            f"to its declared unit '{declared_unit}'"
        )
    return unit.to(declared_unit, value)


# Legacy (v0.x) norm_units strings. The values are (unit, unit of k) where the
# unit of k is what UVPSpec.convert_to_deltasq uses for the k^3 factor: it
# computes k with little_h only when "h^-3" is in norm_units.
_LEGACY_NORM_UNITS = {
    "h^-3 Mpc^3": (u.Mpc**3 / cu.littleh**3, cu.littleh / u.Mpc),
    "Mpc^3": (u.Mpc**3, u.Mpc**-1),
    "Hz str [beam normalization not specified]": (u.Hz * u.sr, u.Mpc**-1),
    "Hz str": (u.Hz * u.sr, u.Mpc**-1),
}
_LEGACY_DELTASQ_SUFFIX = " k^3 / (2pi^2)"


def norm_units_from_legacy_string(norm_units):
    """
    Map a legacy ``norm_units`` string to an astropy unit.

    hera_pspec v0.x stored ``norm_units`` as free-form strings that astropy
    cannot parse safely (``u.Unit("h^-3 Mpc^3")`` silently reads ``h`` as
    hours). This maps the strings v0.x wrote, and only those:

    ==============================================  ================================
    string                                          unit
    ==============================================  ================================
    ``"h^-3 Mpc^3"``                                ``Mpc**3 / littleh**3``
    ``"Mpc^3"``                                     ``Mpc**3``
    ``"Hz str [beam normalization not specified]"`` ``Hz sr``
    ``"Hz str"``                                    ``Hz sr``
    ==============================================  ================================

    For the beam-less form, the bracketed note is dropped: the unit is the
    same, and whether a beam normalisation was applied is not a unit.

    Each form may end in ``" k^3 / (2pi^2)"``, which
    ``UVPSpec.convert_to_deltasq`` appends. The unit is then multiplied by
    the unit of k^3: ``(littleh / Mpc)**3`` for the ``h^-3`` form, else
    ``(1 / Mpc)**3`` (``convert_to_deltasq`` only uses little_h k when
    ``"h^-3"`` is in ``norm_units``). So ``"h^-3 Mpc^3 k^3 / (2pi^2)"`` and
    ``"Mpc^3 k^3 / (2pi^2)"`` map to dimensionless units, and the
    ``Hz str`` forms to ``Hz sr / Mpc**3``.

    Parameters
    ----------
    norm_units : str or bytes
        Legacy ``norm_units`` string.

    Returns
    -------
    unit : astropy.units.UnitBase
        The power spectrum normalisation unit.
    is_deltasq : bool
        Whether the string carried the ``" k^3 / (2pi^2)"`` suffix, i.e.
        the spectrum is Delta^2(k) rather than P(k).

    Raises
    ------
    ValueError
        If ``norm_units`` is not one of the known forms.
    """
    if isinstance(norm_units, bytes):
        norm_units = norm_units.decode()
    base = norm_units
    is_deltasq = norm_units.endswith(_LEGACY_DELTASQ_SUFFIX)
    if is_deltasq:
        base = norm_units[: -len(_LEGACY_DELTASQ_SUFFIX)]
    if base not in _LEGACY_NORM_UNITS:
        known = ", ".join(repr(form) for form in _LEGACY_NORM_UNITS)
        raise ValueError(
            f"Unknown legacy norm_units string {norm_units!r}. Known forms are "
            f"{known}, each optionally followed by {_LEGACY_DELTASQ_SUFFIX!r}."
        )
    unit, k_unit = _LEGACY_NORM_UNITS[base]
    if is_deltasq:
        unit = unit * k_unit**3
    return unit, is_deltasq


def is_legacy_norm_units(norm_units):
    """
    Whether a string is one of the legacy (v0.x) ``norm_units`` forms.

    Parameters
    ----------
    norm_units : str or bytes
        A ``norm_units`` string.

    Returns
    -------
    bool
        True if :func:`norm_units_from_legacy_string` maps the string.
    """
    if isinstance(norm_units, bytes):
        norm_units = norm_units.decode()
    norm_units = norm_units.removesuffix(_LEGACY_DELTASQ_SUFFIX)
    return norm_units in _LEGACY_NORM_UNITS


# Visibility unit strings used by pyuvdata (and "mK", which
# PSpecData.Jy_to_mK sets) that are not astropy unit strings, keyed by their
# lower-case form. "unknown" is what older test objects used for uncalib.
_VIS_UNITS_ALIASES = {
    "uncalib": uncalib,
    "unknown": uncalib,
    "k str": u.K * u.sr,
    "jy": u.Jy,
    "mk": u.mK,
}


@functools.lru_cache(maxsize=256)
def parse_pspec_unit(unit_string):
    """
    Parse a ``UVPSpec.norm_units`` or ``UVPSpec.vis_units`` string.

    The canonical strings (``unit_to_string`` of the unit, e.g.
    ``"Mpc3 / littleh3"``, ``"Hz sr"``, ``"mK"``, ``"uncalib"``) parse with
    astropy, with ``littleh`` and :data:`uncalib` enabled. Legacy strings are
    mapped explicitly, since astropy would misread some of them (``"h^-3"`` is
    hours to astropy):

    - the legacy ``norm_units`` forms of :func:`norm_units_from_legacy_string`
      (a ``" k^3 / (2pi^2)"`` suffix is folded into the unit; use that
      function directly to also learn whether it was there);
    - the visibility unit strings of pyuvdata and hera_pspec, matched
      case-insensitively: ``"uncalib"`` (also ``"UNCALIB"`` and
      ``"unknown"``) -> :data:`uncalib`, ``"K str"`` -> ``K sr``,
      ``"Jy"`` -> ``Jy``, ``"mK"`` -> ``mK``.

    Results are cached.

    Parameters
    ----------
    unit_string : str or bytes
        Unit string.

    Returns
    -------
    unit : astropy.units.UnitBase

    Raises
    ------
    ValueError
        If the string is not a known legacy form and not a valid astropy unit
        string, or if it parses to a unit containing hours (a legacy
        ``"h^..."`` string that is not one of the known forms).
    """
    if isinstance(unit_string, bytes):
        unit_string = unit_string.decode()
    if is_legacy_norm_units(unit_string):
        return norm_units_from_legacy_string(unit_string)[0]
    alias = _VIS_UNITS_ALIASES.get(unit_string.strip().lower())
    if alias is not None:
        return alias
    try:
        unit = parse_unit_string(unit_string)
    except ValueError as err:
        raise ValueError(
            f"Cannot interpret {unit_string!r} as a power spectrum or "
            "visibility unit. Use an astropy unit string (littleh and uncalib "
            "are understood), or one of the legacy forms "
            f"{', '.join(repr(k) for k in _LEGACY_NORM_UNITS)}, "
            "'uncalib', 'K str', 'Jy' or 'mK'."
        ) from err
    if u.hour in unit.bases:
        raise ValueError(
            f"Unit string {unit_string!r} parses as a unit containing hours; "
            "astropy reads 'h' as hours. Write little h as 'littleh'."
        )
    return unit


@functools.lru_cache(maxsize=256)
def canonical_unit_string(unit_string):
    """
    Return the canonical form of a ``norm_units`` or ``vis_units`` string.

    The canonical form is ``unit_to_string(parse_pspec_unit(unit_string))``,
    e.g. ``"h^-3 Mpc^3"`` -> ``"Mpc3 / littleh3"`` and ``"UNCALIB"`` ->
    ``"uncalib"``. Results are cached.

    Parameters
    ----------
    unit_string : str or bytes
        Unit string, canonical or legacy.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If :func:`parse_pspec_unit` cannot interpret the string.
    """
    return unit_to_string(parse_pspec_unit(unit_string))


def canonical_norm_units(norm_units):
    """
    Return the canonical P(k) form of a ``norm_units`` string.

    Like :func:`canonical_unit_string`, except that a legacy string ending in
    ``" k^3 / (2pi^2)"`` (a Delta^2 spectrum, see
    :func:`norm_units_from_legacy_string`) maps to the canonical form of the
    string without the suffix, and the suffix is reported separately: from
    v1.0, ``UVPSpec.norm_units`` holds the P(k) normalisation and
    ``UVPSpec.deltasq`` whether the spectra are Delta^2.

    Parameters
    ----------
    norm_units : str or bytes
        A ``norm_units`` string, canonical or legacy.

    Returns
    -------
    norm_units : str
        Canonical string of the P(k) normalisation, e.g. ``"Mpc3 / littleh3"``.
    is_deltasq : bool
        Whether the string carried the legacy Delta^2 suffix.

    Raises
    ------
    ValueError
        If :func:`parse_pspec_unit` cannot interpret the string.
    """
    if isinstance(norm_units, bytes):
        norm_units = norm_units.decode()
    is_deltasq = is_legacy_norm_units(norm_units) and norm_units.endswith(
        _LEGACY_DELTASQ_SUFFIX
    )
    if is_deltasq:
        norm_units = norm_units[: -len(_LEGACY_DELTASQ_SUFFIX)]
    return canonical_unit_string(norm_units), is_deltasq


def is_littleh(unit):
    """
    Whether a unit contains ``astropy.cosmology.units.littleh``.

    Used to tell power spectra normalised in ``Mpc^3 / littleh^3`` from those
    in ``Mpc^3``. Pass ``UVPSpec.norm_units`` (the P(k) normalisation) rather
    than ``UVPSpec.norm_unit``: after ``UVPSpec.convert_to_deltasq`` the
    littleh factors of ``norm_unit`` cancel against those of k^3.

    Parameters
    ----------
    unit : astropy.units.UnitBase or str
        Unit, or a string :func:`parse_pspec_unit` understands.

    Returns
    -------
    bool
    """
    if isinstance(unit, (str, bytes)):
        unit = parse_pspec_unit(unit)
    return cu.littleh in unit.bases


@functools.lru_cache(maxsize=32)
def cosmology_from_yaml(cosmo_yaml):
    """
    Parse an astropy cosmology from YAML.

    Cached, since files repeat the same string and astropy cosmologies are
    immutable; parsing costs several milliseconds per read otherwise.
    """
    return Cosmology.from_format(cosmo_yaml, format="yaml")


# The "cosmo" dict string and "cosmo_yaml" written for each cosmology, keyed by
# the parameter dict plus the class and name (the YAML holds both). Dumping the
# YAML costs ~0.25 ms per write. The cosmology itself is never rebuilt here:
# building a LambdaCDM leaves a reference cycle that holds the calling frames
# (and so any open HDF5 objects in them) alive until the garbage collector runs.
_COSMO_ATTRS_CACHE = {}
_COSMO_ATTRS_CACHE_SIZE = 32


def cosmo_attrs_for_writing(cosmo):
    """
    Return the HDF5 attribute strings of a cosmology, cached.

    Parameters
    ----------
    cosmo : astropy.cosmology.FLRW
        The cosmology.

    Returns
    -------
    cosmo_params : str
        ``str(conversions.params_from_cosmology(cosmo))``, the v0.x dict
        string of the ``cosmo`` attribute. It has no radiation term, so for
        a cosmology with radiation (such as the default ``Planck18``) v0.x
        readers get an approximation of it (see
        ``conversions.params_from_cosmology``); readers of this version use
        ``cosmo_yaml``.
    cosmo_yaml : str
        ``cosmo.to_format("yaml")``.
    """
    params = conversions.params_from_cosmology(cosmo)
    key = (type(cosmo), cosmo.name, tuple(sorted(params.items())))
    try:
        cached_cosmo, attrs = _COSMO_ATTRS_CACHE[key]
    except KeyError:
        pass
    else:
        # parameters outside the dict (e.g. w0 or Tcmb0) must match too
        if cached_cosmo is cosmo or cached_cosmo.is_equivalent(cosmo):
            return attrs
    attrs = (str(params), cosmo.to_format("yaml"))
    if len(_COSMO_ATTRS_CACHE) >= _COSMO_ATTRS_CACHE_SIZE:
        _COSMO_ATTRS_CACHE.clear()
    _COSMO_ATTRS_CACHE[key] = (cosmo, attrs)
    return attrs


def value_in_unit(value, unit):
    """Return a plain value in ``unit``: Quantities are converted, others kept."""
    if isinstance(value, u.Quantity):
        return value.to_value(unit)
    return value


def spw_range_to_hz(spw_range):
    """
    Return a ``get_spw_ranges`` tuple with its frequencies as floats in Hz.

    Frequency Quantities are converted; plain numbers are taken to be in Hz
    (the pre-v1.0 form of the tuples).
    """
    return tuple(value_in_unit(freq, u.Hz) for freq in spw_range[:2]) + tuple(
        spw_range[2:]
    )


@functools.lru_cache(maxsize=64)
def norm_unit(norm_units, deltasq):
    """
    Return the normalisation unit of a ``norm_units`` string, cached.

    For Delta^2 spectra (``deltasq``), the P(k) normalisation is multiplied
    by the unit of k^3: ``(littleh / Mpc)**3`` if it contains littleh, else
    ``Mpc**-3`` (the k that ``UVPSpec.convert_to_deltasq`` uses).
    """
    unit = parse_pspec_unit(norm_units)
    if deltasq:
        if is_littleh(unit):
            unit = unit * (cu.littleh / u.Mpc) ** 3
        else:
            unit = unit * u.Mpc**-3
    return unit


@functools.lru_cache(maxsize=64)
def pspec_unit(vis_units, norm_units, deltasq):
    """Return ``vis_unit**2 * norm_unit`` for the strings and flag, cached."""
    return parse_pspec_unit(vis_units) ** 2 * norm_unit(norm_units, deltasq)
