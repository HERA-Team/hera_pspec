"""
Private helpers of :mod:`hera_pspec.uvpspec`.

Cosmology attributes of HDF5 files, unit conversion of stored values and the
cached power-spectrum units of a UVPSpec.
"""

import functools

import astropy.cosmology.units as cu
from astropy import units
from astropy.cosmology import Cosmology

from . import conversions, unit_helpers


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
COSMO_ATTRS_CACHE = {}
COSMO_ATTRS_CACHE_SIZE = 32


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
        cached_cosmo, attrs = COSMO_ATTRS_CACHE[key]
    except KeyError:
        pass
    else:
        # parameters outside the dict (e.g. w0 or Tcmb0) must match too
        if cached_cosmo is cosmo or cached_cosmo.is_equivalent(cosmo):
            return attrs
    attrs = (str(params), cosmo.to_format("yaml"))
    if len(COSMO_ATTRS_CACHE) >= COSMO_ATTRS_CACHE_SIZE:
        COSMO_ATTRS_CACHE.clear()
    COSMO_ATTRS_CACHE[key] = (cosmo, attrs)
    return attrs


def value_in_unit(value, unit):
    """Return a plain value in ``unit``: Quantities are converted, others kept."""
    if isinstance(value, units.Quantity):
        return value.to_value(unit)
    return value


def spw_range_to_hz(spw_range):
    """
    Return a ``get_spw_ranges`` tuple with its frequencies as floats in Hz.

    Frequency Quantities are converted; plain numbers are taken to be in Hz
    (the pre-v1.0 form of the tuples).
    """
    return tuple(value_in_unit(freq, units.Hz) for freq in spw_range[:2]) + tuple(
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
    unit = unit_helpers.parse_pspec_unit(norm_units)
    if deltasq:
        if unit_helpers.is_littleh(unit):
            unit = unit * (cu.littleh / units.Mpc) ** 3
        else:
            unit = unit * units.Mpc**-3
    return unit


@functools.lru_cache(maxsize=64)
def pspec_unit(vis_units, norm_units, deltasq):
    """Return ``vis_unit**2 * norm_unit`` for the strings and flag, cached."""
    return unit_helpers.parse_pspec_unit(vis_units) ** 2 * norm_unit(
        norm_units, deltasq
    )
