"""
Define PSpecParameters

PSpecParams are parameters related to power spectra generated
by hera_pspec.
"""

import astropy.cosmology.units as cu
import astropy.units as u


def _parse_unit(unit):
    """
    Turn a unit specification into an astropy unit.

    Parameters
    ----------
    unit : astropy.units.UnitBase or str or None
        The unit. Strings are parsed with ``littleh`` enabled, so
        ``"littleh / Mpc"`` is accepted.

    Returns
    -------
    unit : astropy.units.UnitBase or None
        The parsed unit, or None if ``unit`` is None.
    """
    if unit is None or isinstance(unit, u.UnitBase):
        return unit
    if isinstance(unit, str):
        with u.add_enabled_units([cu.littleh]):
            try:
                return u.Unit(unit)
            except ValueError as err:
                raise ValueError(f"could not parse unit string {unit!r}") from err
    raise TypeError(
        f"unit must be an astropy unit, a string or None, not {type(unit).__name__}"
    )


class PSpecParam:
    def __init__(
        self,
        name,
        description=None,
        value=None,
        expected_type=None,
        form=None,
        *,
        unit=None,
    ):
        """
        PSpecParam init

        Parameters
        ----------
        name : str
            Name of the parameter (the UVPSpec attribute it describes).
        description : str, optional
            Description of the parameter.
        value : object, optional
            Value of the parameter.
        expected_type : type, optional
            Type (or dtype) the stored value is expected to have.
        form : str, optional
            Expected shape of the stored value.
        unit : astropy.units.UnitBase or str, optional
            Unit the stored value is expressed in. Strings are parsed with
            ``astropy.cosmology.units.littleh`` enabled. None (default) means
            the parameter carries no fixed unit, either because it is
            unitless or because its unit is set by other attributes
            (e.g. ``vis_units`` and ``norm_units``).
        """
        self.name = name
        self.description = description
        self.value = value
        self.expected_type = expected_type
        self.form = form
        self.unit = _parse_unit(unit)
        self.__doc__ = (
            f"name : {name}, form : {form}, expected_type : {expected_type}, "
            f"unit : {self.unit} \n {description}"
        )
