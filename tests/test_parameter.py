import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
import pytest

from hera_pspec import parameter


class TestPSpecParamUnit:
    def test_default_is_none(self) -> None:
        param = parameter.PSpecParam("example", description="example")
        assert param.unit is None

    def test_accepts_unit(self) -> None:
        param = parameter.PSpecParam("freq", unit=u.Hz)
        assert param.unit is u.Hz

    @pytest.mark.parametrize(
        "unit, expected",
        [
            ("ns", u.ns),
            ("littleh / Mpc", cu.littleh / u.Mpc),
            (cu.littleh / u.Mpc, cu.littleh / u.Mpc),
        ],
        ids=["string", "littleh_string", "composite_unit"],
    )
    def test_accepts_unit_forms(self, unit, expected) -> None:
        param = parameter.PSpecParam("example", unit=unit)
        assert param.unit == expected

    def test_rejects_unparsable_unit_string(self) -> None:
        """Check that a unit string astropy cannot parse raises a ValueError."""
        with pytest.raises(ValueError, match="could not parse unit string"):
            parameter.PSpecParam("bad", unit="not_a_unit")

    def test_rejects_non_unit(self) -> None:
        """Check that a unit that is neither a string nor a unit raises a TypeError."""
        with pytest.raises(TypeError, match="unit must be"):
            parameter.PSpecParam("bad", unit=1.0)

    def test_positional_args_still_work(self) -> None:
        param = parameter.PSpecParam("example", "desc", 3, int, "(N,)")
        assert param.description == "desc"
        assert param.value == 3
        assert param.expected_type is int
        assert param.form == "(N,)"
        assert param.unit is None

    def test_unit_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            parameter.PSpecParam("example", "desc", 3, np.float64, "(N,)", u.Hz)

    def test_doc_mentions_unit(self) -> None:
        param = parameter.PSpecParam("freq", description="freqs", unit=u.Hz)
        assert "unit : Hz" in param.__doc__
