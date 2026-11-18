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

    @pytest.mark.parametrize(
        "unit, exception, match",
        [
            ("not_a_unit", ValueError, "could not parse unit string"),
            (1.0, TypeError, "unit must be"),
        ],
        ids=["unparsable_string", "non_unit"],
    )
    def test_rejects_bad_unit(self, unit, exception, match) -> None:
        with pytest.raises(exception, match=match):
            parameter.PSpecParam("bad", unit=unit)

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
