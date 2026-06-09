import contextlib
import io
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import h5py
import pytest

from hera_pspec import cli, container, testing
from hera_pspec.container import PSpecContainer
from hera_pspec.data import DATA_PATH
from hera_pspec.uvpspec import UVPSpec


@dataclass
class Result:
    """Minimal stand-in for a CliRunner-style invocation result."""

    exit_code: int
    output: str
    exception: BaseException | None

    @property
    def stdout(self) -> str:  # tests use .stdout and .output interchangeably
        return self.output


def invoke(args: list[str]) -> Result:
    """Invoke the cyclopts app like CliRunner.invoke, returning a Result.

    Always pass an explicit token list (never bare ``cli.app()``) so the strict
    ``-Werror`` warnings job does not trip cyclopts' "Did you mean app([])?" warning.
    """
    buf = io.StringIO()
    exc: BaseException | None = None
    code = 0
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            # exit_on_error=False -> parse errors raise instead of SystemExit(1);
            # print_error=False  -> no rich error panel leaks into captured output.
            cli.app(args, exit_on_error=False, print_error=False)
    except SystemExit as e:  # --help (and any version-style exit) lands here
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        exc = e
    except BaseException as e:  # cyclopts CoercionError/MissingArgumentError/...
        code = 1
        exc = e
    return Result(exit_code=code, output=buf.getvalue(), exception=exc)


@pytest.fixture(scope="module")
def vanilla_uvp() -> UVPSpec:
    uvp, _ = testing.build_vanilla_uvpspec()
    return uvp


@pytest.fixture(scope="module")
def single_baseline_files(tmp_path_factory, vanilla_uvp: UVPSpec) -> list[Path]:
    # Set up by writing out some one-blpair files.
    tmp_path = tmp_path_factory.mktemp("single-bl-files")

    blpairs = vanilla_uvp.get_blpairs()
    files = []
    for i, blpair in enumerate(blpairs):
        sub_uvp = vanilla_uvp.select(blpairs=[blpair], inplace=False)

        fname = tmp_path / f"blpair.{i:02}.h5"
        psc = PSpecContainer(fname, "rw", keep_open=False)
        psc.set_pspec("pspecgroup", "name", sub_uvp)
        psc.set_pspec("pspecgroup", "name2", sub_uvp)

        files.append(fname)

    for fname in files:
        with h5py.File(fname, "a") as fl:
            fl["header"].attrs["extra0"] = [1, 2, 3]
            fl["header"].attrs["extra1"] = "a nice string"

    return files


class TestFastMergeBaselines:
    def test_happy_path(self, vanilla_uvp, single_baseline_files: list[Path]):
        pth = single_baseline_files[0].parent

        result = invoke(
            [
                "fast-merge-baselines",
                "--pattern",
                f"{pth}/blpair.*.h5",
                "--group",
                "pspecgroup",
                "--names",
                "name",
                "--names",
                "name2",
                "--outpath",
                f"{pth}/combined",
                "--no-progress",
                "--extras",
                "extra0",
                "--extras",
                "extra1",
            ]
        )

        if result.exit_code != 0:
            print(result.exception)
            print(result.stdout)
            assert result.exit_code == 0

        # Test that the file we made has all the baselines in it.
        new = PSpecContainer(pth / "combined.pspec.h5", "r", keep_open=False)
        newuvp = new.get_pspec("pspecgroup", "name")
        assert all(blp in vanilla_uvp.get_blpairs() for blp in newuvp.get_blpairs())

        newuvp = new.get_pspec("pspecgroup", "name2")
        assert all(blp in vanilla_uvp.get_blpairs() for blp in newuvp.get_blpairs())

        # Test that the file we made has all the baselines in it.
        with open(pth / "combined.extra0.pkl", "rb") as fl:
            data = pickle.load(fl)
            assert all(blp in data for blp in vanilla_uvp.get_blpairs())

    def test_batch_processing(self, vanilla_uvp, single_baseline_files: list[Path]):
        """Test that batch processing produces the same result as loading all at once."""
        pth = single_baseline_files[0].parent

        # Run with batch_size=2 (small batches to test the batching logic)
        result = invoke(
            [
                "fast-merge-baselines",
                "--pattern",
                f"{pth}/blpair.*.h5",
                "--group",
                "pspecgroup",
                "--names",
                "name",
                "--names",
                "name2",
                "--outpath",
                f"{pth}/combined_batched",
                "--no-progress",
                "--extras",
                "extra0",
                "--extras",
                "extra1",
                "--batch-size",
                "2",
            ]
        )

        if result.exit_code != 0:
            print(result.exception)
            print(result.stdout)
            assert result.exit_code == 0

        # Verify the batched result has all the baselines
        new = PSpecContainer(pth / "combined_batched.pspec.h5", "r", keep_open=False)
        newuvp = new.get_pspec("pspecgroup", "name")
        assert all(blp in vanilla_uvp.get_blpairs() for blp in newuvp.get_blpairs())
        assert len(newuvp.get_blpairs()) == len(vanilla_uvp.get_blpairs())

        newuvp2 = new.get_pspec("pspecgroup", "name2")
        assert all(blp in vanilla_uvp.get_blpairs() for blp in newuvp2.get_blpairs())
        assert len(newuvp2.get_blpairs()) == len(vanilla_uvp.get_blpairs())

        # Test extras were saved correctly
        with open(pth / "combined_batched.extra0.pkl", "rb") as fl:
            data = pickle.load(fl)
            assert all(blp in data for blp in vanilla_uvp.get_blpairs())
            assert len(data) == len(vanilla_uvp.get_blpairs())

    def test_single_batch(self, vanilla_uvp, single_baseline_files: list[Path]):
        """Test that batch_size=1 works correctly (edge case)."""
        pth = single_baseline_files[0].parent

        # Run with batch_size=1 (most extreme batching)
        result = invoke(
            [
                "fast-merge-baselines",
                "--pattern",
                f"{pth}/blpair.*.h5",
                "--group",
                "pspecgroup",
                "--names",
                "name",
                "--outpath",
                f"{pth}/combined_single",
                "--no-progress",
                "--batch-size",
                "1",
            ]
        )

        if result.exit_code != 0:
            print(result.exception)
            print(result.stdout)
            assert result.exit_code == 0

        # Verify the result has all the baselines
        new = PSpecContainer(pth / "combined_single.pspec.h5", "r", keep_open=False)
        newuvp = new.get_pspec("pspecgroup", "name")
        assert all(blp in vanilla_uvp.get_blpairs() for blp in newuvp.get_blpairs())
        assert len(newuvp.get_blpairs()) == len(vanilla_uvp.get_blpairs())


def test_run_help():
    result = invoke(["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output.lower()


def test_run_end_to_end(tmp_path):
    f0 = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    f1 = os.path.join(DATA_PATH, "zen.odd.xx.LST.1.28828.uvOCRSA")
    out = tmp_path / "out.h5"
    result = invoke(
        [
            "run",
            f0,
            f1,
            "--output",
            str(out),
            "--overwrite",
            "--dset-pairs",
            "0",
            "1",
            "--bl-len-range",
            "14",
            "15",
            "--bl-deg-range",
            "50",
            "70",
            "--psname-ext",
            "_0",
            "--spw-ranges",
            "0",
            "25",
            "--file-type",
            "miriad",
        ]
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    psc = container.PSpecContainer(str(out), mode="r")
    assert psc.groups() == ["dset0_dset1"]
    assert psc.spectra("dset0_dset1") == ["dset0_x_dset1_0"]


def test_run_symmetric_taper_flag_passes_bool(monkeypatch, tmp_path):
    """Regression: --no-symmetric-taper must forward Python False (not a truthy str)."""
    captured = {}

    def fake_pspec_run(dsets, filename, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli.pspecdata, "pspec_run", fake_pspec_run)
    result = invoke(
        ["run", "a.uvh5", "--output", str(tmp_path / "o.h5"), "--no-symmetric-taper"]
    )
    assert result.exit_code == 0, result.output
    assert captured["symmetric_taper"] is False
    assert captured["include_crosscorrs"] is True


def test_run_blpair_reshaped(monkeypatch, tmp_path):
    """--blpairs 1 2 3 4 must reshape to ((1, 2), (3, 4))."""
    captured = {}

    def fake_pspec_run(dsets, filename, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli.pspecdata, "pspec_run", fake_pspec_run)
    result = invoke(
        [
            "run",
            "a.uvh5",
            "--output",
            str(tmp_path / "o.h5"),
            "--blpairs",
            "1",
            "2",
            "3",
            "4",
            "--blpairs",
            "5",
            "6",
            "7",
            "8",
        ]
    )
    assert result.exit_code == 0, result.output
    assert captured["blpairs"] == [((1, 2), (3, 4)), ((5, 6), (7, 8))]


def test_bootstrap_help():
    result = invoke(["bootstrap", "--help"])
    assert result.exit_code == 0, result.output
    assert "bootstrap" in result.output.lower()


def test_bootstrap_bool_flags_round_trip(monkeypatch):
    """Regression: time_avg/normal_std/robust_std are real bool flags, not type=bool."""
    captured = {}

    def fake_bootstrap_run(filename, **kwargs):
        captured["filename"] = filename
        captured.update(kwargs)

    monkeypatch.setattr(cli.grouping, "bootstrap_run", fake_bootstrap_run)
    result = invoke(
        ["bootstrap", "x.h5", "--time-avg", "--no-normal-std", "--nsamples", "5"]
    )
    assert result.exit_code == 0, result.output
    assert captured["filename"] == "x.h5"
    assert captured["time_avg"] is True
    assert captured["normal_std"] is False
    assert captured["robust_std"] is False  # argparser default
    assert captured["Nsamples"] == 5
    assert captured["seed"] == 0


def test_bootstrap_blpair_group_parsed(monkeypatch):
    """--blpair-group tokens parse to list[list[int]]."""
    captured = {}

    def fake_bootstrap_run(filename, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli.grouping, "bootstrap_run", fake_bootstrap_run)
    result = invoke(
        ["bootstrap", "x.h5", "--blpair-group", "101 102", "--blpair-group", "103"]
    )
    assert result.exit_code == 0, result.output
    assert captured["blpair_groups"] == [[101, 102], [103]]


def test_auto_noise_help():
    result = invoke(["auto-noise", "--help"])
    assert result.exit_code == 0, result.output
    assert "auto-noise" in result.output.lower()


def _auto_noise_fakes(monkeypatch, calls):
    class FakeUVData:
        def read(self, fname):
            calls["read"] = fname

    class FakeContainer:
        def __init__(self, *a, **k):
            pass

        def groups(self):
            return ["grp1"]

        def spectra(self, group):
            return ["auto_spec"]

        def get_pspec(self, group, spec):
            return f"{group}/{spec}"

        def set_pspec(self, group, spec, uvp, overwrite):
            calls["set"].append((group, spec, overwrite))

        def save(self):
            calls["saved"] = True

    monkeypatch.setattr(cli, "UVData", FakeUVData)
    monkeypatch.setattr(cli.utils, "uvd_to_Tsys", lambda uvd, beam: "TSYS")
    monkeypatch.setattr(
        cli.utils,
        "uvp_noise_error",
        lambda uvp, tsys, err_type: calls["noise"].append((uvp, tsys, tuple(err_type))),
    )
    monkeypatch.setattr(cli.container, "PSpecContainer", FakeContainer)


def test_auto_noise_default_spectra(monkeypatch):
    calls = {"noise": [], "set": [], "saved": False}
    _auto_noise_fakes(monkeypatch, calls)
    result = invoke(["auto-noise", "cont.h5", "autos.uvh5", "beam.fits"])
    assert result.exit_code == 0, result.output
    assert calls["read"] == "autos.uvh5"
    assert calls["noise"] == [("grp1/auto_spec", "TSYS", ("P_N",))]
    assert calls["set"] == [("grp1", "auto_spec", True)]
    assert calls["saved"] is True


def test_auto_noise_explicit_spectra_does_not_raise(monkeypatch):
    """Regression: passing --spectra used to leave `spectra` unbound -> NameError."""
    calls = {"noise": [], "set": [], "saved": False}
    _auto_noise_fakes(monkeypatch, calls)
    result = invoke(
        [
            "auto-noise",
            "cont.h5",
            "autos.uvh5",
            "beam.fits",
            "--spectra",
            "s1",
            "--spectra",
            "s2",
        ]
    )
    assert result.exit_code == 0, result.output
    assert calls["set"] == [("grp1", "s1", True), ("grp1", "s2", True)]


def test_auto_noise_multiple_err_types(monkeypatch):
    """--err-type is repeatable and the full list reaches uvp_noise_error."""
    calls = {"noise": [], "set": [], "saved": False}
    _auto_noise_fakes(monkeypatch, calls)
    result = invoke(
        [
            "auto-noise",
            "cont.h5",
            "autos.uvh5",
            "beam.fits",
            "--err-type",
            "P_N",
            "--err-type",
            "P_SN",
        ]
    )
    assert result.exit_code == 0, result.output
    assert calls["noise"] == [("grp1/auto_spec", "TSYS", ("P_N", "P_SN"))]


def test_generate_pstokes_help():
    result = invoke(["generate-pstokes", "--help"])
    assert result.exit_code == 0, result.output
    assert "generate-pstokes" in result.output.lower()


def test_generate_pstokes_default_pstokes_is_list(monkeypatch):
    """Regression: --pstokes default is ['pI'] (a list), not the string 'pI'."""
    calls = {"construct": 0}

    class FakeOut:
        polarization_array = []  # pI absent -> construct path taken

        def __iadd__(self, other):
            return self

        def write_uvh5(self, outputdata, clobber):
            calls["write"] = (outputdata, clobber)

    class FakeUVData:
        def read(self, fname):
            calls["read"] = fname

    def fake_construct(uvd1, uvd2, pstokes):
        calls["construct"] += 1
        calls["first_pstokes"] = pstokes  # must be 'pI', never 'p'
        return FakeOut()

    monkeypatch.setattr(cli, "UVData", FakeUVData)
    monkeypatch.setattr(cli.pstokes, "construct_pstokes", fake_construct)

    result = invoke(["generate-pstokes", "in.uvh5", "--clobber"])
    assert result.exit_code == 0, result.output
    assert calls["read"] == "in.uvh5"
    assert calls["first_pstokes"] == "pI"
    assert calls["write"] == ("in.uvh5", True)


def test_generate_pstokes_keep_vispols(monkeypatch):
    """keep_vispols=True takes the deepcopy branch, not the construct-for-base branch."""
    calls = {"deepcopy": 0, "construct": 0}

    class FakeOut:
        polarization_array = []

        def __iadd__(self, other):
            return self

        def write_uvh5(self, outputdata, clobber):
            calls["write"] = (outputdata, clobber)

    class FakeUVData:
        def read(self, fname):
            calls["read"] = fname

    def fake_deepcopy(obj):
        calls["deepcopy"] += 1
        return FakeOut()

    def fake_construct(*a, **k):
        calls["construct"] += 1
        return FakeOut()

    monkeypatch.setattr(cli, "UVData", FakeUVData)
    monkeypatch.setattr(cli.copy, "deepcopy", fake_deepcopy)
    monkeypatch.setattr(cli.pstokes, "construct_pstokes", fake_construct)

    result = invoke(["generate-pstokes", "in.uvh5", "--keep-vispols", "--clobber"])
    assert result.exit_code == 0, result.output
    assert calls["deepcopy"] == 1
    assert calls["construct"] == 1  # one missing pol constructed in the loop
    assert calls["write"] == ("in.uvh5", True)
