"""Tests for the hera_pspec `pspec` CLI (scripts → cli migration, #467)."""

import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path

import h5py
import pytest
import typer
from typer.testing import CliRunner

from hera_pspec import cli, container, testing
from hera_pspec.container import PSpecContainer
from hera_pspec.data import DATA_PATH
from hera_pspec.uvpspec import UVPSpec

REPO_ROOT = Path(__file__).resolve().parent.parent


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
        runner = CliRunner()

        pth = single_baseline_files[0].parent

        result = runner.invoke(
            cli.app,
            args=[
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
            ],
        )

        if result.exit_code != 0:
            print(result.exc_info)
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
        runner = CliRunner()

        pth = single_baseline_files[0].parent

        # Run with batch_size=2 (small batches to test the batching logic)
        result = runner.invoke(
            cli.app,
            args=[
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
            ],
        )

        if result.exit_code != 0:
            print(result.exc_info)
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
        runner = CliRunner()

        pth = single_baseline_files[0].parent

        # Run with batch_size=1 (most extreme batching)
        result = runner.invoke(
            cli.app,
            args=[
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
            ],
        )

        if result.exit_code != 0:
            print(result.exc_info)
            print(result.stdout)
            assert result.exit_code == 0

        # Verify the result has all the baselines
        new = PSpecContainer(pth / "combined_single.pspec.h5", "r", keep_open=False)
        newuvp = new.get_pspec("pspecgroup", "name")
        assert all(blp in vanilla_uvp.get_blpairs() for blp in newuvp.get_blpairs())
        assert len(newuvp.get_blpairs()) == len(vanilla_uvp.get_blpairs())


def test_dummy_command():
    runner = CliRunner()

    result = runner.invoke(cli.app, args=["hello"])
    assert "Hi" in result.stdout


# ---------------------------------------------------------------------------
# Adapter tests (Task 1 — register_argparse_command)
# ---------------------------------------------------------------------------

runner = CliRunner()


def _toy_app(with_profiling):
    """Build a 2-command Typer app exposing one adapter-registered `toy` command.

    A second no-op command is required because Typer only treats commands as named
    subcommands when at least two are registered.
    """
    app = typer.Typer()

    @app.command()
    def _noop():  # pragma: no cover - exists only to force subcommand mode
        pass

    captured = {}

    def factory():
        p = argparse.ArgumentParser()
        p.add_argument("x")
        p.add_argument("--y", default="dy")
        return p

    def run(args):
        captured["x"] = args.x
        captured["y"] = args.y
        captured["has_profile"] = hasattr(args, "profile")

    cli.register_argparse_command(
        app,
        name="toy",
        parser_factory=factory,
        runner=run,
        with_profiling=with_profiling,
    )
    return app, captured


def test_adapter_forwards_args_without_profiling():
    app, captured = _toy_app(with_profiling=False)
    result = runner.invoke(app, ["toy", "hello", "--y", "bye"])
    assert result.exit_code == 0, result.output
    assert captured == {"x": "hello", "y": "bye", "has_profile": False}


def test_adapter_adds_profiling_args():
    app, captured = _toy_app(with_profiling=True)
    result = runner.invoke(app, ["toy", "hello"])
    assert result.exit_code == 0, result.output
    assert captured["x"] == "hello"
    assert captured["y"] == "dy"
    assert captured["has_profile"] is True


def test_run_help():
    result = runner.invoke(cli.app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "usage:" in result.output.lower()


def test_run_end_to_end(tmp_path):
    f0 = os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA")
    f1 = os.path.join(DATA_PATH, "zen.odd.xx.LST.1.28828.uvOCRSA")
    out = tmp_path / "out.h5"
    result = runner.invoke(
        cli.app,
        [
            "run",
            f0,
            f1,
            str(out),
            "--overwrite",
            "--dset_pairs",
            "0~1",
            "--bl_len_range",
            "14",
            "15",
            "--bl_deg_range",
            "50",
            "70",
            "--psname_ext",
            "_0",
            "--spw_ranges",
            "0~25",
            "--file_type",
            "miriad",
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    psc = container.PSpecContainer(str(out), mode="r")
    assert psc.groups() == ["dset0_dset1"]
    assert psc.spectra("dset0_dset1") == ["dset0_x_dset1_0"]


def test_bootstrap_help():
    result = runner.invoke(cli.app, ["bootstrap", "--help"])
    assert result.exit_code == 0, result.output
    assert "usage:" in result.output.lower()


def test_bootstrap_runner_filters_profiling_kwargs(tmp_path, monkeypatch):
    """Regression: the runner must strip profiling/logging keys before dispatch.

    Reproduces the args object that _cli_tools.parse_args would produce, then asserts
    _run_bootstrap does not forward 'profile'/'log_*' to grouping.bootstrap_run.
    """
    from hera_cal._cli_tools import add_logging_args, add_profiling_args

    from hera_pspec import grouping

    parser = grouping.get_bootstrap_run_argparser()
    add_profiling_args(parser)
    add_logging_args(parser)
    args = parser.parse_args([str(tmp_path / "x.h5"), "--Nsamples", "5"])

    captured = {}

    def fake_bootstrap_run(filename, **kwargs):
        captured["filename"] = filename
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli.grouping, "bootstrap_run", fake_bootstrap_run)
    cli._run_bootstrap(args)

    assert captured["filename"] == str(tmp_path / "x.h5")
    assert "profile" not in captured["kwargs"]
    assert not any(k.startswith("profile_") for k in captured["kwargs"])
    assert not any(k.startswith("log_") for k in captured["kwargs"])
    assert captured["kwargs"]["Nsamples"] == 5
    # real bootstrap kwargs must still be forwarded
    assert captured["kwargs"]["seed"] == 0
    assert "overwrite" in captured["kwargs"]
    assert "time_avg" in captured["kwargs"]


def test_auto_noise_help():
    result = runner.invoke(cli.app, ["auto-noise", "--help"])
    assert result.exit_code == 0, result.output
    assert "usage:" in result.output.lower()


def test_auto_noise_main_dispatch(monkeypatch):
    """Verify the auto-noise control flow: load autos → Tsys → loop → save."""
    args = argparse.Namespace(
        pspec_container="cont.h5",
        auto_file="autos.uvh5",
        beam="beam.fits",
        groups=None,
        spectra=None,
        err_type="P_N",
    )

    calls = {"noise": [], "set": [], "saved": False}

    class FakeUVData:
        def read(self, fname):
            calls["read"] = fname

    class FakeContainer:
        def __init__(self, *a, **k):
            pass

        def groups(self):
            return ["grp1"]

        def spectra(self, group):
            return ["uvp"]

        def get_pspec(self, group, spec):
            return f"{group}/{spec}"

        def set_pspec(self, group, spec, uvp, overwrite):
            calls["set"].append((group, spec, overwrite))

        def save(self):
            calls["saved"] = True

    def fake_uvd_to_Tsys(uvd, beam):
        calls["beam"] = beam
        return "TSYS"

    monkeypatch.setattr(cli, "UVData", FakeUVData)
    monkeypatch.setattr(cli.utils, "uvd_to_Tsys", fake_uvd_to_Tsys)
    monkeypatch.setattr(
        cli.utils,
        "uvp_noise_error",
        lambda uvp, tsys, err_type: calls["noise"].append((uvp, tsys, err_type)),
    )
    monkeypatch.setattr(cli.container, "PSpecContainer", FakeContainer)

    cli._auto_noise_main(args)

    assert calls["read"] == "autos.uvh5"
    assert calls["beam"] == "beam.fits"
    assert calls["noise"] == [("grp1/uvp", "TSYS", "P_N")]
    assert calls["set"] == [("grp1", "uvp", True)]
    assert calls["saved"] is True


def test_generate_pstokes_help():
    result = runner.invoke(cli.app, ["generate-pstokes", "--help"])
    assert result.exit_code == 0, result.output
    assert "usage:" in result.output.lower()


def test_generate_pstokes_dispatch(monkeypatch):
    """Verify control flow: read input → default outputdata → construct → write."""
    parser = cli.pstokes.generate_pstokes_argparser()
    args = parser.parse_args(["in.uvh5", "--pstokes", "pI", "--clobber"])

    calls = {}

    class FakeUVData:
        def read(self, fname):
            calls["read"] = fname

    class FakeOut:
        polarization_array = []  # pI not present → construct path taken

        def __iadd__(self, other):
            return self

        def write_uvh5(self, outputdata, clobber):
            calls["write"] = (outputdata, clobber)

    def fake_construct(*a, **k):
        calls["construct"] = calls.get("construct", 0) + 1
        return FakeOut()

    monkeypatch.setattr(cli, "UVData", FakeUVData)
    monkeypatch.setattr(cli.pstokes, "construct_pstokes", fake_construct)

    cli._run_generate_pstokes(args)

    assert calls["read"] == "in.uvh5"
    assert calls["construct"] >= 1
    # outputdata defaults to inputdata when not supplied
    assert calls["write"] == ("in.uvh5", True)


def test_generate_pstokes_dispatch_keep_vispols(monkeypatch):
    """keep_vispols=True takes the deepcopy branch instead of constructing the base."""
    parser = cli.pstokes.generate_pstokes_argparser()
    args = parser.parse_args(
        ["in.uvh5", "--pstokes", "pI", "--keep_vispols", "--clobber"]
    )

    calls = {"deepcopy": 0, "construct": 0}

    class FakeOut:
        polarization_array = []  # pI absent → loop constructs it

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

    cli._run_generate_pstokes(args)

    assert calls["deepcopy"] == 1  # deepcopy branch taken, not construct-for-base
    assert calls["construct"] == 1  # one missing pol constructed in the loop
    assert calls["write"] == ("in.uvh5", True)


@pytest.mark.parametrize(
    ("script", "subcommand", "marker"),
    [
        ("pspec_run.py", "run", "--spw_ranges"),
        ("bootstrap_run.py", "bootstrap", "--Nsamples"),
        ("generate_pstokes_run.py", "generate-pstokes", "--keep_vispols"),
        ("auto_noise_run.py", "auto-noise", "--err_type"),
    ],
)
def test_deprecated_shim_warns_and_forwards(script, subcommand, marker):
    script_path = REPO_ROOT / "scripts" / script
    # No -W flag: verify the deprecation warning surfaces under Python's DEFAULT
    # warning filter, which is what a real pipeline invocation gets.
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"{script} --help failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "deprecated" in proc.stderr.lower()
    # the warning names the correct replacement subcommand
    assert f"pspec {subcommand}" in proc.stderr
    # forwarded to the right subcommand's parser (distinctive flag in its --help)
    assert "usage:" in proc.stdout.lower()
    assert marker in proc.stdout


def test_auto_noise_main_dispatch_explicit_spectra(monkeypatch):
    """When --spectra is given, it is used directly.

    Regression: the original auto_noise_run.py left ``spectra`` unbound whenever
    ``args.spectra`` was not None, raising NameError on the first iteration.
    """
    args = argparse.Namespace(
        pspec_container="cont.h5",
        auto_file="autos.uvh5",
        beam="beam.fits",
        groups=["grp1"],
        spectra=["uvp"],
        err_type="P_N",
    )

    calls = {"set": [], "saved": False, "spectra_called": False}

    class FakeUVData:
        def read(self, fname):
            calls["read"] = fname

    class FakeContainer:
        def __init__(self, *a, **k):
            pass

        def groups(self):
            return ["grp1"]

        def spectra(self, group):
            calls["spectra_called"] = True
            return ["SHOULD_NOT_BE_USED"]

        def get_pspec(self, group, spec):
            return f"{group}/{spec}"

        def set_pspec(self, group, spec, uvp, overwrite):
            calls["set"].append((group, spec, overwrite))

        def save(self):
            calls["saved"] = True

    monkeypatch.setattr(cli, "UVData", FakeUVData)
    monkeypatch.setattr(cli.utils, "uvd_to_Tsys", lambda uvd, beam: "TSYS")
    monkeypatch.setattr(cli.utils, "uvp_noise_error", lambda uvp, tsys, err_type: None)
    monkeypatch.setattr(cli.container, "PSpecContainer", FakeContainer)

    cli._auto_noise_main(args)

    # the explicit --spectra list is used; psc.spectra() is NOT consulted
    assert calls["spectra_called"] is False
    assert calls["set"] == [("grp1", "uvp", True)]
    assert calls["saved"] is True
