import os
import pickle
from pathlib import Path

import h5py
import pytest
from typer.testing import CliRunner

from hera_pspec import cli, container, testing
from hera_pspec.container import PSpecContainer
from hera_pspec.data import DATA_PATH
from hera_pspec.uvpspec import UVPSpec

runner = CliRunner()


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


def test_run_help():
    result = runner.invoke(cli.app, ["run", "--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output.lower()


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
            "--output",
            str(out),
            "--overwrite",
            "--dset-pair",
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
            "--spw-range",
            "0",
            "25",
            "--file-type",
            "miriad",
        ],
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
    result = runner.invoke(
        cli.app,
        ["run", "a.uvh5", "--output", str(tmp_path / "o.h5"), "--no-symmetric-taper"],
    )
    assert result.exit_code == 0, result.output
    assert captured["symmetric_taper"] is False
    assert captured["include_crosscorrs"] is True


def test_run_blpair_reshaped(monkeypatch, tmp_path):
    """--blpair 1 2 3 4 must reshape to ((1, 2), (3, 4))."""
    captured = {}

    def fake_pspec_run(dsets, filename, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli.pspecdata, "pspec_run", fake_pspec_run)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "a.uvh5",
            "--output",
            str(tmp_path / "o.h5"),
            "--blpair",
            "1",
            "2",
            "3",
            "4",
            "--blpair",
            "5",
            "6",
            "7",
            "8",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["blpairs"] == [((1, 2), (3, 4)), ((5, 6), (7, 8))]
