import pickle
from pathlib import Path

import h5py
import pytest
from click.testing import Result
from typer.testing import CliRunner

from hera_pspec import cli
from hera_pspec.container import PSpecContainer
from hera_pspec.uvpspec import UVPSpec


@pytest.fixture(scope="module")
def single_baseline_files(
    tmp_path_factory: pytest.TempPathFactory, vanilla_uvp: UVPSpec
) -> list[Path]:
    """Write out one-blpair PSpecContainer files for fast-merge-baselines to combine."""
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


@pytest.fixture
def cli_runner() -> CliRunner:
    """A fresh typer CliRunner for invoking the CLI app."""
    return CliRunner()


def assert_cli_success(result: Result) -> None:
    """Assert a CLI invocation succeeded, printing diagnostics on failure."""
    if result.exit_code != 0:
        print(result.exc_info)
        print(result.stdout)
        assert result.exit_code == 0


class TestFastMergeBaselines:
    @pytest.mark.parametrize(
        "batch_size,names,extras,outpath_suffix",
        [
            (None, ["name", "name2"], ["extra0", "extra1"], "combined"),
            (2, ["name", "name2"], ["extra0", "extra1"], "combined_batched"),
            (1, ["name"], [], "combined_single"),
        ],
        ids=["no_batching", "batch_size_2", "batch_size_1"],
    )
    def test_merge(
        self,
        cli_runner: CliRunner,
        vanilla_uvp: UVPSpec,
        single_baseline_files: list[Path],
        batch_size: int | None,
        names: list[str],
        extras: list[str],
        outpath_suffix: str,
    ) -> None:
        """Check that fast-merge-baselines reproduces all baselines and extras for various batch sizes."""
        pth = single_baseline_files[0].parent

        args = [
            "fast-merge-baselines",
            "--pattern",
            f"{pth}/blpair.*.h5",
            "--group",
            "pspecgroup",
        ]
        for name in names:
            args += ["--names", name]
        args += ["--outpath", f"{pth}/{outpath_suffix}", "--no-progress"]
        for extra in extras:
            args += ["--extras", extra]
        if batch_size is not None:
            args += ["--batch-size", str(batch_size)]

        result = cli_runner.invoke(cli.app, args=args)
        assert_cli_success(result)

        new = PSpecContainer(pth / f"{outpath_suffix}.pspec.h5", "r", keep_open=False)
        for name in names:
            newuvp = new.get_pspec("pspecgroup", name)
            assert all(blp in vanilla_uvp.get_blpairs() for blp in newuvp.get_blpairs())
            assert len(newuvp.get_blpairs()) == len(vanilla_uvp.get_blpairs())

        for extra in extras:
            with open(pth / f"{outpath_suffix}.{extra}.pkl", "rb") as fl:
                data = pickle.load(fl)
                assert all(blp in data for blp in vanilla_uvp.get_blpairs())
                assert len(data) == len(vanilla_uvp.get_blpairs())
