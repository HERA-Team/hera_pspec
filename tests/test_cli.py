from typer.testing import CliRunner
from hera_pspec import cli, testing
from hera_pspec.container import PSpecContainer
from hera_pspec.uvpspec import UVPSpec
import pytest
import h5py
import pickle
from pathlib import Path

@pytest.fixture(scope='module')
def vanilla_uvp() -> UVPSpec:
    uvp, _ = testing.build_vanilla_uvpspec()
    return uvp

@pytest.fixture(scope='module')
def single_baseline_files(tmp_path_factory, vanilla_uvp: UVPSpec) -> list[Path]:
    # Set up by writing out some one-blpair files.
    tmp_path = tmp_path_factory.mktemp('single-bl-files')
    
    blpairs = vanilla_uvp.get_blpairs()
    files = []
    for i, blpair in enumerate(blpairs):
        sub_uvp = vanilla_uvp.select(blpairs=[blpair], inplace=False)
        
        fname = tmp_path / f"blpair.{i:02}.h5"
        psc = PSpecContainer(fname, 'rw', keep_open=False)
        psc.set_pspec('pspecgroup', 'name', sub_uvp)
        psc.set_pspec('pspecgroup', 'name2', sub_uvp)
        
        files.append(fname)
                
    for fname in files:
        with h5py.File(fname, 'a') as fl:
            fl['header'].attrs['extra0'] = [1,2,3]
            fl['header'].attrs['extra1'] = 'a nice string'

    return files

class TestFastMergeBaselines:
    def test_happy_path(self, vanilla_uvp, single_baseline_files: list[Path]):
        runner = CliRunner()
                    
        pth = single_baseline_files[0].parent
        
        result = runner.invoke(
            cli.app,
            args=[
                'fast-merge-baselines',
                '--pattern', f'{pth}/blpair.*.h5',
                '--group', 'pspecgroup',
                '--names', 'name',
                '--names', 'name2',
                '--outpath', f"{pth}/combined",
                '--no-progress',
                '--extras', 'extra0',
                '--extras', 'extra1',
            ]
        )
        
        if result.exit_code != 0:
            print(result.exc_info)
            print(result.stdout)
            assert result.exit_code == 0
        
        # Test that the file we made has all the baselines in it.
        new = PSpecContainer(pth / "combined.pspec.h5", 'r', keep_open=False)
        newuvp = new.get_pspec('pspecgroup', 'name')
        assert all(blp in vanilla_uvp.get_blpairs() for blp in newuvp.get_blpairs())
        
        newuvp = new.get_pspec('pspecgroup', 'name2')
        assert all(blp in vanilla_uvp.get_blpairs() for blp in newuvp.get_blpairs())
        
        # Test that the file we made has all the baselines in it.
        with open(pth / "combined.extra0.pkl", 'rb') as fl:
            data = pickle.load(fl)
            assert all(blp in data for blp in vanilla_uvp.get_blpairs())
        
def test_dummy_command():
    runner = CliRunner()
                
    result = runner.invoke(
        cli.app,
        args=['hello']
    )
    assert "Hi" in result.stdout