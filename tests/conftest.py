"""Pytest configuration and fixtures for tests.

This adds several mock UVPSpec objects that can be used throughout the tests.
"""

import pytest
from hera_pspec.testing import build_vanilla_uvpspec
from hera_pspec import UVPSpec, PSpecData, utils
from hera_pspec import PSpecBeamUV
from pathlib import Path
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData
import copy

DATA_PATH = Path(DATA_PATH)

@pytest.fixture(scope="session")
def beam_nf_dipole() -> PSpecBeamUV:
    beamfile = DATA_PATH / 'HERA_NF_dipole_power.beamfits'
    return PSpecBeamUV(beamfile)
        
@pytest.fixture(scope="session")
def vanilla_uvp() -> UVPSpec:
    return build_vanilla_uvpspec(equal_time_arrays=True)[0]

@pytest.fixture(scope="session")
def vanilla_uvp_with_beam(beam_nf_dipole: PSpecBeamUV) -> UVPSpec:
    return build_vanilla_uvpspec(beam=beam_nf_dipole)[0]

@pytest.fixture(scope="session")
def vanilla_uvp_alternating_times(beam_nf_dipole: PSpecBeamUV) -> UVPSpec:
    """A UVPSpec with alternating times."""
    return build_vanilla_uvpspec(equal_time_arrays=False, beam=beam_nf_dipole)[0]
    
@pytest.fixture(scope="session")
def uvp_example_data() -> UVPSpec:
    # obtain uvp object
    datafile = DATA_PATH / 'zen.2458116.31939.HH.uvh5'

    # read datafile
    uvd = UVData.from_file(datafile)
    # Create a new PSpecData objec
    ds = PSpecData(dsets=[uvd, uvd], wgts=[None, None])

    # choose baselines
    baselines1, baselines2, blpairs = utils.construct_blpairs(
        uvd.get_antpairs()[1:],
        exclude_permutations=False,
        exclude_auto_bls=True
    )
    # compute ps
    return ds.pspec(
        baselines1, baselines2, dsets=(0, 1), pols=[('xx','xx')], 
        spw_ranges=(175, 195), taper='bh',verbose=False
    )
    
@pytest.fixture(scope="session")
def uvp_exact_wfs(uvp_example_data) -> UVPSpec:
    uvp = copy.deepcopy(uvp_example_data)
    ft_file = DATA_PATH / 'FT_beam_HERA_dipole_test'
    
    uvp.get_exact_window_functions(ftbeam=ft_file, inplace=True)
    uvp.check()
    return uvp


    