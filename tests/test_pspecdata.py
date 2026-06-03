import copy
import glob
import json
import os
import warnings
from contextlib import nullcontext

import numpy as np
import pytest
import pyuvdata as uv
from astropy.time import Time
from hera_cal import redcal
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.signal import windows
from uvtools import dspec

from hera_pspec import (
    container,
    conversions,
    pspecbeam,
    pspecdata,
    testing,
    utils,
    uvwindow,
)
from hera_pspec.data import DATA_PATH

# Data files to use in tests
dfiles = ["zen.2458042.12552.xx.HH.uvXAA", "zen.2458042.12552.xx.HH.uvXAA"]
dfiles_std = ["zen.2458042.12552.std.xx.HH.uvXAA", "zen.2458042.12552.std.xx.HH.uvXAA"]

# List of tapering function to use in tests
taper_selection = ["none", "bh7"]
weight_selection = ["identity", "iC", "dayenu"]

# Baseline list shared by the pspec tests
pspec_bls = [(24, 25), (37, 38), (38, 39), (52, 53)]
# taper_selection = ['blackman', 'blackman-harris', 'gaussian0.4', 'kaiser2',
#                   'kaiser3', 'hamming', 'hanning', 'parzen']


def generate_pos_def(n):
    """
    Generate a random positive definite Hermitian matrix.

    Parameters
    ----------
    n : integer
        Size of desired matrix

    Returns
    -------
    A : array_like
        Positive definite matrix
    """
    A = np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))
    A += np.conjugate(A).T
    # Add just enough of an identity matrix to make all eigenvalues positive
    A += -1.01 * np.min(np.linalg.eigvalsh(A)) * np.identity(n)
    return A


def generate_pos_def_all_pos(n):
    """
    Generate a random positive definite symmetric matrix, with all entries
    positive.

    Parameters
    ----------
    n : integer
        Size of desired matrix

    Returns
    -------
    A : array_like
        Positive definite matrix
    """
    A = np.random.uniform(size=(n, n))
    A += A.T
    # Add just enough of an identity matrix to make all eigenvalues positive
    A += -1.01 * np.min(np.linalg.eigvalsh(A)) * np.identity(n)
    return A


def diagonal_or_not(mat, places=7):
    """
    Tests whether a matrix is diagonal or not.

    Parameters
    ----------
    n : array_like
        Matrix to be tested

    Returns
    -------
    diag : bool
        True if matrix is diagonal
    """
    mat_norm = np.linalg.norm(mat)
    diag_mat_norm = np.linalg.norm(np.diag(np.diag(mat)))
    diag = round(mat_norm - diag_mat_norm, places) == 0
    return diag


@pytest.fixture(scope="session")
def _miriad_raw():
    """Session-cached factory: reads miriad file(s) from DATA_PATH.
    Pass a list of filenames → returns a list of UVData.
    Pass a single filename string → returns a single UVData.
    """
    _cache = {}

    def _load(files):
        key = files if isinstance(files, str) else tuple(files)
        if key not in _cache:
            if isinstance(files, str):
                d = uv.UVData()
                d.read_miriad(os.path.join(DATA_PATH, files))
                _cache[key] = d
            else:
                result = []
                for f in files:
                    _d = uv.UVData()
                    _d.read_miriad(os.path.join(DATA_PATH, f))
                    result.append(_d)
                _cache[key] = result
        return _cache[key]

    return _load


@pytest.fixture(scope="session")
def bm_Q():
    beamfile_Q = os.path.join(DATA_PATH, "isotropic_beam.beamfits")
    bm_Q = pspecbeam.PSpecBeamUV(beamfile_Q)
    bm_Q.filename = "isotropic_beam.beamfits"
    return bm_Q


@pytest.fixture
def d(_miriad_raw):
    return copy.deepcopy(_miriad_raw(dfiles))


@pytest.fixture
def d_std(_miriad_raw):
    return copy.deepcopy(_miriad_raw(dfiles_std))


@pytest.fixture
def uvd(_miriad_raw):
    return copy.deepcopy(_miriad_raw("zen.2458042.17772.xx.HH.uvXA"))


@pytest.fixture
def uvd_std(_miriad_raw):
    return copy.deepcopy(_miriad_raw("zen.2458042.17772.std.xx.HH.uvXA"))


@pytest.fixture
def w():
    return [None, None]


@pytest.fixture
def dayenu_r_params() -> dict:
    """Standard dayenu r_params used across tests."""
    return {
        "filter_centers": [0.0],
        "filter_half_widths": [100e-9],
        "filter_factors": [1e-9],
    }


@pytest.fixture
def pspec_blpairs() -> tuple:
    """Standard (bls1, bls2, blpairs) derived from pspec_bls."""
    bls1, bls2, blpairs = utils.construct_blpairs(
        pspec_bls, exclude_auto_bls=True, exclude_permutations=True
    )
    return bls1, bls2, blpairs


@pytest.fixture
def pspec_ds(beam_nf_dipole, uvd):
    """PSpecData with two cross-products of uvd, NF dipole beam, and dataset labels."""
    return pspecdata.PSpecData(
        dsets=[uvd, uvd], wgts=[None, None], beam=beam_nf_dipole, labels=["red", "blue"]
    )


def test_init(uvd):
    # Test creating empty PSpecData
    ds = pspecdata.PSpecData()

    # Test whether unequal no. of weights is picked up
    with pytest.raises(AssertionError, match="The dsets and wgts lists must have equal length"):
        pspecdata.PSpecData(
            [uv.UVData(), uv.UVData(), uv.UVData()], [uv.UVData(), uv.UVData()]
        )

    # Test passing data and weights of the wrong type
    d_arr = np.ones((6, 8))
    d_lst = [[0, 1, 2] for i in range(5)]
    d_float = 12.0
    d_dict = {"(0,1)": np.arange(5), "(0,2)": np.arange(5)}
    with pytest.raises(TypeError, match="dsets, dsets_std, wgts and cals must be UVData"):
        pspecdata.PSpecData(d_arr, d_arr)
    with pytest.raises(TypeError, match="Only UVData objects can be used as datasets"):
        pspecdata.PSpecData(d_lst, d_lst)
    with pytest.raises(TypeError, match="object of type 'float' has no len\\(\\)"):
        pspecdata.PSpecData(d_float, d_float)
    with pytest.raises(TypeError, match="Only UVData objects can be used as datasets"):
        pspecdata.PSpecData([d_float], [d_float])
    with pytest.raises(TypeError, match='Only UVData objects can be used as datasets.'):
        pspecdata.PSpecData(d_dict, d_dict)
    with pytest.raises(TypeError, match="Only UVData objects can be used as datasets"):
        pspecdata.PSpecData([d_dict], [d_dict])

    # Test exception when not a UVData instance
    with pytest.raises(TypeError, match="Only UVData objects can be used as datasets"):
        ds.add([1], [None])

    # Test get weights when fed a UVData for weights
    ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[uvd, uvd])
    key = (0, (24, 25), "xx")
    assert np.all(np.isclose(ds.x(key), ds.w(key)))

    # Test labels when adding dsets
    ds = pspecdata.PSpecData()
    assert len(ds.labels) == 0
    ds.add([uvd, uvd], [None, None])
    assert len(ds.labels) == 2
    ds.add(uvd, None, labels="foo")
    assert len(ds.dsets) == len(ds.labels) == 3
    assert ds.labels == ["dset0", "dset1", "foo"]
    ds.add(uvd, None)
    assert ds.labels == ["dset0", "dset1", "foo", "dset3"]

    # Test some exceptions
    ds = pspecdata.PSpecData()
    with pytest.raises(ValueError, match="Number of delay bins should have been set"):
        ds.get_G(key, key)
    with pytest.raises(ValueError, match="Number of delay bins should have been set"):
        ds.get_H(key, key)


def test_add_data(d):
    """
    Test PSpecData add()
    """
    uv = d[0]
    # proper usage
    ds = pspecdata.PSpecData()
    ds1 = copy.deepcopy(ds)
    ds1.add(dsets=[uv], wgts=None, labels=None)
    # test adding non list objects
    with pytest.raises(TypeError, match="object of type 'int' has no len\\(\\)"):
        ds.add(1, 1)
    # test adding non UVData objects
    with pytest.raises(TypeError, match="Only UVData objects can be used as datasets"):
        ds.add([1], None)
    with pytest.raises(TypeError, match="Only UVData objects .or None. can be used as weights"):
        ds.add([uv], [1])
    with pytest.raises(TypeError, match="Only UVData objects .or None. can be used as error sets"):
        ds.add([uv], None, dsets_std=[1])
    # test adding UVData object with old array shape
    ds2 = copy.deepcopy(uv)
    ds2.data_array = np.tile(uv.data_array, [1, 1, 1, 1])
    with pytest.raises(TypeError, match="Only UVData objects .or None. can be used as weights"):
        ds.add([uv], [1])
    # test adding non UVCal for cals
    with pytest.raises(TypeError, match="Only UVCal objects can be used for calibration"):
        ds.add([uv], None, cals=[1])
    # test TypeError if dsets is dict but other inputs are not
    with pytest.raises(TypeError, match="If 'dsets' is a dict, 'wgts' must also be a dict"):
        ds.add({"d": uv}, [0])
    with pytest.raises(TypeError, match="If 'dsets' is a dict, 'dsets_std' must also be a dict"):
        ds.add({"d": uv}, {"d": uv}, dsets_std=[0])
    with pytest.raises(TypeError, match="If 'cals' is a dict, 'cals' must also be a dict"):
        ds.add({"d": uv}, {"d": uv}, cals=[0])
    # specifying labels when dsets is a dict is a ValueError
    with pytest.raises(ValueError, match="If 'dsets' is a dict, 'labels' cannot be specified"):
        ds.add({"d": uv}, None, labels=["d"])
    # use lists, but not appropriate lengths
    with pytest.raises(AssertionError, match="The dsets and wgts lists must have equal length"):
        ds.add([uv], [uv, uv])
    with pytest.raises(AssertionError, match="The dsets and dsets_std lists must have equal length"):
        ds.add([uv], None, dsets_std=[uv, uv])
    with pytest.raises(AssertionError, match="The dsets and cals lists must have equal length"):
        ds.add([uv], None, cals=[None, None])
    with pytest.raises(AssertionError, match="If labels are specified, the dsets and labels lists must have equal length"):
        ds.add([uv], None, labels=["foo", "bar"])


def test_set_symmetric_taper(d, w, dayenu_r_params):
    """
    Make sure that you can't set a symmtric taper with an truncated R matrix
    """
    ds = pspecdata.PSpecData(dsets=d, wgts=w)
    Nfreq = ds.spw_Nfreqs
    Ntime = ds.Ntimes
    Ndlys = Nfreq - 3
    ds.spw_Ndlys = Ndlys

    # Set baselines to use for tests
    key1 = (0, 24, 38)
    key2 = (1, 25, 38)
    key3 = [(0, 24, 38), (0, 24, 38)]
    key4 = [(1, 25, 38), (1, 25, 38)]

    ds.set_weighting("dayenu")
    ds.set_r_param(key1, dayenu_r_params)
    ds.set_r_param(key2, dayenu_r_params)
    ds1 = copy.deepcopy(ds)
    ds1.set_spw((10, Nfreq - 10))
    ds1.set_symmetric_taper(False)
    ds1.set_filter_extension([10, 10])
    ds1.set_filter_extension((10, 10))
    rm1 = ds.R(key1)
    ds.set_symmetric_taper(True)
    with pytest.raises(ValueError):
        ds1.set_symmetric_taper(True)
    # now make sure warnings are raised when we extend filter with
    # symmetric tapering and that symmetric taper is set to false.
    with pytest.warns(UserWarning, match="filter_extension\\[[01]\\] exceeds"):
        ds.set_filter_extension((10, 10))
    assert not (ds.symmetric_taper)

    """
    Now directly compare results to expectations.
    """
    ds = pspecdata.PSpecData(dsets=d, wgts=w)
    Nfreq = ds.spw_Nfreqs
    Ntime = ds.Ntimes
    Ndlys = Nfreq - 3
    ds.spw_Ndlys = Ndlys
    key1 = (0, 24, 38)
    key2 = (1, 25, 38)

    ds.set_weighting("dayenu")
    ds.set_taper("bh7")
    ds.set_r_param(key1, dayenu_r_params)
    # get the symmetric tapering
    rmat_symmetric = ds.R(key1)
    # now set taper to be asymmetric
    ds.set_symmetric_taper(False)
    rmat_a = ds.R(key1)
    # check against independent solution
    bh_taper = np.sqrt(dspec.gen_window("bh7", Nfreq).reshape(1, -1))
    rmat = dspec.dayenu_mat_inv(
        x=ds.freqs[ds.spw_range[0] : ds.spw_range[1]],
        filter_centers=[0.0],
        filter_half_widths=[100e-9],
        filter_factors=[1e-9],
    )
    wmat = np.outer(np.diag(np.sqrt(ds.Y(key1))), np.diag(np.sqrt(ds.Y(key1))))
    rmat = np.linalg.pinv(wmat * rmat)
    assert np.all(np.isclose(rmat_symmetric, bh_taper.T * rmat * bh_taper, atol=1e-6))
    assert np.all(np.isclose(rmat_a, bh_taper.T**2.0 * rmat, atol=1e-6))
    assert not np.all(np.isclose(rmat_symmetric, rmat_a, atol=1e-6))


def test_labels(d, w):
    """
    Test that dataset labels work.
    """
    # Check that specifying labels does work
    psd = pspecdata.PSpecData(
        dsets=[d[0], d[1]], wgts=[w[0], w[1]], labels=["red", "blue"]
    )
    np.testing.assert_array_equal(psd.x(("red", 24, 38)), psd.x((0, 24, 38)))

    # Check specifying labels using dicts
    dsdict = {"a": d[0], "b": d[1]}
    psd = pspecdata.PSpecData(dsets=dsdict, wgts=dsdict)
    with pytest.raises(ValueError, match="If 'dsets' is a dict, 'labels' cannot be specified"):
        pspecdata.PSpecData(dsets=dsdict, wgts=dsdict, labels=["a", "b"])

    # Check that invalid labels raise errors
    with pytest.raises(KeyError, match="not found"):
        psd.x(("green", 24, 38))


def test_parse_blkey(uvd):
    # make a double-pol UVData
    uvd_orig = copy.deepcopy(uvd)
    uvd_orig.polarization_array[0] = -7
    uvd = uvd_orig + uvd
    # check parse_blkey
    ds = pspecdata.PSpecData(
        dsets=[uvd, uvd], wgts=[None, None], labels=["red", "blue"]
    )
    dset, bl = ds.parse_blkey((0, (24, 25)))
    assert dset == 0
    assert bl == (24, 25)
    dset, bl = ds.parse_blkey(("red", (24, 25), "xx"))
    assert dset == 0
    assert bl == (24, 25, "xx")
    # check PSpecData.x works
    assert ds.x(("red", (24, 25))).shape == (2, 64, 60)
    assert ds.x(("red", (24, 25), "xx")).shape == (64, 60)
    assert ds.w(("red", (24, 25))).shape == (2, 64, 60)
    assert ds.w(("red", (24, 25), "xx")).shape == (64, 60)


def test_str(uvd):
    """
    Check that strings can be output.
    """
    ds = pspecdata.PSpecData()
    print(ds)  # print empty psd
    ds.add(uvd, None)
    print(ds)  # print populated psd


def test_get_Q_alt():
    """
    Test the Q = dC/dp function.
    """
    vect_length = 50
    x_vect = np.random.normal(size=vect_length) + 1.0j * np.random.normal(
        size=vect_length
    )
    y_vect = np.random.normal(size=vect_length) + 1.0j * np.random.normal(
        size=vect_length
    )

    ds = pspecdata.PSpecData()
    ds.spw_Nfreqs = vect_length

    for i in range(vect_length):
        Q_matrix = ds.get_Q_alt(i)
        # Test that if the number of delay bins hasn't been set
        # the code defaults to putting that equal to Nfreqs
        assert ds.spw_Ndlys == ds.spw_Nfreqs

        xQy = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, y_vect))
        yQx = np.dot(np.conjugate(y_vect), np.dot(Q_matrix, x_vect))
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))

        # Test that Q matrix has the right shape
        assert Q_matrix.shape == (vect_length, vect_length)

        # Test that x^t Q y == conj(y^t Q x)
        np.testing.assert_almost_equal(xQy, np.conjugate(yQx))

        # x^t Q x should be real
        np.testing.assert_almost_equal(np.imag(xQx), 0.0)

    x_vect = np.ones(vect_length)
    Q_matrix = ds.get_Q_alt(vect_length // 2)
    xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
    np.testing.assert_almost_equal(xQx, np.abs(vect_length**2.0))
    # Sending in sinusoids for x and y should give delta functions

    # Now do all the same tests from above but for a different number
    # of delay channels
    ds.set_Ndlys(vect_length - 3)
    for i in range(vect_length - 3):
        Q_matrix = ds.get_Q_alt(i)
        xQy = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, y_vect))
        yQx = np.dot(np.conjugate(y_vect), np.dot(Q_matrix, x_vect))
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))

        # Test that Q matrix has the right shape
        assert Q_matrix.shape == (vect_length, vect_length)

        # Test that x^t Q y == conj(y^t Q x)
        np.testing.assert_almost_equal(xQy, np.conjugate(yQx))

        # x^t Q x should be real
        np.testing.assert_almost_equal(np.imag(xQx), 0.0)

    x_vect = np.ones(vect_length)
    Q_matrix = ds.get_Q_alt((vect_length - 2) // 2 - 1)
    xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
    np.testing.assert_almost_equal(xQx, np.abs(vect_length**2.0))
    # Sending in sinusoids for x and y should give delta functions

    # Make sure that error is raised when asking for a delay mode outside
    # of the range of delay bins
    with pytest.raises(IndexError, match="Cannot compute Q matrix for a mode outside"):
        ds.get_Q_alt(vect_length - 1)

    # Ensure that in the special case where the number of channels equals
    # the number of delay bins, the FFT method gives the same answer as
    # the explicit construction method
    multiplicative_tolerance = 0.001
    ds.set_Ndlys(vect_length)
    for alpha in range(vect_length):
        Q_matrix_fft = ds.get_Q_alt(alpha)
        Q_matrix = ds.get_Q_alt(alpha, allow_fft=False)
        Q_diff_norm = np.linalg.norm(Q_matrix - Q_matrix_fft)
        assert Q_diff_norm <= multiplicative_tolerance

    # Check for error handling
    with pytest.raises(ValueError, match="Cannot estimate more delays than there are frequency channels"):
        ds.set_Ndlys(vect_length + 100)


def test_get_Q(uvd):
    """
    Test the Q = dC_ij/dp function.

    A general comment here:
    I would really want to do away with try and exception statements. The reason to use them now
    was that current unittests throw in empty datasets to these functions. Given that we are computing
    the actual value of tau/freq/taper etc. we do need datasets! Currently, if there is no dataset,
    Q_matrix is simply an identity matrix with same dimensions as that of vector length.
    It will be very helpful if we can have more elegant solution for this.

    """
    vect_length = 50
    x_vect = np.random.normal(size=vect_length) + 1.0j * np.random.normal(
        size=vect_length
    )
    y_vect = np.random.normal(size=vect_length) + 1.0j * np.random.normal(
        size=vect_length
    )

    ds = pspecdata.PSpecData()
    ds.spw_Nfreqs = vect_length
    # Test if there is a warning if user does not pass the beam
    key1 = (0, 24, 38)
    key2 = (1, 24, 38)
    uvd = copy.deepcopy(uvd)
    ds_t = pspecdata.PSpecData(dsets=[uvd, uvd])

    for i in range(vect_length):
        try:
            Q_matrix = ds.get_Q(i)
            # Test that if the number of delay bins hasn't been set
            # the code defaults to putting that equal to Nfreqs
            assert ds.spw_Ndlys == ds.spw_Nfreqs
        except IndexError:
            Q_matrix = np.ones((vect_length, vect_length))

        xQy = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, y_vect))
        yQx = np.dot(np.conjugate(y_vect), np.dot(Q_matrix, x_vect))
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))

        # Test that Q matrix has the right shape
        assert Q_matrix.shape == (vect_length, vect_length)

        # Test that x^t Q y == conj(y^t Q x)
        np.testing.assert_almost_equal(xQy, np.conjugate(yQx))

        # x^t Q x should be real
        np.testing.assert_almost_equal(np.imag(xQx), 0.0)

    x_vect = np.ones(vect_length)
    try:
        Q_matrix = ds.get_Q(vect_length / 2)
    except IndexError:
        Q_matrix = np.ones((vect_length, vect_length))
    xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
    np.testing.assert_almost_equal(xQx, np.abs(vect_length**2.0))

    # Now do all the same tests from above but for a different number
    # of delay channels
    ds.set_Ndlys(vect_length - 3)
    for i in range(vect_length - 3):
        try:
            Q_matrix = ds.get_Q(i)
        except IndexError:
            Q_matrix = np.ones((vect_length, vect_length))
        xQy = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, y_vect))
        yQx = np.dot(np.conjugate(y_vect), np.dot(Q_matrix, x_vect))
        xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))

        # Test that Q matrix has the right shape
        assert Q_matrix.shape == (vect_length, vect_length)

        # Test that x^t Q y == conj(y^t Q x)
        np.testing.assert_almost_equal(xQy, np.conjugate(yQx))

        # x^t Q x should be real
        np.testing.assert_almost_equal(np.imag(xQx), 0.0)

    x_vect = np.ones(vect_length)
    try:
        Q_matrix = ds.get_Q((vect_length - 2) / 2 - 1)
    except IndexError:
        Q_matrix = np.ones((vect_length, vect_length))
    xQx = np.dot(np.conjugate(x_vect), np.dot(Q_matrix, x_vect))
    np.testing.assert_almost_equal(xQx, np.abs(vect_length**2.0))

    # Make sure that error is raised when asking for a delay mode outside
    # of the range of delay bins
    with pytest.raises(IndexError):
        ds.get_Q(vect_length - 1)


def test_get_integral_beam(beam_nf_dipole, uvd):
    """
    Test the integral of the beam and tapering function in Q.
    """
    pol = "xx"
    # Test if there is a warning if user does not pass the beam
    uvd = copy.deepcopy(uvd)
    ds_t = pspecdata.PSpecData(dsets=[uvd, uvd])
    ds = pspecdata.PSpecData(dsets=[uvd, uvd], beam=beam_nf_dipole)

    with pytest.warns(UserWarning, match="The beam response could not be calculated"):
        ds_t.get_integral_beam(pol)

    try:
        integral_matrix = ds.get_integral_beam(pol)
        # Test that if the number of delay bins hasn't been set
        # the code defaults to putting that equal to Nfreqs
        assert ds.spw_Ndlys == ds.spw_Nfreqs
    except IndexError:
        integral_matrix = np.ones((ds.spw_Ndlys, ds.spw_Ndlys))

    # Test that integral matrix has the right shape
    assert integral_matrix.shape == (ds.spw_Nfreqs, ds.spw_Nfreqs)


def test_get_unnormed_E(beam_nf_dipole, uvd):
    """
    Test the E function
    """
    # Test that error is raised if spw_Ndlys is not set
    uvd = copy.deepcopy(uvd)
    ds = pspecdata.PSpecData(
        dsets=[uvd, uvd], wgts=[None, None], labels=["red", "blue"]
    )
    ds.spw_Ndlys = None
    with pytest.raises(ValueError, match="Number of delay bins should have been set"):
        ds.get_unnormed_E("placeholder", "placeholder")

    # Test that if R1 = R2, then the result is Hermitian
    ds.spw_Ndlys = 7
    random_R = generate_pos_def_all_pos(ds.spw_Nfreqs)
    wgt_matrix_dict = {}  # The keys here have no significance except they are formatted right
    wgt_matrix_dict[("red", (24, 25))] = random_R
    wgt_matrix_dict[("blue", (24, 25))] = random_R
    ds.set_R(wgt_matrix_dict)
    E_matrices = ds.get_unnormed_E(("red", (24, 25)), ("blue", (24, 25)))
    multiplicative_tolerance = 0.0000001
    for matrix in E_matrices:
        diff_norm = np.linalg.norm(matrix.T.conj() - matrix)
        assert diff_norm <= multiplicative_tolerance

    # Test for the correct shape when exact_norm is True
    ds_c = pspecdata.PSpecData(
        dsets=[uvd, uvd], wgts=[None, None], labels=["red", "blue"], beam=beam_nf_dipole
    )
    ds_c.spw_Ndlys = 10
    random_R = generate_pos_def_all_pos(ds_c.spw_Nfreqs)
    wgt_matrix_dict = {}
    wgt_matrix_dict[("red", (24, 25))] = random_R
    wgt_matrix_dict[("blue", (24, 25))] = random_R
    ds_c.set_R(wgt_matrix_dict)
    E_matrices = ds_c.get_unnormed_E(
        ("red", (24, 25)), ("blue", (24, 25)), exact_norm=True, pol="xx"
    )
    assert E_matrices.shape == (ds_c.spw_Ndlys, ds_c.spw_Nfreqs, ds_c.spw_Nfreqs)

    # Test that if R1 != R2, then i) E^{12,dagger} = E^{21}
    random_R2 = generate_pos_def_all_pos(ds.spw_Nfreqs)
    wgt_matrix_dict = {}
    wgt_matrix_dict[("red", (24, 25))] = random_R
    wgt_matrix_dict[("blue", (24, 25))] = random_R2
    ds.set_R(wgt_matrix_dict)
    E12_matrices = ds.get_unnormed_E(("red", (24, 25)), ("blue", (24, 25)))
    E21_matrices = ds.get_unnormed_E(("blue", (24, 25)), ("red", (24, 25)))
    multiplicative_tolerance = 0.0000001
    for mat12, mat21 in zip(E12_matrices, E21_matrices):
        diff_norm = np.linalg.norm(mat12.T.conj() - mat21)
        assert diff_norm <= multiplicative_tolerance

    # Test that if there is only one delay bin and R1 = R2 = I, then
    # the E matrices are all 0.5s exept in flagged channels.
    ds.spw_Ndlys = 1
    wgt_matrix_dict = {}
    wgt_matrix_dict[("red", (24, 25))] = np.eye(ds.spw_Nfreqs)
    wgt_matrix_dict[("blue", (24, 25))] = np.eye(ds.spw_Nfreqs)
    flags1 = np.diag(ds.Y(("red", (24, 25))))
    flags2 = np.diag(ds.Y(("blue", (24, 25))))
    ds.set_R(wgt_matrix_dict)
    E_matrices = ds.get_unnormed_E(("red", (24, 25)), ("blue", (24, 25)))
    multiplicative_tolerance = 0.0000001
    for matrix in E_matrices:
        for i in range(ds.spw_Nfreqs):
            for j in range(ds.spw_Nfreqs):
                if flags1[i] * flags2[j] == 0:  # either channel flagged
                    np.testing.assert_almost_equal(matrix[i, j], 0.0)
                else:
                    np.testing.assert_almost_equal(matrix[i, j], 0.5)


def test_cross_covar_model(uvd):
    uvd = copy.deepcopy(uvd)
    ds = pspecdata.PSpecData(
        dsets=[uvd, uvd], wgts=[None, None], labels=["red", "blue"]
    )
    key1 = ("red", (24, 25), "xx")
    key2 = ("blue", (25, 38), "xx")
    with pytest.raises(ValueError, match="didn't recognize model"):
        ds.cross_covar_model(key1, key2, model="other_string")
    with pytest.raises(AssertionError, match="key2 must be fed as a tuple"):
        ds.cross_covar_model(key1, "a_string")

    conj1_conj1 = ds.cross_covar_model(key1, key1, conj_1=True, conj_2=True)
    conj1_real1 = ds.cross_covar_model(key1, key1, conj_1=True, conj_2=False)
    real1_conj1 = ds.cross_covar_model(key1, key1, conj_1=False, conj_2=True)
    real1_real1 = ds.cross_covar_model(key1, key1, conj_1=False, conj_2=False)

    # Check matrix sizes
    for matrix in [conj1_conj1, conj1_real1, real1_conj1, real1_real1]:
        assert matrix.shape == (ds.spw_Nfreqs, ds.spw_Nfreqs)
    for j in range(ds.spw_Nfreqs):
        for k in range(ds.spw_Nfreqs):
            # Check that the matrices that ought to be Hermitian are indeed Hermitian
            np.testing.assert_almost_equal(conj1_real1.conj()[k, j], conj1_real1[j, k])
            np.testing.assert_almost_equal(real1_conj1.conj()[k, j], real1_conj1[j, k])
            # Check that real_real and conj_conj are complex conjugates of each other
            # Also check that they are symmetric
            np.testing.assert_almost_equal(real1_real1.conj()[j, k], conj1_conj1[j, k])
            np.testing.assert_almost_equal(real1_real1[k, j], real1_real1[j, k])
            np.testing.assert_almost_equal(conj1_conj1[k, j], conj1_conj1[j, k])

    real1_real2 = ds.cross_covar_model(key1, key2, conj_1=False, conj_2=False)
    real2_real1 = ds.cross_covar_model(key2, key1, conj_1=False, conj_2=False)
    conj1_conj2 = ds.cross_covar_model(key1, key2, conj_1=True, conj_2=True)
    conj2_conj1 = ds.cross_covar_model(key2, key1, conj_1=True, conj_2=True)
    conj1_real2 = ds.cross_covar_model(key1, key2, conj_1=True, conj_2=False)
    conj2_real1 = ds.cross_covar_model(key2, key1, conj_1=True, conj_2=False)
    real1_conj2 = ds.cross_covar_model(key1, key2, conj_1=False, conj_2=True)
    real2_conj1 = ds.cross_covar_model(key2, key1, conj_1=False, conj_2=True)

    # And some similar tests for cross covariances
    for j in range(ds.spw_Nfreqs):
        for k in range(ds.spw_Nfreqs):
            np.testing.assert_almost_equal(real1_real2[k, j], real2_real1[j, k])
            np.testing.assert_almost_equal(conj1_conj2[k, j], conj2_conj1[j, k])
            np.testing.assert_almost_equal(conj1_real2.conj()[k, j], conj2_real1[j, k])
            np.testing.assert_almost_equal(real1_conj2.conj()[k, j], real2_conj1[j, k])


def test_get_unnormed_V(d, w):
    ds = pspecdata.PSpecData(dsets=d, wgts=w, labels=["red", "blue"])
    key1 = ("red", (24, 25), "xx")
    key2 = ("blue", (25, 38), "xx")
    ds.spw_Ndlys = 5

    V = ds.get_unnormed_V(key1, key2)
    # Check size
    assert V.shape == (ds.spw_Ndlys, ds.spw_Ndlys)
    # Test hermiticity. Generally this is only good to about 1 part in 10^15.
    # If this is an issue downstream, should investigate more in the future.
    tol = 1e-10
    frac_non_herm = abs(V.conj().T - V) / abs(V)
    for i in range(ds.spw_Ndlys):
        for j in range(ds.spw_Ndlys):
            assert frac_non_herm[i, j] <= tol


def test_get_MW():
    ds = pspecdata.PSpecData()

    n = 17
    random_G = generate_pos_def_all_pos(n)
    random_H = generate_pos_def_all_pos(n)
    random_V = generate_pos_def_all_pos(n)
    with pytest.raises(AssertionError):
        ds.get_MW(random_G, random_H, mode="L^3")
    with pytest.raises(NotImplementedError, match="Exact norm is not supported for non-I modes"):
        ds.get_MW(random_G, random_H, mode="H^-1", exact_norm=True)

    for mode in ["H^-1", "V^-1/2", "I", "L^-1"]:
        if mode == "H^-1":
            # Test that if we have full-rank matrices, the resulting window functions
            # are indeed delta functions
            M, W = ds.get_MW(random_G, random_H, mode=mode)
            Hinv = np.linalg.inv(random_H)
            for i in range(n):
                np.testing.assert_almost_equal(W[i, i], 1.0)
                for j in range(n):
                    np.testing.assert_almost_equal(M[i, j], Hinv[i, j])

            # When the matrices are not full rank, test that the window functions
            # are at least properly normalized.
            deficient_H = np.ones((3, 3))
            M, W = ds.get_MW(deficient_H, deficient_H, mode=mode)
            norm = np.sum(W, axis=1)
            for i in range(3):
                np.testing.assert_almost_equal(norm[i], 1.0)

            # Check that the method ignores G
            M, W = ds.get_MW(random_G, random_H, mode=mode)
            M_other, W_other = ds.get_MW(random_H, random_H, mode=mode)
            for i in range(n):
                for j in range(n):
                    np.testing.assert_almost_equal(M[i, j], M_other[i, j])
                    np.testing.assert_almost_equal(W[i, j], W_other[i, j])

        elif mode == "V^-1/2":
            # Test that we are checking for the presence of a covariance matrix
            with pytest.raises(ValueError, match="Covariance not supplied for V.-1/2 normalization"):
                ds.get_MW(random_G, random_H, mode=mode)
            # Test that the error covariance is diagonal
            M, W = ds.get_MW(random_G, random_H, mode=mode, band_covar=random_V)
            band_covar = np.dot(M, np.dot(random_V, M.T))
            assert diagonal_or_not(band_covar)

        elif mode == "I":
            # Test that the norm matrix is diagonal
            M, W = ds.get_MW(random_G, random_H, mode=mode)
            assert diagonal_or_not(M)
        elif mode == "L^-1":
            # Test that Cholesky mode is disabled
            with pytest.raises(NotImplementedError, match="Cholesky decomposition mode not currently supported"):
                ds.get_MW(random_G, random_H, mode=mode)

        # Test sizes for everyone
        assert M.shape == (n, n)
        assert W.shape == (n, n)

        # Window function matrices should have each row sum to unity
        # regardless of the mode chosen
        test_norm = np.sum(W, axis=1)
        for norm in test_norm:
            np.testing.assert_almost_equal(norm, 1.0)


@pytest.fixture
def cov_q_setup(d, d_std, w):
    """PSpecData + analytic covariance matrix for test_cov_q tests."""
    ndlys = 13
    dlist = copy.deepcopy(d)
    dlist_std = copy.deepcopy(d_std)
    for _d in dlist:
        _d.flag_array[:] = False
        _d.select(times=np.unique(_d.time_array)[:10], frequencies=_d.freq_array[:16])
    for _d_std in dlist_std:
        _d_std.flag_array[:] = False
        _d_std.select(
            times=np.unique(_d_std.time_array)[:10], frequencies=_d_std.freq_array[:16]
        )
    ds = pspecdata.PSpecData(dsets=dlist, wgts=w, dsets_std=dlist_std)
    ds.set_Ndlys(ndlys)
    chan_x, chan_y = np.meshgrid(range(ds.Nfreqs), range(ds.Nfreqs))
    cov_analytic = np.zeros((ds.spw_Ndlys, ds.spw_Ndlys), dtype=np.complex128)
    for alpha in range(ds.spw_Ndlys):
        for beta in range(ds.spw_Ndlys):
            cov_analytic[alpha, beta] = np.exp(
                -2j * np.pi * (alpha - beta) * (chan_x - chan_y) / ds.spw_Ndlys
            ).sum()
    key1 = (0, 24, 38)
    key2 = (1, 25, 38)
    return ds, cov_analytic, key1, key2


@pytest.mark.parametrize("input_data_weight", weight_selection)
def test_cov_q(cov_q_setup, input_data_weight):
    """
    Test that q_hat_cov has the right shape and accepts keys in correct
    format. Also validate with arbitrary number of delays.
    """
    ds, _, key1, key2 = cov_q_setup

    ds.set_weighting(input_data_weight)
    if input_data_weight == "dayenu":
        with pytest.raises(ValueError):
            ds.R(key1)
        rpk = {
            "filter_centers": [0.0],
            "filter_half_widths": [0.0],
            "filter_factors": [0.0],
        }
        ds.set_r_param(key1, rpk)
        ds.set_r_param(key2, rpk)
    # Run twice: first call may warn for iC (poorly conditioned R), second uses cache.
    for taper_idx in range(len(taper_selection)):
        warn_ctx = (
            pytest.warns(UserWarning, match="Poorly conditioned covariance")
            if input_data_weight == "iC" and taper_idx == 0
            else nullcontext()
        )
        with warn_ctx:
            qc = ds.cov_q_hat(key1, key2, model="dsets")
        assert np.allclose(
            np.array(list(qc.shape)),
            np.array([ds.Ntimes, ds.spw_Ndlys, ds.spw_Ndlys]),
            atol=1e-6,
        )
        qc = ds.cov_q_hat(key1, key2, model="empirical")
        assert np.allclose(
            np.array(list(qc.shape)),
            np.array([ds.Ntimes, ds.spw_Ndlys, ds.spw_Ndlys]),
            atol=1e-6,
        )


def test_cov_q_analytic(cov_q_setup):
    """Test that analytic covariance gives Nchan^2, and validate key-list API."""
    ds, cov_analytic, key1, key2 = cov_q_setup

    ds.set_weighting("identity")
    qc = ds.cov_q_hat(key1, key2, model="dsets")
    assert np.allclose(
        qc, np.repeat(cov_analytic[np.newaxis, :, :], ds.Ntimes, axis=0), atol=1e-6
    )
    qc = ds.cov_q_hat([key1], [key2], time_indices=[0], model="dsets")
    assert np.allclose(
        qc, np.repeat(cov_analytic[np.newaxis, :, :], ds.Ntimes, axis=0), atol=1e-6
    )
    with pytest.raises(ValueError, match="Invalid time index provided"):
        ds.cov_q_hat(key1, key2, time_indices=200)
    with pytest.raises(ValueError, match="time_indices must be an integer or list of integers"):
        ds.cov_q_hat(key1, key2, time_indices="watch out!")


def test_cov_p_hat(d, d_std, w):
    """
    Test cov_p_hat, verify on identity.
    """
    ds = pspecdata.PSpecData(dsets=d, wgts=w, dsets_std=d_std)
    cov_p = ds.cov_p_hat(
        np.sqrt(6.0) * np.identity(10), np.array([5.0 * np.identity(10)])
    )
    for p in range(10):
        for q in range(10):
            if p == q:
                assert np.isclose(30.0, cov_p[0, p, q], atol=1e-6)
            else:
                assert np.isclose(0.0, cov_p[0, p, q], atol=1e-6)


def test_R_truncation(d, w, dayenu_r_params):
    """
    Test truncation of R-matrices. These should give a q_hat that is all
    zeros outside of the with f-start and f-end.
    """
    ds = pspecdata.PSpecData(dsets=d, wgts=w)
    Nfreq = ds.spw_Nfreqs
    Ntime = ds.Ntimes
    Ndlys = Nfreq - 3
    ds.spw_Ndlys = Ndlys

    # Set baselines to use for tests
    key1 = (0, 24, 38)
    key2 = (1, 25, 38)
    key3 = [(0, 24, 38), (0, 24, 38)]
    key4 = [(1, 25, 38), (1, 25, 38)]

    ds.set_weighting("dayenu")
    ds.set_r_param(key1, dayenu_r_params)
    ds.set_r_param(key2, dayenu_r_params)
    ds1 = copy.deepcopy(ds)
    ds1.set_spw((10, Nfreq - 10))
    ds1.set_symmetric_taper(False)
    ds1.set_filter_extension([10, 10])
    ds1.set_filter_extension((10, 10))
    rm1 = ds.R(key1)
    rm2 = ds1.R(key2)
    rm3 = ds1.R(key1)
    assert np.shape(rm2) == (ds1.spw_Nfreqs, ds.spw_Nfreqs)
    # check that all values that are not truncated match values of untrancated matrix.
    assert np.all(np.isclose(rm1[10:-10], rm2, atol=1e-6))
    # make sure no errors are thrown by get_V, get_E, etc...
    ds1.get_unnormed_E(key1, key2)
    ds1.get_unnormed_V(key1, key2)
    h = ds1.get_H(key1, key2)
    g = ds1.get_G(key1, key2)
    ds1.get_MW(g, h)
    # make sure identity weighting isn't broken.
    ds = pspecdata.PSpecData(dsets=d, wgts=w)
    ds1 = copy.deepcopy(ds)
    ds1.set_spw((10, Nfreq - 10))
    ds1.set_weighting("identity")
    ds1.set_symmetric_taper(False)
    ds1.set_filter_extension([10, 10])
    rm1 = ds1.R(key1)


def test_q_hat(d, w):
    """
    Test that q_hat has right shape and accepts keys in the right format.
    """
    # Set weights and pack data into PSpecData
    ds = pspecdata.PSpecData(dsets=d, wgts=w)
    Nfreq = ds.Nfreqs
    Ntime = ds.Ntimes
    Ndlys = Nfreq - 3
    ds.spw_Ndlys = Ndlys

    # Set baselines to use for tests
    key1 = (0, 24, 38)
    key2 = (1, 25, 38)
    key3 = [(0, 24, 38), (0, 24, 38)]
    key4 = [(1, 25, 38), (1, 25, 38)]

    for input_data_weight in weight_selection:
        ds.set_weighting(input_data_weight)
        if input_data_weight == "dayenu":
            with pytest.raises(ValueError, match="r_param not set"):
                ds.R(key1)
            rpk = {
                "filter_centers": [0.0],
                "filter_half_widths": [0.0],
                "filter_factors": [0.0],
            }
            ds.set_r_param(key1, rpk)
            ds.set_r_param(key2, rpk)
        # Loop over list of taper functions
        for taper_idx, taper in enumerate(taper_selection):
            ds.set_taper(taper)
            warn_ctx = (
                pytest.warns(UserWarning, match="Poorly conditioned covariance")
                if input_data_weight == "iC" and taper_idx == 0
                else nullcontext()
            )

            # Calculate q_hat for a pair of baselines and test output shape
            with warn_ctx:
                q_hat_a = ds.q_hat(key1, key2)
            assert q_hat_a.shape == (Ndlys, Ntime)

            # Check that swapping x_1 <-> x_2 results in complex conj. only
            q_hat_b = ds.q_hat(key2, key1)
            q_hat_diff = np.conjugate(q_hat_a) - q_hat_b
            for i in range(Ndlys):
                for j in range(Ntime):
                    np.testing.assert_almost_equal(
                        q_hat_diff[i, j].real, q_hat_diff[i, j].real
                    )
                    np.testing.assert_almost_equal(
                        q_hat_diff[i, j].imag, q_hat_diff[i, j].imag
                    )

            # Check that lists of keys are handled properly
            q_hat_aa = ds.q_hat(key1, key4)  # q_hat(x1, x2+x2)
            q_hat_bb = ds.q_hat(key4, key1)  # q_hat(x2+x2, x1)
            q_hat_cc = ds.q_hat(key3, key4)  # q_hat(x1+x1, x2+x2)

            # Effectively checks that q_hat(2*x1, 2*x2) = 4*q_hat(x1, x2)
            for i in range(Ndlys):
                for j in range(Ntime):
                    np.testing.assert_almost_equal(
                        q_hat_a[i, j].real, 0.25 * q_hat_cc[i, j].real
                    )
                    np.testing.assert_almost_equal(
                        q_hat_a[i, j].imag, 0.25 * q_hat_cc[i, j].imag
                    )

    ds.spw_Ndlys = Nfreq
    # Check that the slow method is the same as the FFT method
    for input_data_weight in weight_selection:
        ds.set_weighting(input_data_weight)
        # Loop over list of taper functions
        for taper in taper_selection:
            ds.set_taper(taper)
            q_hat_a_slow = ds.q_hat(key1, key2, allow_fft=False)
            q_hat_a = ds.q_hat(key1, key2, allow_fft=True)
            assert np.isclose(np.real(q_hat_a / q_hat_a_slow), 1).all()
            assert np.isclose(np.imag(q_hat_a / q_hat_a_slow), 0, atol=1e-6).all()

    # Test if error is raised when one tried FFT approach on exact_norm
    with pytest.raises(NotImplementedError, match="Exact normalization does not support FFT approach"):
        ds.q_hat(key1, key2, exact_norm=True, allow_fft=True)


@pytest.mark.parametrize("taper", taper_selection)
@pytest.mark.parametrize("input_data_weight", weight_selection)
def test_get_H(d, w, input_data_weight, taper):
    """
    Test Fisher/weight matrix calculation.
    """
    ds = pspecdata.PSpecData(dsets=d, wgts=w)
    Nfreq = ds.Nfreqs
    key1 = (0, 24, 38)
    key2 = (1, 25, 38)

    ds.set_weighting(input_data_weight)
    if input_data_weight == "dayenu":
        with pytest.raises(ValueError):
            ds.R(key1)
        rpk = {
            "filter_centers": [0.0],
            "filter_half_widths": [0.0],
            "filter_factors": [0.0],
        }
        ds.set_r_param(key1, rpk)
        ds.set_r_param(key2, rpk)
    ds.set_taper(taper)
    warn_ctx = (
        pytest.warns(UserWarning, match="Poorly conditioned covariance")
        if input_data_weight == "iC"
        else nullcontext()
    )

    ds.set_Ndlys(Nfreq // 3)
    with warn_ctx:
        H = ds.get_H(key1, key2)
    assert H.shape == (Nfreq // 3, Nfreq // 3)

    ds.set_Ndlys()
    H = ds.get_H(key1, key2)
    assert H.shape == (Nfreq, Nfreq)


@pytest.mark.parametrize("taper", taper_selection)
@pytest.mark.parametrize("input_data_weight", weight_selection)
def test_get_G(d, w, input_data_weight, taper):
    """
    Test Fisher/weight matrix calculation.
    """
    ds = pspecdata.PSpecData(dsets=d, wgts=w)
    Nfreq = ds.Nfreqs
    multiplicative_tolerance = 1.0
    key1 = (0, 24, 38)
    key2 = (1, 25, 38)

    ds.set_weighting(input_data_weight)
    if input_data_weight == "dayenu":
        with pytest.raises(ValueError):
            ds.R(key1)
        rpk = {
            "filter_centers": [0.0],
            "filter_half_widths": [0.0],
            "filter_factors": [0.0],
        }
        ds.set_r_param(key1, rpk)
        ds.set_r_param(key2, rpk)
    ds.clear_cache()
    ds.set_taper(taper)
    warn_ctx = (
        pytest.warns(UserWarning, match="Poorly conditioned covariance")
        if input_data_weight == "iC"
        else nullcontext()
    )
    ds.set_Ndlys(Nfreq - 2)
    with warn_ctx:
        G = ds.get_G(key1, key2)
    assert G.shape == (Nfreq - 2, Nfreq - 2)  # Test shape
    matrix_scale = np.min(np.abs(np.linalg.eigvalsh(G)))

    if input_data_weight == "identity":
        # In the identity case, there are three special properties
        # that are respected:
        # i) Symmetry: G_ab = G_ba
        # ii) Cylic property: G = (1/2) tr[R1 Q_a R2 Q_b]
        #                       = (1/2) tr[R2 Q_b R1 Q_a]
        # iii) All elements of G are positive.

        # Test symmetry
        anti_sym_norm = np.linalg.norm(G - G.T)
        assert anti_sym_norm <= matrix_scale * multiplicative_tolerance

        # Test cyclic property of trace, where key1 and key2 can be
        # swapped without changing the matrix. This is secretly the
        # same test as the symmetry test, but perhaps there are
        # creative ways to break the code to break one test but not
        # the other.
        G_swapped = ds.get_G(key2, key1)
        G_diff_norm = np.linalg.norm(G - G_swapped)
        assert G_diff_norm <= matrix_scale * multiplicative_tolerance
        min_diagonal = np.min(np.diagonal(G))

        # Test that all elements of G are positive up to numerical
        # noise with the threshold set to 10 orders of magnitude
        # down from the smallest value on the diagonal
        for i in range(Nfreq - 2):
            for j in range(Nfreq - 2):
                assert G[i, j] >= -min_diagonal * multiplicative_tolerance
    else:
        # In general, when R_1 != R_2, there is a more restricted
        # symmetry where swapping R_1 and R_2 *and* taking the
        # transpose gives the same result
        # UPDATE: Taper now occurs after filter so this
        # symmetry only holds when taper = 'none'.
        # iC uses pseudo-inverse when poorly conditioned, breaking this symmetry.
        if taper == "none" and input_data_weight != "iC":
            G_swapped = ds.get_G(key2, key1)
            G_diff_norm = np.linalg.norm(G - G_swapped.T)
            assert G_diff_norm <= matrix_scale * multiplicative_tolerance


r"""
Under Construction
def test_parseval(ds, d, d_std, w, beam_nf_dipole, bm_Q, uvd, uvd_std):
    # Test that output power spectrum respects Parseval's theorem.
    np.random.seed(10)
    variance_in = 1.
    Nfreq = d[0].Nfreqs
    data = d[0]
    # Use only the requested number of channels
    data.select(freq_chans=range(Nfreq), bls=[(24,24),])
    # Make it so that the test data is unflagged everywhere
    data.flag_array[:] = False
    # Get list of available baselines and LSTs
    bls = data.get_antpairs()
    nlsts = data.Ntimes
    # Simulate data given a Fourier-space power spectrum
    pk = variance_in * np.ones(Nfreq)
    # Make realisation of (complex) white noise in real space
    g = 1.0 * np.random.normal(size=(nlsts,Nfreq)) \
      + 1.j * np.random.normal(size=(nlsts,Nfreq))
    g /= np.sqrt(2.) # Since Re(C) = Im(C) = C/2
    x = data.freq_array[0]
    dx = x[1] - x[0]
    # Fourier transform along freq. direction in each LST bin
    gnew = np.zeros(g.shape).astype(complex)
    fnew = np.zeros(g.shape).astype(complex)
    for i in range(nlsts):
        f = np.fft.fft(g[i]) * np.sqrt(pk)
        fnew[i] = f
        gnew[i] = np.fft.ifft(f)
    # Parseval's theorem test: integral of F^2(k) dk = integral of f^2(x) dx
    k = np.fft.fftshift( np.fft.fftfreq(Nfreq, d=(x[1]-x[0])) )
    fsq = np.fft.fftshift( np.mean(fnew * fnew.conj(), axis=0) )
    gsq = np.mean(gnew * gnew.conj(), axis=0)
    # Realize set of Gaussian random datasets and pack into PSpecData
    data.data_array = np.expand_dims(np.expand_dims(gnew, axis=1), axis=3)
    ds = pspecdata.PSpecData()
    ds.add([data, data], [None, None])
    # Use true covariance instead
    exact_cov = {
        (0,24,24): np.eye(Nfreq),
        (1,24,24): np.eye(Nfreq)
    }
    ds.set_C(exact_cov)

    # Calculate OQE power spectrum using true covariance matrix
    tau = np.fft.fftshift( ds.delays() )
    ps, _ = ds.pspec(bls, input_data_weight='iC', norm='I')
    ps_avg = np.fft.fftshift( np.mean(ps[0], axis=1) )

    # Calculate integrals for Parseval's theorem
    parseval_real = simpson(gsq, x)
    parseval_ft = dx**2. * simpson(fsq, k)
    parseval_phat = simpson(ps_avg, tau)

    # Report on results for different ways of calculating Parseval integrals
    print "Parseval's theorem:"
    print "  \int [g(x)]^2 dx = %3.6e, %3.6e" % (parseval_real.real,
                                                 parseval_real.imag)
    print "  \int [f(k)]^2 dk = %3.6e, %3.6e" % (parseval_ft.real,
                                                 parseval_ft.imag)
    print "  \int p_hat(k) dk = %3.6e, %3.6e" % (parseval_phat.real,
                                                 parseval_phat.imag)

    # Perform approx. equality test (this is a stochastic quantity, so we
    # only expect equality to ~10^-2 to 10^-3
    np.testing.assert_allclose(parseval_phat, parseval_real, rtol=1e-3)
"""


def test_scalar_delay_adjustment(d, w, beam_nf_dipole):
    ds = pspecdata.PSpecData(dsets=d, wgts=w, beam=beam_nf_dipole)
    key1 = (0, 24, 38)
    key2 = (1, 25, 38)

    # Test that when:
    # i) Nfreqs = Ndlys, ii) Sampling, iii) No tapering, iv) R is identity
    # are all satisfied, the scalar adjustment factor is unity
    ds.set_weighting("identity")
    ds.spw_Ndlys = ds.spw_Nfreqs
    adjustment = ds.scalar_delay_adjustment(key1, key2, sampling=True)
    np.testing.assert_almost_equal(adjustment, 1.0)
    ds.set_weighting("iC")
    # if weighting is not identity, then the adjustment should be a vector.
    with pytest.warns(UserWarning, match="Poorly conditioned covariance"):
        adjustment = ds.scalar_delay_adjustment(key1, key2, sampling=True)
    assert len(adjustment == ds.spw_Ndlys)


def test_scalar(d, w, beam_nf_dipole):
    ds = pspecdata.PSpecData(dsets=d, wgts=w, beam=beam_nf_dipole)

    gauss = pspecbeam.PSpecBeamGauss(0.8, np.linspace(115e6, 130e6, 50, endpoint=False))
    ds2 = pspecdata.PSpecData(dsets=d, wgts=w, beam=gauss)

    # Check normal execution
    scalar = ds.scalar(("xx", "xx"))
    scalar_xx = ds.scalar("xx")  # Can use single pol string as shorthand
    assert scalar == scalar_xx
    scalar = ds.scalar(1515)  # polpair-integer = ('xx', 'xx')
    scalar = ds.scalar(("xx", "xx"), taper_override="none")
    scalar = ds.scalar(("xx", "xx"), beam=gauss)
    with pytest.raises(NotImplementedError, match="Polarizations don't match"):
        ds.scalar(("xx", "yy"))

    # Precomputed results in the following test were done "by hand"
    # using iPython notebook "Scalar_dev2.ipynb" in the tests/ directory
    # FIXME: Uncomment when pyuvdata support for this is ready
    # scalar = ds.scalar()
    # np.testing.assert_almost_equal(scalar, 3732415176.85 / 10.**9)

    # FIXME: Remove this when pyuvdata support for the above is ready
    # self.assertRaises(NotImplementedError, ds.scalar)


def test_validate_datasets(d):
    # test freq exception
    uvd = copy.deepcopy(d[0])
    uvd2 = uvd.select(frequencies=np.unique(uvd.freq_array)[:10], inplace=False)
    ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
    with pytest.raises(ValueError, match="all dsets must have the same Nfreqs"):
        ds.validate_datasets()

    # test time exception
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[:10], inplace=False)
    ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
    with pytest.raises(ValueError, match="all dsets must have the same Ntimes"):
        ds.validate_datasets()

    # test label exception
    _labels = ds.labels
    ds.labels = ds.labels[:1]
    with pytest.raises(ValueError, match="self.labels does not have same len"):
        ds.validate_datasets()
    ds.labels = _labels

    # test std exception
    _std = ds.dsets_std
    ds.dsets_std = ds.dsets_std[:1]
    with pytest.raises(ValueError, match="self.dsets_std does not have the same len"):
        ds.validate_datasets()
    ds.dsets_std = _std

    # test wgt exception
    _wgts = ds.wgts
    ds.wgts = ds.wgts[:1]
    with pytest.raises(ValueError, match="self.wgts does not have same len"):
        ds.validate_datasets()
    ds.wgts = _wgts

    # test warnings
    uvd = copy.deepcopy(d[0])
    uvd2 = copy.deepcopy(d[0])
    uvd.select(
        frequencies=np.unique(uvd.freq_array)[:10], times=np.unique(uvd.time_array)[:10]
    )
    uvd2.select(
        frequencies=np.unique(uvd2.freq_array)[10:20],
        times=np.unique(uvd2.time_array)[10:20],
    )
    ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
    ds.validate_datasets()

    # test phasing
    uvd = copy.deepcopy(d[0])
    uvd2 = copy.deepcopy(d[0])
    uvd.phase_to_time(Time(2458042, format="jd"))
    ds = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
    with pytest.raises(ValueError, match="all datasets must have the same phase type"):
        ds.validate_datasets()
    uvd2.phase_to_time(Time(2458042.5, format="jd"))
    ds.validate_datasets()
    # phase_center_catalog should contain only one entry per dataset
    uvd3 = copy.deepcopy(d[0])
    uvd3.phase_center_catalog[1] = uvd3.phase_center_catalog[0]
    ds2 = pspecdata.PSpecData(dsets=[uvd, uvd3], wgts=[None, None])
    with pytest.raises(ValueError, match="phase_center_catalog should contain only one entry"):
        ds2.validate_datasets()
    # phased data
    uvd4 = copy.deepcopy(d[0])

    # test polarization
    ds.validate_pol((0, 1), ("xx", "xx"))

    # test channel widths
    uvd2.channel_width *= 2.0
    ds2 = pspecdata.PSpecData(dsets=[uvd, uvd2], wgts=[None, None])
    with pytest.raises(ValueError, match="all dsets must have the same channel_widths"):
        ds2.validate_datasets()


def test_rephase_to_dset(uvd):
    # get uvd
    uvd1 = copy.deepcopy(uvd)

    # give the uvd an x_orientation to test x_orientation propagation
    uvd1.x_orienation = "east"

    # null test: check nothing changes when dsets contain same UVData object
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd1)], wgts=[None, None]
    )
    # get normal pspec
    bls = [(37, 39)]
    uvp1 = ds.pspec(bls, bls, (0, 1), pols=("xx", "xx"), verbose=False)
    # rephase and get pspec
    ds.rephase_to_dset(0)
    with pytest.warns(UserWarning, match="Skipping dataset 1 b/c it isn't unprojected"):
        ds2 = ds.rephase_to_dset(0, inplace=False)
    uvp2 = ds.pspec(bls, bls, (0, 1), pols=("xx", "xx"), verbose=False)
    blp = (0, ((37, 39), (37, 39)), ("xx", "xx"))
    assert np.isclose(np.abs(uvp2.get_data(blp) / uvp1.get_data(blp)), 1.0).min()


def test_Jy_to_mK(beam_nf_dipole, uvd):
    # test basic execution
    uvd.vis_units = "Jy"
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
        wgts=[None, None],
        beam=beam_nf_dipole,
    )
    ds.Jy_to_mK()
    assert ds.dsets[0].vis_units == "mK"
    assert ds.dsets[1].vis_units == "mK"
    assert (
        uvd.get_data(24, 25, "xx")[30, 30] / ds.dsets[0].get_data(24, 25, "xx")[30, 30]
        < 1.0
    )

    # test feeding beam
    ds2 = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
        wgts=[None, None],
        beam=beam_nf_dipole,
    )
    with pytest.warns(
        UserWarning, match="Feeding a beam model when self.primary_beam already exists"
    ):
        ds2.Jy_to_mK(beam=beam_nf_dipole)
    assert ds.dsets[0] == ds2.dsets[0]

    # test vis_units no Jansky
    uvd2 = copy.deepcopy(uvd)
    uvd2.polarization_array[0] = -6
    uvd2.vis_units = "UNCALIB"
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd2)],
        wgts=[None, None],
        beam=beam_nf_dipole,
    )
    with pytest.warns(
        UserWarning, match="Cannot convert dset 1 Jy -> mK because vis_units = UNCALIB"
    ):
        ds.Jy_to_mK()
    assert ds.dsets[0].vis_units == "mK"
    assert ds.dsets[1].vis_units == "UNCALIB"
    assert (
        ds.dsets[0].get_data(24, 25, "xx")[30, 30]
        != ds.dsets[1].get_data(24, 25, "yy")[30, 30]
    )


def test_trim_dset_lsts():
    fname = os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA")
    uvd1 = UVData()
    uvd1.read_miriad(fname)
    uvd2 = copy.deepcopy(uvd1)
    uvd2.lst_array = (
        uvd2.lst_array + 10.0 * np.median(np.diff(np.unique(uvd2.lst_array)))
    ) % (2.0 * np.pi)

    # test basic execution
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd2)], wgts=[None, None]
    )
    with pytest.warns(
        UserWarning, match="The lst_array is not self-consistent with the time_array"
    ):
        ds.trim_dset_lsts()
    assert ds.dsets[0].Ntimes == 50
    assert ds.dsets[1].Ntimes == 50

    assert np.all(
        (2458042.178948477 < ds.dsets[0].time_array)
        + (ds.dsets[0].time_array < 2458042.1843023109)
    )
    # test exception
    uvd2.lst_array += np.linspace(0, 1e-3, uvd2.Nblts)
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd2)], wgts=[None, None]
    )
    with pytest.raises(ValueError, match="Not all datasets in self.dsets are on the same LST grid"):
        ds.trim_dset_lsts()
    assert ds.dsets[0].Ntimes == 60
    assert ds.dsets[1].Ntimes == 60


def test_get_Q_alt_tensor():
    fname = os.path.join(DATA_PATH, "zen.2458042.17772.xx.HH.uvXA")
    uvd1 = UVData()
    uvd1.read_miriad(fname)
    uvd2 = copy.deepcopy(uvd1)
    uvd2.lst_array = (
        uvd2.lst_array + 10.0 * np.median(np.diff(np.unique(uvd2.lst_array)))
    ) % (2.0 * np.pi)

    # test basic execution
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd1), copy.deepcopy(uvd2)], wgts=[None, None]
    )

    ndly = ds.spw_Ndlys
    ds.spw_Ndlys = None
    Qalt = ds.get_Q_alt_tensor()
    assert ds.spw_Ndlys == ndly


def test_units(beam_nf_dipole, uvd):
    ds = pspecdata.PSpecData()
    # test exception
    with pytest.raises(IndexError, match="No datasets have been added yet"):
        ds.units()
    ds.add(uvd, None)
    # test basic execution
    vis_u, norm_u = ds.units(little_h=False)
    vis_u, norm_u = ds.units()
    assert vis_u == "UNCALIB"
    assert norm_u == "Hz str [beam normalization not specified]"
    ds_b = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=beam_nf_dipole)
    vis_u, norm_u = ds_b.units(little_h=False)
    assert norm_u == "Mpc^3"


def test_delays(uvd):
    ds = pspecdata.PSpecData()
    # test exception
    with pytest.raises(IndexError, match="No datasets have been added yet"):
        ds.delays()
    ds.add([uvd, uvd], [None, None])
    d = ds.delays()
    assert len(d) == ds.dsets[0].Nfreqs


def test_check_in_dset(d):
    # generate ds
    uvd = copy.deepcopy(d[0])
    ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None])

    # check for existing key
    assert ds.check_key_in_dset(("xx"), 0)
    assert ds.check_key_in_dset((24, 25), 0)
    assert ds.check_key_in_dset((24, 25, "xx"), 0)

    # check for non-existing key
    assert not ds.check_key_in_dset("yy", 0)
    assert not ds.check_key_in_dset((24, 26), 0)
    assert not ds.check_key_in_dset((24, 26, "yy"), 0)

    # check exception
    with pytest.raises(KeyError, match="must be a length 1, 2 or 3 tuple"):
        ds.check_key_in_dset((1, 2, 3, 4, 5), 0)

    # test dset_idx
    with pytest.raises(TypeError, match="dset must be either an int or string"):
        ds.dset_idx((1, 2))


def test_C_model():
    # test the key format in ds._C and the shape of stored covariance
    uvd = UVData()
    uvd.read(os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA"))
    cosmo = conversions.Cosmo_Conversions()
    uvb = pspecbeam.PSpecBeamUV(
        os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"), cosmo=cosmo
    )
    ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=uvb)

    spws = utils.spw_range_from_freqs(
        uvd, freq_range=[(160e6, 165e6), (160e6, 165e6)], bounds_error=True
    )
    antpos, ants = uvd.get_enu_data_ants()
    antpos = dict(zip(ants, antpos))
    red_bls = redcal.get_pos_reds(antpos, bl_error_tol=1.0)
    bls1, bls2, blpairs = utils.construct_blpairs(
        red_bls[3], exclude_auto_bls=True, exclude_permutations=True
    )

    ds.set_spw(spws[0])
    key = (0, bls1[0], "xx")
    ds.C_model(key, model="empirical", time_index=0)
    assert (
        (0, 0),
        ((bls1[0][0], bls1[0][1], "xx"), (bls1[0][0], bls1[0][1], "xx")),
        "empirical",
        None,
        False,
        True,
    ) in ds._C.keys()
    ds.C_model(key, model="autos", time_index=0)
    assert (
        (0, 0),
        ((bls1[0][0], bls1[0][1], "xx"), (bls1[0][0], bls1[0][1], "xx")),
        "autos",
        0,
        False,
        True,
    ) in ds._C.keys()
    for Ckey in ds._C.keys():
        assert ds._C[Ckey].shape == (spws[0][1] - spws[0][0], spws[0][1] - spws[0][0])

    ds.set_spw(spws[1])
    key = (0, bls1[0], "xx")
    known_cov = {}
    model = "known"
    Ckey = (
        (0, 0),
        ((bls1[0][0], bls1[0][1], "xx"), (bls1[0][0], bls1[0][1], "xx")),
        "known",
        0,
        False,
        True,
    )
    known_cov[Ckey] = np.diag(np.ones(uvd.Nfreqs))
    ds.C_model(key, model="known", time_index=0, known_cov=known_cov)
    assert Ckey in ds._C.keys()
    assert ds._C[Ckey].shape == (spws[1][1] - spws[1][0], spws[1][1] - spws[1][0])


def test_get_analytic_covariance():
    uvd = UVData()
    uvd.read(os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA"))
    uvd.nsample_array[:] = 1.0
    uvd.flag_array[:] = False
    cosmo = conversions.Cosmo_Conversions()
    uvb = pspecbeam.PSpecBeamUV(
        os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"), cosmo=cosmo
    )

    # extend time axis by factor of 4
    for i in range(2):
        new = copy.deepcopy(uvd)
        new.time_array += new.Ntimes * np.diff(np.unique(new.time_array))[0]
        new.lst_array = uvutils.get_lst_for_time(
            new.time_array, telescope_loc=new.telescope.location
        )
        uvd += new

    # get redundant baselines
    reds, lens, angs = utils.get_reds(uvd, pick_data_ants=True)

    # append roughly 20 blpairs to a list
    bls1, bls2 = [], []
    for red in reds[:3]:
        _bls1, _bls2, _ = utils.construct_blpairs(
            red,
            exclude_auto_bls=False,
            exclude_cross_bls=False,
            exclude_permutations=False,
        )
        bls1.extend(_bls1)
        bls2.extend(_bls2)
    # keep only 20 blpairs for speed (each with 40 independent time samples)
    bls1, bls2 = bls1[:20], bls2[:20]
    Nblpairs = len(bls1)

    # generate a sky and noise simulation: each bl has the same FG signal, constant in time
    # but has a different noise realization
    np.random.seed(0)
    sim1 = testing.sky_noise_sim(
        uvd,
        uvb,
        cov_amp=1000,
        cov_length_scale=10,
        constant_per_bl=True,
        constant_in_time=True,
        bl_loop_seed=0,
        divide_by_nsamp=False,
    )
    np.random.seed(0)
    sim2 = testing.sky_noise_sim(
        uvd,
        uvb,
        cov_amp=1000,
        cov_length_scale=10,
        constant_per_bl=True,
        constant_in_time=True,
        bl_loop_seed=1,
        divide_by_nsamp=False,
    )

    # setup ds
    ds = pspecdata.PSpecData(dsets=[sim1, sim2], wgts=[None, None], beam=uvb)
    ds.Jy_to_mK()

    # assert that imag component of covariance is near zero
    key1 = (0, bls1[0], "xx")
    key2 = (1, bls2[0], "xx")
    ds.set_spw((60, 90))
    M_ = np.diag(np.ones(ds.spw_Ndlys))
    for model in ["autos", "empirical"]:
        (cov_q_real, cov_q_imag, cov_p_real, cov_p_imag) = ds.get_analytic_covariance(
            key1, key2, M=M_, exact_norm=False, pol=False, model=model, known_cov=None
        )
        # assert these arrays are effectively real-valued, even though they are complex type.
        # some numerical noise can leak-in, so check to within a dynamic range of peak real power.
        for cov in [cov_q_real, cov_q_imag, cov_p_real, cov_p_imag]:
            assert np.isclose(cov.imag, 0, atol=abs(cov.real).max() / 1e10).all()

    # Here we generate a known_cov to be passed to ds.pspec, which stores two cov_models named 'dsets' and 'fiducial'.
    # The two models have actually the same data, while in generating output covariance, 'dsets' mode will follow the shorter
    # path where we use some optimization for diagonal matrices, while 'fiducial' mode will follow the longer path
    # where there is no such optimization. This test should show the results from two paths are equivalent.
    known_cov_test = dict()
    C_n_11 = np.diag([2.0] * ds.Nfreqs)
    P_n_11, S_n_11, C_n_12, P_n_12, S_n_12 = (
        np.zeros_like(C_n_11),
        np.zeros_like(C_n_11),
        np.zeros_like(C_n_11),
        np.zeros_like(C_n_11),
        np.zeros_like(C_n_11),
    )
    models = ["dsets", "fiducial"]
    for model in models:
        for blpair in list(zip(bls1, bls2)):
            for time_index in range(ds.Ntimes):
                key1 = (0, blpair[0], "xx")
                dset1, bl1 = ds.parse_blkey(key1)
                key2 = (1, blpair[1], "xx")
                dset2, bl2 = ds.parse_blkey(key2)

                Ckey = ((dset1, dset1), (bl1, bl1)) + (model, time_index, False, True)
                known_cov_test[Ckey] = C_n_11
                Ckey = ((dset1, dset1), (bl1, bl1)) + (model, time_index, False, False)
                known_cov_test[Ckey] = P_n_11
                Ckey = ((dset1, dset1), (bl1, bl1)) + (model, time_index, True, True)
                known_cov_test[Ckey] = S_n_11

                Ckey = ((dset2, dset2), (bl2, bl2)) + (model, time_index, False, True)
                known_cov_test[Ckey] = C_n_11
                Ckey = ((dset2, dset2), (bl2, bl2)) + (model, time_index, False, False)
                known_cov_test[Ckey] = P_n_11
                Ckey = ((dset2, dset2), (bl2, bl2)) + (model, time_index, True, True)
                known_cov_test[Ckey] = S_n_11

                Ckey = ((dset1, dset2), (bl1, bl2)) + (model, time_index, False, True)
                known_cov_test[Ckey] = C_n_12
                Ckey = ((dset2, dset1), (bl2, bl1)) + (model, time_index, False, True)
                known_cov_test[Ckey] = C_n_12
                Ckey = ((dset2, dset1), (bl2, bl1)) + (model, time_index, False, False)
                known_cov_test[Ckey] = P_n_12
                Ckey = ((dset2, dset1), (bl2, bl1)) + (model, time_index, True, True)
                known_cov_test[Ckey] = S_n_12

    uvp_dsets_cov = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        spw_ranges=(60, 90),
        store_cov=True,
        cov_model="dsets",
        known_cov=known_cov_test,
        verbose=False,
        taper="bh",
    )
    uvp_fiducial_cov = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        spw_ranges=(60, 90),
        store_cov=True,
        cov_model="fiducial",
        known_cov=known_cov_test,
        verbose=False,
        taper="bh",
    )
    # check their cov_array are equal
    assert np.allclose(
        uvp_dsets_cov.cov_array_real[0], uvp_fiducial_cov.cov_array_real[0], rtol=1e-05
    )

    # check noise floor computation from auto correlations
    uvp_auto_cov = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        spw_ranges=(60, 90),
        store_cov=True,
        cov_model="autos",
        verbose=False,
        taper="bh",
    )
    # get RMS of noise-dominated bandpowers for uvp_auto_cov
    noise_dlys = np.abs(uvp_auto_cov.get_dlys(0) * 1e9) > 1000
    rms = [
        np.std(
            uvp_auto_cov.get_data(key).real
            / np.sqrt(np.diagonal(uvp_auto_cov.get_cov(key).real, axis1=1, axis2=2)),
            axis=0,
        )
        for key in uvp_auto_cov.get_all_keys()
    ]
    rms = np.mean(rms, axis=0)
    # assert this is close to 1.0
    assert np.isclose(np.mean(rms[noise_dlys]), 1.0, atol=0.1)

    # check signal + noise floor computation
    uvp_fgdep_cov = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        spw_ranges=(60, 90),
        store_cov=True,
        cov_model="foreground_dependent",
        verbose=False,
        taper="bh",
    )
    # get RMS of data: divisor is foreground_dependent covariance this time
    # b/c noise in empirically estimated fg-dep cov yields biased errorbar (tavg is not unbiased, but less-biased)
    rms = []
    for key in uvp_fgdep_cov.get_all_keys():
        rms.append(
            np.std(
                uvp_fgdep_cov.get_data(key)[:, ~noise_dlys].real
                / np.sqrt(
                    np.mean(
                        np.diagonal(uvp_fgdep_cov.get_cov(key).real, axis1=1, axis2=2)[
                            :, ~noise_dlys
                        ],
                        axis=0,
                    )
                ),
                axis=0,
            )
        )
    rms = np.mean(rms, axis=0)
    # assert this is close to 1.0
    assert np.isclose(np.mean(rms), 1.0, atol=0.1)


def test_pspec_basic_execution(pspec_ds):
    """Test basic pspec() execution: output shapes, dtypes, and input parameter variants."""
    uvp = pspec_ds.pspec(
        pspec_bls,
        pspec_bls,
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        little_h=True,
        verbose=False,
    )
    assert len(uvp.bl_array) == len(pspec_bls)
    assert uvp.antnums_to_blpair(((24, 25), (24, 25))) in uvp.blpair_array
    assert uvp.data_array[0].dtype == np.complex128
    assert uvp.data_array[0].shape == (240, 64, 1)
    assert not uvp.exact_windows

    # verify spw_ranges and n_dlys input variants all accepted
    pspec_ds.pspec(pspec_bls, pspec_bls, (0, 1), ("xx", "xx"), spw_ranges=(10, 20))
    pspec_ds.pspec(
        pspec_bls, pspec_bls, (0, 1), ("xx", "xx"), n_dlys=10, spw_ranges=[(10, 20)]
    )
    pspec_ds.pspec(pspec_bls, pspec_bls, (0, 1), ("xx", "xx"), n_dlys=1)


def test_pspec_dayenu_weighting(pspec_ds):
    """Test dayenu (inverse-sinc) weighting: successful run and error handling for bad r_params."""
    rp = {
        "filter_centers": [0.0],
        "filter_half_widths": [250e-9],
        "filter_factors": [1e-9],
    }
    my_r_params = {}
    my_r_params_dset0_only = {}
    for bl in pspec_bls:
        my_r_params[(0,) + bl + ("xx",)] = rp
        my_r_params[(1,) + bl + ("xx",)] = rp
        my_r_params_dset0_only[(0,) + bl + ("xx",)] = rp

    # successful dayenu run
    pspec_ds.pspec(
        pspec_bls,
        pspec_bls,
        (0, 1),
        ("xx", "xx"),
        spw_ranges=(10, 20),
        input_data_weight="dayenu",
        r_params=my_r_params,
    )

    # error: empty r_params dict
    with pytest.raises(ValueError, match="No r_param dictionary supplied for baseline *"):
        pspec_ds.pspec(
            pspec_bls,
            pspec_bls,
            (0, 1),
            ("xx", "xx"),
            spw_ranges=(10, 20),
            input_data_weight="dayenu",
            r_params={},
        )

    # error: r_params missing keys for dset 1
    with pytest.raises(ValueError, match="No r_param dictionary supplied for baseline *"):
        pspec_ds.pspec(
            pspec_bls,
            pspec_bls,
            (0, 1),
            ("xx", "xx"),
            spw_ranges=(10, 20),
            input_data_weight="dayenu",
            r_params=my_r_params_dset0_only,
        )

    # error: grouped baseline format with more than one pair per group is not supported
    with pytest.raises(NotImplementedError, match="Baseline lists bls1 and bls2"):
        pspec_ds.pspec(
            [[(24, 25), (38, 39)]], [[(24, 25), (38, 39)]], (0, 1), [("xx", "xx")]
        )


def test_pspec_isotropic_beam_norm(bm_Q, uvd):
    """Test pspec() normalization with an isotropic beam: checks Q integral shape and
    that exact_norm=True and exact_norm=False agree to within 5%."""
    uvd_temp = copy.deepcopy(uvd)
    bls_Q = [(24, 25)]
    ds_Q = pspecdata.PSpecData(dsets=[uvd_temp, uvd_temp], wgts=[None, None], beam=bm_Q)
    ds_Q.pspec(
        bls_Q,
        bls_Q,
        (0, 1),
        [("xx", "xx")],
        input_data_weight="identity",
        norm="I",
        taper="none",
        verbose=False,
        exact_norm=False,
    )

    Q_sample = ds_Q.get_integral_beam("xx")
    assert np.shape(Q_sample) == (
        ds_Q.spw_range[1] - ds_Q.spw_range[0],
        ds_Q.spw_range[1] - ds_Q.spw_range[0],
    )
    estimated_Q = (1.0 / (4 * np.pi)) * np.ones_like(Q_sample)
    assert np.allclose(np.real(estimated_Q), np.real(Q_sample), rtol=1e-05)

    # exact_norm=True vs exact_norm=False should agree to within 5%
    ds_t = pspecdata.PSpecData(dsets=[uvd_temp, uvd_temp], wgts=[None, None], beam=bm_Q)
    uvp_new = ds_t.pspec(
        bls_Q,
        bls_Q,
        (0, 1),
        [("xx", "xx")],
        input_data_weight="identity",
        norm="I",
        taper="none",
        verbose=False,
        exact_norm=True,
    )
    uvp_ext = ds_t.pspec(
        bls_Q,
        bls_Q,
        (0, 1),
        [("xx", "xx")],
        input_data_weight="identity",
        norm="I",
        taper="none",
        verbose=False,
        exact_norm=False,
    )
    key = (0, (bls_Q[0], bls_Q[0]), "xx")
    diff = np.median(
        (np.real(uvp_new.get_data(key)) - np.real(uvp_ext.get_data(key)))
        / np.real(uvp_ext.get_data(key))
    )
    assert diff <= 0.05


def test_pspec_baseline_formats(pspec_ds):
    """Test pspec() with redundant baseline groups from redcal and with mixed
    grouped/ungrouped baseline list formats."""
    # redundant baseline groups: exclude permutations
    antpos, ants = pspec_ds.dsets[0].get_enu_data_ants()
    antpos = dict(zip(ants, antpos))
    red_bls = [sorted(blg) for blg in redcal.get_pos_reds(antpos)][2]
    bls1, bls2, _ = utils.construct_blpairs(red_bls, exclude_permutations=True)
    uvp = pspec_ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        little_h=True,
        verbose=False,
    )
    assert uvp.antnums_to_blpair(((24, 25), (37, 38))) in uvp.blpair_array
    assert uvp.Nblpairs == 10
    assert uvp.antnums_to_blpair(((24, 25), (52, 53))) in uvp.blpair_array
    assert uvp.antnums_to_blpair(((52, 53), (24, 25))) not in uvp.blpair_array

    # mixed grouped/ungrouped format: [[(bl,)], bl]
    bls1_mixed = [[(24, 25)], (52, 53)]
    bls2_mixed = [[(24, 25)], (52, 53)]
    pspec_ds.pspec(
        bls1_mixed,
        bls2_mixed,
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        little_h=True,
        verbose=False,
    )


def test_pspec_multiple_spws_and_select(beam_nf_dipole, uvd):
    """Test pspec() with multiple spectral windows and verify that select() works
    correctly on the resulting UVPSpec."""
    bls1, bls2, _ = utils.construct_blpairs(
        pspec_bls, exclude_permutations=False, exclude_auto_bls=False
    )

    # two spectral windows + blpair-level select
    uvd_temp = copy.deepcopy(uvd)
    ds = pspecdata.PSpecData(
        dsets=[uvd_temp, uvd_temp], wgts=[None, None], beam=beam_nf_dipole
    )
    uvp = ds.pspec(
        bls1, bls2, (0, 1), ("xx", "xx"), spw_ranges=[(20, 30), (30, 40)], verbose=False
    )
    assert uvp.Nblpairs == 16
    assert uvp.Nspws == 2
    uvp2 = uvp.select(spws=0, bls=[(24, 25)], only_pairs_in_bls=False, inplace=False)
    assert uvp2.Nspws == 1
    assert uvp2.Nblpairs == 7
    uvp.select(spws=0, bls=(24, 25), only_pairs_in_bls=True, inplace=True)
    assert uvp.Nspws == 1
    assert uvp.Nblpairs == 1

    # three spectral windows: verify Nspws, Nspwdlys, shapes, and spw-level select
    uvd_temp = copy.deepcopy(uvd)
    ds = pspecdata.PSpecData(
        dsets=[uvd_temp, uvd_temp], wgts=[None, None], beam=beam_nf_dipole
    )
    uvp = ds.pspec(
        pspec_bls,
        pspec_bls,
        (0, 1),
        ("xx", "xx"),
        spw_ranges=[(10, 24), (30, 40), (45, 64)],
        verbose=False,
    )
    assert uvp.Nspws == 3
    assert uvp.Nspwdlys == 43
    assert uvp.data_array[0].shape == (240, 14, 1)
    assert uvp.get_data((0, 124125124125, ("xx", "xx"))).shape == (60, 14)
    uvp.select(spws=[1])
    assert uvp.Nspws == 1
    assert uvp.Ndlys == 10
    assert len(uvp.data_array) == 1


def test_pspec_polarizations(beam_nf_dipole, uvd):
    """Test pspec() polarization handling: single pol, multi-pol, integer pol codes,
    warnings for unavailable pols, and errors when all pols fail validation."""
    # single available pol
    uvd_temp = copy.deepcopy(uvd)
    ds = pspecdata.PSpecData(
        dsets=[uvd_temp, uvd_temp], wgts=[None, None], beam=beam_nf_dipole
    )
    ds.pspec(
        pspec_bls, pspec_bls, (0, 1), ("xx", "xx"), spw_ranges=[(10, 24)], verbose=False
    )

    # warn and skip unavailable pol in a multi-pol request
    uvd_temp = copy.deepcopy(uvd)
    ds = pspecdata.PSpecData(
        dsets=[uvd_temp, uvd_temp], wgts=[None, None], beam=beam_nf_dipole
    )
    with pytest.warns(
        UserWarning,
        match="Polarization pair: \\('yy', 'yy'\\) failed the validation test",
    ):
        ds.pspec(
            pspec_bls,
            pspec_bls,
            (0, 1),
            [("xx", "xx"), ("yy", "yy")],
            spw_ranges=[(10, 24)],
            verbose=False,
        )

    # integer pol code
    uvd_temp = copy.deepcopy(uvd)
    ds = pspecdata.PSpecData(
        dsets=[uvd_temp, uvd_temp], wgts=[None, None], beam=beam_nf_dipole
    )
    ds.pspec(
        pspec_bls, pspec_bls, (0, 1), (-5, -5), spw_ranges=[(10, 24)], verbose=False
    )

    # error: mismatched bls1/bls2 lengths
    with pytest.raises(AssertionError, match="length of bls1 must equal length of bls2"):
        ds.pspec(pspec_bls[:1], pspec_bls, (0, 1), ("xx", "xx"))

    # error: all requested pols fail validation (yy not present)
    with pytest.warns(
        UserWarning,
        match="Polarization pair: \\('yy', 'yy'\\) failed the validation test",
    ):
        with pytest.raises(ValueError, match="None of the specified polarization pairs"):
            ds.pspec(pspec_bls, pspec_bls, (0, 1), pols=("yy", "yy"))

    # error: dsets have mismatched polarizations
    uvd1 = copy.deepcopy(uvd)
    uvd1.polarization_array = np.array([-6])
    ds = pspecdata.PSpecData(
        dsets=[uvd_temp, uvd1], wgts=[None, None], beam=beam_nf_dipole
    )
    with pytest.warns(
        UserWarning,
        match="Polarization pair: \\('xx', 'xx'\\) failed the validation test",
    ):
        with pytest.raises(ValueError, match="None of the specified polarization pairs"):
            ds.pspec(pspec_bls, pspec_bls, (0, 1), ("xx", "xx"))

    # multi-pol UVData: both xx and yy present
    uvd1 = copy.deepcopy(uvd)
    uvd1.polarization_array = np.array([-6])
    uvd2 = uvd + uvd1
    ds = pspecdata.PSpecData(dsets=[uvd2, uvd2], wgts=[None, None], beam=beam_nf_dipole)
    ds.pspec(
        pspec_bls,
        pspec_bls,
        (0, 1),
        [("xx", "xx"), ("yy", "yy")],
        spw_ranges=[(10, 24)],
        verbose=False,
    )

    # warn and skip xy pol when not present in multi-pol UVData
    with pytest.warns(
        UserWarning,
        match="Polarization pair: \\('xy', 'xy'\\) failed the validation test",
    ):
        ds.pspec(
            pspec_bls,
            pspec_bls,
            (0, 1),
            [("xx", "xx"), ("xy", "xy")],
            spw_ranges=[(10, 24)],
            verbose=False,
        )


@pytest.mark.filterwarnings(
    "ignore:Some integrations have zero nsamples, but non-zero weights"
)
def test_pspec_zero_nsamples(beam_nf_dipole, uvd):
    """Test that baselines with nsample_array=0 produce integration_array=0 in the output."""
    uvd_temp = copy.deepcopy(uvd)
    uvd_temp.nsample_array[uvd_temp.antpair2ind(24, 25, ordered=False)] = 0.0
    ds = pspecdata.PSpecData(
        dsets=[uvd_temp, uvd_temp], wgts=[None, None], beam=beam_nf_dipole
    )
    uvp = ds.pspec([(24, 25)], [(37, 38)], (0, 1), [("xx", "xx")])
    assert np.all(np.isclose(uvp.integration_array[0], 0.0))


def test_pspec_covariance_models(beam_nf_dipole, uvd, uvd_std):
    """Test pspec() covariance storage: empirical, dsets, and foreground_dependent cov_model,
    store_cov and store_cov_diag options, and that dsets/fiducial paths produce equal results."""
    bls1, bls2, _ = utils.construct_blpairs(
        pspec_bls, exclude_permutations=False, exclude_auto_bls=False
    )
    key = (0, (bls1[0], bls2[0]), "xx")

    uvd_temp = copy.deepcopy(uvd)
    uvd_temp_std = copy.deepcopy(uvd_std)
    ds = pspecdata.PSpecData(
        dsets=[uvd_temp, uvd_temp],
        wgts=[None, None],
        dsets_std=[uvd_temp_std, uvd_temp_std],
        beam=beam_nf_dipole,
    )

    # empirical covariance: should be uniform along time axis
    uvp = ds.pspec(
        bls1[:1],
        bls2[:1],
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        little_h=True,
        verbose=False,
        spw_ranges=[(10, 20)],
        filter_extensions=[(2, 2)],
        symmetric_taper=False,
        store_cov=True,
        cov_model="empirical",
    )
    assert hasattr(uvp, "cov_array_real")
    assert np.allclose(uvp.get_cov(key)[0], uvp.get_cov(key)[-1])

    # dsets covariance
    uvp = ds.pspec(
        bls1[:1],
        bls2[:1],
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        little_h=True,
        verbose=False,
        spw_ranges=[(10, 20)],
        exact_norm=True,
        store_cov=True,
        cov_model="dsets",
    )
    assert hasattr(uvp, "cov_array_real")

    # foreground_dependent: store_cov and store_cov_diag should agree on diagonal
    uvp_cov = ds.pspec(
        bls1[:1],
        bls2[:1],
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        little_h=True,
        verbose=False,
        spw_ranges=[(10, 20)],
        exact_norm=True,
        store_cov=True,
        cov_model="foreground_dependent",
    )
    uvp_cov_diag = ds.pspec(
        bls1[:1],
        bls2[:1],
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        little_h=True,
        verbose=False,
        spw_ranges=[(10, 20)],
        exact_norm=True,
        store_cov_diag=True,
        cov_model="foreground_dependent",
    )
    assert np.isclose(
        np.diagonal(uvp_cov.get_cov(key), axis1=1, axis2=2),
        np.real(uvp_cov_diag.get_stats("foreground_dependent_diag", key)) ** 2,
    ).all()


def test_pspec_identity_caching(beam_nf_dipole, uvd):
    """Test that _identity_Y/G/H matrices are cached when baselines are identical
    and unflagged, and not reused when baselines differ in their flag patterns."""
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)],
        wgts=[None, None],
        beam=beam_nf_dipole,
    )

    # identical unflagged baselines: only one cache entry expected
    ds.pspec(
        [(24, 25), (24, 25)],
        [(24, 25), (24, 25)],
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        verbose=False,
        spw_ranges=[(20, 30)],
    )
    assert len(ds._identity_Y) == len(ds._identity_G) == len(ds._identity_H) == 1
    assert list(ds._identity_Y.keys())[0] == ((0, 24, 25, "xx"), (1, 24, 25, "xx"))

    # flagging one baseline breaks the symmetry: two cache entries expected
    ds.dsets[0].flag_array[ds.dsets[0].antpair2ind(37, 38, ordered=False), 25, :] = True
    ds.pspec(
        [(24, 25), (37, 38)],
        [(24, 25), (37, 38)],
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        verbose=False,
        spw_ranges=[(20, 30)],
    )
    assert len(ds._identity_Y) == len(ds._identity_G) == len(ds._identity_H) == 2
    assert ((0, 24, 25, "xx"), (1, 24, 25, "xx")) in ds._identity_Y.keys()
    assert ((0, 37, 38, "xx"), (1, 37, 38, "xx")) in ds._identity_Y.keys()


def test_pspec_exact_windows():
    """Test pspec() with exact_windows=True using both a pre-computed FT beam file
    and a Gaussian beam object."""
    datafile = os.path.join(DATA_PATH, "zen.2458116.31939.HH.uvh5")
    uvd1 = UVData()
    uvd1.read_uvh5(datafile)
    ds = pspecdata.PSpecData(dsets=[uvd1, uvd1], wgts=[None, None])
    baselines1, baselines2, _ = utils.construct_blpairs(
        uvd1.get_antpairs()[1:], exclude_permutations=False, exclude_auto_bls=True
    )

    # FT beam from file
    with pytest.warns(
        UserWarning, match="uvp has no cosmo attribute. Using fiducial cosmology."
    ):
        uvp_w = ds.pspec(
            baselines1,
            baselines2,
            (0, 1),
            ("xx", "xx"),
            spw_ranges=(175, 195),
            exact_windows=True,
            ftbeam=os.path.join(DATA_PATH, "FT_beam_HERA_dipole_test"),
        )
    assert uvp_w.exact_windows

    # Gaussian beam object
    widths = -0.0343 * uvd1.freq_array.flatten() / 1e6 + 11.30
    gaussian_beam = uvwindow.FTBeam.gaussian(
        freq_array=uvd1.freq_array.flatten(), widths=widths, pol="xx"
    )
    with pytest.warns(
        UserWarning, match="uvp has no cosmo attribute. Using fiducial cosmology."
    ):
        ds.pspec(
            baselines1,
            baselines2,
            (0, 1),
            ("xx", "xx"),
            spw_ranges=(175, 195),
            exact_windows=True,
            ftbeam=gaussian_beam,
        )


def test_normalization(beam_nf_dipole, uvd):
    # Test Normalization of pspec() compared to PAPER legacy techniques
    d1 = uvd.select(
        times=np.unique(uvd.time_array)[:-1:2],
        frequencies=np.unique(uvd.freq_array)[40:51],
        inplace=False,
    )
    d2 = uvd.select(
        times=np.unique(uvd.time_array)[1::2],
        frequencies=np.unique(uvd.freq_array)[40:51],
        inplace=False,
    )
    freqs = np.unique(d1.freq_array)

    # Setup baselines
    bls1 = [(24, 25)]
    bls2 = [(37, 38)]

    # Get beam
    beam = copy.deepcopy(beam_nf_dipole)
    cosmo = conversions.Cosmo_Conversions()

    # Set to mK scale
    d1.data_array *= beam.Jy_to_mK(freqs, pol="XX")[None, :, None]
    d2.data_array *= beam.Jy_to_mK(freqs, pol="XX")[None, :, None]

    # Compare using no taper
    OmegaP = beam.power_beam_int(pol="XX")
    OmegaPP = beam.power_beam_sq_int(pol="XX")
    OmegaP = interp1d(beam.beam_freqs / 1e6, OmegaP)(freqs / 1e6)
    OmegaPP = interp1d(beam.beam_freqs / 1e6, OmegaPP)(freqs / 1e6)
    NEB = 1.0
    Bp = np.median(np.diff(freqs)) * len(freqs)
    scalar = (
        cosmo.X2Y(np.mean(cosmo.f2z(freqs))) * np.mean(OmegaP**2 / OmegaPP) * Bp * NEB
    )
    data1 = d1.get_data(bls1[0])
    data2 = d2.get_data(bls2[0])
    legacy = np.fft.fftshift(
        np.conj(np.fft.fft(data1, axis=1))
        * np.fft.fft(data2, axis=1)
        * scalar
        / len(freqs) ** 2,
        axes=1,
    )[0]

    # hera_pspec OQE
    ds = pspecdata.PSpecData(dsets=[d1, d2], wgts=[None, None], beam=beam)
    uvp = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        pols=("xx", "xx"),
        taper="none",
        input_data_weight="identity",
        norm="I",
        sampling=True,
    )
    oqe = uvp.get_data((0, ((24, 25), (37, 38)), ("xx", "xx")))[0]

    # assert answers are same to within 3%
    assert np.isclose(np.real(oqe) / np.real(legacy), 1, atol=0.03, rtol=0.03).all()

    # taper
    window = windows.blackmanharris(len(freqs))
    NEB = Bp / trapezoid(window**2, x=freqs)
    scalar = (
        cosmo.X2Y(np.mean(cosmo.f2z(freqs))) * np.mean(OmegaP**2 / OmegaPP) * Bp * NEB
    )
    data1 = d1.get_data(bls1[0])
    data2 = d2.get_data(bls2[0])
    legacy = np.fft.fftshift(
        np.conj(np.fft.fft(data1 * window[None, :], axis=1))
        * np.fft.fft(data2 * window[None, :], axis=1)
        * scalar
        / len(freqs) ** 2,
        axes=1,
    )[0]

    # hera_pspec OQE
    ds = pspecdata.PSpecData(dsets=[d1, d2], wgts=[None, None], beam=beam)
    uvp = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        taper="blackman-harris",
        input_data_weight="identity",
        norm="I",
    )
    oqe = uvp.get_data((0, ((24, 25), (37, 38)), ("xx", "xx")))[0]

    # assert answers are same to within 3%
    assert np.isclose(np.real(oqe) / np.real(legacy), 1, atol=0.03, rtol=0.03).all()


def test_broadcast_dset_flags():
    # setup
    fname = os.path.join(DATA_PATH, "zen.all.xx.LST.1.06964.uvA")
    uvd = UVData()
    uvd.read_miriad(fname)
    Nfreq = uvd.data_array.shape[2]

    # test basic execution w/ a spw selection
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None]
    )
    ds.broadcast_dset_flags(spw_ranges=[(400, 800)], time_thresh=0.2)
    assert not ds.dsets[0].get_flags(24, 25)[:, 550:650].any()

    # test w/ no spw selection
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None]
    )
    ds.broadcast_dset_flags(spw_ranges=None, time_thresh=0.2)
    assert ds.dsets[0].get_flags(24, 25)[:, 550:650].any()

    # test unflagging
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None]
    )
    ds.broadcast_dset_flags(spw_ranges=None, time_thresh=0.2, unflag=True)
    assert not ds.dsets[0].get_flags(24, 25)[:, :].any()

    # test single integration being flagged within spw
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None]
    )
    ds.dsets[0].flag_array[ds.dsets[0].antpair2ind(24, 25, ordered=False), 600, 0][
        3
    ] = True
    ds.broadcast_dset_flags(spw_ranges=[(400, 800)], time_thresh=0.25, unflag=False)
    assert ds.dsets[0].get_flags(24, 25)[3, 400:800].all()
    assert not ds.dsets[0].get_flags(24, 25)[3, :].all()

    # test pspec run sets flagged integration to have zero weight
    uvd.flag_array[uvd.antpair2ind(24, 25, ordered=False), 400, :][3] = True
    ds = pspecdata.PSpecData(
        dsets=[copy.deepcopy(uvd), copy.deepcopy(uvd)], wgts=[None, None]
    )
    ds.broadcast_dset_flags(spw_ranges=[(400, 450)], time_thresh=0.25)
    uvp = ds.pspec(
        [(24, 25), (37, 38), (38, 39)],
        [(24, 25), (37, 38), (38, 39)],
        (0, 1),
        ("xx", "xx"),
        spw_ranges=[(400, 450)],
        verbose=False,
    )
    # assert flag broadcast above hits weight arrays in uvp
    assert np.all(
        np.isclose(uvp.get_wgts((0, ((24, 25), (24, 25)), ("xx", "xx")))[3], 0.0)
    )
    # assert flag broadcast above hits integration arrays
    assert np.isclose(
        uvp.get_integrations((0, ((24, 25), (24, 25)), ("xx", "xx")))[3], 0.0
    )
    # average spectra
    avg_uvp = uvp.average_spectra(
        blpair_groups=[sorted(np.unique(uvp.blpair_array))],
        time_avg=True,
        inplace=False,
    )
    # repeat but change data in flagged portion
    ds.dsets[0].data_array[uvd.antpair2ind(24, 25, ordered=False), 400:450, :][3] *= 100
    uvp2 = ds.pspec(
        [(24, 25), (37, 38), (38, 39)],
        [(24, 25), (37, 38), (38, 39)],
        (0, 1),
        ("xx", "xx"),
        spw_ranges=[(400, 450)],
        verbose=False,
    )
    avg_uvp2 = uvp.average_spectra(
        blpair_groups=[sorted(np.unique(uvp.blpair_array))],
        time_avg=True,
        inplace=False,
    )
    # assert average before and after are the same!
    assert avg_uvp == avg_uvp2


def test_RFI_flag_propagation(beam_nf_dipole, uvd):
    # generate ds and weights
    uvd = copy.deepcopy(uvd)
    uvd.flag_array[:] = False
    Nfreq = uvd.data_array.shape[1]
    # Basic test of shape
    ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=beam_nf_dipole)
    test_R = ds.R((1, 37, 38, "XX"))
    assert test_R.shape == (Nfreq, Nfreq)

    # First test that turning-off flagging does nothing if there are no flags in the data
    bls1 = [(24, 25)]
    bls2 = [(37, 38)]
    ds = pspecdata.PSpecData(
        dsets=[uvd, uvd], wgts=[None, None], beam=beam_nf_dipole, labels=["red", "blue"]
    )
    uvp_flagged = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        little_h=True,
        verbose=False,
    )
    ds.broadcast_dset_flags(unflag=True)
    uvp_unflagged = ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        input_data_weight="identity",
        norm="I",
        taper="none",
        little_h=True,
        verbose=False,
    )

    qe_unflagged = uvp_unflagged.get_data((0, ((24, 25), (37, 38)), ("xx", "xx")))[0]
    qe_flagged = uvp_flagged.get_data((0, ((24, 25), (37, 38)), ("xx", "xx")))[0]

    # assert answers are same to within 0.1%
    assert np.isclose(
        np.real(qe_unflagged) / np.real(qe_flagged), 1, atol=0.001, rtol=0.001
    ).all()

    # Test that when flagged, the data within a channel really don't have any effect on the final result

    uvd2 = copy.deepcopy(uvd)
    uvd2.flag_array[uvd.antpair2ind(24, 25, ordered=False)] = True
    ds = pspecdata.PSpecData(dsets=[uvd2, uvd2], wgts=[None, None], beam=beam_nf_dipole)
    with pytest.warns(
        UserWarning, match="Some integrations have zero nsamples, but non-zero weights"
    ):
        uvp_flagged = ds.pspec(
            bls1,
            bls2,
            (0, 1),
            ("xx", "xx"),
            input_data_weight="identity",
            norm="I",
            taper="none",
            little_h=True,
            verbose=False,
        )

    uvd2.data_array[uvd.antpair2ind(24, 25, ordered=False)] *= 9234.913
    ds = pspecdata.PSpecData(dsets=[uvd2, uvd2], wgts=[None, None], beam=beam_nf_dipole)
    with pytest.warns(
        UserWarning, match="Some integrations have zero nsamples, but non-zero weights"
    ):
        uvp_flagged_mod = ds.pspec(
            bls1,
            bls2,
            (0, 1),
            ("xx", "xx"),
            input_data_weight="identity",
            norm="I",
            taper="none",
            little_h=True,
            verbose=False,
        )

    qe_flagged_mod = uvp_flagged_mod.get_data((0, ((24, 25), (37, 38)), ("xx", "xx")))[
        0
    ]
    qe_flagged = uvp_flagged.get_data((0, ((24, 25), (37, 38)), ("xx", "xx")))[0]

    # assert answers are same to within 0.1%
    assert np.isclose(
        np.real(qe_flagged_mod), np.real(qe_flagged), atol=0.001, rtol=0.001
    ).all()

    # Test below commented out because this sort of aggressive symmetrization is not yet implemented.
    # # Test that flagging a channel for one dataset (e.g. just left hand dataset x2)
    # # is equivalent to flagging for both x1 and x2.
    # test_wgts_flagged = copy.deepcopy(test_wgts)
    # test_wgts_flagged.data_array[:,:,40:60] = 0. # Flag 20 channels
    # ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[test_wgts_flagged, test_wgts_flagged], beam=beam_nf_dipole)
    # print "mode alpha"
    # uvp_flagged = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='diagonal', norm='I', taper='none',
    #                         little_h=True, verbose=False)
    # ds = pspecdata.PSpecData(dsets=[uvd, uvd], wgts=[None, test_wgts_flagged], beam=beam_nf_dipole)
    # print "mode beta"
    # uvp_flagged_asymm = ds.pspec(bls1, bls2, (0, 1), ('xx','xx'), input_data_weight='diagonal', norm='I', taper='none',
    #                         little_h=True, verbose=False)

    # qe_flagged_asymm = uvp_flagged_asymm .get_data(0, ((24, 25), (37, 38)), 'xx')[0]
    # qe_flagged = uvp_flagged.get_data(0, ((24, 25), (37, 38)), 'xx')[0]

    # #print np.real(qe_flagged_asymm)/np.real(qe_flagged)

    # # assert answers are same to within 3%
    # assert np.isclose(np.real(qe_flagged_asymm)/np.real(qe_flagged), 1, atol=0.03, rtol=0.03).all()

    # print(uvd.data_array.shape)


def test_validate_blpairs(uvd):
    # test exceptions
    uvd = copy.deepcopy(uvd)
    with pytest.raises(TypeError, match="uvd1 must be a UVData instance"):
        pspecdata.validate_blpairs([((1, 2), (2, 3))], None, uvd)
    with pytest.raises(TypeError, match="uvd2 must be a UVData instance"):
        pspecdata.validate_blpairs([((1, 2), (2, 3))], uvd, None)

    bls = [(24, 25), (37, 38)]
    bls1, bls2, blpairs = utils.construct_blpairs(
        bls, exclude_permutations=False, exclude_auto_bls=True
    )
    pspecdata.validate_blpairs(blpairs, uvd, uvd)
    bls1, bls2, blpairs = utils.construct_blpairs(
        bls, exclude_permutations=False, exclude_auto_bls=True, group=True
    )

    pspecdata.validate_blpairs(blpairs, uvd, uvd)

    # test non-redundant
    blpairs = [((24, 25), (24, 38))]
    pspecdata.validate_blpairs(blpairs, uvd, uvd)


@pytest.mark.filterwarnings(
    "ignore:Some integrations have zero nsamples, but non-zero weights"
)
def test_pspec_run(tmp_path):
    fnames = [
        os.path.join(DATA_PATH, d)
        for d in ["zen.even.xx.LST.1.28828.uvOCRSA", "zen.odd.xx.LST.1.28828.uvOCRSA"]
    ]

    beamfile = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    fnames_std = [
        os.path.join(DATA_PATH, d)
        for d in [
            "zen.even.std.xx.LST.1.28828.uvOCRSA",
            "zen.odd.std.xx.LST.1.28828.uvOCRSA",
        ]
    ]

    # test basic execution

    ds = pspecdata.pspec_run(
        fnames,
        str(tmp_path / "out.h5"),
        Jy2mK=False,
        verbose=False,
        overwrite=True,
        dset_pairs=[(0, 1)],
        bl_len_range=(14, 15),
        bl_deg_range=(50, 70),
        psname_ext="_0",
        spw_ranges=[(0, 25)],
    )
    psc = container.PSpecContainer(str(tmp_path / "out.h5"), keep_open=False)
    assert isinstance(psc, container.PSpecContainer)
    assert psc.groups() == ["dset0_dset1"]
    assert psc.spectra(psc.groups()[0]) == ["dset0_x_dset1_0"]
    assert os.path.exists(str(tmp_path / "out.h5"))

    # test Jy2mK, blpairs, cosmo, cov_array, spw_ranges, dset labeling
    cosmo = conversions.Cosmo_Conversions(Om_L=0.0)

    ds = pspecdata.pspec_run(
        fnames,
        str(tmp_path / "out.h5"),
        dsets_std=fnames_std,
        Jy2mK=True,
        beam=beamfile,
        blpairs=[((37, 38), (37, 38)), ((37, 38), (52, 53))],
        verbose=False,
        overwrite=True,
        pol_pairs=[("xx", "xx"), ("xx", "xx")],
        dset_labels=["foo", "bar"],
        dset_pairs=[(0, 0), (0, 1)],
        spw_ranges=[(50, 75), (120, 140)],
        n_dlys=[20, 20],
        cosmo=cosmo,
        trim_dset_lsts=False,
        broadcast_dset_flags=False,
        cov_model="empirical",
        store_cov=True,
    )

    # assert groupname is dset1_dset2
    psc = container.PSpecContainer(str(tmp_path / "out.h5"), keep_open=False)
    assert "foo_bar" in psc.groups()

    # assert uvp names are labeled by dset_pairs
    assert sorted(psc.spectra("foo_bar")) == sorted(["foo_x_bar", "foo_x_foo"])

    # get UVPSpec for further inspection
    uvp = psc.get_pspec("foo_bar", "foo_x_bar")

    # assert Jy2mK worked
    assert uvp.vis_units == "mK"

    # assert only blpairs that were fed are present
    assert uvp.bl_array.tolist() == [137138, 152153]
    assert uvp.polpair_array.tolist() == [1515, 1515]

    # assert weird cosmology was passed
    assert uvp.cosmo == cosmo

    # assert cov_array was calculated b/c std files were passed and store_cov
    assert hasattr(uvp, "cov_array_real")
    # assert dset labeling propagated
    assert set(uvp.labels) == set(["bar", "foo"])

    # assert spw_ranges and n_dlys specification worked
    np.testing.assert_array_equal(
        uvp.get_spw_ranges(),
        [(163476562.5, 165917968.75, 25, 20), (170312500.0, 172265625.0, 20, 20)],
    )

    # test single_dset, time_interleaving, rephasing, flag broadcasting
    uvd = UVData()
    uvd.read_miriad(fnames[0])
    # interleave the data by hand, and add some flags in
    uvd.flag_array[:] = False
    uvd.flag_array[uvd.antpair2ind(37, 38, ordered=False), 10, 0][0] = True
    uvd.flag_array[uvd.antpair2ind(37, 38, ordered=False), 15, 0][:3] = True
    uvd1 = uvd.select(times=np.unique(uvd.time_array)[::2], inplace=False)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)
    ds = pspecdata.pspec_run(
        [copy.deepcopy(uvd)],
        str(tmp_path / "out2.h5"),
        blpairs=[((37, 38), (37, 38)), ((37, 38), (52, 53))],
        interleave_times=True,
        verbose=False,
        overwrite=True,
        spw_ranges=[(0, 25)],
        rephase_to_dset=0,
        broadcast_dset_flags=True,
        time_thresh=0.3,
    )
    psc = container.PSpecContainer(str(tmp_path / "out2.h5"), keep_open=False)
    assert isinstance(psc, container.PSpecContainer)
    assert psc.groups() == ["dset0_dset1"]
    assert psc.spectra(psc.groups()[0]) == ["dset0_x_dset1"]

    # assert dsets are properly interleaved
    assert np.isclose(
        (np.unique(ds.dsets[0].time_array) - np.unique(ds.dsets[1].time_array))[0],
        -np.diff(np.unique(uvd.time_array))[0],
    )

    # assert first integration flagged across entire spw
    assert ds.dsets[0].get_flags(37, 38)[0, 0:25].all()

    # assert first integration flagged *ONLY* across spw
    assert not (
        ds.dsets[0].get_flags(37, 38)[0, :0].any()
        + ds.dsets[0].get_flags(37, 38)[0, 25:].any()
    )

    # assert channel 15 flagged for all ints
    assert ds.dsets[0].get_flags(37, 38)[:, 15].all()

    # assert phase errors decreased after re-phasing
    phserr_before = np.mean(np.abs(np.angle(uvd1.data_array / uvd2.data_array)))
    phserr_after = np.mean(
        np.abs(np.angle(ds.dsets[0].data_array / ds.dsets[1].data_array))
    )
    assert phserr_after < phserr_before

    # test without using future array shape
    uvd_non_future = UVData()
    uvd_non_future.read_miriad(fnames[0])
    ds = pspecdata.pspec_run(
        [copy.deepcopy(uvd_non_future)],
        str(tmp_path / "out2.h5"),
        dsets_std=[copy.deepcopy(uvd_non_future)],
        blpairs=[((37, 38), (37, 38)), ((37, 38), (52, 53))],
        interleave_times=True,
        verbose=False,
        overwrite=True,
        spw_ranges=[(0, 25)],
        rephase_to_dset=0,
        broadcast_dset_flags=True,
        time_thresh=0.3,
    )
    # assert ds passes validation
    psc = container.PSpecContainer(str(tmp_path / "out2.h5"), keep_open=False)
    assert ds.dsets_std[0] is not None
    ds.validate_datasets()

    # repeat feeding dsets_std and wgts
    ds = pspecdata.pspec_run(
        [copy.deepcopy(uvd)],
        str(tmp_path / "out2.h5"),
        dsets_std=[copy.deepcopy(uvd)],
        blpairs=[((37, 38), (37, 38)), ((37, 38), (52, 53))],
        interleave_times=True,
        verbose=False,
        overwrite=True,
        spw_ranges=[(0, 25)],
        rephase_to_dset=0,
        broadcast_dset_flags=True,
        time_thresh=0.3,
    )
    # assert ds passes validation
    psc = container.PSpecContainer(str(tmp_path / "out2.h5"), keep_open=False)
    assert ds.dsets_std[0] is not None
    ds.validate_datasets()

    # test lst trimming
    uvd1 = copy.deepcopy(uvd)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[2:], inplace=False)
    ds = pspecdata.pspec_run(
        [copy.deepcopy(uvd1), copy.deepcopy(uvd2)],
        str(tmp_path / "out2.h5"),
        blpairs=[((37, 38), (37, 38)), ((37, 38), (52, 53))],
        verbose=False,
        overwrite=True,
        spw_ranges=[(50, 100)],
        trim_dset_lsts=True,
    )
    # assert first uvd1 lst_array got trimmed by 2 integrations
    psc = container.PSpecContainer(str(tmp_path / "out2.h5"), keep_open=False)
    assert ds.dsets[0].Ntimes == 8
    assert np.isclose(np.unique(ds.dsets[0].lst_array), np.unique(uvd2.lst_array)).all()

    # test when no data is loaded in dset

    with pytest.warns(
        UserWarning,
        match="pspec_run produced no output because the selected data contains no matching baseline-pairs.",
    ):
        ds = pspecdata.pspec_run(
            fnames,
            str(tmp_path / "out_nobl.h5"),
            Jy2mK=False,
            verbose=False,
            overwrite=True,
            blpairs=[((500, 501), (600, 601))],
        )  # blpairs that don't exist
    assert ds is None
    assert not os.path.exists(str(tmp_path / "out_nobl.h5"))

    # same test but with pre-loaded UVDatas
    uvds = []
    for f in fnames:
        uvd = UVData()
        uvd.read_miriad(f)
        uvds.append(uvd)
    with pytest.warns(
        UserWarning,
        match="pspec_run produced no output because the selected data contains no matching baseline-pairs.",
    ):
        ds = pspecdata.pspec_run(
            uvds,
            str(tmp_path / "out_nobl.h5"),
            dsets_std=fnames_std,
            Jy2mK=False,
            verbose=False,
            overwrite=True,
            blpairs=[((500, 501), (600, 601))],
        )
    assert ds is None
    assert not os.path.exists(str(tmp_path / "out_nobl.h5"))

    # test when data is loaded, but no blpairs match

    with pytest.warns(
        UserWarning,
        match="pspec_run produced no output because the selected data contains no matching baseline-pairs.",
    ):
        ds = pspecdata.pspec_run(
            fnames,
            str(tmp_path / "out_nobl.h5"),
            Jy2mK=False,
            verbose=False,
            overwrite=True,
            blpairs=[((37, 38), (600, 601))],
        )
    assert isinstance(ds, pspecdata.PSpecData)
    assert not os.path.exists(str(tmp_path / "out_nobl.h5"))

    # test glob-parseable input dataset
    dsets = [
        os.path.join(DATA_PATH, "zen.2458042.?????.xx.HH.uvXA"),
        os.path.join(DATA_PATH, "zen.2458042.?????.xx.HH.uvXA"),
    ]

    ds = pspecdata.pspec_run(
        dsets,
        str(tmp_path / "out.h5"),
        Jy2mK=False,
        verbose=True,
        overwrite=True,
        blpairs=[((24, 25), (37, 38))],
    )
    psc = container.PSpecContainer(str(tmp_path / "out.h5"), "rw", keep_open=False)
    uvp = psc.get_pspec("dset0_dset1", "dset0_x_dset1")
    assert uvp.Ntimes == 120

    # test input calibration
    dfile = os.path.join(DATA_PATH, "zen.2458116.30448.HH.uvh5")
    cfile = os.path.join(DATA_PATH, "zen.2458116.30448.HH.flagged_abs.calfits")
    uvc = UVCal()
    uvc.read_calfits(cfile)
    uvc.gain_scale = "Jy"
    uvc.pol_convention = "avg"
    uvc.extra_keywords["filename"] = json.dumps(cfile)
    ds = pspecdata.pspec_run(
        [dfile, dfile],
        str(tmp_path / "out.h5"),
        cals=[copy.deepcopy(uvc), copy.deepcopy(uvc)],
        dsets_std=[dfile, dfile],
        verbose=False,
        overwrite=True,
        blpairs=[((23, 24), (24, 25))],
        pol_pairs=[("xx", "xx")],
        interleave_times=False,
        file_type="uvh5",
        spw_ranges=[(100, 150)],
        cal_flag=True,
    )
    psc = container.PSpecContainer(str(tmp_path / "out.h5"), "rw", keep_open=False)
    uvp = psc.get_pspec("dset0_dset1", "dset0_x_dset1")
    # test calibration flags were propagated to test that cal was applied
    assert ds.dsets[0].flag_array.any()
    assert ds.dsets[1].flag_array.any()
    assert ds.dsets_std[0].flag_array.any()
    assert ds.dsets_std[1].flag_array.any()
    assert ds.dsets[0].extra_keywords["filename"] != '""'
    assert ds.dsets[0].extra_keywords["calibration"] != '""'
    assert "cal: /" in uvp.history

    # test w/ conjugated blpairs
    dfile = os.path.join(DATA_PATH, "zen.2458116.30448.HH.uvh5")
    ds = pspecdata.pspec_run(
        [dfile, dfile],
        str(tmp_path / "out.h5"),
        cals=[copy.deepcopy(uvc), copy.deepcopy(uvc)],
        dsets_std=[dfile, dfile],
        verbose=False,
        overwrite=True,
        blpairs=[((24, 23), (25, 24))],
        pol_pairs=[("xx", "xx")],
        interleave_times=False,
        file_type="uvh5",
        spw_ranges=[(100, 150)],
        cal_flag=True,
    )
    psc = container.PSpecContainer(str(tmp_path / "out.h5"), "rw", keep_open=False)
    uvp = psc.get_pspec("dset0_dset1", "dset0_x_dset1")
    assert uvp.Nblpairs == 1

    # test exceptions
    with pytest.raises(AssertionError, match="dsets must be fed as a list"):
        pspecdata.pspec_run("foo", str(tmp_path / "out.h5"))
    with pytest.raises(AssertionError, match="blpairs must be fed as a list of baseline-pair tuples"):
        pspecdata.pspec_run(
            fnames, str(tmp_path / "out.h5"), blpairs=(1, 2), verbose=False
        )
    with pytest.raises(AssertionError, match="blpairs must be fed as a list of baseline-pair tuples"):
        pspecdata.pspec_run(
            fnames, str(tmp_path / "out.h5"), blpairs=[1, 2], verbose=False
        )
    with pytest.raises(AssertionError):
        pspecdata.pspec_run(fnames, str(tmp_path / "out.h5"), beam=1, verbose=False)

    # test execution with list of files for each dataset and list of cals

    fnames = glob.glob(os.path.join(DATA_PATH, "zen.2458116.*.HH.uvh5"))
    cals = glob.glob(os.path.join(DATA_PATH, "zen.2458116.*.HH.flagged_abs.calfits"))
    with pytest.warns(
        UserWarning,
        match="gain_scale is not set|pol_convention is not specified on the UVCal object|Neither uvd_pol_convention nor uvc_pol_convention are specified",
    ):
        ds = pspecdata.pspec_run(
            [fnames, fnames],
            str(tmp_path / "out5.h5"),
            Jy2mK=False,
            verbose=False,
            overwrite=True,
            file_type="uvh5",
            bl_len_range=(14, 15),
            bl_deg_range=(0, 1),
            psname_ext="_0",
            spw_ranges=[(0, 25)],
            cals=[cals, cals],
        )
    psc = container.PSpecContainer(str(tmp_path / "out5.h5"), "rw", keep_open=False)
    assert isinstance(psc, container.PSpecContainer)
    assert psc.groups() == ["dset0_dset1"]
    assert psc.spectra(psc.groups()[0]) == ["dset0_x_dset1_0"]
    assert os.path.exists(str(tmp_path / "out.h5"))

    # test with cov_model that requires autos w/ fname as filepath
    fnames = glob.glob(os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA"))
    pspecdata.pspec_run(
        [fnames],
        str(tmp_path / "out.h5"),
        spw_ranges=[(50, 70)],
        dset_pairs=[(0, 0)],
        verbose=False,
        overwrite=True,
        file_type="miriad",
        pol_pairs=[("xx", "xx")],
        blpairs=[((37, 38), (37, 38))],
        cov_model="foreground_dependent",
        store_cov=True,
    )
    psc = container.PSpecContainer(str(tmp_path / "out.h5"), keep_open=False)
    uvp = psc.get_pspec("dset0", "dset0_x_dset0")
    assert hasattr(uvp, "cov_array_real")


def test_input_calibration():
    dfiles = sorted(glob.glob(os.path.join(DATA_PATH, "zen.2458116.30*.HH.uvh5")))
    cfiles = sorted(
        glob.glob(os.path.join(DATA_PATH, "zen.2458116.30*.HH.flagged_abs.calfits"))
    )
    for i, f in enumerate(zip(dfiles, cfiles)):
        uvd = UVData()
        uvd.read(f[0])
        dfiles[i] = uvd
        uvc = UVCal()
        uvc.read_calfits(f[1])
        uvc.gain_scale = "Jy"
        uvc.pol_convention = "avg"
        cfiles[i] = uvc

    # test add
    pd = pspecdata.PSpecData()
    pd.add(dfiles, None)  # w/o cal
    pd.add(
        [copy.deepcopy(uv) for uv in dfiles], None, cals=cfiles, cal_flag=False
    )  # with cal
    g = (cfiles[0].get_gains(23, "x") * np.conj(cfiles[0].get_gains(24, "x"))).T
    np.testing.assert_array_almost_equal(
        pd.dsets[0].get_data(23, 24, "xx") / g, pd.dsets[1].get_data(23, 24, "xx")
    )

    # test add with dictionaries
    pd.add(
        {"one": copy.deepcopy(dfiles[0])},
        {"one": None},
        cals={"one": cfiles[0]},
        cal_flag=False,
    )
    np.testing.assert_array_almost_equal(
        pd.dsets[0].get_data(23, 24, "xx") / g, pd.dsets[2].get_data(23, 24, "xx")
    )

    # test dset_std calibration
    pd.add(
        [copy.deepcopy(uv) for uv in dfiles],
        None,
        dsets_std=[copy.deepcopy(uv) for uv in dfiles],
        cals=cfiles,
        cal_flag=False,
    )
    np.testing.assert_array_almost_equal(
        pd.dsets[0].get_data(23, 24, "xx") / g, pd.dsets_std[3].get_data(23, 24, "xx")
    )

    # test exceptions
    pd = pspecdata.PSpecData()
    with pytest.raises(TypeError, match="If 'cals' is a dict, 'cals' must also be a dict"):
        pd.add(
            {"one": copy.deepcopy(dfiles[0])}, {"one": None}, cals="foo", cal_flag=False
        )
    with pytest.raises(AssertionError, match="The dsets and cals lists must have equal length"):
        pd.add(dfiles, [None], cals=[None, None])
    with pytest.raises(TypeError, match="Only UVCal objects can be used for calibration"):
        pd.add(dfiles, [None], cals=["foo"])


def test_window_funcs():
    """
    Test window function computation in ds.pspec()
    This is complementary to test_get_MW above.
    """
    # get a PSpecData
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, "zen.even.xx.LST.1.28828.uvOCRSA"))
    beam = pspecbeam.PSpecBeamUV(
        os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    )
    ds = pspecdata.PSpecData(dsets=[copy.deepcopy(uvd)], beam=beam)
    ds.set_spw((0, 20))
    ds.set_taper("bh")
    bl = (37, 38)
    key = (0, bl, "xx")
    d = uvd.get_data(bl)
    C = np.cov(d[:, :20].T).real
    iC = np.linalg.pinv(C)
    # iterate over various R and M matrices and ensure
    # normalization and dtype is consistent
    for data_weight in ["identity", "iC"]:
        ds.set_weighting(data_weight)
        for norm in ["H^-1", "I", "V^-1/2"]:
            for exact_norm in [True, False]:
                if exact_norm and norm != "I":
                    # exact_norm only supported for norm == 'I'
                    continue
                ds.clear_cache()
                if data_weight == "iC":
                    # fill R with iC
                    ds._R[(0, (37, 38, "xx"), "iC", "bh")] = iC
                # compute G and H
                Gv = ds.get_G(key, key, exact_norm=exact_norm, pol="xx")
                Hv = ds.get_H(key, key, exact_norm=exact_norm, pol="xx")
                Mv, Wv = ds.get_MW(
                    Gv, Hv, mode=norm, exact_norm=exact_norm, band_covar=C
                )
                # assert row-sum is normalized to 1
                assert np.isclose(Wv.sum(axis=1).real, 1).all()
                # assert this is a real matrix, even though imag is populated
                assert np.isclose(Wv.imag, 0, atol=1e-6).all()


def test_get_argparser():
    args = pspecdata.get_pspec_run_argparser()
    a = args.parse_args(
        [
            ["foo"],
            "bar",
            "--dset_pairs",
            "0~0,1~1",
            "--pol_pairs",
            "xx~xx,yy~yy",
            "--spw_ranges",
            "300~400, 600~800",
            "--blpairs",
            "24~25~24~25, 37~38~37~38",
        ]
    )
    assert a.pol_pairs == [("xx", "xx"), ("yy", "yy")]
    assert a.dset_pairs == [(0, 0), (1, 1)]
    assert a.spw_ranges == [(300, 400), (600, 800)]
    assert a.blpairs == [((24, 25), (24, 25)), ((37, 38), (37, 38))]


def test_get_argparser_backwards_compatibility():
    args = pspecdata.get_pspec_run_argparser()
    a = args.parse_args(
        [
            ["foo"],
            "bar",
            "--dset_pairs",
            "0 0, 1 1",
            "--pol_pairs",
            "xx xx, yy yy",
            "--spw_ranges",
            "300 400, 600 800",
            "--blpairs",
            "24 25 24 25, 37 38 37 38",
        ]
    )
    assert a.pol_pairs == [("xx", "xx"), ("yy", "yy")]
    assert a.dset_pairs == [(0, 0), (1, 1)]
    assert a.spw_ranges == [(300, 400), (600, 800)]
    assert a.blpairs == [((24, 25), (24, 25)), ((37, 38), (37, 38))]


"""
# LEGACY MONTE CARLO TESTS
    def validate_get_G(self,tolerance=0.2,NDRAWS=100,NCHAN=8):
        '''
        Test get_G where we interpret G in this case to be the Fisher Matrix.
        Args:
            tolerance, required max fractional difference from analytical
                       solution to pass.
            NDRAWS, number of random data sets to sample frome.
            NCHAN, number of channels. Must be less than test data sets.
        '''
        #read in data.
        dpath=os.path.join(DATA_PATH,'zen.2458042.12552.xx.HH.uvXAA')
        data=uv.UVData()
        wghts=uv.UVData()
        data.read_miriad(dpath)
        wghts.read_miriad(dpath)
        assert(NCHAN<data.Nfreqs)
        #make sure we use fewer channels.
        data.select(freq_chans=range(NCHAN))
        wghts.select(freq_chans=range(NCHAN))
        #********************************************************************
        #set data to random white noise with a random variance and mean.
        ##!!!Set mean to zero for now since analyitic solutions assumed mean
        ##!!!Subtracted data which oqe isn't actually doing.
        #*******************************************************************
        test_mean=0.*np.abs(np.random.randn())
        test_std=np.abs(np.random.randn())
        #*******************************************************************
        #Make sure that all of the flags are set too true for analytic case.
        #*******************************************************************
        data.flag_array[:]=False
        wghts.data_array[:]=1.
        wghts.flag_array[:]=False
        bllist=data.get_antpairs()
        #*******************************************************************
        #These are the averaged "fisher matrices"
        #*******************************************************************
        f_mat=np.zeros((data.Nfreqs,data.Nfreqs),dtype=complex)
        f_mat_true=np.zeros((data.Nfreqs,data.Nfreqs),dtype=complex)
        nsamples=0
        for datanum in range(NDATA):
            #for each data draw, generate a random data set.
            pspec=pspecdata.PSpecData()
            data.data_array=test_std\
            *np.random.standard_normal(size=data.data_array.shape)\
            /np.sqrt(2.)+1j*test_std\
            *np.random.standard_normal(size=data.data_array.shape)\
            /np.sqrt(2.)+(1.+1j)*test_mean
            pspec.add([data],[wghts])
            #get empirical Fisher matrix for baselines 0 and 1.
            pair1=bllist[0]
            pair2=bllist[1]
            k1=(0,pair1[0],pair1[1],-5)
            k2=(0,pair2[0],pair2[1],-5)
            #add to fisher averages.
            f_mat_true=f_mat_true+pspec.get_F(k1,k2,true_fisher=True)
            f_mat=f_mat+pspec.get_F(k1,k2)
            #test identity
            self.assertTrue(np.allclose(pspec.get_F(k1,k2,use_identity=True)/data.Nfreqs**2.,
                            np.identity(data.Nfreqs).astype(complex)))
            del pspec
        #divide out empirical Fisher matrices by analytic solutions.
        f_mat=f_mat/NDATA/data.Nfreqs**2.*test_std**4.
        f_mat_true=f_mat_true/NDATA/data.Nfreqs**2.*test_std**4.
        #test equality to analytic solutions
        self.assertTrue(np.allclose(f_mat,
                        np.identity(data.Nfreqs).astype(complex),
                        rtol=tolerance,
                        atol=tolerance)
        self.assertTrue(np.allclose(f_mat_true,
                                    np.identity(data.Nfreqs).astype(complex),
                                    rtol=tolerance,
                                    atol=tolerance)
        #TODO: Need a test case for some kind of taper.
    def validate_get_MW(self,NCHANS=20):
        '''
        Test get_MW with analytical case.
        Args:
            NCHANS, number of channels to validate.
        '''
        ###
        test_std=np.abs(np.random.randn())
        f_mat=np.identity(NCHANS).astype(complex)/test_std**4.*nchans**2.
        pspec=pspecdata.PSpecData()
        m,w=pspec.get_MW(f_mat,mode='G^-1')
        #test M/W matrices are close to analytic solutions
        #check that rows in W sum to unity.
        self.assertTrue(np.all(np.abs(w.sum(axis=1)-1.)<=tolerance))
        #check that W is analytic soluton (identity)
        self.assertTrue(np.allclose(w,np.identity(nchans).astype(complex)))
        #check that M.F = W
        self.assertTrue(np.allclose(np.dot(m,f_mat),w))
        m,w=pspec.get_MW(f_mat,mode='G^-1/2')
        #check W is identity
        self.assertTrue(np.allclose(w,np.identity(nchans).astype(complex)))
        self.assertTrue(np.allclose(np.dot(m,f_mat),w))
        #check that L^-1 runs.
        m,w=pspec.get_MW(f_mat,mode='G^-1')
    def validate_q_hat(self,NCHAN=8,NDATA=1000,):
        '''
        validate q_hat calculation by drawing random white noise data sets
        '''
        dpath=os.path.join(DATA_PATH,'zen.2458042.12552.xx.HH.uvXAA')
        data=uv.UVData()
        wghts=uv.UVData()
        data.read_miriad(dpath)
        wghts.read_miriad(dpath)
        assert(NCHAN<=data.Nfreqs)
        data.select(freq_chans=range(NCHAN))
        wghts.select(freq_chans=range(NCHAN))
        #***************************************************************
        #set data to random white noise with a random variance and mean
        #q_hat does not subtract a mean so I will set it to zero for
        #the test.
        #****************************************************************
        test_mean=0.*np.abs(np.random.randn())#*np.abs(np.random.randn())
        test_std=np.abs(np.random.randn())
        data.flag_array[:]=False#Make sure that all of the flags are set too true for analytic case.
        wghts.data_array[:]=1.
        wghts.flag_array[:]=False
        bllist=data.get_antpairs()
        q_hat=np.zeros(NCHAN).astype(complex)
        q_hat_id=np.zeros_like(q_hat)
        q_hat_fft=np.zeros_like(q_hat)
        nsamples=0
        for datanum in range(NDATA):
            pspec=pspecdata.PSpecData()
            data.data_array=test_std*np.random.standard_normal(size=data.data_array.shape)/np.sqrt(2.)\
            +1j*test_std*np.random.standard_normal(size=data.data_array.shape)/np.sqrt(2.)+(1.+1j)*test_mean
            pspec.add([data],[wghts])
            for j in range(data.Nbls):
                #get baseline index
                pair1=bllist[j]
                k1=(0,pair1[0],pair1[1],-5)
                k2=(0,pair1[0],pair1[1],-5)
                #get q
                #test identity
                q_hat=q_hat+np.mean(pspec.q_hat(k1,k2,use_fft=False),
                axis=1)
                q_hat_id=q_hat_id+np.mean(pspec.q_hat(k1,k2,use_identity=True),
                axis=1)
                q_hat_fft=q_hat_fft+np.mean(pspec.q_hat(k1,k2),axis=1)
                nsamples=nsamples+1
            del pspec
        #print nsamples
        nfactor=test_std**2./data.Nfreqs/nsamples
        q_hat=q_hat*nfactor
        q_hat_id=q_hat_id*nfactor/test_std**4.
        q_hat_fft=q_hat_fft*nfactor
        #print q_hat
        #print q_hat_id
        #print q_hat_fft
        self.assertTrue(np.allclose(q_hat,
        np.identity(data.Nfreqs).astype(complex)))
        self.assertTrue(np.allclose(q_hat_id,
        np.identity(data.Nfreqs).astype(complex)))
        self.assertTrue(np.allclose(q_hat_fft,
        np.identity(data.Nfreqs).astype(complex)))
"""
