import copy
import os

import numpy as np
import pytest
from astropy import units
from pyuvdata import UVData

from hera_pspec import PSpecData, conversions, pspecbeam, utils, uvwindow
from hera_pspec.data import DATA_PATH

# Data files to use in tests
dfile = "zen.2458116.31939.HH.uvh5"
ftfile = "FT_beam_HERA_dipole_test_xx.hdf5"
basename = "FT_beam_HERA_dipole_test"
outfile = "test.hdf5"


@pytest.fixture()
def make_ft_beam_obj():
    def _factory(spw_range=None):
        return uvwindow.FTBeam.from_file(
            ftfile=os.path.join(DATA_PATH, ftfile), spw_range=spw_range
        )

    return _factory


@pytest.fixture()
def ft_bandwidth():
    return uvwindow.FTBeam.get_bandwidth(os.path.join(DATA_PATH, ftfile))


@pytest.fixture()
def cosmo():
    return conversions.Cosmo_Conversions()


@pytest.fixture()
def uvwindow_obj(make_ft_beam_obj, cosmo):
    return uvwindow.UVWindow(
        ftbeam_obj=make_ft_beam_obj(spw_range=(5, 25)),
        taper="blackman-harris",
        cosmo=cosmo,
        little_h=True,
        verbose=False,
    )


@pytest.fixture()
def lens():
    uvd = UVData()
    uvd.read(os.path.join(DATA_PATH, dfile), read_data=False)
    return utils.get_reds(uvd, bl_error_tol=1.0, pick_data_ants=False)[1]


@pytest.fixture()
def kbins():
    kmax, dk = 1.0, 0.128 / 2
    krange = np.arange(dk * 1.5, kmax, step=dk)
    return ((krange[1:] + krange[:-1]) / 2) * units.h / units.Mpc


def test_FTBeam_init(make_ft_beam_obj):
    ft_beam_obj = make_ft_beam_obj(spw_range=(5, 25))
    data = ft_beam_obj.ft_beam
    freq_array = ft_beam_obj.freq_array
    mapsize = ft_beam_obj.mapsize

    # initialise directly with array
    test = uvwindow.FTBeam(
        data=data,
        pol="xx",
        freq_array=freq_array,
        mapsize=mapsize,
        verbose=False,
        x_orientation="east",
    )
    assert "xx" == test.pol
    assert np.allclose(data, test.ft_beam)

    # raise assertion error if data is not dim 3
    with pytest.raises(AssertionError, match="Wrong dimensions for data input"):
        uvwindow.FTBeam(
            data=data[:, :, 0], pol="xx", freq_array=freq_array, mapsize=mapsize
        )

    # raise assertion error if data has wrong shape
    with pytest.raises(AssertionError, match="Wrong dimensions for data input"):
        uvwindow.FTBeam(
            data=data[:, :, :-1], pol="xx", freq_array=freq_array, mapsize=mapsize
        )

    # raise assertion error if freq_array and data not compatible
    with pytest.raises(AssertionError, match="data must have shape"):
        uvwindow.FTBeam(
            data=data[:12, :, :], pol="xx", freq_array=freq_array, mapsize=mapsize
        )

    # tests related to pol
    test = uvwindow.FTBeam(data=data, pol=-5, freq_array=freq_array, mapsize=mapsize)
    with pytest.raises(AssertionError, match="Wrong polarisation"):
        uvwindow.FTBeam(pol="test", data=data, freq_array=freq_array, mapsize=mapsize)
    with pytest.raises(AssertionError, match="Wrong polarisation"):
        uvwindow.FTBeam(pol=12, data=data, freq_array=freq_array, mapsize=mapsize)
    with pytest.raises(TypeError, match="Must feed pol as str or int"):
        uvwindow.FTBeam(pol=3.4, data=data, freq_array=freq_array, mapsize=mapsize)


def test_FTBeam_from_beam():
    with pytest.raises(NotImplementedError, match="Coming soon"):
        uvwindow.FTBeam.from_beam(beamfile="test")


def test_FTBeam_from_file(make_ft_beam_obj, ft_bandwidth):
    ft_file = os.path.join(DATA_PATH, ftfile)
    spw_range = (5, 25)

    test = uvwindow.FTBeam.from_file(
        ftfile=ft_file, spw_range=spw_range, verbose=False, x_orientation="east"
    )
    assert test.pol == "xx"

    # tests related to ftfile
    with pytest.raises(
        TypeError,
        match=r"expected str, bytes or os\.PathLike object, not float|argument should be a str or an os\.PathLike object where __fspath__ returns a str, not 'float'",
    ):
        uvwindow.FTBeam.from_file(ftfile=12.0)
    with pytest.raises(ValueError, match="Wrong ftfile input"):
        uvwindow.FTBeam.from_file(ftfile="whatever")

    # tests related to spw_range
    test1 = uvwindow.FTBeam.from_file(ftfile=ft_file, spw_range=spw_range)
    assert np.allclose(
        test1.freq_array, make_ft_beam_obj(spw_range=spw_range).freq_array
    )

    test2 = uvwindow.FTBeam.from_file(ftfile=ft_file, spw_range=None)
    assert np.allclose(test2.freq_array, ft_bandwidth)

    with pytest.raises(AssertionError, match="Wrong spw range format"):
        uvwindow.FTBeam.from_file(spw_range=(13), ftfile=ft_file)
    with pytest.raises(AssertionError, match="Wrong spw range format"):
        uvwindow.FTBeam.from_file(spw_range=(20, 10), ftfile=ft_file)
    with pytest.raises(AssertionError, match="Wrong spw range format"):
        uvwindow.FTBeam.from_file(spw_range=(1001, 1022), ftfile=ft_file)


def test_FTBeam_gaussian(make_ft_beam_obj):
    freq_array = make_ft_beam_obj(spw_range=(5, 25)).freq_array

    # fiducial use
    widths = -0.0343 * freq_array / 1e6 + 11.30
    _ = uvwindow.FTBeam.gaussian(freq_array=freq_array, widths=widths, pol="xx")
    # if widths given as unique number, this value is used for all freqs
    _ = uvwindow.FTBeam.gaussian(
        freq_array=freq_array, widths=np.mean(widths), pol="xx"
    )

    # tests on freq_array consistency
    with pytest.raises(AssertionError, match="Must use at least three frequencies"):
        uvwindow.FTBeam.gaussian(
            freq_array=freq_array[:2], pol="xx", widths=np.mean(widths)
        )
    with pytest.raises(
        AssertionError, match="There must be as many frequencies as widths"
    ):
        uvwindow.FTBeam.gaussian(freq_array=freq_array, pol="xx", widths=widths[:10])

    # make sure widths are given in degrees (raises warning)
    with pytest.warns(UserWarning, match="Small widths"):
        _ = uvwindow.FTBeam.gaussian(freq_array=freq_array, pol="xx", widths=0.10)


def test_FTBeam_get_bandwidth(ft_bandwidth):
    test_bandwidth = uvwindow.FTBeam.get_bandwidth(os.path.join(DATA_PATH, ftfile))
    assert np.all(test_bandwidth == ft_bandwidth)
    with pytest.raises(ValueError, match="Wrong ftfile input"):
        uvwindow.FTBeam.get_bandwidth(ftfile="whatever")


def test_FTBeam_update_spw():
    ft_file = os.path.join(DATA_PATH, ftfile)

    # proper usage
    test = uvwindow.FTBeam.from_file(ftfile=ft_file, spw_range=None)
    test.update_spw((5, 25))

    # tests related to spw_range
    test = uvwindow.FTBeam.from_file(ftfile=ft_file, spw_range=None)
    with pytest.raises(AssertionError, match="Wrong spw range format"):
        test.update_spw(spw_range=(13))
    with pytest.raises(AssertionError, match="Wrong spw range format"):
        test.update_spw(spw_range=(20, 10))
    with pytest.raises(AssertionError, match="Wrong spw range format"):
        test.update_spw(spw_range=(1001, 1022))


def test_UVWindow_init(make_ft_beam_obj, cosmo):
    ft_beam_obj_spw = make_ft_beam_obj(spw_range=(5, 25))
    ft_beam_obj = make_ft_beam_obj()

    # fiducial usage
    test = uvwindow.UVWindow(ftbeam_obj=ft_beam_obj_spw)

    # raise error if two ftbeam_obj are not consistent
    with pytest.raises(
        AssertionError, match="Spectral ranges of the two FTBeam objects do not match"
    ):
        uvwindow.UVWindow(ftbeam_obj=(ft_beam_obj_spw, ft_beam_obj))
    ftbeam_test = copy.deepcopy(ft_beam_obj_spw)
    ftbeam_test.mapsize = 2.0
    with pytest.raises(
        AssertionError,
        match="Physical properties of the two FTBeam objects do not match",
    ):
        uvwindow.UVWindow(ftbeam_obj=(ft_beam_obj_spw, ftbeam_test))
    # raise error if ftbeam_obj is wrong input
    with pytest.raises(AssertionError, match="Wrong input given in ftbeam_obj"):
        uvwindow.UVWindow(ftbeam_obj="test")

    # test taper options
    test = uvwindow.UVWindow(ftbeam_obj=ft_beam_obj_spw, taper="blackman-harris")
    assert test.taper == "blackman-harris"
    test = uvwindow.UVWindow(ftbeam_obj=ft_beam_obj_spw, taper=None)
    assert test.taper is None
    with pytest.raises(ValueError, match="Wrong taper"):
        uvwindow.UVWindow(taper="test", ftbeam_obj=ft_beam_obj_spw)

    # test on cosmo
    test = uvwindow.UVWindow(ftbeam_obj=ft_beam_obj_spw, cosmo=cosmo)
    with pytest.raises(AssertionError, match="If no preferred cosmology"):
        uvwindow.UVWindow(cosmo=None, ftbeam_obj=ft_beam_obj_spw)

    # test on verbose
    test = uvwindow.UVWindow(ftbeam_obj=ft_beam_obj_spw, verbose=True)
    assert test.verbose

    # test on little_h
    test = uvwindow.UVWindow(ftbeam_obj=ft_beam_obj_spw, verbose=True)
    assert test.kunits.is_equivalent(units.h / units.Mpc)
    test = uvwindow.UVWindow(ftbeam_obj=ft_beam_obj_spw, little_h=False)
    assert test.kunits.is_equivalent(units.Mpc ** (-1))


def test_UVWindow_from_uvpspec(make_ft_beam_obj):
    ft_beam_obj_spw = make_ft_beam_obj(spw_range=(5, 25))
    spw_range = (5, 25)
    taper = "blackman-harris"

    # obtain uvp object
    datafile = os.path.join(DATA_PATH, dfile)
    uvd = UVData()
    uvd.read_uvh5(datafile)
    # beam
    beamfile = os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits")
    uvb = pspecbeam.PSpecBeamUV(beamfile, cosmo=None)
    Jy_to_mK = uvb.Jy_to_mK(np.unique(uvd.freq_array), pol="xx")
    uvd.data_array *= Jy_to_mK[None, :, None]
    # Create a new PSpecData object, and don't forget to feed the beam object
    ds = PSpecData(dsets=[uvd, uvd], wgts=[None, None], beam=uvb)
    ds_nocosmo = PSpecData(dsets=[uvd, uvd], wgts=[None, None])
    # choose baselines
    baselines1, baselines2, blpairs = utils.construct_blpairs(
        uvd.get_antpairs()[1:], exclude_permutations=False, exclude_auto_bls=True
    )
    # compute ps
    uvp = ds.pspec(
        baselines1,
        baselines2,
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=(175, 195),
        taper=taper,
        verbose=False,
    )
    uvp_nocosmo = ds_nocosmo.pspec(
        baselines1,
        baselines2,
        dsets=(0, 1),
        pols=[("xx", "xx")],
        spw_ranges=spw_range,
        taper=taper,
        verbose=False,
    )
    uvp_crosspol = ds.pspec(
        baselines1,
        baselines2,
        dsets=(0, 1),
        pols=["xx", "yy"],
        spw_ranges=(175, 195),
        taper=taper,
        verbose=False,
    )

    # proper usage
    _ = uvwindow.UVWindow.from_uvpspec(
        uvp, ipol=0, spw=0, verbose=True, ftbeam=os.path.join(DATA_PATH, basename)
    )
    # if cross polarisation
    _ = uvwindow.UVWindow.from_uvpspec(
        uvp_crosspol, ipol=0, spw=0, ftbeam=os.path.join(DATA_PATH, basename)
    )
    # if no cosmo, use default
    with pytest.warns(UserWarning, match="uvp has no cosmo attribute"):
        _ = uvwindow.UVWindow.from_uvpspec(
            uvp_nocosmo,
            ipol=0,
            spw=0,
            verbose=True,
            ftbeam=os.path.join(DATA_PATH, basename),
        )

    # raise error if no ftbeam as option is not implemented yet
    with pytest.raises(NotImplementedError, match="Coming soon"):
        uvwindow.UVWindow.from_uvpspec(
            uvp=uvp, ipol=0, spw=0, ftbeam=None, verbose=False
        )
    # raise error if wrong type for ftbeam
    with pytest.raises(TypeError, match="Check your ftbeam input"):
        uvwindow.UVWindow.from_uvpspec(
            uvp=uvp, ipol=0, spw=0, ftbeam=np.zeros(12), verbose=False
        )
    # raise error if spw not within uvp.Nspws
    with pytest.warns(UserWarning, match="uvp has no cosmo attribute"):
        with pytest.raises(AssertionError, match="Input spw must be smaller or equal"):
            uvwindow.UVWindow.from_uvpspec(
                uvp=uvp_nocosmo, ipol=0, spw=2, ftbeam=os.path.join(DATA_PATH, basename)
            )

    # use FTBeam object directly as input
    freq_array = ft_beam_obj_spw.freq_array
    widths = -0.0343 * freq_array / 1e6 + 11.30
    gaussian_beam = uvwindow.FTBeam.gaussian(
        freq_array=freq_array, widths=widths, pol="xx"
    )
    _ = uvwindow.UVWindow.from_uvpspec(
        uvp, ipol=0, spw=0, verbose=True, ftbeam=gaussian_beam
    )


def test_UVWindow_get_kgrid(uvwindow_obj, lens):
    bl_len = lens[12]
    _ = uvwindow_obj._get_kgrid(bl_len)
    with pytest.raises(
        AssertionError, match="Change width to resolve full window function"
    ):
        uvwindow_obj._get_kgrid(bl_len=bl_len, width=0.0004)


def test_UVWindow_kperp4bl_freq(uvwindow_obj, lens, make_ft_beam_obj):
    bl_len = lens[12]
    ngrid = make_ft_beam_obj().ft_beam.shape[-1]

    _ = uvwindow_obj._kperp4bl_freq(
        freq=uvwindow_obj.freq_array[12], bl_len=bl_len, ngrid=ngrid
    )
    with pytest.raises(AssertionError, match="Choose frequency within spectral window"):
        uvwindow_obj._kperp4bl_freq(freq=1.35 * 1e8, bl_len=bl_len, ngrid=ngrid)
    with pytest.raises(AssertionError, match="Frequency must be given in Hz"):
        uvwindow_obj._kperp4bl_freq(
            freq=uvwindow_obj.freq_array[12] / 1e6, bl_len=bl_len, ngrid=ngrid
        )


def test_UVWindow_interpolate_ft_beam(uvwindow_obj, lens):
    bl_len = lens[12]
    ft_beam = np.copy(uvwindow_obj.ftbeam_obj_pol[0].ft_beam)
    _ = uvwindow_obj._interpolate_ft_beam(bl_len, ft_beam)

    with pytest.raises(AssertionError, match="ft_beam must be dimension 3"):
        uvwindow_obj._interpolate_ft_beam(bl_len=bl_len, ft_beam=ft_beam[0, :, :])
    with pytest.raises(AssertionError, match="ft_beam must have shape"):
        uvwindow_obj._interpolate_ft_beam(bl_len=bl_len, ft_beam=ft_beam[0:10, :, :])
    with pytest.raises(AssertionError, match="ft_beam must have shape"):
        uvwindow_obj._interpolate_ft_beam(bl_len=bl_len, ft_beam=ft_beam[:, :, :].T)


def test_UVWindow_take_freq_FT(uvwindow_obj, lens):
    bl_len = lens[12]
    ft_beam = np.copy(uvwindow_obj.ftbeam_obj_pol[0].ft_beam)
    interp_ft_beam, _ = uvwindow_obj._interpolate_ft_beam(bl_len, ft_beam)
    delta_nu = np.median(np.diff(uvwindow_obj.freq_array))
    _ = uvwindow_obj._take_freq_FT(interp_ft_beam, delta_nu)

    with pytest.raises(AssertionError, match="interp_ft_beam must be dimension 3"):
        uvwindow_obj._take_freq_FT(interp_ft_beam[0, :, :], delta_nu)
    with pytest.raises(AssertionError, match="interp_ft_beam must have shape"):
        uvwindow_obj._take_freq_FT(interp_ft_beam[:, :, :].T, delta_nu)


def test_UVWindow_get_wf_for_tau(uvwindow_obj, lens):
    bl_len = lens[12]
    tau = uvwindow_obj.dly_array[12]
    kperp_bins = uvwindow_obj.get_kperp_bins([bl_len])
    kperp_bins = np.array(kperp_bins.value)
    kpara_bins = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
    kpara_bins = np.array(kpara_bins.value)

    wf_array1 = np.zeros((kperp_bins.size, uvwindow_obj.Nfreqs))
    _ = uvwindow_obj._get_wf_for_tau(tau, wf_array1, kperp_bins, kpara_bins)


def test_UVWindow_get_kperp_bins(uvwindow_obj, lens):
    with pytest.raises(
        AssertionError, match="get_kperp_bins\\(\\) requires array of baseline lengths"
    ):
        uvwindow_obj.get_kperp_bins(bl_lens=[])
    kperps = uvwindow_obj.get_kperp_bins(lens[12])
    assert uvwindow_obj.kunits.is_equivalent(kperps.unit)
    _ = uvwindow_obj.get_kperp_bins(lens)
    with pytest.warns(UserWarning, match="Large number of kperp/kpara bins"):
        _ = uvwindow_obj.get_kperp_bins(np.r_[1.0, lens])


def test_UVWindow_get_kpara_bins(uvwindow_obj):
    HERA_bw = np.linspace(1, 2, 1024, endpoint=False) * 1e8

    with pytest.raises(AssertionError, match="Must feed list of frequencies"):
        uvwindow_obj.get_kpara_bins(freq_array=uvwindow_obj.freq_array[2])
    _ = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
    with pytest.warns(UserWarning, match="Large number of kperp/kpara bins"):
        _ = uvwindow_obj.get_kpara_bins(HERA_bw)
    kparas = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
    assert uvwindow_obj.kunits.is_equivalent(kparas.unit)


def test_UVWindow_get_cylindrical_wf(uvwindow_obj, lens):
    bl_len = lens[12]

    _, _, cyl_wf = uvwindow_obj.get_cylindrical_wf(
        bl_len, kperp_bins=None, kpara_bins=None, return_bins="weighted"
    )
    cyl_wf = uvwindow_obj.get_cylindrical_wf(
        bl_len, kperp_bins=None, kpara_bins=None, return_bins=None
    )
    kperp, kpara, cyl_wf = uvwindow_obj.get_cylindrical_wf(
        bl_len, kperp_bins=None, kpara_bins=None, return_bins="unweighted"
    )
    # check normalisation
    assert np.allclose(np.sum(cyl_wf, axis=(1, 2)), 1.0, atol=1e-3)
    assert kperp.size == cyl_wf.shape[1]
    assert kpara.size == cyl_wf.shape[2]
    assert uvwindow_obj.Nfreqs == cyl_wf.shape[0]
    # test the bins are recovered by get_kperp_bins and get_kpara_bins
    assert np.allclose(kperp, uvwindow_obj.get_kperp_bins(bl_len).value)
    assert np.allclose(
        kpara, uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array).value
    )

    # kperp bins
    kperp2, _, cyl_wf2 = uvwindow_obj.get_cylindrical_wf(
        bl_len,
        kperp_bins=kperp * uvwindow_obj.kunits,
        kpara_bins=None,
        return_bins="unweighted",
    )
    assert np.allclose(cyl_wf2, cyl_wf)
    assert np.allclose(kperp2, kperp)
    # ValueError raised if kperp_bins not linearly spaced
    kperp_log = np.logspace(-2, 0, 100)
    with pytest.raises(
        ValueError, match="get_cylindrical_wf: kperp_bins must be linearly spaced"
    ):
        uvwindow_obj.get_cylindrical_wf(
            bl_len,
            kperp_bins=kperp_log * uvwindow_obj.kunits,
            kpara_bins=None,
            return_bins="unweighted",
        )

    # kpara bins
    _, kpara3, cyl_wf3 = uvwindow_obj.get_cylindrical_wf(
        bl_len,
        kperp_bins=None,
        kpara_bins=kpara * uvwindow_obj.kunits,
        return_bins="unweighted",
    )
    assert np.allclose(cyl_wf3, cyl_wf)
    assert np.allclose(kpara, kpara3)
    # ValueError raised if kpara_bins not linearly spaced
    kpara_log = np.logspace(-1, 1, 100)
    with pytest.raises(
        ValueError, match="get_cylindrical_wf: kpara_bins must be linearly spaced"
    ):
        uvwindow_obj.get_cylindrical_wf(
            bl_len,
            kperp_bins=None,
            kpara_bins=kpara_log * uvwindow_obj.kunits,
            return_bins="unweighted",
        )

    # test filling array by delay symmetry for odd number of delays
    ft_beam_test = uvwindow.FTBeam.from_file(
        ftfile=os.path.join(DATA_PATH, ftfile), spw_range=(5, 24)
    )
    test = uvwindow.UVWindow(ftbeam_obj=ft_beam_test)
    kperp, kpara, cyl_wf = test.get_cylindrical_wf(bl_len, return_bins="unweighted")


def test_UVWindow_cylindrical_to_spherical(uvwindow_obj, lens, kbins):
    bl_len = lens[12]

    kperp, kpara, cyl_wf = uvwindow_obj.get_cylindrical_wf(
        bl_len, kperp_bins=None, kpara_bins=None, return_bins="unweighted"
    )
    ktot = np.sqrt(kperp[:, None] ** 2 + kpara**2)

    # proper usage
    _ = uvwindow_obj.cylindrical_to_spherical(
        cyl_wf=cyl_wf, kbins=kbins, ktot=ktot, bl_lens=bl_len, bl_weights=[2.0]
    )
    _ = uvwindow_obj.cylindrical_to_spherical(
        cyl_wf=cyl_wf[None], kbins=kbins, ktot=ktot, bl_lens=bl_len, bl_weights=None
    )

    # ktot has shape different from cyl_wf
    with pytest.raises(AssertionError, match="k magnitude grid does not match"):
        uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf,
            kbins=kbins,
            ktot=np.sqrt(kperp[:-2, None] ** 2 + kpara**2),
            bl_lens=bl_len,
        )
    # only one k-bin
    with pytest.raises(AssertionError, match="must feed array of k bins"):
        uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf, kbins=kbins[:1], ktot=ktot, bl_lens=bl_len
        )
    # weights have shape different from bl_lens
    with pytest.raises(AssertionError, match="Blpair weights and lengths do not match"):
        uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf, kbins=kbins, ktot=ktot, bl_lens=bl_len, bl_weights=[1.0, 2.0]
        )
    # bl_lens has different size to cyl_wf.shape[0]
    with pytest.raises(AssertionError):
        uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf[None],
            kbins=kbins,
            ktot=ktot,
            bl_lens=lens[:2],
            bl_weights=[1.0, 2.0],
        )
    # raise warning if empty bins
    kbins_test = np.arange(2, 5, step=0.5) * uvwindow_obj.kunits
    uvwindow_obj.verbose = True
    with pytest.warns(UserWarning, match="Some spherical bins are empty"):
        _ = uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf, kbins=kbins_test, ktot=ktot, bl_lens=bl_len
        )

    # raise ValueError if kbins not linearly spaced
    kbins_log = np.logspace(-2, 2, 20)
    with pytest.raises(
        ValueError, match="cylindrical_to_spherical: kbins must be linearly spaced"
    ):
        uvwindow_obj.cylindrical_to_spherical(
            cyl_wf=cyl_wf,
            kbins=kbins_log * uvwindow_obj.kunits,
            ktot=ktot,
            bl_lens=bl_len,
        )


def test_UVWindow_get_spherical_wf(uvwindow_obj, lens, kbins):

    with pytest.warns(
        UserWarning, match="Max spherical k probed is not included in bins"
    ):
        _ = uvwindow_obj.get_spherical_wf(
            kbins=kbins,
            bl_lens=lens[:1],
            bl_weights=[1],
            kperp_bins=None,
            kpara_bins=None,
            return_weighted_k=True,
            verbose=True,
        )
    kperp_bins = uvwindow_obj.get_kperp_bins(lens[:1])
    kpara_bins = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
    dk = np.diff(kbins.value).mean()
    ktot_max = np.sqrt(kperp_bins.value[:, None] ** 2 + kpara_bins.value**2).max()
    full_kbins = (
        np.arange(kbins.value.min(), ktot_max + dk, step=dk) * uvwindow_obj.kunits
    )

    _ = uvwindow_obj.get_spherical_wf(
        kbins=full_kbins,
        kperp_bins=kperp_bins,
        kpara_bins=kpara_bins,
        bl_lens=lens[:1],
        bl_weights=None,
        return_weighted_k=False,
        verbose=None,
    )

    # check inputs
    with pytest.raises(AttributeError, match="Feed k array with units"):
        uvwindow_obj.get_spherical_wf(kbins=kbins.value, bl_lens=lens[:2])
    with pytest.raises(
        AssertionError, match="bl_weights and bl_lens must have same length"
    ):
        uvwindow_obj.get_spherical_wf(kbins=kbins, bl_lens=lens[:2], bl_weights=[1.0])
    with pytest.raises(AssertionError, match="must feed array of k bins"):
        uvwindow_obj.get_spherical_wf(
            kbins=kbins.value[2] * uvwindow_obj.kunits, bl_lens=lens[:1]
        )

    # test kpara bins not outside of spectral window
    kpara_centre = (
        uvwindow_obj.cosmo.tau_to_kpara(
            uvwindow_obj.avg_z, little_h=uvwindow_obj.little_h
        )
        * abs(uvwindow_obj.dly_array).max()
    )
    bad_kpara_bins = (
        np.arange(2.0 * kpara_centre, 10 * kpara_centre, step=kpara_centre)
        * uvwindow_obj.kunits
    )
    bad_kmax = np.sqrt(kperp_bins.value[:, None] ** 2 + bad_kpara_bins.value**2).max()
    bad_full_kbins = (
        np.arange(kbins.value.min(), bad_kmax + dk, step=dk) * uvwindow_obj.kunits
    )
    with pytest.warns(
        UserWarning, match="The bin centre is not included in the array of kpara bins"
    ):
        WF = uvwindow_obj.get_spherical_wf(
            kbins=bad_full_kbins,
            kperp_bins=kperp_bins,
            kpara_bins=bad_kpara_bins,
            bl_lens=lens[:1],
        )
    # ValueError raised if bins not linearly spaced
    kperp_log = np.logspace(-2, 0, 100)
    with pytest.raises(
        ValueError, match="get_spherical_wf: kperp_bins must be linearly spaced"
    ):
        uvwindow_obj.get_spherical_wf(
            kbins=kbins, kperp_bins=kperp_log * uvwindow_obj.kunits, bl_lens=lens[:1]
        )
    kpara_log = np.logspace(-1, 1, 100)
    with pytest.raises(
        ValueError, match="get_spherical_wf: kpara_bins must be linearly spaced"
    ):
        uvwindow_obj.get_spherical_wf(
            kbins=kbins, kpara_bins=kpara_log * uvwindow_obj.kunits, bl_lens=lens[:1]
        )
    kbins_log = np.logspace(-2, 2, 20)
    with pytest.raises(
        ValueError, match="get_spherical_wf: kbins must be linearly spaced"
    ):
        uvwindow_obj.get_spherical_wf(
            kbins=kbins_log * uvwindow_obj.kunits, bl_lens=lens[:1]
        )


def test_UVWindow_check_kunits(uvwindow_obj, kbins):
    uvwindow_obj.check_kunits(kbins)
    with pytest.raises(AttributeError, match="Feed k array with units"):
        uvwindow_obj.check_kunits(kbins.value)


def test_UVWindow_run_and_write(uvwindow_obj, lens, tmp_path):

    kperp_bins = uvwindow_obj.get_kperp_bins(lens[:1])
    kpara_bins = uvwindow_obj.get_kpara_bins(uvwindow_obj.freq_array)
    # proper usage
    uvwindow_obj.run_and_write(
        filepath=str(tmp_path / outfile),
        bl_lens=lens[:1],
        kperp_bins=kperp_bins,
        kpara_bins=kpara_bins,
        clobber=False,
    )

    # raise error if file already exists and clobber is False
    with pytest.raises(IOError, match="exists, not overwriting"):
        uvwindow_obj.run_and_write(filepath=str(tmp_path / outfile), bl_lens=lens[:1], clobber=False)
    # does not raise if clobber is True
    uvwindow_obj.run_and_write(
        filepath=str(tmp_path / outfile),
        bl_lens=[lens[:1]],
        kperp_bins=None,
        kpara_bins=None,
        clobber=True,
    )

    # check inputs
    with pytest.raises(
        AssertionError, match="bl_weights and bl_lens must have same length"
    ):
        uvwindow_obj.run_and_write(
            filepath=str(tmp_path / outfile), bl_lens=lens[:1], bl_weights=[1.0, 1.0], clobber=True
        )
    with pytest.raises(AttributeError, match="Feed k array with units"):
        uvwindow_obj.run_and_write(
            filepath=str(tmp_path / outfile),
            bl_lens=lens[:1],
            kperp_bins=kperp_bins.value,
            clobber=True,
        )
    with pytest.raises(AttributeError, match="Feed k array with units"):
        uvwindow_obj.run_and_write(
            filepath=str(tmp_path / outfile),
            bl_lens=lens[:1],
            kpara_bins=kpara_bins.value,
            clobber=True,
        )
