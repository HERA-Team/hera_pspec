import copy
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pyuvdata import UVData

from hera_pspec import conversions, grouping, plot, pspecbeam, pspecdata, utils
from hera_pspec.data import DATA_PATH

# Data files to use in tests
dfiles = ["zen.all.xx.LST.1.06964.uvA"]


def axes_contains(ax, obj_list):
    """
    Check that a matplotlib.Axes instance contains certain elements.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axes instance.

    obj_list : list of tuples
        List of tuples, one for each type of object to look for. The tuple
        should be of the form (matplotlib.object, int), where int is the
        number of instances of that object that are expected.
    """
    # Get plot elements
    elems = ax.get_children()

    # Loop over list of objects that should be in the plot
    for obj in obj_list:
        objtype, num_expected = obj
        num = sum(1 for elem in elems if isinstance(elem, objtype))
        if num != num_expected:
            return False

    # Return True if no problems found
    return True


@pytest.fixture
def uvd():
    """Load the raw UVData from the test data file."""
    uvdata = UVData()
    uvdata.read_miriad(os.path.join(DATA_PATH, dfiles[0]))
    return uvdata


@pytest.fixture
def pspec_ds(uvd):
    """Build a PSpecData object (beam + two time-interleaved halves of uvd)."""
    bm = pspecbeam.PSpecBeamUV(os.path.join(DATA_PATH, "HERA_NF_dipole_power.beamfits"))
    bm.filename = "HERA_NF_dipole_power.beamfits"
    # Slide the time axis by one integration to avoid noise bias
    uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)
    uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)
    ds = pspecdata.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None], beam=bm)
    ds.rephase_to_dset(0)
    return ds


@pytest.fixture
def uvp(pspec_ds):
    """Compute a UVPSpec over two spectral windows from pspec_ds."""
    bls = [(24, 25), (37, 38), (38, 39)]
    bls1, bls2, _ = utils.construct_blpairs(
        bls, exclude_permutations=False, exclude_auto_bls=True
    )
    return pspec_ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        spw_ranges=[(300, 400), (600, 721)],
        input_data_weight="identity",
        norm="I",
        taper="blackman-harris",
        verbose=False,
    )


def test_plot_average(uvp):
    """
    Test that plotting routine can average over baselines and times.
    """
    # Unpack the list of baseline-pairs into a Python list
    blpairs = np.unique(uvp.blpair_array)
    blps = [blp for blp in blpairs]

    # Plot the spectra averaged over baseline-pairs and times
    f1 = plot.delay_spectrum(
        uvp, [blps], spw=0, pol=("xx", "xx"), average_blpairs=True, average_times=True
    )
    elements = [(mpl.lines.Line2D, 1)]
    assert axes_contains(f1.axes[0], elements)
    plt.close(f1)

    # Average over baseline-pairs but keep the time bins intact
    f2 = plot.delay_spectrum(
        uvp, [blps], spw=0, pol=("xx", "xx"), average_blpairs=True, average_times=False
    )
    elements = [(mpl.lines.Line2D, uvp.Ntpairs)]
    assert axes_contains(f2.axes[0], elements)
    plt.close(f2)

    # Average over times, but keep the baseline-pairs separate
    f3 = plot.delay_spectrum(
        uvp, [blps], spw=0, pol=("xx", "xx"), average_blpairs=False, average_times=True
    )
    elements = [(mpl.lines.Line2D, uvp.Nblpairs)]
    assert axes_contains(f3.axes[0], elements)
    plt.close(f3)

    # Plot the spectra averaged over baseline-pairs and times, but also
    # fold the delay axis
    f4 = plot.delay_spectrum(
        uvp,
        [blps],
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=True,
        average_times=True,
        fold=True,
    )
    elements = [(mpl.lines.Line2D, 1)]
    assert axes_contains(f4.axes[0], elements)
    plt.close(f4)

    # Plot imaginary part
    f4 = plot.delay_spectrum(
        uvp,
        [blps],
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=False,
        average_times=True,
        component="imag",
    )
    elements = [(mpl.lines.Line2D, uvp.Nblpairs)]
    assert axes_contains(f4.axes[0], elements)
    plt.close(f4)

    # Plot abs
    f5 = plot.delay_spectrum(
        uvp,
        [blps],
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=False,
        average_times=True,
        component="abs",
    )
    elements = [(mpl.lines.Line2D, uvp.Nblpairs)]
    assert axes_contains(f4.axes[0], elements)
    plt.close(f5)

    # test errorbar plotting w/ markers

    # bootstrap resample
    (uvp_avg, _, _) = grouping.bootstrap_resampled_error(
        uvp,
        time_avg=True,
        Nsamples=100,
        normal_std=True,
        robust_std=False,
        verbose=False,
    )

    f6 = plot.delay_spectrum(
        uvp_avg,
        uvp_avg.get_blpairs(),
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=False,
        average_times=False,
        component="real",
        error="bs_std",
        lines=False,
        markers=True,
    )
    plt.close(f6)

    # plot errorbar instead of pspec
    f7 = plot.delay_spectrum(
        uvp_avg,
        uvp_avg.get_blpairs(),
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=False,
        average_times=False,
        component="real",
        lines=False,
        markers=True,
        plot_stats="bs_std",
    )
    plt.close(f7)


def test_plot_cosmo(uvp):
    """
    Test that cosmological units can be used on plots.
    """
    # Unpack the list of baseline-pairs into a Python list
    blpairs = np.unique(uvp.blpair_array)
    blps = [blp for blp in blpairs]

    # Set cosmology and plot in non-delay (i.e. cosmological) units
    uvp.set_cosmology(conversions.Cosmo_Conversions())
    f1 = plot.delay_spectrum(
        uvp,
        [blps],
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=True,
        average_times=True,
        delay=False,
    )
    elements = [(mpl.lines.Line2D, 1), (mpl.legend.Legend, 0)]
    assert axes_contains(f1.axes[0], elements)
    plt.close(f1)

    # Plot in Delta^2 units
    f2 = plot.delay_spectrum(
        uvp,
        [blps],
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=True,
        average_times=True,
        delay=False,
        deltasq=True,
        legend=True,
        label_type="blpair",
    )
    # Should contain 1 line and 1 legend
    elements = [(mpl.lines.Line2D, 1), (mpl.legend.Legend, 1)]
    assert axes_contains(f2.axes[0], elements)
    plt.close(f2)

    # Regression test for folded Delta^2 plotting in cosmological units.
    f3 = plot.delay_spectrum(
        uvp,
        [blps],
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=True,
        average_times=True,
        delay=False,
        deltasq=True,
        fold=True,
        legend=True,
        label_type="blpair",
    )
    elements = [(mpl.lines.Line2D, 1), (mpl.legend.Legend, 1)]
    assert axes_contains(f3.axes[0], elements)
    plt.close(f3)


def test_delay_spectrum_misc(uvp):
    # various other tests for plot.delay_spectrum

    blpairs = np.unique(uvp.blpair_array)

    # times selection, label_type
    f1 = plot.delay_spectrum(
        uvp,
        blpairs[:1],
        spw=0,
        pol=("xx", "xx"),
        times=uvp.time_avg_array[:1],
        lines=False,
        markers=True,
        logscale=False,
        label_type="key",
        force_plot=False,
    )
    plt.close(f1)

    # test force plot exception
    large_uvp = copy.deepcopy(uvp)
    for i in range(3):
        _uvp = copy.deepcopy(large_uvp)
        _uvp.time_avg_array += 0  # don't change avg time to make sure this is ok.
        _uvp.time_1_array += (i + 1) ** 2
        _uvp.time_2_array += (i + 1) ** 2
        large_uvp = large_uvp + _uvp
    with pytest.raises(ValueError, match="Trying to plot > 100 spectra"):
        plot.delay_spectrum(large_uvp, large_uvp.get_blpairs(), 0, "xx")

    f2 = plot.delay_spectrum(
        large_uvp,
        large_uvp.get_blpairs(),
        0,
        ("xx", "xx"),
        force_plot=True,
        label_type="blpairt",
        logscale=False,
        lines=True,
        markers=True,
    )
    plt.close(f2)

    # exceptions
    with pytest.raises(ValueError, match="Couldn't understand label_type foo"):
        plot.delay_spectrum(
            large_uvp, large_uvp.get_blpairs()[:3], 0, ("xx", "xx"), label_type="foo"
        )
    with pytest.raises(KeyError, match="Error variable.*not found in stats_array"):
        plot.delay_spectrum(
            large_uvp, [large_uvp.get_blpairs()[0]], 0, ("xx", "xx"), error="not_a_stat"
        )


def test_delay_spectrum_blpair_input_validation(uvp):
    """Exercise the documented blpair input forms for delay_spectrum."""
    blpair = uvp.get_blpairs()[0]

    fig = plot.delay_spectrum(
        uvp,
        [blpair],
        0,
        ("xx", "xx"),
        times=uvp.time_avg_array[:1],
        legend=True,
        lines=False,
        markers=True,
        logscale=False,
    )
    legend = fig.axes[0].get_legend()
    assert legend is None
    assert f"blpair={blpair}" in fig.axes[0].get_title()
    plt.close(fig)

    with pytest.raises(ValueError, match="blpairs.*baseline-pair tuples"):
        plot.delay_spectrum(uvp, [(24, 25), (37, 38)], 0, ("xx", "xx"))

    with pytest.raises(ValueError, match="blpairs.*baseline-pair tuples"):
        plot.delay_spectrum(uvp, [[(24, 25)]], 0, ("xx", "xx"))

    with pytest.raises(TypeError, match="blpairs must be an iterable"):
        plot.delay_spectrum(uvp, [None], 0, ("xx", "xx"))

    with pytest.raises(TypeError, match="blpairs must be baseline-pair tuples"):
        plot.delay_spectrum(uvp, [[None]], 0, ("xx", "xx"))


def test_delay_spectrum_auto_title_legend(uvp):
    """Smart title/legend should separate static and varying metadata."""
    blpair = uvp.get_blpairs()[0]

    fig, ax = plt.subplots()
    plot.delay_spectrum(
        uvp,
        [blpair],
        0,
        ("xx", "xx"),
        times=uvp.time_avg_array[:2],
        legend=True,
        ax=ax,
        lines=False,
        markers=True,
        logscale=False,
    )
    legend = ax.get_legend()
    assert legend is not None
    legend_text = [text.get_text() for text in legend.get_texts()]
    assert len(legend_text) == 2
    assert all("lst=" in text for text in legend_text)
    assert all("blpair=" not in text for text in legend_text)
    assert all("pol=" not in text for text in legend_text)
    assert "spw=0" in ax.get_title()
    assert f"blpair={blpair}" in ax.get_title()
    assert "pol=('xx', 'xx')" in ax.get_title()
    assert "lst=" not in ax.get_title()
    plt.close(fig)

    fig = plot.delay_spectrum(
        uvp,
        uvp.get_blpairs()[:2],
        0,
        ("xx", "xx"),
        times=uvp.time_avg_array[:1],
        legend=True,
        lines=False,
        markers=True,
        logscale=False,
    )
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_text = [text.get_text() for text in legend.get_texts()]
    assert len(legend_text) == 2
    assert all("blpair=" in text for text in legend_text)
    assert "spw=0" in ax.get_title()
    assert "pol=('xx', 'xx')" in ax.get_title()
    assert "lst=" in ax.get_title()
    plt.close(fig)

    # When average_blpairs=True, blpair should not appear in the title
    # even if all series share the same (averaged) blpair group label.
    all_blpairs = uvp.get_blpairs()
    fig = plot.delay_spectrum(
        uvp, [all_blpairs], 0, ("xx", "xx"), average_blpairs=True, average_times=True
    )
    ax = fig.axes[0]
    assert "blpair=" not in ax.get_title()
    assert "spw=0" in ax.get_title()
    plt.close(fig)


def test_delay_spectrum_title_legend_opt_out_and_override(uvp):
    """Auto title/legend generation should be suppressible and overridable."""
    blpair = uvp.get_blpairs()[0]

    fig = plot.delay_spectrum(
        uvp,
        [blpair],
        0,
        ("xx", "xx"),
        times=uvp.time_avg_array[:2],
        legend=True,
        lines=False,
        markers=True,
        logscale=False,
        title_legend=False,
    )
    ax = fig.axes[0]
    assert ax.get_legend() is None
    assert ax.get_title() == ""
    plt.close(fig)

    fig = plot.delay_spectrum(
        uvp,
        [blpair],
        0,
        ("xx", "xx"),
        times=uvp.time_avg_array[:2],
        legend=True,
        lines=False,
        markers=True,
        logscale=False,
        label_type="blpairt",
    )
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert str(blpair) in legend.get_texts()[0].get_text()
    plt.close(fig)

    title, labels, show_legend = plot._get_delay_spectrum_title_and_labels(
        [], "auto", True
    )
    assert title == ""
    assert labels == []
    assert show_legend is False


def test_plot_waterfall(uvp):
    """
    Test that waterfall can be plotted.
    """
    # Unpack the list of baseline-pairs into a Python list
    blpairs = np.unique(uvp.blpair_array).tolist()
    blps = [uvp.blpair_to_antnums(blp) for blp in blpairs]

    # Set cosmology and plot in non-delay (i.e. cosmological) units
    uvp.set_cosmology(conversions.Cosmo_Conversions(), overwrite=True)
    f1 = plot.delay_waterfall(
        uvp, [blps], spw=0, pol=("xx", "xx"), average_blpairs=True, delay=False
    )
    plt.close(f1)

    # Plot in Delta^2 units
    f2 = plot.delay_waterfall(
        uvp,
        [blps],
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=True,
        delay=False,
        deltasq=True,
    )
    plt.close(f2)

    # Try some other arguments
    f3 = plot.delay_waterfall(
        uvp,
        [blpairs],
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=False,
        delay=True,
        log=False,
        vmin=-1.0,
        vmax=3.0,
        cmap="RdBu",
        fold=True,
        component="abs",
    )
    plt.close(f3)

    # Try with imaginary component
    f4 = plot.delay_waterfall(
        uvp,
        [blpairs],
        spw=0,
        pol=("xx", "xx"),
        average_blpairs=False,
        delay=True,
        log=False,
        vmin=-1.0,
        vmax=3.0,
        cmap="RdBu",
        fold=True,
        component="imag",
    )
    plt.close(f4)

    # Try some more arguments
    fig, axes = plt.subplots(1, len(blps))
    plot.delay_waterfall(
        uvp,
        [blps],
        spw=0,
        pol=("xx", "xx"),
        lst_in_hrs=False,
        times=np.unique(uvp.time_avg_array)[:10],
        axes=axes,
        component="abs",
        title_type="blvec",
    )
    plt.close()

    # exceptions
    large_uvp = copy.deepcopy(uvp)
    for i in range(1, 4):
        _uvp = copy.deepcopy(uvp)
        _uvp.blpair_array += i * 20
        large_uvp += _uvp
    with pytest.raises(ValueError, match="Nblps > 20 and force_plot == False"):
        plot.delay_waterfall(large_uvp, large_uvp.get_blpairs(), 0, ("xx", "xx"))
    _ = plot.delay_waterfall(
        large_uvp, large_uvp.get_blpairs(), 0, ("xx", "xx"), force_plot=True
    )
    plt.close()


def test_uvdata_waterfalls(uvd, tmp_path):
    """
    Test waterfall plotter
    """
    for d in ["data", "flags", "nsamples"]:
        outdir = tmp_path / d
        outdir.mkdir()
        basename = str(outdir / "waterfall_{bl}_{pol}")
        plot.plot_uvdata_waterfalls(
            uvd, basename, vmin=0, vmax=100, data=d, plot_mode="real"
        )
        figfiles = glob.glob(str(outdir / "waterfall_*_*.png"))
        assert len(figfiles) == 15


def test_delay_wedge(pspec_ds):
    """Tests for plot.delay_wedge"""
    # construct new uvp
    reds, lens, angs = utils.get_reds(pspec_ds.dsets[0], pick_data_ants=True)
    bls1, bls2, blps, _, _ = utils.calc_blpair_reds(
        pspec_ds.dsets[0],
        pspec_ds.dsets[1],
        exclude_auto_bls=False,
        exclude_permutations=True,
    )
    uvp = pspec_ds.pspec(
        bls1,
        bls2,
        (0, 1),
        ("xx", "xx"),
        spw_ranges=[(300, 350)],
        input_data_weight="identity",
        norm="I",
        taper="blackman-harris",
        verbose=False,
    )

    # test basic delay_wedge call
    _ = plot.delay_wedge(
        uvp,
        0,
        ("xx", "xx"),
        blpairs=None,
        times=None,
        fold=False,
        delay=True,
        rotate=False,
        component="real",
        log10=False,
        loglog=False,
        red_tol=1.0,
        center_line=False,
        horizon_lines=False,
        title=None,
        ax=None,
        cmap="viridis",
        figsize=(8, 6),
        deltasq=False,
        colorbar=False,
        cbax=None,
        vmin=None,
        vmax=None,
        edgecolor="none",
        flip_xax=False,
        flip_yax=False,
        lw=2,
    )
    plt.close()

    # specify blpairs and times
    _ = plot.delay_wedge(
        uvp,
        0,
        ("xx", "xx"),
        blpairs=uvp.get_blpairs()[-5:],
        times=uvp.time_avg_array[:1],
        fold=False,
        delay=True,
        component="imag",
        rotate=False,
        log10=False,
        loglog=False,
        red_tol=1.0,
        center_line=False,
        horizon_lines=False,
        title=None,
        ax=None,
        cmap="viridis",
        figsize=(8, 6),
        deltasq=False,
        colorbar=False,
        cbax=None,
        vmin=None,
        vmax=None,
        edgecolor="none",
        flip_xax=False,
        flip_yax=False,
        lw=2,
    )
    plt.close()

    # fold, deltasq, cosmo and log10, loglog
    _ = plot.delay_wedge(
        uvp,
        0,
        ("xx", "xx"),
        blpairs=None,
        times=None,
        fold=True,
        delay=False,
        component="abs",
        rotate=False,
        log10=True,
        loglog=True,
        red_tol=1.0,
        center_line=False,
        horizon_lines=False,
        title="hello",
        ax=None,
        cmap="viridis",
        figsize=(8, 6),
        deltasq=True,
        colorbar=False,
        cbax=None,
        vmin=None,
        vmax=None,
        edgecolor="none",
        flip_xax=False,
        flip_yax=False,
        lw=2,
    )
    plt.close()

    # colorbar, vranges, flip_axes, edgecolors, lines
    _ = plot.delay_wedge(
        uvp,
        0,
        ("xx", "xx"),
        blpairs=None,
        times=None,
        fold=False,
        delay=False,
        component="abs",
        rotate=False,
        log10=True,
        loglog=False,
        red_tol=1.0,
        center_line=True,
        horizon_lines=True,
        title="hello",
        ax=None,
        cmap="viridis",
        figsize=(8, 6),
        deltasq=True,
        colorbar=True,
        cbax=None,
        vmin=6,
        vmax=15,
        edgecolor="grey",
        flip_xax=True,
        flip_yax=True,
        lw=2,
        set_bl_tick_minor=True,
    )
    plt.close()

    # feed axes, red_tol
    fig, ax = plt.subplots()
    cbax = fig.add_axes([0.85, 0.1, 0.05, 0.9])
    cbax.axis("off")
    plot.delay_wedge(
        uvp,
        0,
        ("xx", "xx"),
        blpairs=None,
        times=None,
        fold=False,
        delay=True,
        component="abs",
        rotate=True,
        log10=True,
        loglog=False,
        red_tol=10.0,
        center_line=False,
        horizon_lines=False,
        ax=ax,
        cmap="viridis",
        figsize=(8, 6),
        deltasq=False,
        colorbar=True,
        cbax=cbax,
        vmin=None,
        vmax=None,
        edgecolor="none",
        flip_xax=False,
        flip_yax=False,
        lw=2,
        set_bl_tick_major=True,
    )
    plt.close()

    # test exceptions
    with pytest.raises(ValueError, match="at least two baseline pairs"):
        plot.delay_wedge(
            uvp,
            0,
            ("xx", "xx"),
            blpairs=[uvp.get_blpairs()[-1]],
            times=uvp.time_avg_array[:1],
        )

    with pytest.raises(ValueError, match="Did not understand component foo"):
        plot.delay_wedge(uvp, 0, ("xx", "xx"), component="foo")
    plt.close()
