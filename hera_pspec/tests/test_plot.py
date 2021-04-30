import unittest
import pytest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, copy, sys
from .. import pspecdata, pspecbeam, conversions, plot, utils, grouping
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData
import glob

# Data files to use in tests
dfiles = [
    'zen.all.xx.LST.1.06964.uvA',
]

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
    contains_all = False
    for obj in obj_list:
        objtype, num_expected = obj
        num = 0
        for elem in elems:
            if isinstance(elem, objtype): num += 1
        if num != num_expected:
            return False

    # Return True if no problems found
    return True

class Test_Plot(unittest.TestCase):

    def setUp(self):
        """
        Load data and calculate power spectra.
        """
        # Instantiate empty PSpecData
        self.ds = pspecdata.PSpecData()

        # Load datafiles
        uvd = UVData()
        uvd.read_miriad(os.path.join(DATA_PATH, dfiles[0]))
        self.uvd = uvd

        # Load beam file
        beamfile = os.path.join(DATA_PATH, 'HERA_NF_dipole_power.beamfits')
        self.bm = pspecbeam.PSpecBeamUV(beamfile)
        self.bm.filename = 'HERA_NF_dipole_power.beamfits'

        # We only actually have 1 data file here, so slide the time axis by one
        # integration to avoid noise bias
        uvd1 = uvd.select(times=np.unique(uvd.time_array)[:-1:2], inplace=False)
        uvd2 = uvd.select(times=np.unique(uvd.time_array)[1::2], inplace=False)

        # Create a new PSpecData object
        self.ds = pspecdata.PSpecData(dsets=[uvd1, uvd2], wgts=[None, None],
                                      beam=self.bm)
        self.ds.rephase_to_dset(0) # Phase to the zeroth dataset

        # Construct list of baseline pairs to calculate power spectra for
        bls = [(24,25), (37,38), (38,39),]
        self.bls1, self.bls2, blp = utils.construct_blpairs(
                        bls, exclude_permutations=False, exclude_auto_bls=True)

        # Calculate the power spectrum
        self.uvp = self.ds.pspec(self.bls1, self.bls2, (0, 1),
                                 ('xx','xx'), spw_ranges=[(300, 400), (600,721)],
                                 input_data_weight='identity', norm='I',
                                 taper='blackman-harris', verbose=False)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_plot_average(self):
        """
        Test that plotting routine can average over baselines and times.
        """
        # Unpack the list of baseline-pairs into a Python list
        blpairs = np.unique(self.uvp.blpair_array)
        blps = [blp for blp in blpairs]

        # Plot the spectra averaged over baseline-pairs and times
        f1 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                  average_blpairs=True, average_times=True)
        elements = [(matplotlib.lines.Line2D, 1),]
        assert axes_contains(f1.axes[0], elements)
        plt.close(f1)

        # Average over baseline-pairs but keep the time bins intact
        f2 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                  average_blpairs=True, average_times=False)
        elements = [(matplotlib.lines.Line2D, self.uvp.Ntimes),]
        assert axes_contains(f2.axes[0], elements)
        plt.close(f2)

        # Average over times, but keep the baseline-pairs separate
        f3 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                  average_blpairs=False, average_times=True)
        elements = [(matplotlib.lines.Line2D, self.uvp.Nblpairs),]
        assert axes_contains(f3.axes[0], elements)
        plt.close(f3)

        # Plot the spectra averaged over baseline-pairs and times, but also
        # fold the delay axis
        f4 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                  average_blpairs=True, average_times=True,
                                  fold=True)
        elements = [(matplotlib.lines.Line2D, 1),]
        assert axes_contains(f4.axes[0], elements)
        plt.close(f4)

        # Plot imaginary part
        f4 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                  average_blpairs=False, average_times=True,
                                  component='imag')
        elements = [(matplotlib.lines.Line2D, self.uvp.Nblpairs),]
        assert axes_contains(f4.axes[0], elements)
        plt.close(f4)

        # Plot abs
        f5 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                  average_blpairs=False, average_times=True,
                                  component='abs')
        elements = [(matplotlib.lines.Line2D, self.uvp.Nblpairs),]
        assert axes_contains(f4.axes[0], elements)
        plt.close(f5)

        # test errorbar plotting w/ markers

        # bootstrap resample
        (uvp_avg, _,
         _) = grouping.bootstrap_resampled_error(self.uvp, time_avg=True,
                                                 Nsamples=100, normal_std=True,
                                                 robust_std=False, verbose=False)

        f6 = plot.delay_spectrum(uvp_avg, uvp_avg.get_blpairs(), spw=0,
                                pol=('xx','xx'), average_blpairs=False,
                                average_times=False,
                                component='real', error='bs_std', lines=False,
                                markers=True)
        plt.close(f6)

        # plot errorbar instead of pspec
        f7 = plot.delay_spectrum(uvp_avg, uvp_avg.get_blpairs(), spw=0,
                                pol=('xx','xx'), average_blpairs=False,
                                average_times=False,
                                component='real', lines=False,
                                markers=True, plot_stats='bs_std')
        plt.close(f7)


    def test_plot_cosmo(self):
        """
        Test that cosmological units can be used on plots.
        """
        # Unpack the list of baseline-pairs into a Python list
        blpairs = np.unique(self.uvp.blpair_array)
        blps = [blp for blp in blpairs]

        # Set cosmology and plot in non-delay (i.e. cosmological) units
        self.uvp.set_cosmology(conversions.Cosmo_Conversions())
        f1 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                  average_blpairs=True, average_times=True,
                                  delay=False)
        elements = [(matplotlib.lines.Line2D, 1), (matplotlib.legend.Legend, 0)]
        self.assertTrue( axes_contains(f1.axes[0], elements) )
        plt.close(f1)

        # Plot in Delta^2 units
        f2 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                  average_blpairs=True, average_times=True,
                                  delay=False, deltasq=True, legend=True,
                                  label_type='blpair')
        # Should contain 1 line and 1 legend
        elements = [(matplotlib.lines.Line2D, 1), (matplotlib.legend.Legend, 1)]
        self.assertTrue( axes_contains(f2.axes[0], elements) )
        plt.close(f2)


    def test_delay_spectrum_misc(self):
        # various other tests for plot.delay_spectrum

        # Unpack the list of baseline-pairs into a Python list
        blpairs = np.unique(self.uvp.blpair_array)
        blps = [blp for blp in blpairs]

        # times selection, label_type
        f1 = plot.delay_spectrum(self.uvp, blpairs[:1], spw=0, pol=('xx','xx'),
                                 times=self.uvp.time_avg_array[:1], lines=False,
                                 markers=True, logscale=False, label_type='key',
                                 force_plot=False)
        plt.close(f1)

        # test force plot exception
        uvp = copy.deepcopy(self.uvp)
        for i in range(3):
            # build-up a large uvpspec object
            _uvp = copy.deepcopy(uvp)
            _uvp.time_avg_array += (i+1)**2
            uvp = uvp + _uvp
        pytest.raises(ValueError, plot.delay_spectrum, uvp, uvp.get_blpairs(), 0, 'xx')

        f2 = plot.delay_spectrum(uvp, uvp.get_blpairs(), 0, ('xx','xx'),
                                 force_plot=True, label_type='blpairt',
                                 logscale=False, lines=True, markers=True)
        plt.close(f2)

        # exceptions
        pytest.raises(ValueError, plot.delay_spectrum, uvp,
                         uvp.get_blpairs()[:3], 0, ('xx','xx'),
                         label_type='foo')


    def test_plot_waterfall(self):
        """
        Test that waterfall can be plotted.
        """
        # Unpack the list of baseline-pairs into a Python list
        blpairs = np.unique(self.uvp.blpair_array).tolist()
        blps = [self.uvp.blpair_to_antnums(blp) for blp in blpairs]

        # Set cosmology and plot in non-delay (i.e. cosmological) units
        self.uvp.set_cosmology(conversions.Cosmo_Conversions(), overwrite=True)
        f1 = plot.delay_waterfall(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                   average_blpairs=True, delay=False)
        plt.close(f1)

        # Plot in Delta^2 units
        f2 = plot.delay_waterfall(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                                   average_blpairs=True, delay=False,
                                   deltasq=True)
        plt.close(f2)

        # Try some other arguments
        f3 = plot.delay_waterfall(self.uvp, [blpairs,], spw=0, pol=('xx','xx'),
                                   average_blpairs=False, delay=True,
                                   log=False, vmin=-1., vmax=3.,
                                   cmap='RdBu', fold=True, component='abs')
        plt.close(f3)

        # Try with imaginary component
        f4 = plot.delay_waterfall(self.uvp, [blpairs,], spw=0, pol=('xx','xx'),
                                   average_blpairs=False, delay=True,
                                   log=False, vmin=-1., vmax=3.,
                                   cmap='RdBu', fold=True, component='imag')
        plt.close(f4)

        # Try some more arguments
        fig, axes = plt.subplots(1, len(blps))
        plot.delay_waterfall(self.uvp, [blps,], spw=0, pol=('xx','xx'),
                             lst_in_hrs=False,
                             times=np.unique(self.uvp.time_avg_array)[:10],
                             axes=axes, component='abs', title_type='blvec')
        plt.close()

        # exceptions
        uvp = copy.deepcopy(self.uvp)
        for i in range(1, 4):
            _uvp = copy.deepcopy(self.uvp)
            _uvp.blpair_array += i * 20
            uvp += _uvp
        pytest.raises(ValueError, plot.delay_waterfall, uvp,
                         uvp.get_blpairs(), 0, ('xx','xx'))
        fig = plot.delay_waterfall(uvp, uvp.get_blpairs(), 0, ('xx','xx'),
                                   force_plot=True)
        plt.close()

    def test_uvdata_waterfalls(self):
        """
        Test waterfall plotter
        """
        uvd = copy.deepcopy(self.uvd)

        basename = "test_waterfall_plots_3423523923_{bl}_{pol}"

        for d in ['data', 'flags', 'nsamples']:
            print("running on {}".format(d))
            plot.plot_uvdata_waterfalls(uvd, basename, vmin=0, vmax=100,
                                        data=d, plot_mode='real')

            figfiles = glob.glob("test_waterfall_plots_3423523923_*_*.png")
            assert len(figfiles) == 15
            for f in figfiles:
                os.remove(f)

    def test_delay_wedge(self):
        """ Tests for plot.delay_wedge """
        # construct new uvp
        reds, lens, angs = utils.get_reds(self.ds.dsets[0], pick_data_ants=True)
        bls1, bls2, blps, _, _ = utils.calc_blpair_reds(self.ds.dsets[0],
                                                        self.ds.dsets[1],
                                                        exclude_auto_bls=False,
                                                        exclude_permutations=True)
        uvp = self.ds.pspec(bls1, bls2, (0, 1), ('xx','xx'),
                            spw_ranges=[(300, 350)],
                            input_data_weight='identity', norm='I',
                            taper='blackman-harris', verbose=False)

        # test basic delay_wedge call
        f1 = plot.delay_wedge(uvp, 0, ('xx','xx'), blpairs=None, times=None,
                              fold=False, delay=True, rotate=False,
                              component='real', log10=False, loglog=False,
                              red_tol=1.0, center_line=False,
                              horizon_lines=False, title=None, ax=None,
                              cmap='viridis', figsize=(8, 6), deltasq=False,
                              colorbar=False, cbax=None, vmin=None, vmax=None,
                              edgecolor='none', flip_xax=False, flip_yax=False,
                              lw=2)
        plt.close()

        # specify blpairs and times
        f2 = plot.delay_wedge(uvp, 0, ('xx','xx'),
                              blpairs=uvp.get_blpairs()[-5:],
                              times=uvp.time_avg_array[:1],
                              fold=False, delay=True, component='imag',
                              rotate=False, log10=False, loglog=False, red_tol=1.0,
                              center_line=False, horizon_lines=False, title=None,
                              ax=None, cmap='viridis',
                              figsize=(8, 6), deltasq=False, colorbar=False,
                              cbax=None, vmin=None, vmax=None,
                              edgecolor='none', flip_xax=False, flip_yax=False,
                              lw=2)
        plt.close()

        # fold, deltasq, cosmo and log10, loglog
        f3 = plot.delay_wedge(uvp, 0, ('xx','xx'), blpairs=None, times=None,
                              fold=True, delay=False, component='abs',
                              rotate=False, log10=True, loglog=True, red_tol=1.0,
                              center_line=False, horizon_lines=False,
                              title='hello', ax=None, cmap='viridis',
                              figsize=(8, 6), deltasq=True, colorbar=False,
                              cbax=None, vmin=None, vmax=None,
                              edgecolor='none', flip_xax=False, flip_yax=False,
                              lw=2)
        plt.close()

        # colorbar, vranges, flip_axes, edgecolors, lines
        f4 = plot.delay_wedge(uvp, 0, ('xx','xx'), blpairs=None, times=None,
                              fold=False, delay=False, component='abs',
                              rotate=False, log10=True, loglog=False, red_tol=1.0,
                              center_line=True, horizon_lines=True, title='hello',
                              ax=None, cmap='viridis', figsize=(8, 6),
                              deltasq=True, colorbar=True, cbax=None, vmin=6,
                              vmax=15, edgecolor='grey', flip_xax=True,
                              flip_yax=True, lw=2, set_bl_tick_minor=True)
        plt.close()

        # feed axes, red_tol
        fig, ax = plt.subplots()
        cbax = fig.add_axes([0.85, 0.1, 0.05, 0.9])
        cbax.axis('off')
        plot.delay_wedge(uvp, 0, ('xx','xx'), blpairs=None, times=None,
                         fold=False, delay=True, component='abs',
                         rotate=True, log10=True, loglog=False, red_tol=10.0,
                         center_line=False, horizon_lines=False, ax=ax,
                         cmap='viridis', figsize=(8, 6), deltasq=False,
                         colorbar=True, cbax=cbax, vmin=None, vmax=None,
                         edgecolor='none', flip_xax=False, flip_yax=False,
                         lw=2, set_bl_tick_major=True)
        plt.close()

        # test exceptions
        pytest.raises(ValueError, plot.delay_wedge, uvp, 0, ('xx','xx'),
                         component='foo')
        plt.close()

if __name__ == "__main__":
    unittest.main()
