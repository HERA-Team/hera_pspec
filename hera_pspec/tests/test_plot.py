import unittest
import nose.tools as nt
import matplotlib
import numpy as np
import os, copy, sys
from hera_pspec import pspecdata, pspecbeam, conversions, plot
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData

# Data files to use in tests
dfiles = [
    'zen.all.xx.LST.1.06964.uvA',
]

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
        
        # Load beam file
        beamfile = os.path.join(DATA_PATH, 'NF_HERA_Beams.beamfits')
        self.bm = pspecbeam.PSpecBeamUV(beamfile)
        self.bm.filename = 'NF_HERA_Beams.beamfits'
        
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
        self.bls1, self.bls2, blp = pspecdata.construct_blpairs(
                        bls, exclude_permutations=False, exclude_auto_bls=True)
        
        # Calculate the power spectrum
        self.uvp = self.ds.pspec(self.bls1, self.bls2, (0, 1), 
                                 spw_ranges=[(300, 400), (600,721)], 
                                 input_data_weight='identity', norm='I', 
                                 taper='blackman-harris', verbose=False)
        
    def tearDown(self):
        pass

    def runTest(self):
        pass
    
    def axes_contains(self, ax, obj_list):
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
            
    def test_plot_average(self):
        """
        Test that plotting routine can average over baselines and times.
        """
        # Unpack the list of baseline-pairs into a Python list
        blpairs = np.unique(self.uvp.blpair_array)        
        blps = [blp for blp in blpairs]
        
        # Plot the spectra averaged over baseline-pairs and times
        ax1 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol='xx', 
                                  average_blpairs=True, average_times=True)
        elements = [(matplotlib.lines.Line2D, 1),]
        self.assertTrue( self.axes_contains(ax1, elements) )
        
        # Average over baseline-pairs but keep the time bins intact
        ax2 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol='xx', 
                                  average_blpairs=True, average_times=False)
        elements = [(matplotlib.lines.Line2D, self.uvp.Ntimes),]
        self.assertTrue( self.axes_contains(ax2, elements) )
        
        # Average over times, but keep the baseline-pairs separate
        ax3 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol='xx', 
                                  average_blpairs=False, average_times=True)
        elements = [(matplotlib.lines.Line2D, self.uvp.Nblpairs),]
        self.assertTrue( self.axes_contains(ax3, elements) )
        
        # Plot the spectra averaged over baseline-pairs and times, but also 
        # fold the delay axis
        ax4 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol='xx', 
                                  average_blpairs=True, average_times=True,
                                  fold=True)
        elements = [(matplotlib.lines.Line2D, 1),]
        self.assertTrue( self.axes_contains(ax4, elements) )
        
    def test_plot_cosmo(self):
        """
        Test that cosmological units can be used on plots.
        """
        # Unpack the list of baseline-pairs into a Python list
        blpairs = np.unique(self.uvp.blpair_array)        
        blps = [blp for blp in blpairs]
        
        # Set cosmology and plot in non-delay (i.e. cosmological) units
        self.uvp.set_cosmology(conversions.Cosmo_Conversions())
        ax1 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol='xx', 
                                  average_blpairs=True, average_times=True, 
                                  delay=False)
        elements = [(matplotlib.lines.Line2D, 1), (matplotlib.legend.Legend, 0)]
        self.assertTrue( self.axes_contains(ax1, elements) )
        
        # Plot in Delta^2 units
        ax2 = plot.delay_spectrum(self.uvp, [blps,], spw=0, pol='xx', 
                                  average_blpairs=True, average_times=True, 
                                  delay=False, deltasq=True, legend=True)
        
        # Should contain 1 line and 1 legend
        elements = [(matplotlib.lines.Line2D, 1), (matplotlib.legend.Legend, 1)]
        self.assertTrue( self.axes_contains(ax2, elements) )
        
if __name__ == "__main__":
    unittest.main()
