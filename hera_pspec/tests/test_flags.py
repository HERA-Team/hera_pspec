from __future__ import print_function, division
import unittest
import nose.tools as nt
import numpy as np
from pyuvdata import UVData
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec.flags import uvd_to_array, stacked_array, construct_factorizable_mask, long_waterfall, flag_channels

dfiles = ['zen.even.xx.LST.1.28828.uvOCRSA', 'zen.odd.xx.LST.1.28828.uvOCRSA']
baseline = (38, 68, 'xx')

class Test_Flags(unittest.TestCase):

    def setUp(self):
        
        # Load datafiles into UVData objects
        self.d = []
        for dfile in dfiles:
            _d = UVData()
            _d.read_miriad(os.path.join(DATA_PATH, dfile))
            self.d.append(_d)
        # data to use when testing the plotting function
        self.data_list = [self.d[0].get_flags(38, 68, 'xx'), self.d[1].get_flags(38, 68, 'xx')]
        
    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_uvd_to_array(self):
        """
        testing the uvd to array function
        """
        nsamples, flags = uvd_to_array(self.d, baseline)
        # making sure that length of lists is always equal
        nt.assert_equal(len(nsamples), len(flags))
        # making sure an error comes up if len(uvdlist) = 0
        nt.assert_raises(ValueError, uvd_to_array, [], baseline)
        # error if a UVData object is input instead of a list
        nt.assert_raises(TypeError, uvd_to_array, self.d[0], baseline)

    def test_stacked_array(self):
        """
        testing the array stacking function
        """           
        flags_list = uvd_to_array(self.d, baseline)[1]
        long_array_flags = stacked_array(flags_list)

        # make sure # rows in output = sum of # rows in each input array
        nt.assert_equal(long_array_flags.shape[0], sum([flag_array.shape[0] \
                                       for flag_array in flags_list]))
        for flag_array in flags_list:
            # ensuring that the number of columns is unchanged
            nt.assert_equal(long_array_flags.shape[1], flag_array.shape[1])
        # ensuring that arrays are stacked in order as expected
        nt.assert_true(np.array_equal( \
            long_array_flags[0 : flags_list[0].shape[0], :], flags_list[0]))
        nt.assert_true(np.array_equal( \
            long_array_flags[ flags_list[0].shape[0] : flags_list[0].shape[0] + \
                             flags_list[1].shape[0], :], flags_list[1]))

    def test_construct_factorizable_mask(self):
        """
        testing mask generator function
        """
        # testing unflagging
        unflagged_uvdlist = construct_factorizable_mask(self.d, unflag=True, \
                                                        inplace=False)
        for uvd in unflagged_uvdlist:
            unflagged_mask = uvd.get_flags((38, 68, 'xx'))
            nt.assert_equal(np.sum(unflagged_mask), 0)
        # ensuring that greedy flagging works as expected in extreme cases
        allflagged_uvdlist = construct_factorizable_mask( \
            self.d, greedy_threshold=0.0001, first='row', inplace=False)
        for uvd in allflagged_uvdlist:
            flagged_mask = uvd.get_flags((38, 68, 'xx'))
        # everything flagged since the greedy threshold is extremely low
            nt.assert_equal(np.sum(flagged_mask), \
                            np.sum(np.ones(flagged_mask.shape)))
        # ensuring that n_threshold parameter works as expected in extreme cases
        allflagged_uvdlist2 = construct_factorizable_mask( \
            self.d, n_threshold=35, first='row', inplace=False)
        for uvd in allflagged_uvdlist2:
            flagged_mask = uvd.get_flags((38, 68, 'xx'))
            nt.assert_equal(np.sum(flagged_mask), \
                            np.sum(np.ones(flagged_mask.shape)))
        # ensuring that greedy flagging is occurring within the intended spw: 
        greedily_flagged_uvdlist = construct_factorizable_mask( \
            self.d, n_threshold = 6, greedy_threshold = 0.35, first='col', \
            spw_ranges=[(0, 300), (500, 700)], inplace=False)
        for i in range(len(self.d)):
            # checking that outside the spw range, flags are all equal
            nt.assert_true(np.array_equal( \
                greedily_flagged_uvdlist[i].get_flags((38, 68, 'xx'))[:, 300:500], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 300:500]))
            nt.assert_true(np.array_equal( \
                greedily_flagged_uvdlist[i].get_flags((38, 68, 'xx'))[:, 700:], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 700:]))
            # flags are actually retained
            original_flags_ind = np.where(self.d[i].get_flags((38, 68, 'xx')) == True)
            new_flags = greedily_flagged_uvdlist[i].get_flags((38, 68, 'xx'))
            old_flags = self.d[i].get_flags((38, 68, 'xx'))
            nt.assert_true(np.array_equal( \
                new_flags[original_flags_ind], old_flags[original_flags_ind]))
            # checking that inplace objects match in important areas
            nt.assert_true(np.array_equal( \
                greedily_flagged_uvdlist[i].get_data((38, 68, 'xx')), \
                                          self.d[i].get_data((38, 68, 'xx'))))
            nt.assert_true(np.array_equal( \
                greedily_flagged_uvdlist[i].get_nsamples((38, 68, 'xx')), \
                                          self.d[i].get_nsamples((38, 68, 'xx'))))
            # making sure flags are actually independent in each spw
            masks = [new_flags[:, 0:300], new_flags[:, 500:700]]
            for mask in masks:
                Nfreqs = mask.shape[1]
                Ntimes = mask.shape[0]
                N_flagged_rows = np.sum( \
                    1*(np.sum(mask, axis=1)/Nfreqs > 0.999999999))
                N_flagged_cols = np.sum( \
                    1*(np.sum(mask, axis=0)/Ntimes > 0.999999999))
                nt.assert_true(int(np.sum( \
                    mask[np.where(np.sum(mask, axis=1)/Nfreqs < 0.99999999)]) \
                                   /(Ntimes-N_flagged_rows)) == N_flagged_cols)

    # copied from test_plot.py for testing the long_waterfall plotting function
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

    def test_long_waterfall(self):
        """
        testing the long waterfall plotting function
        """
        main_waterfall, freq_histogram, time_histogram, data = long_waterfall( \
            self.data_list, title='Flags Waterfall')
        # making sure the main waterfall has the right number of dividing lines
        main_waterfall_elems = [(matplotlib.lines.Line2D, \
                                 round(data.shape[0]/60, 0))]
        nt.assert_true(self.axes_contains(main_waterfall, main_waterfall_elems))
        # making sure the time graph has the appropriate line element
        time_elems = [(matplotlib.lines.Line2D, 1)]
        nt.assert_true(self.axes_contains(time_histogram, time_elems))
        # making sure the freq graph has the appropriate line element
        freq_elems = [(matplotlib.lines.Line2D, 1)]
        nt.assert_true(self.axes_contains(freq_histogram, freq_elems))
                       
    def test_flag_channels(self):
        """
        testing the channel-flagging function
        """
        # ensuring that flagging is occurring: 
        column_flagged_uvdlist = flag_channels( \
            self.d, [(200, 451), (680, 881)], inplace=False)
        for i in range(len(self.d)):
            # checking that outside the spw ranges, flags are all equal
            nt.assert_true(np.array_equal( \
                column_flagged_uvdlist[i].get_flags((38, 68, 'xx'))[:, :200], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, :200]))
            nt.assert_true(np.array_equal( \
                column_flagged_uvdlist[i].get_flags((38, 68, 'xx'))[:, 451:680], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 451:680]))
            nt.assert_true(np.array_equal( \
                column_flagged_uvdlist[i].get_flags((38, 68, 'xx'))[:, 881:], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 881:]))
            # checking that inside the ranges, everything is flagged
            nt.assert_true(np.array_equal( \
                column_flagged_uvdlist[i].get_flags((38, 68, 'xx'))[:, 200:451], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 200:451]))
            nt.assert_true(np.array_equal( \
                column_flagged_uvdlist[i].get_flags((38, 68, 'xx'))[:, 680:881], \
                                  self.d[i].get_flags((38, 68, 'xx'))[:, 680:881]))
            # checking that inplace objects match in important areas
            nt.assert_true(np.array_equal( \
                column_flagged_uvdlist[i].get_data((38, 68, 'xx')), \
                                          self.d[i].get_data((38, 68, 'xx'))))
            nt.assert_true(np.array_equal( \
                column_flagged_uvdlist[i].get_nsamples((38, 68, 'xx')), \
                                          self.d[i].get_nsamples((38, 68, 'xx'))))