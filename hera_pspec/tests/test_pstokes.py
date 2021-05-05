import unittest
import pytest
import os, sys
from hera_pspec.data import DATA_PATH
from .. import pstokes
import pyuvdata
import pyuvdata.utils as uvutils
import copy
import numpy as np

dset1 = os.path.join(DATA_PATH, 'zen.all.xx.LST.1.06964.uvA')
dset2 = os.path.join(DATA_PATH, 'zen.all.yy.LST.1.06964.uvA')
multipol_dset = os.path.join(DATA_PATH, 'zen.2458116.31193.HH.uvh5')
multipol_dset_cal = os.path.join(DATA_PATH, 'zen.2458116.31193.HH.flagged_abs.calfits')


class Test_pstokes(unittest.TestCase):

    def setUp(self):
        # Loading pyuvdata objects
        self.uvd1 = pyuvdata.UVData()
        self.uvd1.read_miriad(dset1)
        self.uvd2 = pyuvdata.UVData()
        self.uvd2.read_miriad(dset2)

    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_combine_pol(self):
        uvd1 = self.uvd1
        uvd2 = self.uvd2

        # basic execution on pol strings
        out1 = pstokes._combine_pol(uvd1, uvd2, 'XX', 'YY')
        # again w/ pol ints
        out2 = pstokes._combine_pol(uvd1, uvd2, -5, -6)
        # assert equivalence
        assert out1 == out2

        # check exceptions
        pytest.raises(AssertionError, pstokes._combine_pol, dset1, dset2, 'XX', 'YY' )
        pytest.raises(AssertionError, pstokes._combine_pol, uvd1, uvd2, 'XX', 1)

    def test_construct_pstokes(self):
        uvd1 = self.uvd1
        uvd2 = self.uvd2

        # test to form I and Q from single polarized UVData objects
        uvdI = pstokes.construct_pstokes(dset1=uvd1, dset2=uvd2, pstokes='pI')
        uvdQ = pstokes.construct_pstokes(dset1=uvd1, dset2=uvd2, pstokes='pQ')

        # check exceptions
        pytest.raises(AssertionError, pstokes.construct_pstokes, uvd1, 1)

        # check baselines
        uvd3 = uvd2.select(ant_str='auto', inplace=False)
        pytest.raises(ValueError, pstokes.construct_pstokes, dset1=uvd1, dset2=uvd3 )

        # check frequencies
        uvd3 = uvd2.select(frequencies=np.unique(uvd2.freq_array)[:10], inplace=False)
        pytest.raises(ValueError, pstokes.construct_pstokes, dset1=uvd1, dset2=uvd3)

        uvd3 = uvd1.select(frequencies=np.unique(uvd1.freq_array)[:10], inplace=False)
        uvd4 = uvd2.select(frequencies=np.unique(uvd2.freq_array)[10:20], inplace=False)
        pytest.raises(ValueError, pstokes.construct_pstokes, dset1=uvd3, dset2=uvd4)

        # check times
        uvd3 = uvd2.select(times=np.unique(uvd2.time_array)[0:3], inplace=False)
        pytest.raises(ValueError, pstokes.construct_pstokes, dset1=uvd1, dset2=uvd3)

        uvd3 = uvd1.select(times=np.unique(uvd1.time_array)[0:3], inplace=False)
        uvd4 = uvd2.select(times=np.unique(uvd2.time_array)[1:4], inplace=False)
        pytest.raises(ValueError, pstokes.construct_pstokes, dset1=uvd3, dset2=uvd4)

        # combining two polarizations (dset1 and dset2) together
        uvd3 = uvd1 + uvd2

        # test to form I and Q from dual polarized UVData objects
        uvdI = pstokes.construct_pstokes(dset1=uvd3, dset2=uvd3, pstokes='pI')

        # check except for same polarizations
        pytest.raises(AssertionError, pstokes.construct_pstokes, dset1=uvd1, dset2=uvd1, pstokes='pI')

    def test_construct_pstokes_multipol(self):
        """test construct_pstokes on multi-polarization files"""
        uvd = pyuvdata.UVData()
        uvd.read(multipol_dset)
        uvc = pyuvdata.UVCal()
        uvc.read_calfits(multipol_dset_cal)
        uvutils.uvcalibrate(uvd, uvc)
        wgts = [(0.5, 0.5), (0.5, -0.5)]

        for i, ps in enumerate(['pI', 'pQ']):
            uvp = pstokes.construct_pstokes(dset1=uvd, dset2=uvd, pstokes=ps)
            # assert polarization array is correct
            assert uvp.polarization_array == np.array([i + 1])
            # assert data are properly summmed
            pstokes_vis = uvd.get_data(23, 24, 'xx') * wgts[i][0] + uvd.get_data(23, 24, 'yy') * wgts[i][1]
            assert np.isclose(pstokes_vis, uvp.get_data(23, 24, ps)).all()

    def test_filter_dset_on_stokes_pol(self):
        dsets = [self.uvd1, self.uvd2]
        out = pstokes.filter_dset_on_stokes_pol(dsets, 'pI')
        assert out[0].polarization_array[0] == -5
        assert out[1].polarization_array[0] == -6
        pytest.raises(AssertionError, pstokes.filter_dset_on_stokes_pol, dsets, 'pV')
        dsets = [self.uvd2, self.uvd1]
        out2 = pstokes.filter_dset_on_stokes_pol(dsets, 'pI')
        assert out == out2

    def test_generate_pstokes_argparser(self):
        # test argparser for noise error bars.
        ap = pstokes.generate_pstokes_argparser()
        args=ap.parse_args(["input.uvh5", "--pstokes", "pI", "pQ", "--clobber"])
        assert args.inputdata == "input.uvh5"
        assert args.outputdata is None
        assert args.clobber


if __name__ == "__main__":
    unittest.main()
