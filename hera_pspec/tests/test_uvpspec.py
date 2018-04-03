import unittest
import nose.tools as nt
import numpy as np
import os
import sys
from hera_pspec.data import DATA_PATH
from hera_pspec import UVPSpec
from hera_pspec import parameter

class Test_uvpspec(unittest.TestCase):

    def setUp(self):
        uvp = UVPSpec()

        Ntimes = 10
        Nfreqs = 50
        Ndlys = Nfreqs
        Nspws = 1
        Nspwdlys = Nspws * Nfreqs

        # [((1, 2), (1, 2)), ((2, 3), (2, 3)), ((1, 3), (1, 3))]
        blpairs = [1002001002, 2003002003, 1003001003]
        bls = [1002, 2003, 1003]
        Nbls = len(bls)
        Nblpairs = len(blpairs)
        Nblpairts = Nblpairs * Ntimes

        blpair_array = np.tile(blpairs, Ntimes)
        bl_array = np.array(bls)
        bl_vecs = np.array([[  2.96372145e+00,   3.10408473e+06,   8.34928223e+06],
                            [ -3.10411538e+06,  -6.33088145e+00,   5.24518137e+06],
                            [ -8.34927222e+06,  -5.24518578e+06,  -8.38190317e-09]])
        time_array = np.repeat(np.linspace(2458042.1, 2458042.2, Ntimes), Nblpairs)
        time_1_array = time_array
        time_2_array = time_array
        lst_array = np.repeat(np.ones(Ntimes, dtype=np.float), Nblpairs)
        lst_1_array = lst_array
        lst_2_array = lst_array
        spws = np.arange(Nspws)
        spw_array = np.tile(spws, Ndlys)
        freq_array = np.repeat(np.linspace(100e6, 105e6, Nfreqs, endpoint=False), Nspws)
        dly_array = np.repeat(np.fft.fftfreq(Nfreqs, np.median(np.diff(freq_array))), Nspws)
        pol_array = np.array([-5])
        Npols = len(pol_array)
        units = 'unknown'
        channel_width = np.median(np.diff(freq_array))

        telescope_location = np.array([5109325.85521063, 2005235.09142983, -3239928.42475397])

        data_array, flag_array, integration_array = {}, {}, {}
        for s in spws:
            data_array[s] = np.ones((Nblpairts, Ndlys, Npols), dtype=np.complex) * blpair_array[:, None, None] / 1e9
            flag_array[s] = np.ones((Nblpairts, Ndlys, Npols), dtype=np.bool)
            integration_array[s] = np.ones((Nblpairts, Npols), dtype=np.float)

        params = ['Ntimes', 'Nfreqs', 'Nspws', 'Nspwdlys', 'Nblpairs', 'Nblpairts', 'Npols', 'Ndlys',
                  'blpair_array', 'time_1_array', 'time_2_array', 'lst_1_array', 'lst_2_array',
                  'spw_array', 'dly_array', 'freq_array', 'pol_array', 'data_array', 'flag_array',
                  'integration_array', 'bl_array', 'bl_vecs', 'telescope_location', 'units',
                  'channel_width']

        for p in params:
            setattr(uvp, p, locals()[p])

        self.uvp = uvp


    def tearDown(self):
        pass

    def runTest(self):
        pass

    def test_param(self):
        a = parameter.PSpecParam("example", description="example", expected_form=int)





