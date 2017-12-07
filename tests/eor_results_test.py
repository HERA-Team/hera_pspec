import unittest
from capo.eor_results import (
        get_pk_from_npz as load_pk, get_k3pk_from_npz as load_k3pk,
        read_bootstraps, random_avg_bootstraps
        )
from capo.cosmo_units import f212z, c
from capo import pspec
import numpy as np
import os
import glob

test_data_dir='test_data/'
test_data_file= test_data_dir+'test_eor_results.npz'

class Test_eor_loader(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_z_from_pk(self):
        ref_z=f212z(119.7e6)
        out_z,_,_,_ = load_pk(test_data_file,verbose=False)
        self.assertEqual(ref_z, out_z,
                msg='Expected z {0} but ' \
                        'retruned z {1}'.format(ref_z,out_z)
                        )

    def test_k_from_pk(self):
        ref_k=.2
        _,out_k,_,_ = load_pk(test_data_file,verbose=False)
        self.assertEqual(ref_k, out_k,
                msg='Expected k_par {0} but ' \
                        'retruned k_par {1}'.format(ref_k,out_k)
                        )

    def test_pk_from_pk(self):
        ref_pk = 36752
        ref_pk_err = 13987
        _,_,out_pk,out_pk_err = load_pk(test_data_file,verbose=False)
        self.assertTrue(
                np.allclose([ref_pk,ref_pk_err], [out_pk,out_pk_err],),
                msg='Expected pk, pk_err {0} +/- {1} but ' \
                        'retruned pk,pk_err {2} +/- {3}'\
                        ''.format(ref_pk,ref_pk_err,out_pk,out_pk_err)
                        )

    def test_z_from_k3pk(self):
        ref_z=f212z(119.7e6)
        out_z,_,_,_ = load_k3pk(test_data_file,verbose=False)
        self.assertEqual(ref_z, out_z,
                msg='Expected z {0} but ' \
                        'retruned z {1}'.format(ref_z,out_z)
                        )

    def test_k_from_k3pk(self):
        ref_k_par=.2
        ref_z=f212z(119.7e6)
        ref_umag = 30/(3e8/(119.7*1e6))
        ref_k_perp = ref_umag*pspec.dk_du(ref_z)
        ref_k_mag = np.sqrt( ref_k_par**2 + ref_k_perp**2)
        _,out_k_mag,_,_ = load_k3pk(test_data_file,verbose=False)
        self.assertEqual(ref_k_mag, out_k_mag,
                msg='Expected k_mag {0} but ' \
                        'retruned k_mag {1}'.format(ref_k_mag,out_k_mag)
                        )

    def test_k3pk_from_k3pk(self):
        ref_k3pk = 992
        ref_k3pk_err = 1003
        _,_,out_k3pk,out_k3pk_err = load_k3pk(test_data_file,verbose=False)
        self.assertTrue(
                np.allclose([ref_k3pk,ref_k3pk_err], [out_k3pk,out_k3pk_err],),
                msg='Expected pk, pk_err {0} +/- {1} but ' \
                        'retruned pk,pk_err {2} +/- {3}'\
                        ''.format(ref_k3pk,ref_k3pk_err,out_k3pk,out_k3pk_err)
                        )
class Test_bootstrap_functions(unittest.TestCase):

    def setUp(self):
        self.path = os.path.dirname(os.path.realpath(__file__))



    def tearDown(self):
        self.path =''

    def test_none_input(self):
        with self.assertRaises(TypeError):
            random_avg_bootstraps()

    def test_blank_list(self):
        with self.assertRaises(TypeError):
            random_avg_bootstraps([])

    def test_empty_string_input(self):
        with self.assertRaises(TypeError):
            random_avg_bootstraps('')

    def test_no_boot_axis_input(self):
        with self.assertRaises(TypeError):
            random_avg_bootstraps({'test':[]})

    def test_no_time_axis_input(self):
        with self.assertRaises(TypeError):
            random_avg_bootstraps({'test':[]},boot_axis=1)

    def test_boot_axis_type(self):
        with self.assertRaises(TypeError):
            random_avg_bootstraps({'test':[]},boot_axis=1.2,time_axis=1)
            random_avg_bootstraps({'test':[]},boot_axis=1+1j,time_axis=1)

    def test_time_axis_type(self):
        with self.assertRaises(TypeError):
            random_avg_bootstraps({'test':[]},boot_axis=1,time_axis=1.2)
            random_avg_bootstraps({'test':[]},boot_axis=1,time_axis=1+1j)

    def test_num_boots(self):
        test_files= glob.glob(os.path.join(self.path,
                'test_data/inject_test1/pspec*.npz'))
        # /test_boot{0:02d}'.format(n)) for n in range(5)]
        ref_boot = 5
        out_dict = read_bootstraps( test_files)
        out_boot = np.shape(out_dict['nocov_vs_t'])[0]
        self.assertEqual(ref_boot, out_boot)

    def test_load_final_boot(self):
        ref_freq=0.11970443349753694
        test_files= glob.glob(os.path.join(self.path,
                'test_data/inject_test1/pspec*.npz'))
        out_dict = read_bootstraps( test_files)
        out_freq = out_dict['freq'][0]
        self.assertEqual(out_freq,ref_freq)

    def test_num_ks(self):
        test_files= glob.glob(os.path.join(self.path,
                     'test_data/inject_test1/pspec*.npz'))
        ref_ks = 21
        out_dict = read_bootstraps( test_files)
        out_ks = np.shape(out_dict['pk_vs_t'])[1]
        self.assertEqual(out_ks, ref_ks)

    def test_num_boots(self):
        test_files= glob.glob(os.path.join(self.path,
                     'test_data/inject_test1/pspec*.npz'))
        ref_dict = read_bootstraps(test_files)
        in_dict = {'pIs':ref_dict['nocov_vs_t']}
        ref_boots=15
        out_dict = random_avg_bootstraps( in_dict,
                boot_axis=0,time_axis=-1,nboot=ref_boots)
        out_boots = np.shape(out_dict['pIs'])[0]
        self.assertEqual(ref_boots, out_boots)

    def test_num_ks_avged(self):
        test_files= glob.glob(os.path.join(self.path,
                     'test_data/inject_test1/pspec*.npz'))
        ref_dict = read_bootstraps(test_files)
        in_dict = {'pIs':ref_dict['nocov_vs_t']}
        ref_boot=15
        ref_ks=21
        out_dict = random_avg_bootstraps( in_dict,
                boot_axis=0,time_axis=-1, nboot=ref_boot)
        out_ks = np.shape(out_dict['pIs'])[1]
        self.assertEqual(out_ks, ref_ks)

if __name__== '__main__':
    unittest.main()
