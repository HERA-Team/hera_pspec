import unittest
import nose.tools as nt
import numpy as np
from hera_pspec import grouping

class Test_bootstrap(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def runTest(self):
        pass
    
    def test_group_baselines(self):
        """
        Test baseline grouping behavior.
        """
        # Generate example lists of baselines
        bls1 = [(0,i) for i in range(1)]
        bls2 = [(0,i) for i in range(2)]
        bls3 = [(0,i) for i in range(4)]
        bls4 = [(0,i) for i in range(5)]
        bls5 = [(0,i) for i in range(13)]
        bls6 = [(0,i) for i in range(521)]
        
        # Check that error is raised when more groups requested than baselines
        nt.assert_raises(ValueError, grouping.group_baselines, bls1, 2)
        nt.assert_raises(ValueError, grouping.group_baselines, bls2, 5)
        nt.assert_raises(ValueError, grouping.group_baselines, bls4, 6)
        
        # Check that keep_remainder=False results in equal-sized blocks
        g1a = grouping.group_baselines(bls4, 2, keep_remainder=False, randomize=False)
        g1b = grouping.group_baselines(bls5, 5, keep_remainder=False, randomize=False)
        g1c = grouping.group_baselines(bls6, 10, keep_remainder=False, randomize=False)
        g2a = grouping.group_baselines(bls4, 2, keep_remainder=False, randomize=True)
        g2b = grouping.group_baselines(bls5, 5, keep_remainder=False, randomize=True)
        g2c = grouping.group_baselines(bls6, 10, keep_remainder=False, randomize=True)
        
        # Loop over groups and check that blocks are equal in size
        gs = [g1a, g1b, g1c, g2a, g2b, g2c]
        for g in gs:
            self.assert_(np.unique([len(grp) for grp in g]).size == 1)
        
        # Check that total no. baselines is preserved with keep_remainder=False
        for bls in [bls1, bls2, bls3, bls4, bls5, bls6]:
            for ngrp in [1, 2, 5, 10, 45]:
                for rand in [True, False]:
                    try:
                        g = grouping.group_baselines(bls, ngrp, 
                                                 keep_remainder=True, 
                                                 randomize=rand)
                    except:
                        continue
                    count = np.sum([len(_g) for _g in g])
                    self.assertEqual(count, len(bls))
        
        # Check that random seed works
        g1 = grouping.group_baselines(bls5, 3, randomize=True, seed=10)
        g2 = grouping.group_baselines(bls5, 3, randomize=True, seed=11)
        g3 = grouping.group_baselines(bls5, 3, randomize=True, seed=10)
        for i in range(len(g1)):
            for j in range(len(g1[i])):
                self.assertEqual(g1[i][j], g3[i][j])
    
    def test_sample_baselines(self):
        """
        Test baseline sampling (with replacement) behavior.
        """
        # Generate example lists of baselines
        bls1 = [(0,i) for i in range(1)]
        bls2 = [(0,i) for i in range(2)]
        bls3 = [(0,i) for i in range(4)]
        bls4 = [(0,i) for i in range(5)]
        bls5 = [(0,i) for i in range(13)]
        bls6 = [(0,i) for i in range(521)]
        
        # Example grouped list
        g1 = grouping.group_baselines(bls5, 3, randomize=False)
        
        # Check that returned length is the same as input length
        for bls in [bls1, bls2, bls3, bls4, bls5, bls6]:
            samp = grouping.sample_baselines(bls)
            self.assertEqual(len(bls), len(samp))
        
        # Check that returned length is the same for groups too
        samp = grouping.sample_baselines(g1)
        self.assertEqual(len(g1), len(samp))
        
    
if __name__ == "__main__":
    unittest.main()
