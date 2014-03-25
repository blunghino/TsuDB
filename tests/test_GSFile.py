# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:01:17 2014

@author: blunghino
"""

import unittest

import numpy as np
from numpy import asarray as ary
from numpy.testing import assert_allclose, assert_array_equal, assert_array_almost_equal

import TsuDB as tdb
from TsuDB import TsuDBGSFile

class TestGSFileClass(unittest.TestCase):
    
    # test data a
    a_csv_file_name = 'GS_PapuaNewGuinea_Waipo_157m.csv'
    a_dist_means = ary([1.63])
    a_dist_stds = ary([.73])
    a_bins_phi_1 = -0.75
    a_bins_phi_mid_3 = -0.375
    a_typegs = 'STLD'
    b_csv_file_name = 'GS_Sumatra_Lhokkruet2_T1.csv'
    b_dist_means = ary([1.275675, 1.80715, 1.666925])
    b_dist_stds = ary([0.887167727, 0.883166074, 1.012755422])
    b_dist_moments_2 = ary([-0.591407243, -0.92246897, -1.372314436])
    b_bin_units = 'phi mid'
    c_csv_file_name = 'GS_Japan_Sendai_T3-10.csv'
    c_dist_moments_3 = ary([22.694552, 51.320702, 32.55427, 53.349303, 
                                63.232381, 68.214429, 50.28316, 2.352845])
    c_bulk_mean =  1.657406209748447
    c_bulk_mean_sand =  1.544082097212654
    c_bulk_dist = ary([ 
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                        0.5494,
                        2.6345,
                        7.0543,
                       13.5909,
                       20.4264,
                       22.6927,
                       18.1712,
                        8.7561,
                        2.5106,
                        1.3141,
                        0.1261,
                        0.0054,
                        0.0088,
                        0.1019,
                        0.0369,
                        0.0496,
                        0.0574,
                        0.0743,
                        0.0906,
                        0.0918,
                        0.0873,
                        0.0887,
                        0.0919,
                        0.0874,
                        0.0854,
                        0.0868,
                        0.0926,
                        0.0972,
                        0.0978,
                        0.0969,
                        0.0897,
                        0.0825,
                        0.0740,
                        0.0666,
                        0.0618,
                        0.0582,
                        0.0558,
                        0.0535,
                        0.0513,
                        0.0454,
                        0.0384,
                        0.0318,
                        0.0218,
                        0.0116,
                        0.0029,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0
                    ])
    c_bins_phi_mid = ary([
                       -1.8750,
                       -1.6250,
                       -1.3750,
                       -1.1250,
                       -0.8750,
                       -0.6250,
                       -0.3750,
                       -0.1250,
                        0.1250,
                        0.3750,
                        0.6250,
                        0.8750,
                        1.1250,
                        1.3750,
                        1.6250,
                        1.8750,
                        2.1250,
                        2.3750,
                        2.6250,
                        2.8750,
                        3.1250,
                        3.3750,
                        3.6250,
                        3.8750,
                        4.1250,
                        4.3750,
                        4.6250,
                        4.8750,
                        5.1250,
                        5.3750,
                        5.6250,
                        5.8750,
                        6.1250,
                        6.3750,
                        6.6250,
                        6.8750,
                        7.1250,
                        7.3750,
                        7.6250,
                        7.8750,
                        8.1250,
                        8.3750,
                        8.6250,
                        8.8750,
                        9.1250,
                        9.3750,
                        9.6250,
                        9.8750,
                       10.1250,
                       10.3750,
                       10.6250,
                       10.8749,
                       11.1250,
                       11.3749,
                       11.6250,
                       11.8752,
                       12.1251,
                       12.3751,
                       12.6250,
                       12.8747,
                       13.1252,
                       13.3755,
                       13.6250,
                ])
    c_id = 'Sendai, T3-10'
    c_get_depth_bin_edges = ary([0, 1, 2, 3, 4, 5, 6, 7.25, 8.5])
    d_csv_file_name = 'GS_Japan_Sendai_T3-77.csv'
    d_mid_depth = ary([np.nan, 5.5])
    d_min_depth = ary([np.nan, 5])
    
    a = TsuDBGSFile(a_csv_file_name)
    b = TsuDBGSFile(b_csv_file_name)
    c = TsuDBGSFile(c_csv_file_name)
    d = TsuDBGSFile(d_csv_file_name)
        
    def test_gsfile_init(self):
        """
        test that init reads various csv field correctly
        """
        self.assertEqual(self.a.typegs, self.a_typegs)
        self.assertEqual(self.b.bin_units, self.b_bin_units)
        self.assertEqual(self.a.bins_phi[1], self.a_bins_phi_1)
        self.assertEqual(self.a.bins_phi_mid[3], self.a_bins_phi_mid_3)
        self.assertIsInstance(self.a.trench_name, list)
        self.assertEqual(self.c.id, self.c_id)
        self.assertIsInstance(self.a.mid_depth, np.ndarray)
        
    def test_gsfile_dist_means(self):
        assert_allclose(self.b.dist_means(), self.b_dist_means)
        assert_allclose(self.a.dist_means(), self.a_dist_means, rtol=.01)
    
    def test_gsfile_dist_stds(self):
        assert_allclose(self.a.dist_stds(), self.a_dist_stds, rtol=.01)
        assert_allclose(self.b.dist_stds(), self.b_dist_stds)
        
    def test_gsfile_dist_moments(self):
        assert_allclose(self.b.dist_moments()[2], self.b_dist_moments_2)
        assert_allclose(self.c.dist_moments()[3], self.c_dist_moments_3, rtol=1e-05)
        
    def test_gsfile_get_depth_bin_edges(self):
        """
        check that _get_depth_bin_edges method returns the correct bin edges
        """
        assert_array_equal(self.c._get_depth_bin_edges(), self.c_get_depth_bin_edges)
        
    def test_gsfile_with_some_depths_empty(self):
        """
        check that empty depth fields are handled ok
        """
        assert_array_equal(self.d.mid_depth, self.d_mid_depth)
        assert_array_equal(self.d.min_depth, self.d_min_depth)
        
    def test_gsfile_convert_bins_to_phi_mid(self):
        """
        check that conversion from phi to phi mid produces the correct results
        """
        assert_array_almost_equal(self.c.bins_phi_mid, self.c_bins_phi_mid, decimal=3)
        
    def test_gsfile_bulk_dist(self):
        """
        check that bulk dist is calculated correctly
        """
        assert_array_almost_equal(self.c.bulk_dist(), self.c_bulk_dist, decimal=3)
        
    def test_gsfile_bulk_mean(self):
        """
        check that bulk mean is calculated correctly
        """
        self.assertAlmostEqual(self.c.bulk_mean(), self.c_bulk_mean, places=5)
    
    def test_gsfile_bulk_mean_sand_only(self):
        """
        check that bulk mean works for specific gs fraction
        """
        self.assertAlmostEqual(self.c.bulk_mean(gs_min_max=(4, -1)), self.c_bulk_mean_sand, places=3)
    
        
if __name__ == '__main__':
    unittest.main()

