# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:57:07 2013

@author: Brent Lunghino

use unittest to run tests on TsuDB.
run from command prompt with: python test_Transect.py
or to run all tests in directory "tests": python -m unittest discover -v tests
"""
import unittest

import numpy as np
from numpy import asarray as ary
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestTransectClass(unittest.TestCase):
    
    ## dataset 1
    slc1 = ary([1, 1, 1, 1, 1], dtype=np.float64)
    tsc1 = ary([1, 1, 1, 2, 2], dtype=np.float64)
    dts1 = ary([0, 25, 50, 100, 0], dtype=np.float64)
    x1 = np.arange(5)
    correct_Tsort_ind1 = ary([0, 1, 2, 4, 3])
    correct_Tsort_dds1 = ary([np.nan, 25., 25., np.nan, 100.])
    correct_Tsort_dx1 = ary([np.nan, 1, 1, np.nan, -1])
    correct_Tsort_return_flow_dx1 = ary([-1, -1, np.nan, 1, np.nan])
    correct_Tsort_middx1 = ary([np.nan, .5, 1.5, np.nan, 3.5])
    correct_Tsort_midds1 = ary([np.nan, 12.5, 37.5, np.nan, 50])
    correct_Tsort_return_flow_middx1 = ary([.5, 1.5, np.nan, 3.5, 
                                                 np.nan])
    correct_Tsort_return_flow_midds1 = ary([12.5, 37.5, np.nan, 50, 
                                                 np.nan])
    ## dataset 2
    slc2 = ary([1, 1, 1, 2, 1, 1, 2, 2, 2, 2])
    tsc2 = ary([1, 1, 1, 1, 2, 1, 2, 1, 2, 1])
    dts2 = ary([-10, 20, 10, 3, 10, 25, 0, 3, 20, 2])
    x2 = np.arange(10)
    correct_Tsort_ind2 = ary([0, 2, 1, 5, 4, 9, 3, 7, 6, 8])
    correct_mxw2 = ary([5, 5, 5, 9, 5, 5, 9, 9, 9, 9])
    correct_Tsort_sds2 = ary([-10, 10, 20, 25, 10, 2, 3, 3, 0, 20])
    correct_Tsort_tnum2 = ary([1., 1, 1, 1, 2, 3, 3, 3, 4, 4])
    correct_Tsort_smxt2 = ary([np.nan, np.nan, np.nan, 5, 4, 9, 
                                    np.nan, np.nan, np.nan, 8])
    correct_Tsort_smx2 = ary([5, 5, 5, 5, 4, 9, 9, 9, 8, 8])
        
    def test_Tsort_with_dataset_1(self):
        """
        make sure that _Tsort gives the expected results from dataset 1
        """
        t1 = tdb.Transect(self.x1, self.slc1, self.tsc1, self.dts1)
        assert_array_equal(t1.sx, self.correct_Tsort_ind1)
        assert_array_equal(t1.dds, self.correct_Tsort_dds1)
        assert_array_equal(t1.dx, self.correct_Tsort_dx1)
        assert_array_equal(t1.middx, self.correct_Tsort_middx1)
        assert_array_equal(t1.midds, self.correct_Tsort_midds1)
        
    def test_Tsort_return_flow_with_dataset_1(self):
        """
        check that _Tsort_return_flow gives the expected results from dataset 1
        """
        t1 = tdb.Transect(self.x1, self.slc1, self.tsc1, self.dts1, 
                          return_flow=True)
        assert_array_equal(t1.sx, self.correct_Tsort_ind1)
        assert_array_equal(t1.dx, self.correct_Tsort_return_flow_dx1)
        assert_array_equal(t1.middx, self.correct_Tsort_return_flow_middx1)
        assert_array_equal(t1.midds, self.correct_Tsort_return_flow_midds1)
        
    def test_Tsort_with_dataset_2(self):
        """
        test that _Tsort gives the expected results from dataset 2
        """
        t2 = tdb.Transect(self.x2, self.slc2, self.tsc2, self.dts2)
        assert_array_equal(t2.ind, self.correct_Tsort_ind2)
        assert_array_equal(t2.sds, self.correct_Tsort_sds2)
        assert_array_equal(t2.tnum, self.correct_Tsort_tnum2)
        assert_array_equal(t2.smxt, self.correct_Tsort_smxt2)
        assert_array_equal(t2.smx, self.correct_Tsort_smx2)
        
    def test_Transect_with_no_dts(self):
        """
        test that Transect class initializes correctly when DTS is not passed
        """
        t2 = tdb.Transect(self.x2, self.slc2, self.tsc2)
        assert_array_equal(t2.mxw, self.correct_mxw2)
        assert_array_equal(t2.w, self.slc2)
                          
    def test_exclude_transect_with_dataset_2(self):
        """
        test that _exclude_transect excludes only the specified transects
        """
        excludeA = ('Jantang', 1)
        excludeB = [('Jantang', 1), ('Kuala Merisi', 2)]
        slKey = {1: "Jantang", 2: "Kuala Merisi"}
        t2A = tdb.Transect(self.x2, self.slc2, self.tsc2, self.dts2, 
                           exclude=excludeA, slKey=slKey)
        t2B = tdb.Transect(self.x2, self.slc2, self.tsc2, self.dts2, 
                           exclude=excludeB, slKey=slKey)
        assert_array_equal(t2A.sds, ary([10, 2, 3, 3, 0, 20]))
        assert_array_equal(t2A.x, ary([3., 4, 6, 7, 8, 9]))
        assert_array_equal(t2B.sds, ary([10, 2, 3, 3]))
        assert_array_equal(t2B.ind, ary([5, 4, 7, 8]))
        
        
if __name__ == '__main__':
    unittest.main()