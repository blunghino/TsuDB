# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:48:20 2014

@author: blunghino
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestDenan(unittest.TestCase):
    
    arr1 = np.asarray([0, 2., 2., np.nan, np.nan, 3.5, -1])
    arr2 = np.ones_like(arr1)
    arr3 = np.ones_like(arr1)
    arr3[0] = np.nan
    arr4 = np.zeros_like(arr1)
    
    def test_denan_with_one_array_with_nans(self):
        """
        for a single array, denan should remove the nan
        """
        assert_array_equal(self.arr3[1:], tdb.denan(self.arr3))
        
    def test_denan_with_one_array_with_no_nans(self):
        """
        for a single array with no nans, it should be returned unchanged
        """
        assert_array_equal(self.arr4, tdb.denan(self.arr4))
        
    def test_denan_with_multiple_arrays_with_no_nans(self):
        """
        for multiple arrays with no nans, arrays should be unchanged
        """
        out = tdb.denan(self.arr2, self.arr4)
        assert_array_equal(self.arr4, out[1])
        assert_array_equal(self.arr2, out[0])
        
    def test_denan_with_multiple_arrays_with_nans(self):
        """
        for multiple arrays with nans, denan should remove the nans accordingly
        """
        out = tdb.denan(self.arr1, self.arr2, self.arr3, self.arr4)
        assert_array_equal(out[0], np.asarray([2., 2, 3.5, -1]))
        assert_array_equal(out[3], np.zeros(4))
        
    def test_denan_with_multiple_arrays_with_nrounds(self):
        """
        check that nrounds controls the nans that are removed
        """
        out = tdb.denan(self.arr1, self.arr2, self.arr3, self.arr4, n_rounds=2)
        assert_array_equal(out[3], np.zeros(5))
        assert_array_equal(out[0], np.asarray([0, 2, 2, 3.5, -1]))
        self.assertTrue(np.isnan(out[2][0]))      
        
        
if __name__ == "__main__":
    unittest.main()