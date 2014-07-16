# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:01:21 2014

@author: blunghino
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestAverageInBins(unittest.TestCase):
    
    def setUp(self):
        self.x = np.arange(10)
        self.y = np.asarray([12 if a % 3 == 0 else 0 for a in self.x])
    
    def test_average_in_bins_with_percents(self):
        """
        check that the function works when `block` is a percent
        """
        xp, yp = tdb.average_in_bins(self.x, self.y, block=25, percents=True)
        assert_array_equal(xp, np.asarray([1.125, 3.375, 5.625, 7.875]))
        assert_array_equal(yp, np.asarray([4, 6, 6, 4]))
    
    def test_average_in_bins_with_blocks(self):
        """
        check that the function works when `block` is an increment of the data
        """
        xp, yp = tdb.average_in_bins(self.x, self.y, block=2, percents=False)
        assert_array_equal(xp, np.asarray([1, 3, 5, 7, 9]))
        assert_array_equal(yp, np.asarray([4, 6, 6, 0, 12]))                            
        
if __name__ == "__main__":
    unittest.main()