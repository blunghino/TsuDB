# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:37:49 2014

@author: blunghino
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestPercentileByBin(unittest.TestCase):
    
    def test_percentile_by_bin(self):
        """
        ensure that the correct median value is returned
        """
        bin_data = np.asarray(list(range(3))*3, dtype=float)
        bin_data[1:-1] += .1
        bin_data[-1] = 3
        perc_data = np.asarray([0,5,-1,0,29,-2,0,10,-1.3])
        bins, percs = tdb.percentile_by_bin(bin_data, perc_data)
        assert_array_equal(percs, np.asarray([0,10,-1.3]))
        
if __name__ == "__main__":
    unittest.main()