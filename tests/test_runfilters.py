# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:37:49 2014

@author: blunghino
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestRunfilters(unittest.TestCase):
    
    f1 = np.ones(4)
    data = np.arange(4) * 11
    f2 = np.ones(4) * 2
    f2[-1] = 0
    f3 = np.asarray([0, 1, 0, 1])
    f4 = np.zeros(4)
        
    def test_runfilters_filter_multiple(self):
        out = tdb.runfilters([self.data, self.f1, self.f3], 3, verbose=False)
        assert_array_equal(out[0], np.asarray([11, 33]))
        assert_array_equal(out[1], np.ones(2))
        
    def test_runfilters_only_one_filter(self):
        out = tdb.runfilters([self.data, self.f4, self.f3], 1, verbose=False)
        assert_array_equal(out[0], np.asarray([11, 33]))
        
    def test_runfilters_filter_non_bool(self):
        out = tdb.runfilters([self.data, self.f1, self.f2], 1, verbose=False)
        assert_array_equal(out[0], self.data[:-1])
    

if __name__ == "__main__":
    unittest.main()