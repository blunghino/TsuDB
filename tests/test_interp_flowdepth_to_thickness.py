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


class TestInterpFlowdepthToThickness(unittest.TestCase):
    
    slc1 = ary([1, 1, 1, 1, 1, 1], dtype=np.float64)
    tsc1 = ary([1, 1, 1, 1, 1, 1], dtype=np.float64)
    dts1 = ary([0, 25, 50, 100, 75, 125], dtype=np.float64)
    t = ary([1, 2, 1, np.nan, 3, 5])
    f = ary([20, np.nan, 10, 30, np.nan, np.nan])
    tint = ary([1, 2, 1, 3.])
    fint = ary([20, 15, 10, 20.])
    dtsint = ary([0, 25, 50, 75.])
    tnumint = ary([1, 1, 1, 1])

    def test_interp_flowdepth_to_thickness(self):
        """
        test that the expected output is returned from good data
        """
        F = tdb.Transect(self.f, self.slc1, self.tsc1, self.dts1)
        T = tdb.Transect(self.t, self.slc1, self.tsc1, self.dts1)
        out = tdb.interp_flowdepth_to_thickness(T, F)
        assert_array_equal(out[0], self.tint)
        assert_array_equal(out[1], self.fint)
        assert_array_equal(out[2], self.dtsint)
        assert_array_equal(out[3], self.tnumint)
        
        
if __name__ == '__main__':
    unittest.main()