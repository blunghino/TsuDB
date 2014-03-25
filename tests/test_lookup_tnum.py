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


class TestLookuptnum(unittest.TestCase):
    
    slc1 = ary([1, 1, 1, 1, 1], dtype=np.float64)
    tsc1 = ary([1, 1, 1, 2, 2], dtype=np.float64)
    dts1 = ary([0, 25, 50, 100, 0], dtype=np.float64)
    x1 = np.arange(5)
    
    def test_lookup_tnum(self):
        """
        check that the generator yields the correct values
        """
        t = tdb.Transect(self.x1, self.slc1, self.tsc1, self.dts1)
        gen = tdb.lookup_tnum(1, t)
        tnums = {g for g in gen}
        self.assertEqual(tnums, {1, 2})
        
        
if __name__ == '__main__':
    unittest.main()