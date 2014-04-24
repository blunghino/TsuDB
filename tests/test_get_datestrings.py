# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:43:09 2014

@author: blunghino
"""

import unittest
import datetime as dt

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestGetDateStrings(unittest.TestCase):
    
    def setUp(self):
        d1 = dt.datetime(2005, 3, 28)
        d2 = dt.datetime(2001, 1, 1)
        d3 = dt.datetime(2001, 6, 23)
        self.timestamps = np.asarray([d.timestamp() for d in (d1, d2, d3)])
        self.timestamps[1] = np.nan
        self.dates = np.asarray(['2005-03-28', 'nan', '2001-06-23'])
    
    def test_getdatestrings_with_assorted_timestamps(self):
        out = tdb.get_datestrings(self.timestamps)
        assert_array_equal(out, self.dates)
        
if __name__ == "__main__":
    unittest.main()       