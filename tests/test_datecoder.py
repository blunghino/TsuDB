# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:37:49 2014

@author: blunghino
"""

import unittest
import datetime

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import TsuDB as tdb


class TestDatecoder(unittest.TestCase):
    
    correct_outputformat_1 = np.asarray([np.nan, 1104220800, 
                                              1299830400, 1111993200, 
                                              np.nan, np.nan, 
                                              993279600, 900658800])
    correct_outputformat_0 = np.asarray(
        [np.nan, 
         datetime.datetime(2004, 12, 28, 0, 0),
         datetime.datetime(2011, 3, 11, 0, 0), 
         datetime.datetime(2005, 3, 28, 0, 0),
         np.nan,
         np.nan,
         datetime.datetime(2001, 6, 23, 0, 0),
         datetime.datetime(1998, 7, 17, 0, 0)]
         )
         
    def setUp(self):
        self.dates = ['', '38349.0', '40613.0', '2005-03-28', '1946-04-01?',  
                      '?', '2001-06-23', '35993.0']
                      
    def test_datecoder_mixed_string_float_input(self):
        """
        check that datecoder can handle a mix of strings and floats
        """
        mixed_dates = ['', 38349., 40613., '2005-03-28', '1946-04-01?',  
                      '?', '2001-06-23', 35993.]
        out = tdb.datecoder(mixed_dates)
        assert_array_almost_equal(out, self.correct_outputformat_1, decimal=3)
                      
    def test_datecoder_timestamp_output(self):
        """
        check that datecoder gives the correct timestamp output
        """
        out = tdb.datecoder(self.dates)
        assert_array_almost_equal(out, self.correct_outputformat_1, decimal=3)
        
    def test_datecoder_datetime_output(self):
        """
        check that datecoder gives the correct string output
        """
        out = tdb.datecoder(self.dates, outputformat=0)
        assert_array_equal(out, self.correct_outputformat_0)
        
        
if __name__ == "__main__":
    unittest.main()