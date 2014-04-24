# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:37:49 2014

@author: blunghino
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestGetevents(unittest.TestCase):
    
    event_lookup = {993279600: 'Peru, 2001', 1299830400: 'Japan, 2011'}
    slCode = np.asarray([1, 1, 1, 2, 60, 3]) 
    tsunami = np.asarray([993279600, 993279600, 1299830400, 993279600, 
                          1299830400, np.nan])
    adict = {'event_lookup': event_lookup, 'SLCode': slCode, 
             'Tsunami': tsunami}
                      
    def test_getevents_with_one_sublocation(self):
        """
        test that get events gives the currect output for a single sublocation
        """
        out = tdb.getevents([1], self.adict)
        self.assertEqual(out, ['Peru, 2001'])

    def test_getevents_with_multiple_sublocations(self):
        """
        test that get events gives the correct output when passed multiple 
        sublocations
        """        
        out = tdb.getevents([1, 1, 1, 3, 2, 60], self.adict)
        self.assertEqual(out, ['Peru, 2001', 'Peru, 2001', 'Peru, 2001', 
                               '', 'Peru, 2001', 'Japan, 2011'])
        
    def test_getevents_want_second_event_at_given_sublocation(self):
        """
        test that the second event corresponding to a single sublocation
        will not be returned. This is how the function is intended to behave
        but it could cause inconsistent results
        """
        out = tdb.getevents(np.ones(100), self.adict)
        self.assertRaises(ValueError, out.index, 'Japan, 2011')
        
            

if __name__ == "__main__":
    unittest.main()