# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:37:49 2014

@author: blunghino
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestGetValuesOnTransectWithTuple(unittest.TestCase):
    
    x = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
    y = np.arange(6)
    transect = np.asarray([1, 1, 2, 1, 1, 1.])
    slKey = {1: 'Sendai', 2: 'Jantang', 3: 'Nothing'}
    slCode = np.asarray([1, 1, 1, 2, 2, 1.])
    subloc = ['Sendai', 'Sendai', 'Sendai', 'Jantang', 'Jantang', 'Sendai']
    adict = {'Transect': transect, 'SLCode': slCode, 
             'Sublocation': subloc, 'SLKey': slKey, 'X': x, 'Y': y}
    
    def test_get_values_on_transect_with_tuple_with_nparray(self):
        """
        test that the correct array is returned when target field is a nparray
        """ 
        out = tdb.get_values_on_transect_with_tuple(('Sendai', 1), self.adict, 'Y')
        assert_array_equal(out, np.asarray([0, 1, 5.]))
    
    def test_get_values_on_transect_with_tuple_with_list(self):
        """
        test that the correct list is returned when target field is a list
        """               
        out = tdb.get_values_on_transect_with_tuple(('Sendai', 1), self.adict, 'X')
        self.assertEqual(out, ['Zero', 'One', 'Five'])
        
    def test_get_values_on_transect_with_tuple_with_slcode(self):
        """
        check with an SLCode not string in transect_tuple (1 instead of 'Sendai')
        """
        out = tdb.get_values_on_transect_with_tuple((1, 2), self.adict, 'X')
        self.assertEqual(out, ['Two'])
        
    def test_get_values_on_transect_with_tuple_with_multiple_keys(self):
        """
        test that multiple fields can be returned in a tuple
        """
        out = tdb.get_values_on_transect_with_tuple(('Sendai', 1), self.adict, 'X', 'Y')
        assert_array_equal(out[1], np.asarray([0, 1, 5.]))
        self.assertEqual(out[0], ['Zero', 'One', 'Five'])
        
    def test_get_values_on_transect_with_tuple_none_if_no_transect(self):
        """
        test that the function returns None if the transect does not exist
        """
        out = tdb.get_values_on_transect_with_tuple(('Sendai', 3), self.adict, 'Y')
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()