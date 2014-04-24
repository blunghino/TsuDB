# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:37:49 2014

@author: blunghino
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestGetBFromA(unittest.TestCase):
    
    keyA = list(range(6))
    A = [0, 4, 5, 3]
    keyB = ['a', 'b', 'c', 'd', 'e', 'f']
    B = ['a', 'e', 'f', 'd']
    keyC = [1, 1, 2, 3, 4, 4]
    C = [1, 4, 4, 3]

    Adict = {'keyC': keyC, 'keyA': keyA, 'keyB': keyB}
    
    def test_getBfromA_with_lists(self):
        """
        test that the correct list is returned when a list is passed
        """
        out1 = tdb.get_B_from_A(self.A, self.Adict, keyA='keyA', keyB='keyB')
        out2 = tdb.get_B_from_A(self.B, self.Adict, keyA='keyB', keyB='keyA')
        self.assertEqual(out1, self.B)
        self.assertEqual(out2, self.A)
        
    def test_getBfromA_with_empty_list(self):
        """
        test that an empty list is returned when an empty list is passed
        """
        out = tdb.get_B_from_A([], self.Adict, keyA='keyA', keyB='keyB')
        self.assertEqual(out, [])
        
    def test_getBfromA_with_nparray(self):
        """
        test that the correct list is returned when a numpy array is passed
        """
        out1 = tdb.get_B_from_A(np.asarray(self.A), self.Adict, keyA='keyA', 
                               keyB='keyB')
        out2 = tdb.get_B_from_A(np.asarray(self.B), self.Adict, keyA='keyB',
                                keyB='keyA')
        self.assertEqual(out1, self.B)
        self.assertEqual(out2, self.A)
        
    def test_getBfromA_with_repeat_values(self):
        """
        test that repeat values are handled (assume sorted for repeats)
        """
        out1 = tdb.get_B_from_A(self.C, self.Adict, keyA='keyC', keyB='keyB')
        out2 = tdb.get_B_from_A(self.B, self.Adict, keyA='keyB', keyB='keyC')
        self.assertEqual(out1, self.B)
        self.assertEqual(out2, self.C)
        
if __name__ == "__main__":
    unittest.main()