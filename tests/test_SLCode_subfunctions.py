# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:00:21 2014

@author: blunghino
"""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestSLCodesubfunctions(unittest.TestCase):

    sublocations = ['Lagundri Bay', 'Sarangbuang', 'Pulau Asu', 
                             'Sarangbuang', 'Tuangku', 'Tuangku', 'Tuangku']
    slcodes = np.asarray([ 1.,  3.,  2.,  3.,  4.,  4.,  4.])
    slkey = {1: 'Lagundri Bay', 2: 'Pulau Asu', 3: 'Sarangbuang', 
                      4: 'Tuangku'}
                      
    def test_slcoder(self):
        slcodes_out, slkey_out = tdb.SLcoder(self.sublocations)
        self.assertEqual(slkey_out, self.slkey)
        assert_array_equal(slcodes_out, self.slcodes)
        
    def test_sldecoder(self):
        sublocations_out = tdb.SLdecoder(self.slcodes, self.slkey)
        self.assertEqual(sublocations_out, self.sublocations)
        
    def test_lookup_slcode(self):
        code_out = tdb.lookup_SLCode('Sarangbuang', self.slkey)
        self.assertEqual(code_out, 3)
        
        
if __name__ == "__main__":
    unittest.main()