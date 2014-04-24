# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:37:49 2014

@author: blunghino
"""

import unittest
import os

import numpy as np
from numpy.testing import assert_array_equal

import TsuDB as tdb


class TestCsv2dic(unittest.TestCase):

    csv_test_file_path = os.path.join(os.path.dirname(__file__), 'test_csv2dic_partial_r6.csv')
    correct_dic_keys = {
        'ID',	'Location',	'Sublocation',	'Date',	'Tsunami', 'Comments',
        'FACSID',	'Lat',	'Long',	'Elevation', 'ElevationDatum',	
        'ElevationBy',	'ObservationType',	'Transect',	'ShoreNormal',	
        'ShorelineOrientation',	'TransectOrientation',	
        'Distance2transect',	'Distance2shore',	'DepositStart',	
        'DepositLimit',	'InundationLimit',	'Runup',	'FlowDepth',	
        'HeightAtShore',	'FlowDirection',	'MaxFlowDirection',	
        'Nwaves',	'Modern',	'ProjectedFlowDepth',	'Method',	
        'Thickness',	'MaxThickness',	'GSFileOriginal',	
        'GSFileUniform',	'TypeGS',	'WhosGS',	'DepthTopGS',	
        'DepthBottomGS',	'NSamplesGS',	'Layers',	'MaxLayers',	
        'MudCap',	'MudIntLayers',	'Ngrading',	'Sgrading',	'Igrading',	
        'Massive', 'UpperContact',	'BasalContact',	'RipUps', 'HeavyMin',	
        'ShellFragments', 'WholeShell',	'CoarseClasts',	'OrganicDebris',
        'Boulder',	'Aaxis', 'Baxis', 'Caxis', 'BoulderOrientation',
        'Sediment',	'Underlying', 'Surface', 'Topography',	'Notes',
        'floats', 'event_lookup', 'emap', 'incomplete_transect', 
        'datum_lookup', 'attributes', 'SLCode', 'SLKey', 'typegs_lookup', 
        'mw_lookup',
    }
    correct_transect = np.asarray([1, 1, 1, np.nan, 1, 2])
    correct_underlying = ['soil', 'tan sand with roots', '', '', '', '']
    dic = tdb.csv2dic(csv_test_file_path)
  
    def test_csv2dic_has_all_keys(self):
        """
        test that all expected dictionary keys were created
        """
        self.assertEqual(set(self.dic.keys()), self.correct_dic_keys)
        
    def test_csv2dic_has_correct_types(self):
        """
        test that dictionary values are the correct types
        """
        self.assertIsInstance(self.dic['ShorelineOrientation'][0], float)
        self.assertIsInstance(self.dic['Notes'][-1], str)
        self.assertIsInstance(self.dic['ID'], list)
        self.assertIsInstance(self.dic['Date'], np.ndarray)
        
    def test_csv2dic_gets_correct_values(self):
        """
        test that values are correct
        """
        assert_array_equal(self.dic['Transect'], self.correct_transect)
        self.assertEqual(self.dic['Underlying'], self.correct_underlying)
        
        
if __name__ == "__main__":
    unittest.main()