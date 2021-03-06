# -*- coding: utf-8 -*-
"""
Tools to work with tsunami data stored in excel database.

Written in Python 3.3

First version ReadTsuDepCSV.py created on Thu May 30 12:23:05 2013
TsuDB.py series created on Wed Jul 17 11:15:25 2013
200 series incorporates new features into the transect class
200 series replaces the original CSV reader (starting at v201)
300 series incorporates grain size data features (eg GSData class)
300 series allows TsuDB to be imported as a module to support unittest testing
    (starting at 302)
300 series replaces csv reader to read into dictionary straight from .xlsx 
    database (starting at 303)
Starting at version 307, all changes tracked with git
    
to test all functionality:
    at the command prompt in the py directory under the main project directory
    $ python -m unittest discover -v
    
use requirements.txt to install all third party requirements with pip:
    pip install -r requirements.txt
    
also requires GS_tools module to be added to PYTHONPATH environment variable
GS_tools module available at https://github.com/blunghino/GS_tools

@author: Brent Lunghino, USGS, CNTS
"""

import os
import sys
import csv
import pickle
import warnings
import datetime as dt
from pprint import pprint
from tkinter import filedialog

import xlrd
import numpy as np
import matplotlib as mpl
from scipy.stats import linregress, nanmean, gmean
from scipy import nanmax, nanmin, nanargmax
from scipy.optimize import curve_fit
from matplotlib import cm, pyplot as plt

from GS_tools.gsfile import GSFile

## optional imports required only for a couple plotting subfunctions
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    pass
try:
    from distance_on_unit_sphere import distance_on_unit_sphere
except ImportError:
    pass

###############################################################################
##   SUBFUNCTIONS
###############################################################################       
def notnan(x):
    """
    returns a matrix of True/False values, false where x is nan
    (inverts the np.isnan function)
    
    !! In many situations can use np.isfinite(x) instead
    """
    return np.logical_not(np.isnan(x))
    
###############################################################################
def attributefilter(x, filtr, invert=False, nanzero=False):
    """
    filters data from array x using filtr
    
    where filtr is false, the corresponding value in array x will be deleted
    x and filtr must share length
    if invert = True, inverts the function to delete where filtr is true
    nanzero changes NaN values in the filter to be zero 
    (NaN is true in boolean logic)
    """
    if nanzero:
        filtr = filtr*notnan(filtr)
    if not invert:
        return x[filtr.astype(np.bool)]
    elif invert:
        return x[np.logical_not(filtr.astype(np.bool))]
        
###############################################################################
def denan(*args, n_rounds=None, verbose=False):
    """
    find where any NaNs exist and remove those locations from all vectors
    
    pass any number of vectors (of uniform length) to denan. 
    denan will find all occurences of NaN in each vector and remove data at the 
    indices where NaNs occur in that vector from all vectors
    
    by specifying n_rounds the number of vectors that denan will filter NaNs
    from can be controlled. for example, if you are passing 5 vectors
    containing NaNs to denan, but you don't want to filter based on NaNs from
    the 5th vector, specify n_rounds=4
    """ 
    if len(args) > 1:
        ## change to a list to allow elements to be modified
        args = list(args)
        if n_rounds is None:
            ## remove elements based on all arrays in
            n_rounds = len(args)
        for ii in range(n_rounds):
            ## find where NaNs occur in args[ii]
            Nnan = notnan(args[ii])
            if verbose:
                print('size after denan round '+str(ii+1)+': '+str(len(args))+\
                      'x'+str(len(Nnan.nonzero()[0])))      
            for jj in range(len(args)):
                ## remove where Nnan is false from each array in args
                args[jj] = attributefilter(args[jj], Nnan)               
        return args
    else:
        arg = args[0]
        return arg[notnan(arg)]
    
###############################################################################
def runfilters(x, num_filters, invert=False, verbose=False):    
    """
    passes array of vectors to attribute filter
    
    the vectors to be used as boolean filters are at the end of the array,
    runfilters will run through the number of filters specified starting at 
    x[-1] and working all the way to x[-num_filters]
    """
    if num_filters == 0: 
        return x
    else:
        for ii in range(num_filters):
            ## eg for 2 filters with ii = 1 => ii = -2 => use second array 
            ## from end in x
            ii = (ii+1)*-1
            for jj in range(len(x)):
                x[jj] = attributefilter(x[jj], x[ii], invert=invert)
            if verbose:
                print('size after filter '+str(ii*-1)+': '+str(len(x))+'x'+ \
                      str(len(x[0])))
        return x

###############################################################################    
def datecoder(dates, dateformat=0, outputformat=1):
    """
    converts a list or vector of dates in excel date form into strings or 
    datetime timestamps
    
    dateformat is an xlrd argument that specifies the format that the dates
    are saved in in the xls file.
    if outputformat == 1, return datetime timestamps
    if outputformat == 0, return strings
    """
    ## convert to datetime object format
    for ii in range(len(dates)):
        ## try removing the time portion of the excel date
        try:
            dates[ii] = dt.datetime(*xlrd.xldate_as_tuple(float(dates[ii]),
                                                          dateformat)[0:6])
        except ValueError:
            try:
                dates[ii] = dt.datetime.strptime(dates[ii], '%Y-%m-%d')
            ## bad excel date value
            except ValueError or TypeError:
                dates[ii] = np.nan
        ## convert to timestamp 
        if outputformat:
            try:
                dates[ii] = dates[ii].replace(tzinfo=dt.timezone.utc).timestamp()
            except AttributeError:
                continue
    return np.asarray(dates)

###############################################################################
def SLcoder(textin):
    """
    creates a list of coded values and a key to decode them
    
    accepts a list. returns a numpy array of equal length containing integer
    values representing each unique item in the input list
    also returns a dictionary connecting each key to the item it codes for
    """
    coded = np.zeros(len(textin))
    key = {}
    for ii, string in enumerate(sorted(set(textin))):
        ii += 1
        filtr = [jj for jj in range(len(textin)) if textin[jj] == string]
        coded[filtr] = ii
        key[ii] = string
    return coded, key
    
###############################################################################
def initialize_Adict():
    """
    create a dictionary and populate it with TsuDB data not stored in
    the database
    
    this is where to add metadata or other information that should be
    universally accessible to the database but doesn't fit into the excel file.
    
    a great place to create ugly but convenient solutions to problems
    """
    ## attributes to be converted from strings to floats when they are read in
    floats = ('ElevationDatum', 'ElevationBy', 'ObservationType', 'Transect',
              'Lat', 'Long', 'Elevation', 'ShorelineOrientation', 'Layers',
              'TransectOrientation', 'Distance2transect', 'Distance2shore',
              'DepositStart', 'DepositLimit', 'InundationLimit', 'Runup', 
              'FlowDepth', 'HeightAtShore', 'FlowDirection', 'Method',
              'MaxFlowDirection', 'Thickness', 'MaxThickness', 'Modern',
              'Aaxis', 'Baxis', 'Caxis', 'BoulderOrientation', 'TypeGS',
              'MaxLayers', 'MudCap', 'MudIntLayers', 'Ngrading', 'Sgrading', 
              'Igrading', 'Massive',  'RipUps', 'ShoreNormal', 'Nwaves',     
              'HeavyMin', 'ShellFragments', 'WholeShell', 'CoarseClasts',
              'OrganicDebris', 'ProjectedFlowDepth', 'Boulder', 'UpperContact', 
              'BasalContact', 'NSamplesGS', 'DepthTopGS', 'DepthBottomGS',)
    ## lookup dictionary to convert coded text to integer values
    typegs_lookup = {'T': 1., 'VC': 2., 'PC': 3., 'GC': 4., 'RC': 5., 'ST': 1., 
                     'LD': 2., 'SV': 3., 'STLD': 4., 'CS': 5., 'DGS': 6., 
                     'G': 1., 'A': 2., 'E': 3., 'R': 4., 'I': 5.}
    ## lookup dictionary to convert coded elevation datums to strings
    datum_lookup = {1: 'NAVD88', 2: 'MLLW', 3: 'MSL', 4: 'Tokyo Peil'}
    ## lookup dictionary to convert event datetime timestamps to event names
    event_lookup = {
                    993254400: 'Peru, 2001', 
                    1299801600: 'Japan, 2011', 
                    1104019200: 'Indian Ocean, 2004',
                    1111968000: 'Sumatra, 2005',
                    900633600: 'Papua New Guinea, 1998', 
                    1267228800: 'Chile, 2010', 
                    1254182400: 'Samoa, 2009',
                    }
    ## dictionary of EQ magnitudes coded by datetime timestamps
    mw_lookup = {
        993254400: 8.4,
        1299801600: 9.,
        1104019200: 9.1,
        1111968000: 8.6,
        900633600: 7.,
        1267228800: 8.8,
        1254182400: 8.1,
    }
    ## use to map events to pyplot color symbol strings, allows consistent 
    ## symbology in pyplot figures
    emap = {
            'Peru, 2001': 'kd', 
            'Japan, 2011': 'bH', 
            'Indian Ocean, 2004': 'rv', 
            'Papua New Guinea, 1998': 'gs', 
            'Chile, 2010': 'yo', 
            'Samoa, 2009': 'mp',
            'Sumatra, 2005': 'c^'
            }
    ## approximate tsunami earthquake source locations (lat, lon, depth [km])
    sources = {
        'Peru, 2001': (-16.38, -73.5, 32), 
        'Japan, 2011': (38.297, 142.373, 29), 
        'Indian Ocean, 2004': (3.295, 95.982, 30), 
        'Papua New Guinea, 1998': (-2.975, 142.692, 10), 
        'Chile, 2010': (-36.122, -72.898, 22.9), 
        'Samoa, 2009': (-15.489, -172.095, 18),
        'Sumatra, 2005': (2.05, 97.06, 33.7)
    }
    ## list of tuples specifying incomplete transects to exclude from some 
    ## plots. This list is based off spatial coverage of thickness data.
    incomplete_transect = [('Jantang', 4), ('Jantang', 5), ('Lhok Kruet', 4),  
                           ('Pulau Breuh', 1), ('Pulut', 1), ('Bangkaru', 1),
                           ('Mankeri', 1), ('Kulmunai Kuddi', 1),
                           ('Ibral Nagar Nalaveli', 1), ('Kulmunai Kuddi', 2),
                           ('Ibral Nagar Nalaveli', 5), ('Singkel', 1),
                           ('Pulau Asu', 1), ('Salaut', 3), ('Babi', 1),
                           ('Lagundri Bay', 1), ('Pulut', 1), ('Tuangku', 2)]
    ## list of high resolution grain size data (sampling interval <= 1 cm)
    high_res_gs = [
        'GS_Japan_Sendai_T3-6.csv', 
        'GS_Japan_Sendai_T3-8.csv', 
        'GS_Japan_Sendai_T3-10.csv', 
        'GS_Japan_Sendai_T3-11.csv', 
        'GS_Japan_Sendai_T3-16.csv', 
        'GS_Japan_Sendai_T3-23.csv', 
        'GS_Japan_Sendai_T3-27.csv', 
        'GS_Chile_Coliumo_Trench3.csv', 
        'GS_Chile_Coliumo_Trench7.csv',
        'GS_PapuaNewGuinea_Arop_Hole7.csv',
        'GS_PapuaNewGuinea_SissanoVillage_327mML.csv',
        'GS_Peru_Amecosupe_trenchH13-20.csv',
        'GS_Sumatra_Babi_SUM15.csv',
        'GS_Sumatra_Busung1_T2_1.csv',
        'GS_Sumatra_Busung1_T2_2.csv',
        'GS_Sumatra_Jantang3_T13.csv',
        'GS_Sumatra_Jantang3_T3.csv',
        'GS_Sumatra_Jantang3_T4.csv',
        'GS_Sumatra_Jantang3_T5.csv',
        'GS_Sumatra_Jantang3_T6.csv',
        'GS_Sumatra_Jantang3_T7.csv',        
        'GS_Sumatra_Jantang3_T8.csv',
        'GS_Sumatra_LagundriBay_SUM2.csv',
        'GS_Sumatra_LagundriBay_SUM6.csv',
        'GS_Sumatra_Lhokkruet_SUM21.csv',
        'GS_Sumatra_LangiIsland_SUM33.csv',
        'GS_Sumatra_Tuangku_SUM13.csv',
        'GS_Sumatra_Salaut_SUM18.csv',
        'GS_WesternSamoa_Satitoa_T13.csv', 
        'GS_WesternSamoa_Satitoa_T19.csv', 
    ]
    ## initialize master dictionary
    dic = {'floats': floats, 'event_lookup': event_lookup, 'emap': emap, 
           'incomplete_transect': incomplete_transect, 'mw_lookup': mw_lookup,
           'typegs_lookup': typegs_lookup, 'datum_lookup': datum_lookup,
           'high_res_gs': high_res_gs, 'sources': sources}
    return dic
        
###############################################################################
def xls2dic(xls_file_path='', n_sheets_to_skip=4):
    """
    create dictionary of tsunami deposit database by reading in excel file
    
    each sheet read in must be organized by columns and have the same headers
    """
    dir_0 = os.getcwd()
    ## if no path or invalid path ask user to manually find excel file
    if not xls_file_path or not os.path.isfile(xls_file_path):
        xls_file_path = filedialog.askopenfilename()
    path, xls_file = os.path.split(xls_file_path)
    os.chdir(path)
    ## initialize dictionary and bring in default dictionary entries
    dic = initialize_Adict()
    ## initialize xlrd Book object
    book = xlrd.open_workbook(xls_file)
    biglist = []
    ## the first `n_sheets_to_skip` sheets are organizational and are ignored,
    ## the rest contain data by location and all have the same column headers
    for ii in range(n_sheets_to_skip, book.nsheets):
        sheet = book.sheet_by_index(ii)        
        if ii == n_sheets_to_skip:
            ## header strings from the first row of the first sheet
            biglist.append([c.value for c in sheet.row(0)])
        ## all other rows contain data
        for jj in range(1, sheet.nrows):
            ## get data from each cell in row jj
            smalllist = []
            for c in sheet.row(jj):
                ## repace cells that raise a UnicodeEncodeError with NaNs
                try:
                    test = str(c.value).encode('cp1252')
                    smalllist.append(c.value)
                except UnicodeEncodeError:
                    smalllist.append('NaN')
            biglist.append(smalllist)
    dic["attributes"] = biglist[0]
    for ii in range(len(biglist[0])):
        ## dictionary value is from each row in column ii  
        ## dictionary key is from the headers in column ii
        dic.update({biglist[0][ii]: [line[ii] for line in biglist[1:]]})
    ## convert lists of strings of numbers stored dictionary to float arrays
    for k in dic['floats']:
        v = dic[k]
        try:
            dic[k] = np.asarray(v, dtype=float)
        except ValueError:
            ## change strings to floats
            for ii, string in enumerate(v):
                try:
                    v[ii] = float(string)
                except ValueError:
                    ##  Convert blanks to nan
                    if string == '' or string == ' ': 
                        v[ii] = np.nan
                    ## coded text to numeric value
                    elif string in dic['typegs_lookup'].keys():
                        v[ii] = dic['typegs_lookup'][string]
                    else:
                        print('ValueError raised for unexpected string.\
                                      \nReplaced "%s" with "nan"' % string)
                        v[ii] = np.nan
            dic[k] = np.asarray(v, dtype=float)
    ## covert string dates to datetime numbers
    dic["Date"] = datecoder(dic["Date"])
    dic["Tsunami"] = datecoder(dic["Tsunami"])
    ## encode sublocations
    dic["SLCode"], dic["SLKey"] = SLcoder(dic["Sublocation"])
    dic['unique'] = np.arange(dic['Modern'].size, dtype=float)
    ## change back to original working directory
    os.chdir(dir_0)
    return dic
    
###############################################################################   
def savedict(Adict, dict_name='TsuDB_Adict_', askfilename=False):
    """
    pickle a dictionary with name and timestamp. 
    save as with askfilename=True.
    
    returns the name of the created pickle file
    """
    ## save as
    if askfilename:
        filename = filedialog.asksaveasfilename()
    ## create automatic unique file name
    else:
        ## right meow!
        meow = dt.datetime.strftime(dt.datetime.today(), '%Y-%m-%d')
        filename = dict_name + meow + '.pkl'
        while os.path.isfile(filename):
            meow = dt.datetime.strftime(dt.datetime.today(), '%Y-%m-%d_%H%M%S')
            filename = dict_name + meow + '.pkl'
    ## write to pickle file
    with open(filename, 'wb') as output:
        pickle.dump(Adict, output)
        output.close()    
    return filename

###############################################################################
def opendict(filename):
    """
    unpickle a pickle file containing a dictionary (or anything really)
    """
    ## find file to open
    if not filename or not os.path.isfile(filename):
        path, filename = os.path.split(filedialog.askopenfilename())
        try:
            os.chdir(path)
        except OSError:
            print('OSError. No valid input.')
            sys.exit()
    ## read data
    with open(filename, 'rb') as picklein:
        dic = pickle.load(picklein)
        picklein.close()    
    return dic
    
###############################################################################
class Transect:
    """
    Sort data within each transect by distance to shore
    Set transect specific properties for observations that fall on a transect.
    
    
    To initialize a Transect object requires a minimum of 3 vectors of equal
    length:
        X is any data
        SLC is the code specifying the sublocations (no NaNs allowed)
        TSC is the transect number (no NaNs allowed)
        *DTS is the distance to shore (optional, no NaNs allowed)
    
    available attributes:
    .x is the attribute passed to Transect in its original order
    .w is the sublocation in its original order
    .t is the transect number in its original order
    .mxw is the maximum value of .x at any given sublocation
    .ds is the distance to shore in its original order
    .ind is the indices required to sort within each transect by increasing 
    distance to shore
    .sx, .sw, .st, .sds are .x, .w, .t, .ds sorted by .ind
    .smx is the max value of .sx along each transect, in sorted order
    .smxt is the max value of .sx along each transect, all other values are nan
    .dx is the difference between two adjacent values of .sx along a transect
    where the last value of .dx on a transect is NaN (if return_flow=False)
    .middx is the average of two adjacent .sx values on a transect
    .dds is the difference between two adjacent values of .sds along a transect
    .midds is the midpoint between .sds values along a transect
    .tnum is an arbitrary number unique to each transect (sorted order)
    
    """
    def __init__(self, X, SLC, TSC, DTS=None, return_flow=False, exclude=None, 
                 slKey=None):
        """
        initialize transect object
        
        `return_flow` is True if you want to sort transects from landward to 
        seaward
        
        `slKey` is only needed if `exclude` is not None and must be a
        dictionary relating sublocation numbers to sublocation strings 
        (normally created by TsuDB.SLcoder function)
        
        `exclude` is used to exclude specific transects. it is a list of tuples
        with each tuple in the form (sublocation, transect) 
        eg ('Kuala Merisi', 1). transects in this list will be excluded.
        """
        self.x = np.asarray(X, dtype=np.float64)
        self.w = SLC
        self.t = TSC
        if 0 in set(self.t):
            warnings.warn("zero transect value passed to Transect class")                   
        self.mxw = np.ones_like(self.x) * np.nan
        for sloc in set(self.w):
            self.mxw[self.w == sloc] = max(self.x[self.w == sloc])            
        if DTS is not None:
            self.ds = DTS
            self.smx = np.empty_like(self.x)
            self.dx = np.empty_like(self.x)
            self.dds = np.empty_like(self.x)
            self.midds = np.empty_like(self.x)
            self.middx = np.empty_like(self.x)
            self.tnum = np.zeros_like(self.x)
            self.smxt = np.ones_like(self.x) * np.nan
            if not return_flow:
                self.ind = self._Tsort()
            else:
                self.ind = self._Tsort_return_flow()
        if exclude:
            self._exclude_transect(exclude, slKey)
    
    def _Tsort(self):
        """
        gets np.argsort indices to sort data within transects 
        sort order w then t then ds
        """
        tnum = 1
        index = np.argsort(self.w)
        transects = self.t[index]
        slocs = self.w[index]
        for sloc in set(slocs):
            subt = transects.copy()
            ## set everywhere that is not sloc equal to zero
            subt[slocs != sloc] = 0
            ## get argsort for subt at sloc
            ind1 = np.argsort(subt[subt > 0])
            ## sort index at sloc
            index[subt > 0] = index[subt > 0][ind1]
            ## make copies of ds and t sorted by index 
            dists = self.ds[index]
            subt = self.t[index]
            ## again, set everywhere that is not sloc equal to zero
            subt[slocs != sloc] = 0
            ## sort x by index
            self.sx = self.x[index]
            for transect in set(subt[subt > 0]):
                ## isolate each transect in subt
                filtr = subt == transect
                ## find the maximum on the transect
                self.smx[filtr] = nanmax(self.sx[filtr])
                subds = dists.copy()
                ## get the argsort for distance to shore at transect
                ind1 = np.argsort(subds[filtr])
                ## sort index at transect
                index[filtr] = index[filtr][ind1]
                ## apply index to sort other attributes
                self.sx = self.x[index]
                self.sds = self.ds[index]
                ## set unique transect number for this transect
                self.tnum[filtr] = tnum
                tnum += 1
                ## find where the max occurs
                try:
                    nanargmx = nanargmax(self.sx[filtr])
                except ValueError:
                    nanargmx = np.nan
                for ii in range(len(ind1)):
                    ## pointer to element ii on a transect in the sorted array
                    pointer = filtr.nonzero()[0][0] + ii
                    if ii == 0:
                        ## NaNs at the most seaward point on the transect
                        self.dx[pointer] = np.nan                        
                        self.middx[pointer] = np.nan
                        self.dds[pointer] = np.nan
                        self.midds[pointer] = np.nan
                    else:
                        ## calculate intermediate values for all other points
                        self.dx[pointer] = self.sx[pointer] - \
                            self.sx[pointer-1]
                        self.middx[pointer] = (self.sx[pointer] +
                                               self.sx[pointer-1]) / 2
                        self.dds[pointer] = self.sds[pointer] - \
                            self.sds[pointer-1]
                        self.midds[pointer] = self.sds[pointer-1] + \
                            self.dds[pointer] / 2
                    if ii == nanargmx:
                        ## this is where the maximum occurs
                        self.smxt[pointer] = self.sx[pointer]
        ## sort by index        
        self.sx = self.x[index]
        self.sds = self.ds[index]
        self.sw = self.w[index]
        self.st = self.t[index]
        return index
        
    def _Tsort_return_flow(self):
        """
        gets np.argsort indices based off sort of w, t then ds
        
        differs from _Tsort because it calculates intermediate values based on
        offshore flow direction... eg: dx would be change from landward point 
        to seaward point
        
        see _Tsort for more detailed comments on the code
        """
        tnum = 1
        index = np.argsort(self.w)
        transects = self.t[index]
        slocs = self.w[index]
        for sloc in set(slocs):
            subt = transects.copy()
            subt[slocs != sloc] = 0
            ind1 = np.argsort(subt[subt > 0])
            index[subt > 0] = index[subt > 0][ind1]
            dists = self.ds[index]
            subt = self.t[index]
            subt[slocs != sloc] = 0
            self.sx = self.x[index]
            for transect in set(subt[subt > 0]):
                filtr = subt == transect
                self.smx[filtr] = nanmax(self.sx[filtr])
                subds = dists.copy()
                ind1 = np.argsort(subds[filtr])
                index[filtr] = index[filtr][ind1]
                self.sx = self.x[index]
                self.sds = self.ds[index]
                self.tnum[filtr] = tnum
                tnum += 1
                nanargmx = nanargmax(self.sx[filtr])
                for ii in range(len(ind1)):
                    ## pointer to element ii on a transect in the sorted array
                    pointer = filtr.nonzero()[0][0] + ii
                    ## This section is different than in _Tsort
                    ## assumes seaward flow direction
                    if ii == len(ind1)-1:
                        ## NaNs at the most landward point on the transect
                        self.dx[pointer] = np.nan                        
                        self.middx[pointer] = np.nan
                        self.dds[pointer] = np.nan
                        self.midds[pointer] = np.nan
                    else:
                        ## calculate intermediate values everywhere else
                        self.dx[pointer] = self.sx[pointer] - \
                        self.sx[pointer+1]
                        self.middx[pointer] = (self.sx[pointer] + \
                        self.sx[pointer+1]) / 2
                        self.dds[pointer] = self.sds[pointer] - \
                        self.sds[pointer+1]
                        self.midds[pointer] = self.sds[pointer+1] + \
                        self.dds[pointer] / 2                       
                    if ii == nanargmx:
                        self.smxt[pointer] = self.sx[pointer]
        self.sx = self.x[index]
        self.sds = self.ds[index]
        self.sw = self.w[index]
        self.st = self.t[index]
        return index
        
    def _exclude_transect(self, args, slKey):
        """
        remove data at transects with located at args
        args = (sloc, tsc) or sequence of (sloc, tsc)
        where sloc is the string sublocation and TSC is the integer transect 
        number as written in the Excel database
        """
        if isinstance(args, tuple):
            args = [args]
        for arg in args:
            t = arg[1]
            ## reverse dictionary lookup to get code from string
            w = [k for k, v in slKey.items() if v == arg[0]][0]
            ## create filters
            f1 = self.w == w
            f2 = self.t == t
            f3 = np.logical_not(f1 * f2)
            f4 = self.sw == w
            f5 = self.st == t
            f6 = np.logical_not(f4 * f5)
            for attr in dir(self):
                if attr[0] == '_':
                    ## methods start with an _
                    continue
                elif attr in ('t', 'w', 'x', 'ds', 'ind', 'mxw'):
                    ## not transect sorted apply filter f3
                    a = getattr(self, attr)
                    setattr(self, attr, a[f3])
                else:
                    ## transect sorted apply filter f6
                    a = getattr(self, attr)
                    setattr(self, attr, a[f6])

###############################################################################
def locationcharacteristics(Adict, file_name='', by_sublocation=False):
    """
    calculate and write characteristics by location or by sublocation
    fields include min, max, average, and N
    """
    floats = ('Elevation', 'Layers', 'MaxLayers', 'Distance2shore',
              'DepositStart', 'DepositLimit', 'InundationLimit', 'Runup',
              'FlowDepth',  'ProjectedFlowDepth', 'Thickness', 'MaxThickness',
              'MudIntLayers', 'Massive', 'Ngrading', 'Sgrading', 'Igrading')
    binaries = ('MudCap', 'RipUps', 'HeavyMin', 'ShellFragments', 'WholeShell',
                'CoarseClasts', 'OrganicDebris', 'Boulder')
    if by_sublocation:
        ## characteristics by sublocation
        code, key = Adict['SLCode'], Adict['SLKey']
    else:
        ## characteristics by location
        code, key = SLcoder(Adict['Location'])
    if not file_name:
        ## save as box to specify a file name to write to
        file_name = filedialog.asksaveasfilename()
    with open(file_name, 'w') as file:
        cw = csv.writer(file, dialect='excel', lineterminator='\n')
        cw.writerow([''] + [v for _, v in sorted(key.items())])
        ## for each attribute field (decimal)
        for f in floats:
            mn, mx, mean, N = [], [], [], []
            tempcode = None
            ## for each location or sublocation
            if f in ('Elevation', 'Distance2shore'):
                thk = Adict['Thickness'].copy()
                thk[thk == 0] = np.nan
                tempf, tempcode = denan(Adict[f], code, thk, verbose=False)[:2]
                f = 'Deposit' + f
            for k in sorted(key.keys()):
                if tempcode is None:
                    temp = Adict[f][code == k]
                else:
                    temp = tempf[tempcode == k]
                if len(temp) == 0:
                    temp = np.array([np.nan])
                mn.append(nanmin(temp))
                mx.append(nanmax(temp))
                N.append(len(temp[notnan(temp)]))
                if N[-1] > 0:
                    mean.append(np.nansum(temp)/N[-1])
                else:
                    mean.append('nan')
            cw.writerow([f])
            cw.writerow(['Min'] + mn)
            cw.writerow(['Max'] + mx)
            cw.writerow(['Mean'] + mean)
            cw.writerow(['N'] + N)
            cw.writerow([''])
        ## for each binary attribute field
        for b in binaries:
            N = []
            ## for each location or sublocation
            for k in sorted(key.keys()):
                temp = Adict[b][code == k]
                N.append(np.nansum(temp))
            cw.writerow([b])
            cw.writerow(['N'] + N)
            cw.writerow([''])
    print('Location characteristics output written to %s' % file_name)

###############################################################################
def percent_normal_graded(uniformgsfile, tolerance=.1, min_size=None,
                          csv_dir='', tsunami_only=True):
    """
    find the percent of sampling intervals that have normal grading

    uniformgsfile is the name of a uniform format csv file with the path
        configured in TsuDBGSFile

    tolerance is the threshold difference of mean grain sizes (in phi)
        required to consider an interval normal graded

    min_size = None means all sizes are considered, a number sets the highest
        phi value to be allowed

    csv_dir specifies the directory where the uniformgsfile is located

    tsunami_only specifies whether to consider only samples of a tsunami deposit

    returns
        ng: number of normal graded intervals
        ng/len(diffs): proportion normal graded
    """
    gsf = TsuDBGSFile(uniformgsfile, project_directory=csv_dir)
    means = gsf.dist_means(min_size=min_size)
    if tsunami_only:
        means = means[gsf.layer > 0]
    diffs = means[:-1] - means[1:]
    ng = len(diffs[diffs > tolerance])
    return ng, ng/len(diffs)

###############################################################################
## Plotting routine subfunctions...
###############################################################################
def interp_flowdepth_to_thickness(THK, FLD, keep_nans=False):
    """
    POORLY NAMED - works for any Transect data of interest
    
    interpolates between `FLD` observations 
    such that every `THK` observation has a matching `FLD`
    
    `THK` and `FLD` must be objects of the Transect class that were created 
    from the same lists of sublocations, transect numbers, & distances to shore
    
    if `keep_nans` is True, the output arrays will be the same size as the  
    original thickness array
    """
    THKint = []
    FLDint = []
    DTSint = []
    tnumint = []
    ## find correct thickness, flowdepth pairs
    for ii, t in enumerate(THK.sx):
        if not np.isnan(t):
            if not np.isnan(FLD.sx[ii]):
                ## direct matches
                FLDint.append(FLD.sx[ii])
            else:
                ## interpolate to create a match
                filtr1 = FLD.tnum == FLD.tnum[ii]
                filtr2 = np.isfinite(FLD.sx[filtr1])
                fld = FLD.sx[filtr1][filtr2]
                if len(fld) == 0:
                    ## no flow depths on transect
                    if keep_nans:
                        FLDint.append(np.nan)
                        THKint.append(np.nan)
                        DTSint.append(np.nan)
                        tnumint.append(np.nan)
                    continue
                else:
                    dts = FLD.sds[filtr1][filtr2]
                    d2 = THK.sds[ii]
                    ind = dts.searchsorted(d2)
                    if ind == 0 or ind == len(fld):
                        ## thickness measurement is before first flow depth
                        ## thickness measurement is after last flow depth
                        if keep_nans:
                            FLDint.append(np.nan)
                            THKint.append(np.nan)
                            DTSint.append(np.nan)
                            tnumint.append(np.nan)
                        continue
                    else:
                        ## thickness measurement between two flow depths
                        d1 = dts[ind-1]
                        d3 = dts[ind]
                        f1 = fld[ind-1]
                        f3 = fld[ind]
                        ## interpolate
                        f2 = f1+((f3-f1)*(d2-d1)/(d3-d1))
                        FLDint.append(f2)
            ## all thicknesses
            THKint.append(t)
            DTSint.append(THK.sds[ii])
            tnumint.append(THK.tnum[ii])
        elif keep_nans:
            FLDint.append(np.nan)
            THKint.append(np.nan)
            DTSint.append(np.nan)
            tnumint.append(np.nan)
    return THKint, FLDint, DTSint, tnumint
    
###############################################################################
def SLdecoder(codein, key):
    """
    uses dictionary keys to return the items that 'codein' values code for
    """
    if isinstance(codein, (int, float)):
        ## single sublocation to decode
        return key[codein]
    else:
        ## sequence of sublocations to decode
        return [key[code] for code in codein]

###############################################################################        
def getevents(slocs, Adict, return_mw=False):
    """   
    using Adict['event_lookup'] gets the tsunami event associated with 
    each sublocation code in slocs
    returns a list of strings if passed a list else a numpy array...

    
    slocs must be a list or np.ndarray
    
    note that sublocation code does not code to a unique event in cases where
    multiple events have been studied at the same sub location!!!!
    """
    SLCode = Adict['SLCode']
    tsunami = Adict['Tsunami']
    event = []
    if return_mw:
        lookup_key = 'mw_lookup'
    else:
        lookup_key = 'event_lookup'
    for s in slocs:
        t = nanmin(tsunami[SLCode == s])
        try:
            event.append(Adict[lookup_key][t])
        except KeyError:
            event.append('' if not return_mw else np.nan)
    return event
    
###############################################################################
def get_B_from_A(A, Adict, keyA='ID', keyB='GSFileUniform'):
    """
    accepts list A of UNIQUE Adict[keyA] values
    returns a list of the associated Adict[keyB] values
    
    default keyA and keyB will retreive gsfileuniform from associated ids
    
    NOT necessary when using numpy arrays!!!! only for list Adict values
    """
    out = list(range(len(A)))
    to_search = list(Adict[keyA])
    to_retreive = list(Adict[keyB])
    for ii, a in enumerate(A):
        try:
            ind = to_search.index(a)
            out[ii] = to_retreive.pop(ind)
            to_search.pop(ind)
        except ValueError:
            out[ii] = ''
    return out
    
###############################################################################
def get_datestrings(timestamps):
    """
    return a numpy array of date strings converted from timestamps
    """
    out = np.empty(timestamps.shape, dtype='<U10')
    for ii, date in enumerate(timestamps):
        try:
            out[ii] = dt.datetime.fromtimestamp(date).strftime('%Y-%m-%d')
        ## nan encountered
        except OSError:
            out[ii] = 'nan'
        except TypeError:
            out[ii] = 'nan'
    return out
    
###############################################################################
def get_values_on_transect_with_tuple(transect_tuple, Adict, *keys):
    """
    accepts a transect tuple (sloc, tsc) like ('Sendai', 1)
    returns a list of the values associated with that transect at Adict[key] 
    for each key in keys    
    
    sloc can be string or slcode
    """
    sloc, tsc = transect_tuple
    ## if sloc is a string, look up the SLCode value for it
    if isinstance(sloc, str):
        sloc = lookup_SLCode(sloc, Adict['SLKey'])
    ## filters for SLCode and Transect number
    f1 = Adict['SLCode'] == sloc
    f2 = Adict['Transect'] == tsc
    f = f1 * f2
    if not f.any():
        return None
    ## create a list of outs that is the same length as keys
    outs = list(keys)
    for ii, key in enumerate(keys):
        ## get target array or list from Adict
        get_from = Adict[key]
        if isinstance(get_from, np.ndarray):
            ## assign to outs[ii] the values of get_from associated with the 
            ## desired transect by filtering numpy array
            outs[ii] = get_from[f]
        else:
            ## for lists, filter with f manually
            outs[ii] = [x for ii, x in enumerate(get_from) if f[ii]]
    ## if only one object in outs list, return it independently - not in a list
    if len(outs) == 1:
        return outs[0]
    return outs
    
###############################################################################
def get_gsmeans(Adict, return_std_instead=False, gs_min_max=None, mm=False):
    """
    reads in ALL GSFiles and returns an array of bulk means for each trench
    
    gs_min_max is a sequence of length 2 specifying the minimum and maximum
    grain size to be used in the calculations (grain size in phi)
    eg gs_min_max=(4,-1) to specify only sand grains
    """
    gsmeans = np.ones_like(Adict["Thickness"]) * np.nan
    for ii, gs in enumerate(Adict['GSFileUniform']):
        if gs:
            try:
                if return_std_instead:
                    gsmeans[ii] = TsuDBGSFile(gs).bulk_std(gs_min_max=gs_min_max)
                else:
                    gsmeans[ii] = TsuDBGSFile(gs).bulk_mean(gs_min_max=gs_min_max)
            except FileNotFoundError or ValueError as e:
                print(e)
                print('TsuDB.get_gsmeans: Error at', gs)
                continue
    if mm is True:
        gsmeans = 2.**-gsmeans
    return gsmeans
    
###############################################################################
def get_maxsuspensiongradedlayerthickness(Adict):
    """
    reads in all grain size data files and returns an array of thickness of 
    the thickest suspension graded layer in the trench/core
    """
    sglt = np.ones_like(Adict['Thickness']) * np.nan
    for ii, filename in enumerate(Adict['GSFileUniform']):
        if filename and filename in Adict['high_res_gs']:
            try:
                ## create a TsuDBGSFile object for each grain size data file
                gs = TsuDBGSFile(filename)
            except FileNotFoundError as e:
                print(e)
                print('TsuDB.get_maxsuspensiongradedlayerthickness: Error at',
                      gs)
                continue
            temp = gs.thickness_of_layers_in_layer_type(layer_type=1)
            if temp is not None:
                sglt[ii] = max(temp)
    return sglt
    
###############################################################################
def get_flowdepth_for_GS_file(csv_file_names, Adict, verbose=True):
    """
    use a list of csv file names to get a list of projected flow depths for the
    associated trenches
    """
    ## find indices for all gs files
    unique = get_B_from_A(csv_file_names, Adict, 'GSFileUniform', 'unique')
    ## initial values for projected flow depth
    pfd = get_B_from_A(unique, Adict, 'unique', 'ProjectedFlowDepth')
    ## transect sort and interpolate flow depths
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["unique"],
                Adict["ProjectedFlowDepth"],
                n_rounds=3
                )
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    UNI = Transect(out[3], SLC, TSC, DTS)
    FLD = Transect(out[4], SLC, TSC, DTS)
    UNIint, FLDint, DTSint, tnumint = interp_flowdepth_to_thickness(UNI, FLD)
    FLDint = np.asarray(FLDint)
    UNIint = np.asarray(UNIint)
    ## look at flow depth value for each trench
    for ii, ind in enumerate(unique):
        ## no exact flow depth value, must get interpolated value
        if pfd[ii] == '' or np.isnan(pfd[ii]):
            ## file not in database
            if ind == '':
                if verbose:
                    print('File not found: {}'.format(csv_file_name[ii]))
            ## find interpolated value
            else:
                try:
                    pfd[ii] = FLDint[UNIint == ind][0]
                ## no matching flow depth in interpolated values
                except IndexError:
                    pfd[ii] = np.nan
    if verbose:
        pprint([(c, p) for c, p in zip(csv_file_names, pfd)])
    
    return pfd
    
###############################################################################
def lookup_SLCode(string, slKey):
    """
    reverse dictionary lookup to get code from string
    """
    key = [k for k, v in slKey.items() if v == string]
    try:
        return key[0]
    ## no matches found
    except IndexError:
        raise

###############################################################################
def lookup_tnum(slcode, Trnsct, negative_slcodes_where_no_transect=True):
    """
    generator for iterating through all transects at sublocation slcode
    """
    try:
        tnums = Trnsct.tnum[Trnsct.sw == slcode]
    except AttributeError or IndexError:
        raise
    if tnums.size == 0 and negative_slcodes_where_no_transect:
        ## if no transect exists, we can instead yield -1 * slcode to indicate
        ## to the call to lookup_tnum that no transect data exists here
        tnums = {-slcode}
    else:
        tnums = set(tnums)
    for tnum in tnums:
        yield tnum
    
###############################################################################
def figsaver(fig, save_fig, fig_title='untitled', overwrite=False, dpi=450,
             transparent=False):
    """
    save figure "fig"
    
    if save_fig is "True" the program will open a saveas window allowing the 
    user to specify a filename, type, and directory
    
    if save_fig is a string it must specify a file format supported by
    matplotlib. eg: save_fig='jpg'
    this will initiate autosave which will create a new directory in the cwd
    and save the figure automatically using the string fig_title
    
    if overwrite is True the function will allow old figures with the same name
    to be overwritten
    """
    if isinstance(save_fig, str):
        meow = dt.datetime.strftime(dt.datetime.today(), '%Y-%m-%d')
        directory = os.path.join(os.getcwd(), 'TsuDB_figures_on_{}'.format(meow))
        if not os.path.exists(directory):
            print('Making directory', directory)
            os.mkdir(directory)
        fname = fig_title.replace(" ", "_")+'.'+save_fig
        path = os.path.join(directory, fname)
        if not overwrite and os.path.isfile(path):
            path = filedialog.asksaveasfilename()
            if not path:
                print('\nsavefig canceled\n')
                return
        try:
            plt.savefig(path, dpi=dpi, transparent=transparent)
        except ValueError:
            print('\nsavefig canceled... ValueError...\n')
            print('\nInvalid file extension string. Try like save_fig="jpg"\n')
            return
    else:
        path = filedialog.asksaveasfilename()
        if not path:
            print('\nsavefig canceled\n')
            return
        plt.savefig(path, dpi=dpi, transparent=transparent)
    print('Saved figure "' + fig_title + '" as ' + path)
    
###############################################################################
def average_in_bins(x, y, block=10, percents=True):
    """
    average `y` over every `block` percent of `x`
    `block` must be the result of 100 evenly divided by any positive integer.
    returns two numpy arrays of equal length, the mid points of `block` spaced
    bins of `x` 
    and average values of `y` over the blocks
    """
    min_x = nanmin(x)
    max_x = nanmax(x)
    if not percents:
        ## created block spaced values from min_x to max_x
        xp = np.arange(np.floor(min_x), np.ceil(max_x), block)
    else:
        ## create block percent spaced values from 0 to 100 percent of max_x
        xp = np.asarray([max_x * p / 100 for p in np.arange(0., 100., block)])
        ## change block to be an increment of x, not a percent
        block = xp[1] - xp[0]
    yp = np.zeros_like(xp)
    for ii, p in enumerate(xp):
        ## set up filters
        if ii == 0:
            ## include lower bound in first bin
            f1 = x >= p
        else:
            f1 = x > p
        f2 = x <= p + block
        filtr = f1 * f2
        ## get mean in filtered part of y
        yp[ii] = nanmean(y[filtr])
    ## midpoint of bins
    xp = xp + (block/2)
    return xp, yp

###############################################################################
def percentile_by_bin(bin_data, perc_data, percentile=50, bin_size=1):
    """
    calculate the `percentile` percentile value of `bin_data` in each 
    `bin_size` bin of `bin_data`
    returns `bins`: the centers of `bin_data` in `bin_size` size bins, and 
    `percs`: the percentile values for each bin
    """
    bins = np.arange(min(bin_data), max(bin_data), bin_size)
    percs = np.ones_like(bins) * np.nan
    for ii, b in enumerate(bins):
        if ii == 0:
            f1 = bin_data >= b
        else:
            f1 = bin_data > b
        f2 = bin_data <= b + bin_size
        try:
            percs[ii] = np.percentile(perc_data[f1 * f2], percentile)
        except ValueError:
            ## no data in a bin...
            percs[ii] = np.nan
    bins = bins + (bin_size/2)
    ## remove nans from result
    filtr = notnan(percs)
    bins = bins[filtr]
    percs = percs[filtr]
    return bins, percs

###############################################################################
def filter_outliers(x, n_sigma=3):
    """
    create a filter that is false for all values from numpy array x that 
    are > n_sigma standard deviations from the mean
    """
    return abs(x) < n_sigma*x.std() + x.mean()
    
###############################################################################
def last4(labs):
    """
    return the last 4 characters of each element in a sequence of strings
    """
    return [L[-4:] for L in labs]
    
###############################################################################
def sort_legend_labels(hands, labs, func=last4):
    """
    sort legend handles and labels
    func can be used to specify what to sort on
    func must take labs as an argument and returns a np array of same length
    """        
    hands = np.asarray(hands)
    labs = np.asarray(labs)
    ## sort based on the results of a function of labs
    if func:
        sorton = func(labs)
    ## sort based on labs
    else:
        sorton = labs
    ## get argsort of sorton
    ind = np.argsort(sorton)
    return list(hands[ind]), list(labs[ind])

###############################################################################
def distance_to_source(Adict, lats, lons, events, sources=None, 
                       Re=6373.):
    """
    calculate the distance between points with latitudes `lats` and 
    longitudes `lons` and sources at `events` 
    if `sources` is None, uses the sources hardcoded into Adict
    `Re` is the radius of the sphere (radius of earth in km by default)
    """
    if sources is None:
        sources = Adict['sources']
    names = []
    source_lat = np.zeros(len(sources))
    source_lon = source_lat.copy()
    ## iterate on sources dictionary and unpack individual variables from 
    ## dictionary value tuples
    for ii, (k, (s_lat, s_lon, _)) in enumerate(sources.items()):
        names.append(k)
        source_lat[ii] = s_lat
        source_lon[ii] = s_lon
    names = np.asarray(names)
    distances = np.zeros_like(lats)
    ## calculate distances to all lats associated with each event
    for ii, e in enumerate(events):
        ## set up lat lon variables to pass as arguments to distance calc
        lat1 = lats[ii]
        lon1 = lons[ii]
        f = names == e
        lat2 = source_lat[f]
        lon2 = source_lon[f]
        ## call distance calc function and multiply by radius of earth
        distances[ii] = Re * distance_on_unit_sphere(lat1, lon1, lat2, lon2)
    return distances
    
###############################################################################
def axint(ticklabel, position):
    """
    for use with matplotlib.ticker.FuncFormatter
    """
    return int(ticklabel)
    
###############################################################################
def axround(ticklabel, position):
    """
    for use with matplotlib.ticker.FuncFormatter
    """
    return round(ticklabel, 1)
        
###############################################################################
def insetmap(fig, lat, long, full_globe=False, lbwh=[0.66, 0.70, .23, .23], 
              map_style=1, zorder=2, mfc='red', resolution='l', frame_on=True,
              parallels_meridians=None, sources=None):
    """
    overlays a map showing the points in "lat" and "long" on figure 'fig'
    
    lat and long are vectors of latitude and longitude, the points to plot
    full_globe set to "True" plots the whole world, "False" plots only the area
    of the world around your lat long points
    lbwh is a list of length 4 specifying the location of the left, bottom,
    width, and height of the inset axis. the values are specified as fractions
    of the total size of the figure
    set map_style = 2 for an etopo map background
    set parallels_meridians to a integer number of degrees to draw parallels 
    and meridians on the map
    """
    ## check if you have Basemap
    try:
        check = Basemap
    except NameError:
        return fig
    ## add axis to figure to hold inset mad
    inset = fig.add_axes(lbwh, zorder=zorder, frame_on=frame_on)
    ## calculate map corners to frame data nicely
    if not full_globe:
        mnlt, mnlg, mxlt, mxlg = min(lat), min(long), max(lat), max(long)
        pad = min(10, 90-mxlt, 90+mnlt)
        lllat = min(mnlt-pad, mnlt+pad)
        urlat = max(mxlt-pad, mxlt+pad)
        if mxlg - mnlg > 180 and mnlg < 0 and mxlg > 0:
            long[long < 0] += 360
            mxlg, mnlg = min(long), max(long)
            urlong = mnlg + pad
            lllong = mxlg - pad
        else:
            lllong = min(mnlg-pad, mnlg+pad)
            urlong = max(mxlg-pad, mxlg+pad)
        ## initialize basemap
        m = Basemap(resolution=resolution, area_thresh=10, llcrnrlat=lllat, 
                    llcrnrlon=lllong, urcrnrlat=urlat, urcrnrlon=urlong)
    ## full globe map
    else:
        ## initialize basemap
        m = Basemap(resolution=resolution, area_thresh=10, lat_0=0, lon_0=160)
        long[long < 0] += 360
    ## convert lat and long to map coordinates
    y, x = m(lat, long)
    ## set map style
    if map_style == 1:
        m.drawcoastlines(zorder=zorder+1)
        m.fillcontinents('gray', zorder=zorder)
    else:
        m.etopo(zorder=zorder)
    if parallels_meridians:
        m.drawmeridians(np.arange(-180,360, parallels_meridians), 
                        zorder=zorder+1, labels=[0,0,0,1])
        m.drawparallels(np.arange(-80, 90, parallels_meridians), 
                        zorder=zorder+1, labels=[1,0,0,0])
    inset.frame_on = frame_on
    ## plot data on map
    m.plot(x, y, marker='o', mec='k', mfc=mfc, zorder=zorder+2, markersize=6, 
           linestyle='None')
    ## annotate map with sources
    if sources:
        names = []
        source_lat = np.zeros(len(sources))
        source_lon = source_lat.copy()
        ## find locations of sources
        for ii, (k, (s_lat, s_lon, _)) in enumerate(sources.items()):
            names.append(k)
            source_lat[ii] = s_lat
            source_lon [ii] = s_lon
        source_lon[source_lon < 0] += 360
        source_y, source_x = m(source_lat, source_lon)
        ## customize positions for labeling so it looks good
        for ii, s in enumerate(names):
            arrow = None
            up = 0
            if s in ('Peru, 2001', 'Chile, 2010'):
                rl = -1
                ha = 'right'
            elif s in ('Indian Ocean, 2004'):
                rl = 0
                up = -3
                ha = 'center'
            elif s in ('Sumatra, 2005'):
                rl = 0
                up = -4
                ha = 'center'
            else:
                rl = 1
                ha = 'left'
            plt.annotate(s, (source_x[ii], source_y[ii]), xytext=(15*rl,25*up), 
                         textcoords='offset points', horizontalalignment=ha,
                         arrowprops=dict(arrowstyle='-', color='w'), color='w')
    return fig
    
###############################################################################
## Plotting routines    
###############################################################################    
def data_globe(Adict, save_fig=False, attribute="Modern", label_tsunamis=True,
               fig_title='Global distribution of tsunami deposit data'):
    """
    plot location data for any Adict key 'attribute' on world map
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["Lat"], 
                Adict["Long"], 
                Adict[attribute]
                )
    out = runfilters(out, 1)
    ## label tsunami sources on the map
    if label_tsunamis:
        sources = Adict['sources']
    else:
        sources = None
    fig = plt.figure(figsize=(16,10))
    ## call insetmap to add axes to the figure and plot the data
    fig = insetmap(fig, out[0], out[1], full_globe=False, lbwh=[.05,.05,.9,.9], 
                   map_style=2, parallels_meridians=40, 
                   sources=sources)
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')                
    return fig     
    
###############################################################################
def localslope_thickness(Adict, save_fig=False, agu_print=True, annotate=True,
                         lin_regress=False, fig_title=' Slope vs Thickness'):
    """
    plot data from Adict- slope vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["Elevation"], 
                Adict["Modern"],
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)
    ELV = Transect(out[5], SLC, TSC, DTS)
    filtr = np.isfinite(THK.dx)
    d_thk = THK.dx[filtr]
    m = ELV.dx[filtr] / ELV.dds[filtr]
    cm = ELV.sx / ELV.sds
    if agu_print:
        fig = plt.figure(figsize=(14,12))
        ax = plt.subplot(111)
        plt.scatter(m, d_thk)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel('Local Slope (m/m)')
        plt.ylabel('Change in Deposit Thickness (cm)')
        if annotate:
            fs = 32
            n = len(m[np.isfinite(m)])
            m_p = m > 0
            m_n = m < 0
            dt_p = d_thk > 0
            dt_n = d_thk < 0 
            I = len(m[m_p * dt_p]) * 100 / n
            IV = len(m[m_p * dt_n]) * 100 / n
            III = len(m[m_n * dt_n]) * 100 / n
            II = len(m[m_n * dt_p]) * 100 / n
            plt.text(.95, .95, 'I: {:.0f} %'.format(I), fontsize=fs,
                     ha='right', va='bottom', transform=ax.transAxes)
            plt.text(.95, .05, 'IV: {:.0f} %'.format(IV), fontsize=fs,
                     ha='right', va='top', transform=ax.transAxes)
            plt.text(.05, .05, 'III: {:.0f} %'.format(III), fontsize=fs,
                     ha='left', va='top', transform=ax.transAxes)
            plt.text(.05, .95, 'II: {:.0f} %'.format(II), fontsize=fs,
                     ha='left', va='bottom', transform=ax.transAxes)                     
        elif lin_regress:
            M, B, R = linregress(m[np.isfinite(m)], 
                                 d_thk[np.isfinite(m)])[:3]
            ## little m is topographic local slope
            line = m*M + B
            plt.plot(m, line, 'k-')
            plt.text(.05, .95, r'$\mathdefault{R^2}$ = %.03f' % R**2, fontsize=14,
                     transform=ax.transAxes, color='k', ha='left', va='bottom') 
    else:
        fig = plt.figure(figsize=(20, 6))
        plt.subplot(133)
        plt.scatter(cm, THK.sx)
        plt.axis(ymin=0)
        plt.axvline(0, color='black')
        plt.xlabel('Cumulative transect slope')
        plt.ylabel('Deposit Thickness (cm)')
        plt.title('Cumulative Transect Slope vs Thickness of Deposit')
        plt.subplot(132)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.scatter(m, d_thk)
        plt.xlabel('Local slope')
        plt.ylabel('Change in thickness (cm)')
        plt.title('Local Slope vs Change in Thickness of Deposit')
        plt.subplot(131)
        plt.axvline(0, color='black')
        plt.scatter(m, THK.sx[filtr])
        plt.axis(ymin=0)
        plt.xlabel('Local slope')
        plt.ylabel('Deposit Thickness (cm)')
        plt.title('Local Slope vs Thickness of Deposit')
        plt.tight_layout()    
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def elevation_thickness(Adict, save_fig=False, agu_print=False, annotate=True,
                         lin_regress=False, exclude=True,
                         fig_title=' Elevation vs Thickness'):
    """
    plot data from Adict- elevation vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    if exclude is True:
        exclude = Adict['incomplete_transect']
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["Elevation"], 
                Adict["Modern"],
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS, exclude=exclude, 
                   slKey=Adict["SLKey"])
    ELV = Transect(out[5], SLC, TSC, DTS, exclude=exclude, 
                   slKey=Adict["SLKey"])
    filtr = np.isfinite(THK.dx)
    d_thk = THK.dx[filtr]
    d_elv = ELV.dx[filtr]
    if agu_print:
        fig = plt.figure(figsize=(14,12))
        ax = plt.subplot(111)
        plt.scatter(d_elv, d_thk)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel('Change in Elevation (m)')
        plt.ylabel('Change in Deposit Thickness (cm)')
        if annotate:
            fs = 32
            n = len(d_elv[np.isfinite(d_elv)])
            de_p = d_elv > 0
            de_n = d_elv < 0
            dt_p = d_thk > 0
            dt_n = d_thk < 0 
            I = len(d_elv[de_p * dt_p]) * 100 / n
            IV = len(d_elv[de_p * dt_n]) * 100 / n
            III = len(d_elv[de_n * dt_n]) * 100 / n
            II = len(d_elv[de_n * dt_p]) * 100 / n
            plt.text(.95, .95, 'I: {:.0f} %'.format(I), fontsize=fs,
                     ha='right', va='bottom', transform=ax.transAxes)
            plt.text(.95, .05, 'IV: {:.0f} %'.format(IV), fontsize=fs,
                     ha='right', va='top', transform=ax.transAxes)
            plt.text(.05, .05, 'III: {:.0f} %'.format(III), fontsize=fs,
                     ha='left', va='top', transform=ax.transAxes)
            plt.text(.05, .95, 'II: {:.0f} %'.format(II), fontsize=fs,
                     ha='left', va='bottom', transform=ax.transAxes)                     
    else:
        fig = plt.figure(figsize=(20, 6))
        plt.subplot(133)
        plt.scatter(ELV.sx, THK.sx)
        plt.axis(ymin=0)
        plt.axvline(0, color='black')
        plt.xlabel('Elevation (m)')
        plt.ylabel('Deposit Thickness (cm)')
        plt.title('Elevation vs Thickness of Deposit')
        plt.subplot(132)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.scatter(d_elv, d_thk)
        plt.xlabel('Change in Elevation (m)')
        plt.ylabel('Change in Thickness (cm)')
        plt.title('Change in Elevation vs Change in Thickness of Deposit')
        plt.subplot(131)
        plt.axvline(0, color='black')
        plt.scatter(d_elv, THK.sx[filtr])
        plt.axis(ymin=0)
        plt.xlabel('Change in Elevation (m)')
        plt.ylabel('Deposit Thickness (cm)')
        plt.title('Change in Elevation vs Thickness of Deposit')
        plt.tight_layout()    
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def elevation_meangs(Adict, save_fig=False, agu_print=False, annotate=True,
                         lin_regress=False, plot_std_instead=False, 
                         exclude=True, sand_only=False,
                         fig_title=' Elevation vs Mean Grain Size'):
    """
    plot data from Adict- elevation vs mean grain size
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    if plot_std_instead and fig_title:
        fig_title = 'Standard Deviation in Grain Size vs Thickness'
    print('Running plotting routine:', fig_title)
    if exclude is True:
        exclude = Adict['incomplete_transect']
    if sand_only:
        # specify min and max grain size to use in gs mean calculation
        gs_min_max = (4, -1)
    else:
        gs_min_max = None
    gsmeans = get_gsmeans(Adict, gs_min_max=gs_min_max, 
                          return_std_instead=plot_std_instead)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                gsmeans,
                Adict["Elevation"], 
                Adict["Modern"],
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    MGS = Transect(out[3], SLC, TSC, DTS, exclude=exclude, slKey=Adict["SLKey"])
    ELV = Transect(out[4], SLC, TSC, DTS, exclude=exclude, slKey=Adict["SLKey"])
    filtr = np.isfinite(MGS.dx)
    d_mgs = MGS.dx[filtr]
    d_elv = ELV.dx[filtr]
    if agu_print:
        fig = plt.figure(figsize=(14,12))
        ax = plt.subplot(111)
        plt.scatter(d_elv, d_mgs)
        ax.invert_yaxis()
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.xlabel('Change in Elevation (m)')
        plt.ylabel('Change in Mean Grain Size (phi)')
        if annotate:
            fs = 32
            n = len(d_elv[np.isfinite(d_elv)])
            de_p = d_elv > 0
            de_n = d_elv < 0
            dg_p = d_mgs > 0
            dg_n = d_mgs < 0 
            I = len(d_elv[de_p * dg_p]) * 100 / n
            IV = len(d_elv[de_p * dg_n]) * 100 / n
            III = len(d_elv[de_n * dg_n]) * 100 / n
            II = len(d_elv[de_n * dg_p]) * 100 / n
            plt.text(.95, .95, 'I: {:.0f} %'.format(I), fontsize=fs,
                     ha='right', va='bottom', transform=ax.transAxes)
            plt.text(.95, .05, 'IV: {:.0f} %'.format(IV), fontsize=fs,
                     ha='right', va='top', transform=ax.transAxes)
            plt.text(.05, .05, 'III: {:.0f} %'.format(III), fontsize=fs,
                     ha='left', va='top', transform=ax.transAxes)
            plt.text(.05, .95, 'II: {:.0f} %'.format(II), fontsize=fs,
                     ha='left', va='bottom', transform=ax.transAxes)                     
    else:
        fig = plt.figure(figsize=(20, 6))
        ax = plt.subplot(133)
        plt.scatter(ELV.sx, MGS.sx)
        ax.invert_yaxis()
        plt.axis(ymin=0)
        plt.axvline(0, color='black')
        plt.xlabel('Elevation (m)')
        plt.ylabel('Mean Grain Size (phi)')
        plt.title('Elevation vs Mean Grain Size of Deposit')
        ax = plt.subplot(132)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.scatter(d_elv, d_mgs)
        ax.invert_yaxis()
        plt.xlabel('Change in Elevation (m)')
        plt.ylabel('Change in Mean Grain Size (phi)')
        plt.title('Change in Elevation vs Change in Mean Grain Size of Deposit')
        ax = plt.subplot(131)
        plt.axvline(0, color='black')
        plt.scatter(d_elv, MGS.sx[filtr])
        ax.invert_yaxis()
        plt.axis(ymin=0)
        plt.xlabel('Change in Elevation (m)')
        plt.ylabel('Mean Grain Size (phi)')
        plt.title('Change in Elevation vs Mean Grain Size of Deposit')
        plt.tight_layout()    
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def toposlope_depositslope(Adict, save_fig=False, lin_regress=True, 
                             remove_outliers=3,
                    fig_title=' Topographic Slope vs Deposit Thickness Slope'):
    """
    plot data from Adict- slope vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    
    remove_outliers value is the number of standard deviations from the mean 
    to keep, if 0 no outliers are removed
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["Elevation"], 
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)
    ELV = Transect(out[5], SLC, TSC, DTS)
    m_t = ELV.dx / ELV.dds
    m_d = THK.dx * .01 / THK.dds
    m_t, m_d = m_t[np.isfinite(m_t)], m_d[np.isfinite(m_d)]
    if remove_outliers:
        f1 = filter_outliers(m_t, remove_outliers)
        f2 = filter_outliers(m_d, remove_outliers)
        m_t = m_t[f1 * f2]
        m_d = m_d[f1 * f2]
    if lin_regress:
        m, b, r = linregress(m_t, m_d)[:3]
        line = m*m_t + b
    fig = plt.figure(figsize=(14,12))
    ax = plt.subplot(111)
    plt.scatter(m_t, m_d)
    if lin_regress:
        plt.plot(m_t, line, 'k')
        plt.text(.05, .95, r'$\mathdefault{R^2}$ = %.03f' % r**2, fontsize=14,
                 transform=ax.transAxes, color='k', ha='left', va='bottom') 
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title(fig_title)
    plt.xlabel('Topographic slope (m/m)')
    plt.ylabel('Deposit thickness slope (m/m)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def flowdepth_slope(Adict, save_fig=False, 
                     fig_title='Slope vs Flow Depth'):
    """
    plot data from Adict- slope vs flow depth
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)    
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["FlowDepth"], 
                Adict["Elevation"], 
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = Transect(out[2], SLC, TSC, out[2])
    FLD = Transect(out[3], SLC, TSC, out[2])    
    ELV = Transect(out[4], SLC, TSC, out[2])
    M = ELV.sx / DTS.smxt
    cm = ELV.sx / ELV.sds
    m = ELV.dx / ELV.dds
    fig = plt.figure(figsize=(20, 6))
    plt.subplot(133)    
    plt.scatter(M, FLD.smx)
    plt.xlabel('Transect slope')
    plt.ylabel('Maximum flow depth (m)')
    plt.title('Transect Slope vs Maximum Flow Depth')
    plt.axvline(0, color='black')
    plt.axis(ymin=0)
    plt.subplot(132)
    plt.scatter(cm, FLD.sx)
    plt.xlabel('Cumulative transect slope')
    plt.ylabel('Flow depth (m)')
    plt.axvline(0, color='black')
    plt.axis(ymin=0)
    plt.title('Cumulative Transect Slope vs Flow Depth')
    plt.subplot(131)    
    plt.axvline(0, color='black')
    plt.scatter(m, FLD.sx)
    plt.axis(ymin=0)
    plt.xlabel('Local slope')
    plt.ylabel('Flow depth (m)')
    plt.title('Local Slope vs Flow Depth')
    plt.tight_layout()
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################
def sedimentconcentration_histogram(Adict, save_fig=False, 
                                fig_title='Sediment Concentration Histogram'):
    """
    plot data from Adict- histogram of sediment concentrations
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["ProjectedFlowDepth"], 
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)    
    FLD = Transect(out[5], SLC, TSC, DTS)
    THKint, FLDint = interp_flowdepth_to_thickness(THK, FLD)[:2]
    FLDint = np.asarray(FLDint)
    THKint = np.asarray(THKint)
    conc = THKint/FLDint
    weights = (100/len(conc)) * np.ones_like(conc)
    g_avg = gmean(conc[conc > 0])
    a_avg = np.mean(conc)
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.hist(conc, bins=25, weights=weights)
    plt.text(.95, .92, 'Geometric Mean = {:.2}'.format(g_avg), ha='right',
             transform=ax.transAxes)
    plt.text(.95, .85, 'Arithmetic Mean = {:.2}'.format(a_avg), ha='right',
             transform=ax.transAxes)
    plt.ylabel('Frequency (%)')
    plt.xlabel('Sediment Concentration (%)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')   
    return fig
    
###############################################################################
def sedimentconcentration_distance(Adict, save_fig=False, 
                      fig_title='Sediment Concentration vs Distance to Shore'):
    """
    plot data from Adict- sediment concentration vs distance from shore
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["ProjectedFlowDepth"], 
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)    
    FLD = Transect(out[5], SLC, TSC, DTS)
    THKint, FLDint, DTSint, TNUMint = interp_flowdepth_to_thickness(THK, FLD)
    FLDint = np.asarray(FLDint)
    THKint = np.asarray(THKint)
    conc = THKint / FLDint
    f1 = conc > 0
    DTSint = np.asarray(DTSint)
    m, b, r = linregress(DTSint[f1], conc[f1])[:3]
    r2 = round(r**2, 2)
    line = m*DTSint[f1] + b
    labs, hands = [], []
    FLD_tnum = list(FLD.tnum)
    sloc = [FLD.sw[FLD_tnum.index(t)] for t in TNUMint]
    event = np.asarray(getevents(sloc, Adict))
    emap = Adict['emap']
    fig = plt.figure()
    ax = plt.subplot(111)
    for e in set(event):
        if e:
            p, = plt.plot(DTSint[event == e], conc[event == e], emap[e], ms=12)
            if e not in labs:
                labs.append(e)
                hands.append(p)
    ind = np.argsort(DTSint[f1])
    plt.plot(DTSint[f1][ind], line[ind], 'k-')
    plt.text(.98, .98, r'$\mathdefault{R^2 =}$ '+str(r2), fontsize=14, 
             transform=ax.transAxes, ha='right', va='top')
    plt.ylabel('Sediment Concentration (%)')
    plt.xlabel('Distance to Shore (m)')
    plt.legend(hands, labs, numpoints=1, loc=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')   
    return fig  
    
###############################################################################
def flowdepth_thickness_semilog(Adict, save_fig=False, japan_only=False,
                                agu_print=True, lines=True,
                                fig_title='Flow Depth vs Thickness Semi Log'):
    """
    plot data from Adict- flow depth vs thickness on a semilog plot
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["ProjectedFlowDepth"], 
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)    
    FLD = Transect(out[5], SLC, TSC, DTS)
    THKint, FLDint, _, tnumint = interp_flowdepth_to_thickness(THK, FLD)
    FLDint = np.asarray(FLDint)
    THKint = np.asarray(THKint)
    tnumint = np.asarray(tnumint)
    if japan_only:
        japan_sloc = nanmin([k for (k, v) in Adict['SLKey'].items() if v == 'Sendai'])
        japan_tnum = nanmin(FLD.tnum[FLD.sw == japan_sloc])
        filtr = tnumint == japan_tnum
        THKint = THKint[filtr]
        FLDint = FLDint[filtr]
        tnumint = tnumint[filtr]
    w = np.sort(FLDint)
    labs, hands = [], []
    FLD_tnum = list(FLD.tnum)
    sloc = [FLD.sw[FLD_tnum.index(t)] for t in tnumint]
    event = np.asarray(getevents(sloc, Adict))
    emap = Adict['emap']
    fig = plt.figure(figsize=(19, 10))
    ax = plt.subplot(111)
    for e in set(event):
        if e:
            p, = plt.plot(FLDint[event == e], THKint[event == e], emap[e], ms=12)
            if e not in labs:
                labs.append(e)
                hands.append(p)
    hands, labs = sort_legend_labels(hands, labs)
    ## plot a bunch of regression lines
    if not agu_print:    
        ## linear regression on all data
        m, b, r = linregress(FLDint, THKint)[:3]
        r2 = round(r**2, 2)
        line1 = m*w + b
        p, = plt.plot(w, line1, '-', color='Teal', lw=2)
        print('Linear Regression: y = {m:.2}*x + {b:.2}'.format(m=m, b=b))
        labs.append('Linear Regression: r^2 = {}'.format(r2))
        hands.append(p)
        ## function to use with scipy.optimize.curve_fit
        def func(x, a, b):
            return a * x**(3/2) + b 
        ## exponential curve fit (x^(3/2))
        popt, pcov = curve_fit(func, FLDint, THKint)
        line4 = func(w, *popt)
        p, = plt.plot(w, line4, '-', color='Orange', lw=2)
        hands.append(p)
        labs.append('Exponential: y = {0:.2}*x^(3/2) + {1:.2}'.format(*popt))
        print('Exponential Curve: y = {0:.2}*x^(3/2) + {1:.2}'.format(*popt))
        ## percentile curve fits
        for ls, perc in [('--', 10), ('-', 50), ('-.', 90)]:
            label = '{}th Percentile Linear Regression'.format(perc)
            bins, percs = percentile_by_bin(FLDint, THKint, percentile=perc, 
                                            bin_size=1)
            m_, b_, r_ = linregress(bins, percs)[:3]
            line = m_*bins + b_
            p, = plt.plot(bins, line, color='k', lw=2, ls=ls)
            hands.append(p)
            labs.append('{0} ({1:.2})'.format(label, r_**2))
            print('{label}: y = {m:.2}*x + {b:.2}'.format(label=label, m=m_, b=b_))
        ## Goto, 2014 relationship
        line2 = 1.9*w + .29
        p, = plt.plot(w, line2, 'r-', lw=2)
        labs.append('Goto, 2014 Median Linear Regression (2% line)')
        hands.append(p)
        ## Takashimizu, 2012 relationship
        line3 = -.189*w**3 + .211*w**2 + 4.46*w - .00826
        p, = plt.plot(w, line3, 'g-', lw=2)
        labs.append('Takashimizu, 2012 Polynomial Regression')
        hands.append(p)
    elif lines:
        ## Goto, 2014 relationship
        line2 = 2.8*w + 4.4
        p, = plt.plot(w, line2, color='Orange', lw=2)
        labs.append('90th Percentile Linear Regression: Goto et al., 2014')
        hands.append(p)
        ## 90th percentile curve
        perc = 90
        label = '{}th Percentile Linear Regression'.format(perc)
        bins, percs = percentile_by_bin(FLDint, THKint, percentile=perc, 
                                        bin_size=1)
        m_, b_, r_ = linregress(bins, percs)[:3]
        bins = np.hstack((np.asarray([FLDint.min()]), bins))
        line = m_*bins + b_
        p, = plt.plot(bins, line, color='k', lw=2)
        hands.append(p)
        labs.append(label + ': Data')#+ r': $\mathdefault{R^2 =}$ ' + str(round(r_**2,2)))
        print('{label}: y = {m:.2}*x + {b:.2}'.format(label=label, m=m_, b=b_))
        print('R^2 = ', str(round(r_**2, 2)))
    ## format axes
    ax.set_xscale('log')
    ax.set_ylim(bottom=0, top=100) 
#    plt.text(.98, .98, r'$\mathdefault{R^2 =}$ '+str(r2), fontsize=14, 
#             transform=ax.transAxes, ha='right', va='top')
    plt.xlabel('Flow Depth (m)')
    plt.ylabel('Deposit Thickness (cm)')
    plt.legend(hands, labs, numpoints=1, loc=2, frameon=False, fontsize=22)
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')   
    return fig
    
###############################################################################
def flowdepth_thickness(Adict, save_fig=False, poly_fit=None, exclude=None, 
                          fig_title='Flow Depth vs Thickness', 
                          annotate=False, agu_print=True, 
                          block_average=None):
    """
    plot data from Adict- flow depth vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    :rtype : matplotlib.figure.Figure
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["ProjectedFlowDepth"], 
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)    
    FLD = Transect(out[5], SLC, TSC, DTS)
    THKint, FLDint, DTSint, tnumint = interp_flowdepth_to_thickness(THK, FLD)
    FLDint = np.asarray(FLDint)
    THKint = np.asarray(THKint)
    DTSint = np.asarray(DTSint)   
    m_, b_, r_ = linregress(FLDint, THKint)[:3]
    r2_ = round(r_**2, 2)
    line3_ = m_*FLDint + b_
    if agu_print:
        labs, hands = [], []
        sloc = [FLD.sw[FLD.tnum == t][0] for t in tnumint]
        event = np.asarray(getevents(sloc, Adict))
        emap = Adict['emap']
        fig = plt.figure(figsize=(16, 12))
        ax = plt.subplot(111)
        for e in set(event):
            if e:
                p, = plt.plot(FLDint[event == e], THKint[event == e], emap[e], 
                              ms=12)
                if e not in labs:
                    labs.append(e)
                    hands.append(p)
        if poly_fit is None:
            plt.plot(FLDint, line3_, 'k-', zorder=-1)        
            plt.text(.98, .98, r'$\mathdefault{R^2 =}$ '+str(r2_), fontsize=14, 
                 transform=ax.transAxes, ha='right', va='top')
        else:
            ## TODO make this go to the 3/2 power with curve_fit (see *semilog)
            ind = np.argsort(FLDint)
            P = np.polyfit(FLDint[ind], THKint[ind], poly_fit)
            F = np.poly1d(P)
            xp = np.linspace(FLDint.min(), FLDint.max(), 500)
            plt.plot(xp, F(xp), 'k-')
        if block_average is not None:
            japan_sloc = nanmin([k for (k, v) in Adict['SLKey'].items() if v == 'Sendai'])
            f1 = sloc == japan_sloc
            f2 = sloc != japan_sloc
            ## calculate average for each block_average sized block
            japan_bin_fld, japan_av_thk = average_in_bins(FLDint[f1], 
                                                          THKint[f1],
                                                          percents=False, 
                                                          block=block_average)
            all_bin_fld, all_av_thk = average_in_bins(FLDint[f2], 
                                                      THKint[f2],
                                                      percents=False,
                                                      block=block_average)
            p1, p2 = plt.plot(japan_bin_fld, japan_av_thk, 'c*', all_bin_fld, 
                              all_av_thk, 'w*', ms=25, mew=2)
            b = int(block_average)
            hands += [p1, p2]
            labs += ['Averages of {} m Flow Depth Bins: Japan Only'.format(b),
                     'Averages of {} m Flow Depth Bins: All Excluding Japan'.format(b)]
        ax.tick_params(axis='both', which='major', labelsize=18)        
        hands, labs = sort_legend_labels(hands, labs)
        plt.legend(hands, labs, numpoints=1, frameon=False, loc=2)
        plt.xlabel('Flow Depth (m)', fontsize=18)
        plt.ylabel('Deposit Thickness (cm)', fontsize=18)
        plt.axis(xmax=20)
    else:
        flowmx, thickmx = denan(FLD.smx, THK.smxt)
        m, b, r = linregress(flowmx, thickmx)[:3]
        r2 = round(r**2, 2)
        line1 = m*FLD.smx + b
        Fsx, Tsx, Dsx = denan(FLD.sx, THK.sx, THK.sds)
        M, B, R = linregress(Fsx, Tsx)[:3]
        R2 = round(R**2, 2)
        LINE2 = M*Fsx + B
        cmap = 'RdBu_r'
        vmin, vmax = 0, 1000
        fig = plt.figure(figsize=(23, 6)) #(18, 8) for 2 panels
        cbax = fig.add_axes([.925, .2, .01, .6])
        ax1 = plt.subplot(131)
        plt.scatter(FLD.smx, THK.sx, c=THK.sds, s=30, cmap=cmap, vmin=vmin, 
                    vmax=vmax) 
        plt.scatter(flowmx, thickmx, c='none', s=60, edgecolors='k', 
                    linewidths=3, zorder=10)
        if annotate:
            for X, Y in zip(FLD.smx, THK.smxt):
                ax1.annotate('{}'.format(Y), xy=(X,Y), ha='right')
        plt.plot(FLD.smx, line1, 'k', zorder=11)
        plt.text(.98, .98, r'$R^2 =$ '+str(r2), fontsize=14, 
                 transform=ax1.transAxes,
                 horizontalalignment='right', verticalalignment='top')
        plt.xlabel('Maximum flow depth (m)')
        plt.ylabel('Deposit Thickness (cm)')
        plt.title('Maximum '+fig_title)
        plt.axis(xmin=0, ymin=0)
        ax2 = plt.subplot(132)
        plt.scatter(Fsx, Tsx, c=Dsx, s=50, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.plot(Fsx, LINE2, 'k')
        plt.text(.98, .98, r'$R^2 =$ '+str(R2), fontsize=14, 
                 transform=ax2.transAxes,
                 horizontalalignment='right', verticalalignment='top')
        plt.xlabel('Flow depth (m)')
        plt.ylabel('Deposit Thickness (cm)')
        plt.title(fig_title)
        plt.axis(xmin=0, ymin=0)
        ax3 = plt.subplot(133)
        plt.scatter(FLDint, THKint, c=DTSint, s=50, cmap=cmap, vmin=vmin, 
                    vmax=vmax)
        plt.plot(FLDint, line3_, 'k')
        plt.text(.98, .98, r'$R^2 =$ '+str(r2_), fontsize=14, 
                 transform=ax3.transAxes,
                 horizontalalignment='right', verticalalignment='top')
        plt.axis(xmin=0, ymin=0)
        plt.xlabel('Flow Depth (m)')
        plt.ylabel('Deposit Thickness (cm)')
        plt.title('Interpolated '+fig_title)
        cb = plt.colorbar(cax=cbax, extend='max')
        cb.set_label('Distance from shore (m)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################
def maxthickness_maxflowdepth(Adict, save_fig=False, agu_print=True, 
                                exclude=True, semi_log=False,
                                lin_regress=True, fig_title=\
                                'Maximum Flow Depth vs Maximum Thickness'):  
    """
    plot data from Adict- flow depth vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    if exclude is True:
        exclude = Adict['incomplete_transect']
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["ProjectedFlowDepth"], 
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS, exclude=exclude, 
                    slKey=Adict["SLKey"])
    FLD = Transect(out[5], SLC, TSC, DTS, exclude=exclude, 
                   slKey=Adict["SLKey"])
    tmx, fmx, sloc = denan(THK.smxt, FLD.smx, FLD.sw, n_rounds=2)
    event = np.asarray(getevents(sloc, Adict))
    emap = Adict['emap']
    hands, labs = [], []
    fig = plt.figure(figsize=(16, 12))
    ax = plt.subplot(111)
    if agu_print:
        for e in set(event):
            if e:
                p, = plt.plot(fmx[event == e], tmx[event == e], emap[e], ms=12)
                labs.append(e)
                hands.append(p)
        hands, labs = sort_legend_labels(hands, labs)
    else:
        for s in sorted(set(sloc)):
            p, = plt.plot(fmx[sloc == s], tmx[sloc == s], 
                          '*Dv^dpHso<>'[int(s%11)], ms=12)
            labs.append(SLdecoder(s, Adict["SLKey"]))
            hands.append(p)
    if lin_regress:
        m, b, r = linregress(fmx, tmx)[:3]
        r2 = round(r**2, 2)
        plt.plot(fmx, m*fmx+b, 'k-', zorder=-1)
        plt.text(.98, .98, r'$\mathdefault{R^2 =}$ '+str(r2), fontsize=18, 
                 transform=ax.transAxes, ha='right', va='top')
    if semi_log:
        ax.set_xscale('log')
    else:
        ax.set_xlim([0, 20])             
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))
    hands, labs = sort_legend_labels(hands, labs)
    plt.legend(hands, labs, numpoints=1, frameon=False, fontsize=22,
               loc=2 if lin_regress else 1)

    plt.xlabel('Maximum Flow Depth (m)')
    plt.ylabel('Maximum Deposit Thickness (cm)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def flowdepth_maxsuspensiongradedlayerthickness(Adict, save_fig=False, 
                                                   exclude=True, fig_title= \
            'Maximum Flow Depth vs Maximum Suspension Graded Layer Thickness'):
    print('Running plotting routine:', fig_title)    
    if exclude:
        exclude = Adict['incomplete_transect']
    sglt = get_maxsuspensiongradedlayerthickness(Adict)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"],
                Adict['ProjectedFlowDepth'], 
                sglt,
                Adict['Modern'],
                n_rounds=3)
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    FLD = Transect(out[3], SLC, TSC, DTS, exclude=exclude, 
                   slKey=Adict["SLKey"])
    SGLT = Transect(out[4], SLC, TSC, DTS, exclude=exclude, 
                    slKey=Adict['SLKey'])
    sglt_int, fld_int, _, tnum_int = interp_flowdepth_to_thickness(SGLT, FLD)
    sglt_int = np.asarray(sglt_int)
    fld_int = np.asarray(fld_int)
    FLD_tnum = list(FLD.tnum)
    slocs = [FLD.sw[FLD_tnum.index(t)] for t in tnum_int]
    events = np.asarray(getevents(slocs, Adict))
    hands, labs = [], []
    emap = Adict['emap']
    fig = plt.figure()
    for e in set(events):
        if e:
            p, = plt.plot(fld_int[events == e], sglt_int[events == e], emap[e], 
                          ms=12)
            if e not in labs:
                labs.append(e)
                hands.append(p)
    plt.xlabel('Flow Depth (m)')
    plt.ylabel('Suspension Graded Layer Thickness (cm)')
    plt.ylim(bottom=0, top=nanmax(sglt_int)+1)
    plt.legend(hands, labs, numpoints=1, frameon=False)
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################
def meangs_thickness(Adict, save_fig=False, exclude=True, sand_only=False,
                     plot_std_instead=False, agu_print=True,
                     fig_title='Mean grain size vs Thickness'):
    """
    plot data from Adict - mean grain size vs thickness
    """
    if plot_std_instead and fig_title:
        fig_title = 'Standard Deviation in Grain Size vs Thickness'
    print('Running plotting routine:', fig_title)
    if exclude is True:
        exclude = Adict['incomplete_transect']
    if sand_only:
        # specify min and max grain size to use in gs mean calculation
        gs_min_max = (4, -1)
    else:
        gs_min_max = None
    gsmeans = get_gsmeans(Adict, gs_min_max=gs_min_max, 
                          return_std_instead=plot_std_instead)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                gsmeans,
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS, exclude=exclude, 
                    slKey=Adict["SLKey"])
    MGS = Transect(out[5], SLC, TSC, DTS, exclude=exclude, 
                   slKey=Adict["SLKey"])
    events = np.asarray(getevents(MGS.sw, Adict))
    fig = plt.figure(figsize=(12, 10))
    handles, labels = [], []
    for e in set(events):
        if e:
            p, = plt.plot(MGS.sx[events == e], THK.sx[events == e], 
                          Adict['emap'][e], ms=12)
            handles.append(p)
            labels.append(e)
    plt.ylim(bottom=0)
    plt.ylabel('Thickness (cm)')
    if plot_std_instead:
        plt.xlabel(r'Standard Deviation in Grain Size ($\mathsf{\phi}$)')
    else:
        plt.xlabel(r'Mean Grain Size ($\mathsf{\phi}$)')
    if not agu_print:
        plt.title(fig_title)
    if sand_only:
        plt.xlim(gs_min_max)
        loc = 2
    else:
        loc = 1
    handles, labels = sort_legend_labels(handles, labels)
    plt.legend(handles, labels, numpoints=1, frameon=False, loc=loc)
    if save_fig:
        figsaver(fig, save_fig, fig_title)    
    print('******************************************************************')
    return fig
    
###############################################################################
def meangs_flowdepth(Adict, save_fig=False, exclude=True, lin_regress=True,
                     interpolate_flowdepth=True, sand_only=False,
                     plot_std_instead=False, agu_print=True, mm=False,
                     fig_title='Flow depth vs mean grain size'):
    """
    plot data from Adict - thickness vs flow depth showing mean grain size
    
    set plot_std_instead to True to change the figure to plot standard 
    deviation in grain size rather than mean grain size.
    
    set sand_only to True to plot only data from the sand fraction
    """
    if plot_std_instead and fig_title:
        fig_title = 'Flow Depth vs Standard Deviation in Grain Size'
    print('Running plotting routine:', fig_title)
    if exclude is True:
        exclude = Adict['incomplete_transect']
    if sand_only:
        # specify min and max grain size to use in gs mean calculation
        gs_min_max = (4, -1)
    else:
        gs_min_max = None
    gsmeans = get_gsmeans(Adict, return_std_instead=plot_std_instead,
                          gs_min_max=gs_min_max, mm=mm)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["ProjectedFlowDepth"],
                gsmeans,
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    FLD = Transect(out[3], SLC, TSC, DTS, exclude=exclude, 
                   slKey=Adict["SLKey"])
    MGS = Transect(out[4], SLC, TSC, DTS, exclude=exclude, 
                   slKey=Adict["SLKey"])
    fig = plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    if interpolate_flowdepth:
        mgs, fld = interp_flowdepth_to_thickness(MGS, FLD, keep_nans=True)[:2]
    else:
        fld = FLD.sx
        mgs = MGS.sx
    mgs, fld, slocs = denan(np.asarray(mgs), np.asarray(fld), MGS.sw)
    events = np.asarray(getevents(slocs, Adict))
    handles, labels = [], []
    for e in set(events):
        if e:
            p, = plt.plot(fld[events == e], mgs[events == e], Adict['emap'][e], 
                          ms=12)
            handles.append(p)
            labels.append(e)
    if not agu_print:
        plt.title(fig_title)
    plt.xlabel('Flow Depth (m)')
    if plot_std_instead and mm:
        plt.ylabel('Standard Deviation in Grain Size (mm)')
    elif plot_std_instead and not mm:
        plt.ylabel(r'Standard Deviation in Grain Size ($\mathsf{\phi}$)')
    elif mm:
        plt.ylabel('Mean Grain Size (mm)')
    else:
        plt.ylabel(r'Mean Grain Size ($\mathsf{\phi}$)')
    plt.xlim(xmin=0)
    dx = 0
    if sand_only:
        plt.ylim(gs_min_max)
        loc = 2
    elif mm:
        loc = 2
        dx = .85
    else:
        ax.invert_yaxis() 
        loc = 4
    handles, labels = sort_legend_labels(handles, labels)
    plt.legend(handles, labels, numpoints=1, frameon=False, loc=loc, fontsize=22)
    fld = np.asarray(fld)
    if lin_regress is True:
        m, b, r = linregress(fld, mgs)[:3]
        plt.plot(fld, fld*m+b, 'k-', zorder=-1)
        plt.text(.02+dx, .06, 
                 r'$\mathdefault{R^2 =}$ %0.2f' % r**2,
                 fontsize=18, transform=ax.transAxes, ha='left', va='top')
    elif lin_regress == 'exponential':
        ## function to use with scipy.optimize.curve_fit
        def func(x, a, b, c):
            return a * np.exp(-b *x) + c 
        ## exponential curve fit (x^(3/2))
        popt, pcov = curve_fit(func, fld, mgs)
        x = np.arange(max(fld)+1)
        line4 = func(x, *popt)
        p, = plt.plot(x, line4, '-', color='k', lw=2)
    if save_fig:
        figsaver(fig, save_fig, fig_title)    
    print('******************************************************************')
    return fig
    
###############################################################################
def meangs_flowdepth_thickness(Adict, save_fig=False, exclude=True, agu_print=True,
                               japan_only=False, plot_std_instead=False,
                                 interpolate_flowdepth=True, sand_only=False,
                     fig_title='Flow depth vs Thickness with mean grain size'):
    """
    plot data from Adict - thickness vs flow depth showing mean grain size
    """
    if plot_std_instead and fig_title:
        fig_title = 'Flow Depth vs Thickness with Standard Deviation in Grain Size'
    print('Running plotting routine:', fig_title)
    if exclude is True:
        exclude = Adict['incomplete_transect']
    if sand_only:
        # specify min and max grain size to use in gs mean calculation
        gs_min_max = (4, -1)
    else:
        gs_min_max = None
    gsmeans = get_gsmeans(Adict, gs_min_max=gs_min_max, 
                          return_std_instead=plot_std_instead)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["ProjectedFlowDepth"],
                gsmeans,
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS, exclude=exclude, 
                    slKey=Adict["SLKey"])
    FLD = Transect(out[5], SLC, TSC, DTS, exclude=exclude, 
                   slKey=Adict["SLKey"])
    MGS = Transect(out[6], SLC, TSC, DTS, exclude=exclude, 
                   slKey=Adict["SLKey"])
    fig = plt.figure(figsize=(19, 13))
    # colorbar settings
    vmin = nanmin(MGS.sx)
    vmax = nanmax(MGS.sx)
    extend = 'neither'
    if interpolate_flowdepth:
        thk, fld = interp_flowdepth_to_thickness(THK, FLD, keep_nans=True)[:2]
        thk = np.asarray(thk)
        fld = np.asarray(fld)
    else:
        thk = THK.sx
        fld = FLD.sx
    if japan_only:
        japan_sloc = nanmin([k for (k, v) in Adict['SLKey'].items() if v == 'Sendai'])
        t = nanmin(FLD.tnum[FLD.sw == japan_sloc])
        p = plt.scatter(fld[FLD.tnum == t], thk[FLD.tnum == t], c=MGS.sx[FLD.tnum == t],
                        s=50, vmin=vmin, vmax=vmax, cmap='hot_r')
    else:
        p = plt.scatter(fld, thk, c=MGS.sx, s=100, vmin=vmin, vmax=vmax, 
                        cmap='hot_r')
    if not agu_print:
        plt.title(fig_title)
    plt.xlabel('Flow Depth (m)')
    plt.ylabel('Deposit Thickness (cm)')
#    plt.xlim(xmin=0)
    plt.xscale('log')
    plt.ylim(ymin=0)
    cbar = plt.colorbar(orientation='horizontal', shrink=.7, extend=extend)
    cbar.ax.invert_xaxis()
    cbar.ax.xaxis.set_major_locator(mpl.ticker.NullLocator())
    if plot_std_instead:
        cbar.set_label(r'Standard Deviation in Grain Size ($\mathsf{\phi}$)')
    else:
        cbar.set_label(r'Mean Grain Size ($\mathsf{\phi}$)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)    
    print('******************************************************************')
    return fig
    
###############################################################################
def slope_flowdepth_thickness(Adict, save_fig=False,
                                fig_title='Flow depth vs Thickness'):
    """
    plot data from Adict-  flow depth normalized by local slope vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["Elevation"], 
                Adict["ProjectedFlowDepth"],
                Adict["Modern"], 
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)
    ELV = Transect(out[5], SLC, TSC, DTS)
    FLD = Transect(out[6], SLC, TSC, DTS)
    m = ELV.dx / ELV.dds
    cm = ELV.sx / ELV.sds
    cmap = 'seismic'
    mn = -.06
    mx = .06
    fig = plt.figure(figsize=(18, 7))
    cbax = fig.add_axes([0.925, 0.2, 0.01, 0.6])
    plt.subplot(121)
    plt.scatter(FLD.sx, THK.sx, s=50, c=m, cmap=cmap, vmin=mn, vmax=mx)
    plt.axis(ymin=0, xmin=0)
    plt.xlabel('Flow depth (m)')
    plt.ylabel('Deposit Thickness (cm)')
    plt.title(fig_title+' showing Local Slope')
    plt.subplot(122)
    plt.scatter(FLD.sx, THK.sx, s=50, c=cm, cmap=cmap, vmin=mn, vmax=mx)
    plt.axis(ymin=0, xmin=0)
    plt.xlabel('Flow depth (m)')
    plt.ylabel('Deposit Thickness (cm)')
    plt.title(fig_title+' showing Cumulative Slope')
    cb = plt.colorbar(cax=cbax, extend='max')
    cb.set_label('Slope') 
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
  
###############################################################################
def percentIL_thickness(Adict, save_fig=False, inset_map=False, 
                        normalize_thickness=True, min_transect_points=1,
                        average_perc=10, fig_title=\
             'Distance to shore as percent of inundation limit vs thickness'):
    """
    plot data from Adict- percent of inundation limit vs thickness
    
    if normalize_thickness, make a subplot with thicknesses normalized to max 
    thickness
    
    min_transect_points specifies the number of points on a transect required 
    to include a data for that transect
    
    average_perc specifies the fraction interval over which to average, if
    None, do not average
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["InundationLimit"], 
                Adict["Lat"],
                Adict["Long"],
                Adict["Modern"],
                n_rounds=6
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]    
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)
    INL = Transect(out[5], SLC, TSC, DTS)
    LAT = out[6]
    LON = out[7]
    filtr = THK.tnum >= 0
    for t in set(THK.tnum):
        f = THK.tnum == t
        if len(f.nonzero()[0]) < min_transect_points:
            filtr[f] = False
    x = 100 * THK.sds / INL.sx
    x = x[filtr]
    if normalize_thickness:
        thk_norm = THK.sx / THK.smx
        nplots = 3
    else:
        nplots = 2
    fig = plt.figure(figsize=(15, 12))
    plt.subplot(nplots, 1, 1)
    plt.scatter(x, THK.sx[filtr])
    if average_perc is not None:
        ## calculate average for each average_perc sized block
        fractions, av_thk = average_in_bins(x, THK.sx[filtr], 
                                            block=average_perc)
        plt.plot(fractions, av_thk, 'r*', ms=16)
    plt.xlim([0,100])
    plt.ylim(ymin=0)
    plt.title(fig_title)
    plt.ylabel('Deposit Thickness (cm)')
    plt.xlabel('Distance from shore (% of inundation limit)')
    plt.subplot(nplots, 1, 2)
    plt.scatter(x, THK.smxt[filtr])
    plt.xlim([0,100])
    plt.ylim(ymin=0)
    plt.xlabel('Distance from shore (% of inundation limit)')
    plt.ylabel('Maximum Deposit Thickness (cm)')
    if normalize_thickness:
        plt.subplot(nplots, 1, 3)
        plt.scatter(x, thk_norm[filtr])
        if average_perc is not None:
            ## calculate average for each average_perc sized block
            fractions, av_norm_thk = average_in_bins(x, thk_norm[filtr], 
                                                     block=average_perc)
            plt.plot(fractions, av_norm_thk, 'r*', ms=16)
        plt.xlim([0,100])
        plt.ylim([0,1])
        plt.xlabel('Distance from shore (% of inundation limit)')
        plt.ylabel('Deposit Thickness \n(fraction of maximum \ntransect thickness)')
    if inset_map:
        fig = insetmap(fig, LAT, LON, full_globe=True)    
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################    
def percentIL_flowdepth(Adict, save_fig=False, normalize_flowdepth=True,
                        min_transect_points=1, average_perc=10, fig_title=\
            'Distance to shore as percent inundation limit vs flow depth'):
    """
    plot data from Adict- percent of inundation limit vs flow depth
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["FlowDepth"], 
                Adict["InundationLimit"], 
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]    
    FLD = Transect(out[3], SLC, TSC, DTS)
    INL = Transect(out[4], SLC, TSC, DTS)
    filtr = FLD.tnum >= 0
    for t in set(FLD.tnum):
        f = FLD.tnum == t
        if len(f.nonzero()[0]) < min_transect_points:
            filtr[f] = False
    x = 100 * FLD.sds / INL.sx
    x = x[filtr]
    if normalize_flowdepth:
        fld_norm = FLD.sx / FLD.smx
        nplots = 3
    else:
        nplots = 2    
    fig = plt.figure(figsize=(15, 12))
    plt.subplot(nplots, 1, 1)
    plt.scatter(x, FLD.sx[filtr])
    if average_perc is not None:
        ## calculate average for each average_perc sized block
        percs, av_fld = average_in_bins(x, FLD.sx[filtr], block=average_perc)
        ## midpoint of blocks scaled to percent                                         
        percs = percs+(average_perc/2)
        plt.plot(percs, av_fld, 'r*', ms=16)
    plt.xlim([0,100])
    plt.ylim(ymin=0)
    plt.title(fig_title)
    plt.ylabel('Flow depth (m)')
    plt.xlabel('Distance from shore (% of inundation limit)')
    plt.subplot(nplots, 1, 2)
    plt.scatter(x, FLD.smxt[filtr])
    plt.xlim([0,100])
    plt.ylim(ymin=0)
    plt.xlabel('Distance from shore (% of inundation limit)')
    plt.ylabel('Maximum flow depth (m)')
    if normalize_flowdepth:
        plt.subplot(nplots, 1, 3)
        plt.scatter(x, fld_norm[filtr])
        if average_perc is not None:
            ## calculate average for each average_perc sized block
            percs, av_norm_fld = average_in_bins(x, fld_norm[filtr], 
                                                 block=average_perc)
            ## midpoint of blocks scaled to percent                                         
            percs = percs+(average_perc/2)
            plt.plot(percs, av_norm_fld, 'r*', ms=16)
        plt.xlim([0,100])
        plt.ylim([0,1])
        plt.xlabel('Distance from shore (% of inundation limit)')
        plt.ylabel('Flow depth \n(fraction of maximum \ntransect flowdepth)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################    
def distance_flowdepth_slope(Adict, save_fig=False, xmax=5000, 
                               fig_title=\
                                  'Distance to Shore vs Flow Depth and Slope'):
    """
    plot data from Adict- percent inundation limit vs max flow depth and slope
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["FlowDepth"], 
                Adict["Elevation"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    FLD = Transect(out[3], SLC, TSC, out[2])
    ELV = Transect(out[4], SLC, TSC, out[2])
    ELV.dds[ELV.dds == 0] = np.nan            
    m = ELV.dx / ELV.dds
    cm = ELV.sx / ELV.sds
    mn = -.175 #min(nanmin(cm), nanmin(m))
    mx = .175
    cmap = 'seismic'
    s = 50
    fig = plt.figure(figsize=(18, 10))
    plt.suptitle(fig_title, fontsize=16)
    cbax = fig.add_axes([.925, .2, .01, .6])
    plt.subplot(223)
    plt.scatter(FLD.sds, FLD.smxt, c=m, s=s, cmap=cmap, vmin=mn, vmax=mx)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('Distance from shore (m)')
    plt.ylabel('Maximum flow depth (m)')
    plt.subplot(221)
    plt.scatter(FLD.sds, FLD.sx, c=m, s=s, cmap=cmap, vmin=mn, vmax=mx)
    plt.xlim([0, xmax])
    plt.ylim(ymin=0)
    plt.title('Local slope')
    plt.xlabel('Distance from shore (m)')
    plt.ylabel('Flow depth (m)')
    plt.subplot(224)
    plt.scatter(FLD.sds, FLD.smxt, c=cm, s=s, cmap=cmap, vmin=mn, vmax=mx)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('Distance from shore (m)')
    plt.ylabel('Maximum flow depth (m)')
    plt.subplot(222)
    plt.scatter(FLD.sds, FLD.sx, c=cm, s=s, cmap=cmap, vmin=mn, vmax=mx)
    plt.xlim([0, xmax])
    plt.ylim(ymin=0)
    plt.title('Cumulative transect slope')
    plt.xlabel('Distance from shore (m)')
    plt.ylabel('Flow depth (m)')
    cb = plt.colorbar(cax=cbax, extend='max')
    cb.set_label('Slope')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################    
def distance_thickness_slope(Adict, save_fig=False, xmax=5000, 
                               fig_title=\
                               'Distance to Shore vs Thickness and Slope'):
    """
    plot data from Adict- percent transect vs thickness and slope
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"],
                Adict["MaxThickness"],
                Adict["Elevation"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, out[2])
    ELV = Transect(out[5], SLC, TSC, out[2])
    ELV.dds[ELV.dds == 0] = np.nan            
    m = ELV.dx / ELV.dds
    cm = ELV.sx / ELV.sds
    mn = -.06
    mx = .06
    cmap = 'bwr'
    s = 50
    fig = plt.figure(figsize=(18, 10))
    plt.suptitle(fig_title, fontsize=16)
    cbax = fig.add_axes([.925, .2, .01, .6])
    plt.subplot(223)
    plt.scatter(THK.sds, THK.smxt, c=m, s=s, cmap=cmap, vmin=mn, vmax=mx)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('Distance from shore (m)')
    plt.ylabel('Maximum thickness (cm)')
    plt.subplot(221)
    plt.scatter(THK.sds, THK.sx, c=m, s=s, cmap=cmap, vmin=mn, vmax=mx)
    plt.xlim([0, xmax])
    plt.ylim(ymin=0)
    plt.title('Local slope')
    plt.xlabel('Distance from shore (m)')
    plt.ylabel('Deposit Thickness (cm)')
    plt.subplot(224)
    plt.scatter(THK.sds, THK.smxt, c=cm, s=s, cmap=cmap, vmin=mn, vmax=mx)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('Distance from shore (m)')
    plt.ylabel('Maximum thickness (cm)')
    plt.subplot(222)
    p = plt.scatter(THK.sds, THK.sx, c=cm, s=s, cmap=cmap, vmin=mn, vmax=mx)
    plt.xlim([0, xmax])
    plt.ylim(ymin=0)
    plt.title('Cumulative transect slope')
    plt.xlabel('Distance from shore (m)')
    plt.ylabel('Deposit Thickness (cm)')
    cb = plt.colorbar(cax=cbax, extend='both')
    cb.set_label('Slope')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
        
###############################################################################    
def distance_thickness(Adict, save_fig=False, fig_title=\
                        'Thickness vs Distance to shore'):
    """
    plot data from Adict- distance to shore vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["Thickness"],
                Adict["MaxThickness"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    THK = Transect((out[3]+out[4])/2., out[0], out[1], out[2])
    cmap = cm.get_cmap('spectral')
    n = int(nanmax(THK.tnum))
    colors = [cmap((ii+1)/(n+1)) for ii in range(n)]
    sym = 'v*o^s<pd>h'
    handles, labels = [], []
    fig = plt.figure(figsize=(23, 12))
    for ii in range(n):
        filtr = THK.tnum == ii+1
        plt.plot(THK.sds[filtr], THK.sx[filtr], '-', c=colors[ii])
        h, = plt.plot(THK.sds[filtr], THK.sx[filtr], sym[ii%10], c=colors[ii])
        handles.append(h)
        labels.append(SLdecoder(THK.sw[filtr][0], Adict["SLKey"]))
    plt.title(fig_title)
    plt.xlabel('Distance to shore (m)')
    plt.ylabel('Deposit Thickness (cm)')
    plt.legend(handles, labels, numpoints=1, fontsize=10, loc=1, frameon=False)
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################    
def distance_thickness_panels(Adict, save_fig=False, tolerance=.1,
                                agu_print=None, lin_regress=False, 
                                exclude=None, verbose=True, fig_title=\
                                'Thickness vs Distance to Shore (Panels)'):
    """
    plot data from Adict- distance to shore vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["Thickness"],
                Adict["MaxThickness"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    THK = Transect((out[3]+out[4])/2., out[0], out[1], out[2], exclude=exclude, 
                    slKey=Adict["SLKey"])
    if verbose:
        thinning, thickening = [], []
    tnums = list(set(THK.tnum))
    n = int(np.ceil(np.sqrt(len(tnums))))
    fig = plt.figure(figsize=(22, 13))      
    for ii, tnum in enumerate(sorted(tnums)):
        filtr = THK.tnum == tnum
        x = THK.sds[filtr]
        y = THK.sx[filtr]
        N = len(y)
        sloc = SLdecoder(nanmax(THK.sw[filtr]), Adict["SLKey"])
        ax = plt.subplot(n, n, ii+1)
        plt.plot(x, y, 'b--')
        plt.plot(x, y, 'k.')
        if lin_regress:
            if N > 2:
                m, b, r = linregress(x, y)[:3]
                line = m*x + b
                r2 = round(r**2, 2)
                plt.plot(x, line, 'r:')
                plt.text(.02, .96, 
                         'N ='+str(N)+r',  $\mathdefault{R^2}$='+str(r2), 
                         transform=ax.transAxes, fontsize=9, va='top',  
                         ha='left', color='r')
                if verbose and r2 >= tolerance:
                    if m > 0:
                        thickening.append(sloc)
                    elif m < 0:
                        thinning.append(sloc)
            else:
                plt.text(.02, .96, 'N ='+str(N), transform=ax.transAxes, 
                         fontsize=9, color='r', ha='left', va='top')
            if verbose:
                print('N landward thinning = '+str(len(thinning)))
                print('N landward thickening = '+str(len(thickening)))
        plt.axis(xmin=0, ymin=0)
        ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(3))
        xformatter = mpl.ticker.FuncFormatter(axint)
        yformatter = mpl.ticker.FuncFormatter(axround)
        ax.xaxis.set_major_formatter(xformatter)
        ax.yaxis.set_major_formatter(yformatter)
        plt.title(sloc)
        plt.xlabel('Distance to Shore (m)', fontsize=9)
        plt.ylabel('Deposit Thickness (cm)', fontsize=9)
    plt.tight_layout()
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def distance_changeinthickness(Adict, save_fig=False, fig_title=\
                           'Change in Deposit Thickness vs Distance to Shore'):
    """
    plot data from Adict- distance to shore vs change in thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["Thickness"],
                Adict["MaxThickness"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    THK = Transect((out[3]+out[4])/2., out[0], out[1], out[2])
    fig = plt.figure(figsize=(16, 10))
    plt.axhline(color='black', zorder=-1)
    plt.scatter(THK.midds, THK.dx, s=50, c=THK.dds, cmap='RdBu_r')
    plt.axis(xmin=0)
    plt.xlabel('Distance from shore (m)')
    plt.ylabel(r'$\Delta$ thickness (cm)')
    plt.title(fig_title)
    cbar = plt.colorbar(orientation='vertical', fraction=.075, pad=.1, 
                        aspect=30, shrink=.75)
    cbar.set_label(r'$\Delta$ distance from previous point on transect (m)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def distance_changeinthickness_panels(Adict, save_fig=False, 
                                         fig_title=\
                           'Change in Deposit Thickness vs Distance to Shore'):
    """
    plot data from Adict- distance to shore vs change in thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["Thickness"],
                Adict["MaxThickness"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    THK = Transect((out[3]+out[4])/2., out[0], out[1], out[2])
    tnums = []
    for tnum in sorted(set(THK.tnum)):
        if len(THK.sx[THK.tnum == tnum]) > 1:
            tnums.append(tnum)
    n = int(np.ceil(np.sqrt(len(tnums))))
    fig = plt.figure(figsize=(22, 13))
    for ii, tnum in enumerate(tnums):
        sloc = SLdecoder(nanmax(THK.sw[THK.tnum == tnum]), Adict["SLKey"])
        plt.subplot(n, n, ii+1)
        plt.plot(THK.midds[THK.tnum == tnum], THK.dx[THK.tnum == tnum], 'b--')
        plt.plot(THK.midds[THK.tnum == tnum], THK.dx[THK.tnum == tnum], 'k.')
        plt.axhline(color='black', zorder=-1)
        plt.axis(xmin=0)
        plt.title(sloc)
        plt.xlabel('Distance from shore (m)')
        plt.ylabel(r'$\Delta$ thickness (cm)')
    plt.tight_layout()
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################    
def distance_elevation_panels(Adict, save_fig=False,
                                agu_print=None, exclude=None, fig_title=\
                                'Thickness vs Distance to Shore (Panels)'):
    """
    plot data from Adict- distance to shore vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["Elevation"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    ELV = Transect(out[3], out[0], out[1], out[2], exclude=exclude, 
                    slKey=Adict["SLKey"])
    tnums = list(set(ELV.tnum))
    n = int(np.ceil(np.sqrt(len(tnums))))
    fig = plt.figure(figsize=(50, 30), dpi=72)      
    for ii, tnum in enumerate(sorted(tnums)):
        filtr = ELV.tnum == tnum
        x = ELV.sds[filtr]
        y = ELV.sx[filtr]
        sloc = SLdecoder(nanmax(ELV.sw[filtr]), Adict["SLKey"])
        ax = plt.subplot(n, n, ii+1)
        plt.plot(x, y, 'b--')
        plt.plot(x, y, 'k.')
        plt.axis(xmin=0, ymin=0)
        ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(3))
        xformatter = mpl.ticker.FuncFormatter(axint)
        yformatter = mpl.ticker.FuncFormatter(axround)
        ax.xaxis.set_major_formatter(xformatter)
        ax.yaxis.set_major_formatter(yformatter)
        plt.title(sloc)
        plt.xlabel('Distance to Shore (m)', fontsize=9)
        plt.ylabel('Elevation (m)', fontsize=9)
    plt.tight_layout()
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def percenttransect_thickness(Adict, save_fig=False, normalize_thickness=False, 
                              fig_title=\
                                 'Thickness vs Distance along transect'):
    """
    plot data from Adict- percent of total transect vs thickness
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["Thickness"],
                Adict["MaxThickness"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    THK = Transect((out[3]+out[4])/2., out[0], out[1], out[2])
    if normalize_thickness:
        thk = THK.sx / THK.smx
    else:
        thk = THK.sx
    DTS = Transect(out[2], out[0], out[1], out[2])
    percent = 100. * DTS.sx / DTS.smx
    fig = plt.figure(figsize=(13, 8))
    plt.scatter(DTS.sx, thk, c=percent, s=40, cmap='RdBu_r')
    plt.axis(ymin=0, xmin=0)
    plt.title(fig_title)
    plt.xlabel('Distance on transect (m)')
    if normalize_thickness:
        plt.ylabel('Deposit Thickness (fraction of maximum thickness on transect)')
    else:
        plt.ylabel('Deposit Thickness (cm)')
    cbar = plt.colorbar(orientation='horizontal', fraction=.075, pad=.1, 
                        aspect=30, shrink=.75, ticks=[0, 25, 50, 75, 100])
    cbar.set_label('Percent of total distance along transect')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def percenttransect_flowdepth(Adict, save_fig=False, inset_map=False, 
                                fig_title=\
                                'Flow Depth vs Distance along transect'):
    """
    plot data from Adict- percent of total transect vs flow depth
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["FlowDepth"],
                Adict["Lat"],
                Adict["Long"],
                Adict["Modern"],
                n_rounds=4
                )
    out = runfilters(out, 1)
    FLD = Transect(out[3], out[0], out[1], out[2])
    DTS = Transect(out[2], out[0], out[1], out[2])
    LAT = out[4]
    LON = out[5]
    percent = 100. * DTS.sx / DTS.smx
    fig = plt.figure(figsize=(13, 8))
    plt.scatter(DTS.sx, FLD.sx, s=40, c=percent, cmap='RdBu_r')
    plt.title(fig_title)
    plt.xlabel('Distance on transect (m)')
    plt.ylabel('Flow depth (m)')
    plt.axis(ymin=0, xmin=0)
    cbar = plt.colorbar(orientation='horizontal', fraction=.075, pad=.1,
                        aspect=30, shrink=.75, ticks=[0, 25, 50, 75, 100])
    cbar.set_label('Percent of total distance along transect')
    if inset_map:
        fig = insetmap(fig, LAT, LON, lbwh=[.14, .655, .23, .23])
#        ax1.get_frame().set_alpha(0.)
    if save_fig:
        figsaver(fig, save_fig, fig_title, transparent=True)
    print('******************************************************************')
    return fig   

###############################################################################
def distancetosource_maxflowdepth(Adict, save_fig=False, agu_print=True, 
                       fig_title='Maximum Flow Depth vs Distance From Source'):
    """
    plot data from Adict- distance to source vs max flow depth
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["FlowDepth"],
                Adict["Lat"],
                Adict["Long"],
                Adict["Modern"],
                n_rounds=4
                )    
    out = runfilters(out, 1)
    FLD = Transect(out[3], out[0], out[1], out[2])
    LAT = Transect(out[4], out[0], out[1], out[2])
    LON = Transect(out[5], out[0], out[1], out[2])
    filtr = np.isfinite(FLD.smxt)
    slocs = FLD.sw[filtr]
    maxf = FLD.smxt[filtr]
    latf = np.ones_like(maxf) * np.nan
    lonf = latf.copy()
    for ii, tnum in enumerate(FLD.tnum[filtr]):
        f = LAT.tnum == tnum
        lats = LAT.sx[f]
        lons = LON.sx[f]
        try:
            latf[ii] = lats[np.isfinite(lats)][0]
            lonf[ii] = lons[np.isfinite(lons)][0]
        ## no lat lons on transect
        except IndexError:
            continue
    events = np.asarray(getevents(slocs, Adict))
    emap = Adict['emap']
    d = distance_to_source(Adict, latf, lonf, events)
    hands, labs = [], []
    fig = plt.figure(figsize=(16,12))
    for e in set(events):
        if e:
            f = events == e
            p, = plt.plot(d[f], maxf[f], emap[e], ms=12)
            if any(np.isfinite(latf[f])):
                hands.append(p)
                labs.append(e)
    plt.xlabel('Distance From Source (km)')
    plt.ylabel('Maximum Flow Depth (m)')
    hands, labs = sort_legend_labels(hands, labs)
    plt.legend(hands, labs, numpoints=1, loc=2, fontsize=22, frameon=False)
    if save_fig:
        figsaver(fig, save_fig, fig_title)   
    print('******************************************************************')
    return fig
    
###############################################################################
def distance_flowdepth(Adict, save_fig=False, inset_map=False, agu_print=True,
                        fig_title='Flow Depth vs Distance to shore'):
    """
    plot data from Adict- distance to shore vs flow depth
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["FlowDepth"],
                Adict["Lat"],
                Adict["Long"],
                Adict["Modern"],
                n_rounds=4
                )
    out = runfilters(out, 1)
    FLD = Transect(out[3], out[0], out[1], out[2])
    LAT = out[4]
    LON = out[5]
    handles, labels = [], []

    fig = plt.figure(figsize=(23, 12))
    if agu_print:    
        event = np.asarray(getevents(FLD.sw, Adict))
        emap = Adict['emap']
        for e in set(event):
            if e:
                p, = plt.plot(FLD.sds[event == e], FLD.sx[event == e], emap[e],
                              ms=12)
                labels.append(e)
                handles.append(p)
        handles, labels = sort_legend_labels(handles, labels)
    else:
        cmap = cm.get_cmap('spectral')
        n = int(nanmax(FLD.tnum))
        colors = [cmap((ii+1)/(n+1)) for ii in range(n)]
        sym = 'v*o^s<pd>h'
        for ii in range(n):
            filtr = FLD.tnum == ii+1
            plt.plot(FLD.sds[filtr], FLD.sx[filtr], '-', c=colors[ii])
            p, = plt.plot(FLD.sds[filtr], FLD.sx[filtr], sym[ii%10], c=colors[ii])
            handles.append(p)
            labels.append(SLdecoder(FLD.sw[filtr][0], Adict["SLKey"]))
        plt.title(fig_title)
    plt.xlabel('Distance From Shore (m)')
    plt.ylabel('Flow Depth (m)')
    plt.legend(handles, labels, numpoints=1, frameon=not agu_print,
               fontsize=22 if agu_print else 10)
    if inset_map:
        fig = insetmap(fig, LAT, LON, full_globe=False, map_style=2)
    if save_fig:
        figsaver(fig, save_fig, fig_title)   
    print('******************************************************************')
    return fig

###############################################################################
def distance_flowdepth_panels(Adict, save_fig=False, tolerance=.1,
                                agu_print=None, lin_regress=False, 
                                exclude=None, verbose=True, fig_title=\
                                'Flow Depth vs Distance to Shore (Panels)'):
    """
    plot data from Adict- distance to shore vs flow depth
    dependent on dictionary Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["ProjectedFlowDepth"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    FLD = Transect(out[3], out[0], out[1], out[2])
    if verbose:
        increasing, decreasing = [], []
    tnums = list(set(FLD.tnum))
    n = int(np.ceil(np.sqrt(len(tnums))))
    fig = plt.figure(figsize=(22, 13))
    for ii, tnum in enumerate(sorted(tnums)):
        filtr = FLD.tnum == tnum
        x = FLD.sds[filtr]
        y = FLD.sx[filtr]
        N = len(y)
        sloc = SLdecoder(nanmax(FLD.sw[filtr]), Adict["SLKey"])
        ax = plt.subplot(n, n, ii+1)
        plt.plot(x, y, 'b--')
        plt.plot(x, y, 'k.')
        if lin_regress:
            if N > 2:
                m, b, r = linregress(x, y)[:3]
                line = m*x + b
                r2 = round(r**2, 2)
                plt.plot(x, line, 'r:', zorder=11)
                plt.text(.02, .02, 'N ='+str(N)+r',  $\mathdefault{R^2}$='+str(r2), 
                         transform=ax.transAxes, va='bottom', ha='left', 
                         fontsize=9, color='r')
                if verbose and r2 > tolerance:
                    if m > 0:
                        increasing.append(sloc)
                    elif m < 0:
                        decreasing.append(sloc)
            else:
                plt.text(.02, .04, 'N ='+str(N), transform=ax.transAxes, 
                         fontsize=9, color='r', ha='left', va='bottom')        
            if verbose:
                print('N landward increasing = '+str(len(increasing)))
                print('N landward decreasing = '+str(len(decreasing)))
        plt.axis(xmin=0, ymin=0)
        ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(3))
        yformatter = mpl.ticker.FuncFormatter(axround)
        xformatter = mpl.ticker.FuncFormatter(axint)
        ax.xaxis.set_major_formatter(xformatter)
        ax.yaxis.set_major_formatter(yformatter)
        plt.title(sloc)
        plt.xlabel('Distance to Shore (m)', fontsize=11)
        plt.ylabel('Flow Depth (m)', fontsize=11)
    plt.tight_layout()
    if save_fig:
        figsaver(fig, save_fig, fig_title)   
    print('******************************************************************')
    return fig

###############################################################################
def distance_meangs_panels(Adict, save_fig=False, gs_min_max=None,
                             fig_title='Distance vs Mean grain size: panels'):
    """
    plot data from Adict- distance to shore vs meangs
    dependent on dictionary Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    gsmeans = get_gsmeans(Adict, gs_min_max=gs_min_max)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                gsmeans,
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    MGS = Transect(out[3], out[0], out[1], out[2])
    tnums = list(set(MGS.tnum))
    n = int(np.ceil(np.sqrt(len(tnums))))
    fig = plt.figure(figsize=(22, 13))
    for ii, tnum in enumerate(sorted(tnums)):
        filtr = MGS.tnum == tnum
        x = MGS.sds[filtr]
        y = MGS.sx[filtr]
        N = len(y)
        sloc = SLdecoder(nanmax(MGS.sw[filtr]), Adict["SLKey"])
        ax = plt.subplot(n, n, ii+1)
        plt.plot(x, y, 'b--')
        plt.plot(x, y, 'k.')
        plt.axis(xmin=0, ymin=0)
        ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(3))
        yformatter = mpl.ticker.FuncFormatter(axround)
        xformatter = mpl.ticker.FuncFormatter(axint)
        ax.xaxis.set_major_formatter(xformatter)
        ax.yaxis.set_major_formatter(yformatter)
        plt.title(sloc)
        plt.xlabel('Distance to Shore (m)', fontsize=11)
        plt.ylabel(r'Bulk mean grain size ($\mathsf{\phi}$)', fontsize=11)
    plt.tight_layout()
    if save_fig:
        figsaver(fig, save_fig, fig_title)   
    print('******************************************************************')
    return fig

###############################################################################
def volume_flowdepth(Adict, save_fig=False, agu_print=True,
                      exclude=True,
                      fig_title='Volume of Deposit vs Flow Depth'):
    """
    plot data from Adict - volume of deposit along transect vs most seaward
    flow depth on the transect
    dependent on global variable Adict containing TDB data keyed by attribute
    Volume calculated by sum((thickness2+thickness1)/2 * (distance2-distance1))
    """
    print('Running plotting routine:', fig_title)
    if exclude is True:
        exclude = Adict['incomplete_transect']
    out = denan(Adict['SLCode'],
                Adict['Transect'],
                Adict['Distance2shore'],
                Adict['Thickness'],
                Adict['MaxThickness'],
                Adict['ProjectedFlowDepth'],
                Adict['Modern'],
                n_rounds=5
                )
    THK = Transect((out[4]+out[3])/2., out[0], out[1], out[2], exclude=exclude, 
                    slKey=Adict["SLKey"])
    FLD = Transect(out[5], out[0], out[1], out[2], exclude=exclude, 
                   slKey=Adict["SLKey"])
    ## Area of each trapezoidal deposit (in m^2)
    A = THK.middx * THK.dds * .01
    ## initiate lists to append to    
    Fshore = []
    FmaxW = []
    FmaxT = []
    FmaxTW = []
    sloc = []
    sloc_strings = []
    V = []
    for tnum in set(THK.tnum):
        filtr = THK.tnum == tnum
        ## Volume is the sum of each deposit section area
        volume = np.nansum(A[filtr])
        if not np.isnan(volume):
            V.append(volume)    
            w = FLD.sw[filtr][0]
            ## find the maximum flow depth ocurring on each transect
            FmaxT.append(FLD.smx[filtr][0])
            if not agu_print:
                ## find the maximum flow depth occurring at each sublocation
                FmaxW.append(nanmax(Adict["ProjectedFlowDepth"][Adict["SLCode"] == w]))
                ## max of transect if it exists else max of sublocation
                if not np.isnan(FmaxT[-1]):
                    FmaxTW.append(FmaxT[-1])
                else:
                    FmaxTW.append(FmaxW[-1])
                ## find the most seaward flow depth on each transect
                f2 = np.isfinite(FLD.sx[filtr])
                if any(f2):
                    Fshore.append(FLD.sx[filtr][f2 == True][0])
                else:
                    Fshore.append(np.nan)
            ## get sublocation name for labeling
            sloc.append(w)
            sloc_strings.append(SLdecoder(w, Adict["SLKey"]))
    if agu_print:
        event = getevents(sloc, Adict)
        ## remove locations with no data
        for ii, f in enumerate(FmaxT):
            if np.isnan(f):
                V.pop(ii)
                FmaxT.pop(ii)
                event.pop(ii)
        emap = Adict['emap']
        hands, labs = [], []
        fig = plt.figure(figsize=(13, 9))
        ax = plt.subplot(111)
        ax.set_xlim([0, 20])
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))        
        plt.xlabel('Maximum Flow Depth (m)', fontsize=18)
        plt.ylabel(r'Volume of Deposit (m$\mathdefault{^3}$ per m alongshore)', 
                   fontsize=18)
        for ii, e in enumerate(event):
            if e:
                p, = plt.plot(FmaxT[ii], V[ii], emap[e], ms=12, label=e)
                if e not in labs:
                    labs.append(e)
                    hands.append(p)
        hands, labs = sort_legend_labels(hands, labs)
        plt.legend(hands, labs, frameon=False, numpoints=1, loc=2) 
    else:
        sloc = sloc_strings    
        sym = 'v^*osp<>dh'
        cmap = cm.get_cmap('spectral')
        n = len(V)
        colors = [cmap((ii+1)/(n+1)) for ii in range(n)]
        fig = plt.figure(figsize=(18, 12))
        plt.suptitle(fig_title)
        limits = [0, 20, 0, 250]
        plt.subplot(221)    
        plt.xlabel('Maximum flow depth on transect (m)')
        plt.ylabel(r'Volume of deposit (m$^3$ per m alongshore)')
        plt.axis(limits)
        plt.subplot(222)
        plt.ylabel(r'Volume of deposit (m$^3$ per m alongshore)')
        plt.xlabel('Most seaward flow depth on transect (m)')
        plt.axis(limits)
        plt.subplot(223)
        plt.ylabel(r'Volume of deposit (m$^3$ per m alongshore)')
        plt.xlabel('Maximum flow depth at study site (m)')
        plt.axis(limits)
        ax = plt.subplot(224)
        plt.ylabel(r'Volume of deposit (m$^3$ per m alongshore)')
        plt.xlabel('Maximum flow depth on transect else at study site (m)')
        plt.axis(limits)
        for ii, v in enumerate(V):
            plt.subplot(221)
            plt.plot(FmaxT[ii], v, sym[ii%10], c=colors[ii])    
            plt.subplot(222)
            plt.plot(Fshore[ii], v, sym[ii%10], c=colors[ii])
            plt.subplot(223)
            plt.plot(FmaxW[ii], v, sym[ii%10], c=colors[ii])
            plt.subplot(224)
            plt.plot(FmaxTW[ii], v, sym[ii%10], c=colors[ii], label=sloc[ii])
        hands, labs = ax.get_legend_handles_labels()
        fig.legend(hands, labs, numpoints=1, fontsize=10, loc=7, 
                   borderaxespad=1)
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################
def averagethickness_flowdepth(Adict, save_fig=False, max_on_transect=False,
                               exclude=True, agu_print=False, fig_title=\
                                 'Average Thickness vs Maximum Flow Depth'):
    """
    plot the average deposit thickness by sublocation versus the max flow depth
    dependent on global variable Adict containing TDB data keyed by attribute
    max flow depth is the maximum flow depth occurring on the transect when
    available else the maximum flow depth at the sublocation
    Thickness is averaged by distance
    (deposit area divided by total transect distance)
    """
    print('Running plotting routine:', fig_title)
    if exclude is True:
        exclude = Adict['incomplete_transect']
    out = denan(Adict['SLCode'],
                Adict['Transect'],
                Adict['Distance2shore'],
                Adict['Thickness'],
                Adict['MaxThickness'],
                Adict['ProjectedFlowDepth'],
                Adict['Modern'],
                n_rounds=5
                )
    THK = Transect((out[4]+out[3])/2., out[0], out[1], out[2], exclude=exclude, 
                    slKey=Adict["SLKey"])
    FLD = Transect(out[5], out[0], out[1], out[2], exclude=exclude, 
                   slKey=Adict["SLKey"])
    ## Area of each trapezoidal deposit (mixed units)
    A = THK.middx * THK.dds
    ## initiate lists to append to    
    FmaxW = []
    FmaxTW = []
    avT = []
    sloc = []
    for tnum in set(THK.tnum):
        filtr = THK.tnum == tnum
        ## average thickness is the sum of the areas divided by the sum of the
        ## distances between each area
        a = np.nansum(A[filtr])/np.nansum(THK.dds[filtr])
        if not np.isnan(a):
            ## get sublocation name for labeling
            w = FLD.sw[filtr][0]
            ## find the maximum flow depth occurring at each sublocation
            f = nanmax(Adict["ProjectedFlowDepth"][Adict["SLCode"] == w])
            if not np.isnan(f):
                sloc.append(w)
                avT.append(a)
                FmaxW.append(f)
                ## find the maximum flow depth ocurring on each transect
                FmaxT = FLD.smx[filtr][0]
                ## max of transect if it exists else max of sublocation
                if not np.isnan(FmaxT):
                    FmaxTW.append(FmaxT)
                else:
                    FmaxTW.append(FmaxW[-1])
    if max_on_transect:
        fmx = np.asarray(FmaxTW)
    else:
        fmx = np.asarray(FmaxW)
    avT = np.asarray(avT)
    fig = plt.figure(figsize=(13, 9))
    ax = plt.subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xlim([0, 20])
    ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(5))    
    if agu_print:
        event = getevents(sloc, Adict)
        emap = Adict['emap']
        hands, labs = [], []
        for ii, e in enumerate(event):
            if e:
                p, = plt.plot(fmx[ii], avT[ii], emap[e], label=e, ms=12)
                if e not in labs:
                    labs.append(e)
                    hands.append(p)
        hands, labs = sort_legend_labels(hands, labs)
        plt.legend(hands, labs, numpoints=1, frameon=False, loc=2, fontsize=22)
    else:
        m, b, r = linregress(fmx, avT)[:3]
        line = m*fmx + b
        r2 = round(r**2, 3)
        sloc = SLdecoder(sloc, Adict["SLKey"])
        cmap = cm.get_cmap('spectral')
        n = len(avT)
        colors = [cmap((ii+1)/(n+1)) for ii in range(n)]      
        sym = 'v^*osp<>dh'
        for ii, av in enumerate(avT):
            plt.plot(fmx[ii], av, sym[ii%10], c=colors[ii], label=sloc[ii])
        plt.plot(fmx, line, 'k-', zorder=-1)
        plt.text(.98, .98, r'$\mathdefault{R^2} = $'+str(r2), 
                 transform=ax.transAxes, ha='right', va='top')
        hands, labs = ax.get_legend_handles_labels()
        fig.legend(hands, labs, numpoints=1, fontsize=10, loc=7, 
                   borderaxespad=1)
    plt.xlabel('Maximum Flow Depth (m)')
    plt.ylabel('Mean Deposit Thickness (cm)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)      
    print('******************************************************************')
    return fig

###############################################################################
def volume_slope(Adict, save_fig=False, 
                  fig_title='Slope vs Deposit volume'):
    """
    plot the relationship between the volume of the deposit and the slope over
    which it is deposited
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict['SLCode'],
                Adict['Transect'],
                Adict['Distance2shore'],
                Adict['Thickness'],
                Adict['MaxThickness'],
                Adict['Elevation'],
                Adict['Modern'],
                )
    THK = Transect((out[4]+out[3])/2., out[0], out[1], out[2])
    DTS = Transect(out[2], out[0], out[1], out[2])
    ELV = Transect(out[5], out[0], out[1], out[2])
    V = np.zeros_like(THK.x) * np.nan
    ## Area of each trapezoidal deposit (in m2)
    A = THK.middx * THK.dds * .01
    for tnum in set(THK.tnum):
        filtr = THK.tnum == tnum
        ## Volume is the sum of each deposit section area
        V[filtr] = np.nansum(A[filtr])
    ## Total slope of deposit
    m = ELV.sx / DTS.smxt
    fig = plt.figure(figsize=(14, 8))
    plt.scatter(m, V, c=DTS.smxt, cmap='gray_r', s=50)
    plt.title(fig_title)
    plt.xlabel('Slope of transect')
    plt.ylabel(r'Volume of deposit along transect (m$^3$ per m alongshore)')
    plt.axis(ymin=0)
    plt.axvline(0, color='black')
    cb = plt.colorbar(fraction=.1, shrink=.75)
    cb.set_label('Transect length (m)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def localslope_thickness_proximities(Adict, save_fig=False, fig_title=\
                   'Local slope vs Thickness at varying proximity thresholds'):
    """
    plot data from Adict- slope vs thickness
    test different proximity
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["Elevation"], 
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)
    ELV = Transect(out[5], SLC, TSC, DTS)
    m = ELV.dx / ELV.dds
    fig = plt.figure(figsize=(9, 12))
    plt.subplot(421)
    plt.axvline(0, color='black')
    plt.scatter(m, THK.sx)
    plt.axis(ymin=0)
    plt.xlabel('Local slope')
    plt.ylabel('Deposit Thickness (cm)')
    plt.title('Local slope vs. thickness of deposit')
    plt.subplot(422)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.scatter(m, THK.dx)
    plt.xlabel('Local slope')
    plt.ylabel('Change in thickness (cm)')
    plt.title('Local slope vs. change in thickness of deposit')
    for ii, r in enumerate((25, 10, 5)):
        m = ELV.dx / ELV.dds
        m[ELV.dds > r] = np.nan
        ii = (ii+1)*2 + 1
        plt.subplot(4, 2, ii)
        plt.axvline(0, color='black')
        plt.scatter(m, THK.sx)
        plt.axis(ymin=0)
        plt.xlabel('Local slope')
        plt.ylabel('Deposit Thickness (cm)')
        plt.title(str(r)+' meter proximity')
        ii += 1
        plt.subplot(4, 2, ii)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.scatter(m, THK.dx)
        plt.xlabel('Local slope')
        plt.ylabel('Change in thickness (cm)')
        plt.title(str(r)+' meter proximity')
    plt.tight_layout()    
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def elevation_thickness_proximities(Adict, save_fig=False, annotate=True,
                                    fig_title=\
                   'Elevation vs Thickness at varying proximity thresholds'):
    """
    plot data from Adict- Elevation vs thickness
    test different proximity
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["Elevation"], 
                Adict["Modern"],
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)
    ELV = Transect(out[5], SLC, TSC, DTS)
    fig = plt.figure(figsize=(22, 6))
    for ii, r in enumerate((np.Inf, 25, 10, 5)):
        filtr = np.isfinite(THK.dx)
        d_thk = THK.dx[filtr]
        d_elv = ELV.dx[filtr]
        d_ds = ELV.dds[filtr]
        d_elv[d_ds > r] = np.nan
        d_thk[d_ds > r] = np.nan
        ii += 1
        ax = plt.subplot(1, 4, ii)
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.scatter(d_elv, d_thk)
        plt.xlabel('Change in Elevation (m)')
        plt.ylabel('Change in thickness (cm)')
        plt.title(str(r)+' meter proximity')
        if annotate:
            fs = 14
            n = len(d_elv[np.isfinite(d_elv)])
            de_p = d_elv > 0
            de_n = d_elv < 0
            dt_p = d_thk > 0
            dt_n = d_thk < 0 
            I = len(d_elv[de_p * dt_p]) * 100 / n
            IV = len(d_elv[de_p * dt_n]) * 100 / n
            III = len(d_elv[de_n * dt_n]) * 100 / n
            II = len(d_elv[de_n * dt_p]) * 100 / n
            plt.text(.95, .95, 'I: {:.0f} %'.format(I), fontsize=fs,
                     ha='right', va='bottom', transform=ax.transAxes)
            plt.text(.95, .05, 'IV: {:.0f} %'.format(IV), fontsize=fs,
                     ha='right', va='top', transform=ax.transAxes)
            plt.text(.05, .05, 'III: {:.0f} %'.format(III), fontsize=fs,
                     ha='left', va='top', transform=ax.transAxes)
            plt.text(.05, .95, 'II: {:.0f} %'.format(II), fontsize=fs,
                     ha='left', va='bottom', transform=ax.transAxes)
    plt.tight_layout()    
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################
def thickness_nextthickness(Adict, save_fig=False, 
                              fig_title='Thicknes vs next Thickness'):
    """
    plot each thickness on a transect with the next thickness following it.
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine: '+fig_title)
    out = denan(Adict['SLCode'],
                Adict['Transect'],
                Adict['Distance2shore'],
                Adict['Thickness'],
                Adict['MaxThickness'],
                Adict['Modern']
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)
    mx = np.ceil(nanmax(THK.smxt)/10) * 10.
    cmap = cm.get_cmap('spectral')
    n = int(nanmax(THK.tnum))
    colors = [cmap((ii+1)/(n+1)) for ii in range(n)]
    handles, labels = [], []
    fig = plt.figure(figsize=(17, 12))
    ax = plt.subplot(111)
    plt.plot(np.arange(mx+1), np.arange(mx+1), 'k--')
    for ii, tnum in enumerate(sorted(set(THK.tnum))):
        sym = 'v^*osp<>dh'[ii%10]
        filtr = THK.tnum == tnum
        p, = plt.plot(THK.sx[filtr][:-1], THK.sx[filtr][1:], sym, 
                      c=colors[ii])
        handles.append(p)
        labels.append(SLdecoder(THK.sw[filtr][0], Adict["SLKey"]))
    plt.text(.02, .98, 'Landward Thickening', transform=ax.transAxes, 
             fontsize=14, ha='left', va='top')
    plt.text(.98, .02, 'Landward Thinning', transform=ax.transAxes, 
             fontsize=14, ha='right', va='bottom')
    plt.text(.97, .99, 'Constant Thickness', transform=ax.transAxes,
             ha='right', va='top', rotation=45)
    ax.set_aspect('equal')
    plt.xlabel(r'Thickness$_i$ (cm)')
    plt.ylabel(r'Thickness$_{i+1}$ (cm)')
    fig.legend(handles, labels, fontsize=9.5, numpoints=1, loc=7, 
               borderaxespad=6)
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################
def thickness_nextthickness_flowdepth(Adict, save_fig=False, 
                                       fig_title='Thicknes vs next Thickness'):
    """
    plot each thickness on a transect with the following thickness 
    showing flow depth
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine: '+fig_title)
    out = denan(Adict['SLCode'],
                Adict['Transect'],
                Adict['Distance2shore'],
                Adict['Thickness'],
                Adict['MaxThickness'],
                Adict['ProjectedFlowDepth'],
                Adict['Modern'],
                n_rounds=5
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)
    FLD = Transect(out[5], SLC, TSC, DTS)\
    ## do calculations to interpolate flow depths
    THKint, FLDint, _, tnumint = interp_flowdepth_to_thickness(THK, FLD)
    THKint = np.asarray(THKint)
    FLDint = np.asarray(FLDint)
    x1, y1, z1 = [], [], []
    x2, y2, z2 = [], [], []
    for tnum in sorted(set(THK.tnum)):
        f1 = THK.tnum == tnum
        thk = THK.sx[f1]
        fmx = nanmax(FLD.smxt[f1])
        f2 = tnumint == tnum
        thk_i = THKint[f2]
        fld_i = FLDint[f2]
        n = len(thk)
        if n > 1:
            for ii, t in enumerate(thk):
                if ii == 0:
                    x1.append(t)
                elif ii == n - 1:
                    y1.append(t)
                    z1.append(fmx)
                else:
                    x1.append(t)
                    y1.append(t)
                    z1.append(fmx)
        n = len(thk_i)
        if n > 1:
            for ii, t in enumerate(thk_i):
                if ii == 0:
                    x2.append(t)
                    z2.append((fld_i[ii]+fld_i[ii+1])/2)
                elif ii == n - 1:
                    y2.append(t)
                else:
                    x2.append(t)
                    y2.append(t)
                    z2.append((fld_i[ii]+fld_i[ii+1])/2)
    mx = np.ceil(max(attributefilter(np.asarray(y1), np.isfinite(z1)))/10) * 10
    vmax = nanmax(z1)
    vmin = nanmin(z1)
    s = 40
    cmap = 'gray_r'
    
    def setup_axes(ax):
        ax.set_aspect('equal')
        plt.plot(np.arange(mx+1), np.arange(mx+1), 'k--')
        plt.axis([0, mx, 0, mx])
        plt.text(.02, .98, 'Landward Thickening', transform=ax.transAxes, 
                 fontsize=14, ha='left', va='top')
        plt.text(.98, .02, 'Landward Thinning', transform=ax.transAxes, 
                 fontsize=14, ha='right', va='bottom')
        plt.text(.97, .99, 'Constant Thickness', transform=ax.transAxes,
                 ha='right', va='top', rotation=45)
        plt.xlabel(r'Thickness$_i$ (cm)')
        plt.ylabel(r'Thickness$_{i+1}$ (cm)')
        return ax
        
    fig = plt.figure(figsize=(18, 9))
    setup_axes(plt.subplot(122))
    plt.scatter(x2, y2, s=s, c=z2, cmap=cmap, vmax=vmax, vmin=vmin)
    plt.title('Colormap shows interpolated flow depth')
    setup_axes(plt.subplot(121))
    plt.scatter(x1, y1, s=s, c=z1, cmap=cmap, vmax=vmax, vmin=vmin)
    plt.title('Colormap shows maximum flow depth')
    cax = fig.add_axes([.925, .2, .01, .6])
    cb = plt.colorbar(cax=cax)
    cb.set_label('Flow Depth (m)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################
def mw_thickness(Adict, save_fig=False, fig_title='Mw vs Thickness',
                 annotate=False):
    """
    plot Mw vs thickness
    """
    print('Running plotting routine: '+fig_title)
    out = denan((Adict['Thickness']+Adict['MaxThickness'])/2,
                     Adict['SLCode'], Adict["Modern"])
    thk, slc, _ = runfilters(out, 1)
    mw = np.asarray(getevents(slc, Adict, return_mw=True))
    thk, mw = denan(thk, mw)
    m, b, r = linregress(mw, thk)[:3]
    r2 = round(r**2, 2)
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.scatter(mw, thk)
    plt.ylim(bottom=0)
    plt.text(.02, .98, r'$\mathdefault{R^2 =}$ ' +str(r2), fontsize=14,
                 transform=ax.transAxes, ha='left', va='top')
    plt.plot(mw, mw*m+b, 'k')
    plt.xlabel('Mw')
    plt.ylabel('Deposit Thickness (cm)')
    if annotate:
        for m in set(mw):
            t = thk[mw == m]
            plt.annotate('n = %i' % len(t), (m, t.max()), xytext=(-30,20),
                         textcoords='offset points')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def flowdepth_nextflowdepth(Adict, save_fig=False, 
                              fig_title='Flow Depth vs next Flow Depth'):
    """
    plot each flow depth on a transect with the next flow depth following it.
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running plotting routine: '+fig_title)
    out = denan(
        Adict['SLCode'],
        Adict['Transect'],
        Adict['Distance2shore'],
        Adict['ProjectedFlowDepth'],
        Adict['Modern']
    )
    out = runfilters(out, 1)
    FLD = Transect(out[3], out[0], out[1], out[2])
    mx = np.ceil(nanmax(FLD.smxt)/10) * 10.
    cmap = cm.get_cmap('spectral')
    n = int(nanmax(FLD.tnum))
    colors = [cmap((ii+1)/(n+1)) for ii in range(n)]
    sym = 'v^*osp<>dh'
    handles, labels = [], []
    fig = plt.figure(figsize=(17, 12))
    ax = plt.subplot(111)
    plt.plot(np.arange(mx+1), np.arange(mx+1), 'k--')
    for ii, tnum in enumerate(sorted(set(FLD.tnum))):
        filtr = FLD.tnum == tnum
        p, = plt.plot(FLD.sx[filtr][:-1], FLD.sx[filtr][1:], sym[ii%10], 
                      c=colors[ii])
        handles.append(p)
        labels.append(SLdecoder(FLD.sw[filtr][0], Adict["SLKey"]))
    plt.text(.02, .98, 'Landward Increasing', transform=ax.transAxes, 
             fontsize=14, ha='left', va='top')
    plt.text(.98, .02, 'Landward Decreasing', transform=ax.transAxes, 
             ha='right', va='bottom', fontsize=14)
    plt.text(.97, .99, 'Constant Flow Depth', transform=ax.transAxes,
             ha='right', va='top', rotation=45)
    ax.set_aspect('equal')
    plt.xlabel(r'Flow Depth$_i$ (m)')
    plt.ylabel(r'Flow Depth$_{i+1}$ (m)')
    fig.legend(handles, labels, fontsize=9.5, numpoints=1, loc=7, 
               borderaxespad=6)
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig

###############################################################################
def topo_classified_thickness_flowdepth(Adict, save_fig=False,
                            fig_title='Topography, Thickness, and Flow Depth'):
    """
    Regarding your analysis, have you tried some averaging of deposit
    characteristics (thickness) in the cross shore direction to average 
    across local topography and then compare that to flow depth? - GG
    """
    print('Running plotting routine:', fig_title, 'NOT YET IMPLEMENTED')
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["ProjectedFlowDepth"],
                Adict["Elevation"],
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)    
    FLD = Transect(out[5], SLC, TSC, DTS)
    ELV = Transect(out[6], SLC, TSC, DTS)
    THKint, FLDint, DTSint, tnumint = interp_flowdepth_to_thickness(THK, FLD)
    FLDint = np.asarray(FLDint)
    THKint = np.asarray(THKint)
    DTSint = np.asarray(DTSint)
    km_sloc = nanmin([k for (k, v) in Adict['SLKey'].items() if v == 'Kuala Merisi'])
    f = ELV.tnum == nanmin(ELV.tnum[ELV.sw == km_sloc])
    m = ELV.dx / ELV.dds
    ## get sign on all slope points
    m_sign = np.zeros_like(m[f])
    for ii, s in enumerate(m[f]):
        if np.isnan(s):
            m_sign[ii] = np.nan
        elif s < 0:
            m_sign[ii] = -1
        else:
            m_sign[ii] = 1
    inflection = np.abs(np.diff(m_sign))
    dts = ELV.sds[f]
    print(ELV.sx[f][13:23])
    print(dts[13:23])
    print(m[f][13:23])
    print(m_sign[13:23])
    print(inflection[13:23])
    
    fig, ax1 = plt.subplots()
    ax1.plot(dts, ELV.sx[f], 'g-')
    ax2 = ax1.twinx()
    ax2.axhline(0, color='k')
    ax2.plot(ELV.midds[f], m[f], 'r.', dts[1:], inflection/-20, 'b.')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return None

###############################################################################
def volume_flowdepth_segments(Adict, save_fig=False,
                            fig_title='Volume vs Flow Depth'):
    """
    Regarding your analysis, have you tried some averaging of deposit
    characteristics (thickness) in the cross shore direction to average 
    across local topography and then compare that to flow depth? - GG
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict["SLCode"], 
                Adict["Transect"], 
                Adict["Distance2shore"], 
                Adict["Thickness"], 
                Adict["MaxThickness"], 
                Adict["ProjectedFlowDepth"],
                Adict["Elevation"],
                Adict["Modern"], 
                n_rounds=3
                )
    out = runfilters(out, 1)
    SLC = out[0]
    TSC = out[1]
    DTS = out[2]
    THK = Transect((out[3]+out[4])/2., SLC, TSC, DTS)    
    FLD = Transect(out[5], SLC, TSC, DTS)
    ELV = Transect(out[6], SLC, TSC, DTS)
    THKint, FLDint, DTSint, tnumint = interp_flowdepth_to_thickness(THK, FLD)
    FLDint = np.asarray(FLDint)
    THKint = np.asarray(THKint)
    DTSint = np.asarray(DTSint)
    
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################
def sand_mud_volume(Adict, dts=2750, sublocation='Sendai', 
                     elevation_datum='Tokyo Peil', save_fig=False, 
                     fig_title='Sand vs Mud Volume'):
    """
    compare sand vs mud components of deposit
    
    dependent on global variable Adict containing TDB data keyed by attribute
        
    dts specifies the distance beyond which the deposit is mud
    sublocation specifies the 
    
    ONLY works at sublocations which have only one transect.
    """
    print('Running plotting routine:', fig_title)
    out = denan(Adict['SLCode'],
                Adict['Transect'],
                Adict['Distance2shore'],
                Adict['Thickness'],
                Adict['MaxThickness'],
                Adict['Elevation'],
                Adict['Modern'],
                )
    THK = Transect((out[4]+out[3])/2., out[0], out[1], out[2])
    ELV = Transect(out[5], out[0], out[1], out[2])
    units = r' $\mathdefault{\frac{m^3}{m_{alongshore}}}$'
    ## set up filter for desired sublocation, reverse dictionary lookup  
    key = [k for k, v in Adict['SLKey'].items() if v == sublocation]
    f = THK.sw == key[0]
    x = THK.sds[f]
    y = THK.sx[f]
    middx = THK.middx[f]
    dds = THK.dds[f]
    ii = x.searchsorted(dts)
    xs = x[:ii+1]
    ys = y[:ii+1]
    xm = x[ii:]
    ym = y[ii:]
    ## Volume of each trapezoidal deposit (in m3/m alongshore)
    V = np.nansum(middx*dds*.01)
    Vs = np.nansum(middx[:ii+1]*dds[:ii+1]*.01)
    Vm = np.nansum(middx[ii+1:]*dds[ii+1:]*.01)
    ## must specify all the vertices to fill in the deposit area on the plot
    vert_sand = [(x[0], 0)] + list(zip(xs, ys)) + [(x[ii], 0)]
    vert_mud = [(x[ii], 0)] + list(zip(xm, ym)) + [(x[-1], 0)]
    poly_sand = mpl.patches.Polygon(vert_sand, fc='y', alpha=.5)
    poly_mud = mpl.patches.Polygon(vert_mud, fc='brown', alpha=.5)
    fig = plt.figure(figsize=(16, 12))
    ## specify subplot shape
    gs = mpl.gridspec.GridSpec(3, 1)
    ax0 = plt.subplot(gs[:-1, :])
    ax0.add_patch(poly_sand)
    ax0.add_patch(poly_mud)
    ax0.plot(x, y, 'bo')
    ## write volumes on the figure
    ax0.text(.15, .85, 'Sand volume = '+str(round(Vs, 2))+units, 
             transform=ax0.transAxes,
             horizontalalignment='center')
    ax0.text(.85, .85, 'Mud volume = '+str(round(Vm, 2))+units, 
             transform=ax0.transAxes,
             horizontalalignment='center')
    ax0.text(.5, .95, 'Total volume = '+str(round(V, 2))+units, 
             transform=ax0.transAxes,
             horizontalalignment='center')
    plt.title(sublocation+' - '+fig_title, fontsize=16)
    plt.ylabel('Deposit thickness (cm)')
    ax1 = plt.subplot(gs[2, :])    
    ax1.plot(ELV.sds[f], ELV.sx[f])
    plt.xlabel('Distance to shore (m)')
    plt.ylabel('Elevation, '+elevation_datum+' (m)')
    if save_fig:
        figsaver(fig, save_fig, fig_title)
    print('******************************************************************')
    return fig
    
###############################################################################    
def distance_thickness_widget(Adict,
                                fig_title='Thickness vs Distance to shore'):
    """
    interactively plot transect data distance to shore vs thickness
    
    dependent on global variable Adict containing TDB data keyed by attribute
    
    DETAILED comments on widget function in thickness_nextthickness_widget
    """
    print('Running interactive plot:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["Thickness"],
                Adict["MaxThickness"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    THK = Transect((out[3]+out[4])/2., out[0], out[1], out[2])
    cmap = cm.get_cmap('spectral')
    n = int(nanmax(THK.tnum))
    colors = [cmap((ii+1)/(n+1)) for ii in range(n)]
    sym = 'v*o^s<pd>h'
    handles, line_handles, labels, symbols = [], [], [], []
    fig = plt.figure(figsize=(23, 12))
    hidden_ax = fig.add_axes([.1, .2, .5, .6])
    ax = fig.add_axes([.05, .1, .6, .8])
    for ii in range(n):
        filtr = THK.tnum == ii+1
        thk = THK.sx[filtr]
        dts = THK.sds[filtr]
        lh, = plt.plot(dts, thk, '-', c=colors[ii], visible=False)
        h, = plt.plot(dts, thk, sym[ii%10], c=colors[ii], visible=False)
        sh, = hidden_ax.plot(dts, thk, sym[ii%10],c=colors[ii], zorder=-10)
        handles.append(h)
        symbols.append(sh)
        line_handles.append(lh)
        labels.append(SLdecoder(THK.sw[filtr][0], Adict["SLKey"]))
    plt.title(fig_title)
    plt.xlabel('Distance to shore (m)')
    plt.ylabel('Deposit Thickness (cm)')
    ax.set_xlim(0, 1000)
    plt.legend(symbols, labels, numpoints=1, fontsize=10, loc=1, frameon=False)
    bax = fig.add_axes([.7, .2, .25, .6])    
    boxes = sorted(set(labels))
    buttons = mpl.widgets.CheckButtons(bax, boxes, 
                                              [False for box in boxes])
                                              
    def hide_function(box):
        n = range(len(handles))
        if box == 'Sendai':
            ax.set_xlim(0, 5000)
            ax.set_ylim(0, 35)
        elif box == 'Kuala Merisi':
            ax.set_xlim(0, 2500)
            ax.set_ylim(0, 35)
        elif box in ('Constitucion', 'La Trinchera'):
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 80)
        else:
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 35)
        hands = [handles[ii] for ii in n if labels[ii] == box]
        l_hands = [line_handles[ii] for ii in n if labels[ii] == box]
        for h in hands:
            h.set_visible(not h.get_visible())
        for lh in l_hands:
            lh.set_visible(not lh.get_visible())
        plt.draw()
        
    buttons.on_clicked(hide_function)
    print('Displaying figure...')
    plt.show()
    print('******************************************************************')    

###############################################################################
def thickness_nextthickness_widget(Adict, fig_title=\
                                    'Thickness vs Next Thickness on Transect'):
    """
    interactively plot Thickness vs Next Thickness on Transect
    dependent on global variable Adict containing TDB data keyed by attribute
    """
    print('Running interactive plot:', fig_title)
    out = denan(Adict["SLCode"],
                Adict["Transect"],
                Adict["Distance2shore"],
                Adict["Thickness"],
                Adict["MaxThickness"],
                Adict["Modern"]
                )
    out = runfilters(out, 1)
    THK = Transect((out[3]+out[4])/2., out[0], out[1], out[2])
    ## for plotting 1:1 line (constant thickness)
    mx = np.ceil(nanmax(THK.smxt)/10) * 10.
    cmap = cm.get_cmap('spectral')
    tnums = set(THK.tnum)
    n = len(tnums)
    colors = [cmap((ii+1)/(n+1)) for ii in range(n)]
    sym = 'v^*osp<>dh'
    handles, labels, leg_handles = [], [], []
    fig = plt.figure(figsize=(22, 10))
    ## create a hidden axis to allow legend with symbols by plotting duplicate 
    ## plot behind the main plot axis... the main plot axis will originally 
    ## have all its plots set to invisible!
    hidden_ax = fig.add_axes([.5, .5, .01, .01])
    ax = plt.subplot(111)
    ## set up plot axis
    ax.set_aspect('equal')
    plt.plot(np.arange(mx+1), np.arange(mx+1), 'k--')
    plt.text(.02, .98, 'Landward Thickening', transform=ax.transAxes, 
             fontsize=14, ha='left', va='top')
    plt.text(.98, .02, 'Landward Thinning', transform=ax.transAxes, 
             fontsize=14, ha='right', va='bottom')
    plt.text(.97, .99, 'Constant Thickness', transform=ax.transAxes,
             ha='right', va='top', rotation=45)
    plt.xlabel(r'Thickness$_i$ (cm)')
    plt.ylabel(r'Thickness$_{i+1}$ (cm)')
    plt.title(fig_title)
    ## plot points
    for ii, tnum in enumerate(sorted(tnums)):
        filtr = THK.tnum == tnum
        thk = THK.sx[filtr]
        if len(thk) > 1:
            p, = plt.plot(thk[:-1], thk[1:], sym[ii%10], c=colors[ii], 
                          visible=False)
            lh, = hidden_ax.plot(thk[:-1], thk[1:], sym[ii%10], c=colors[ii], 
                                 zorder=-1)
            ## handles for plot                         
            handles.append(p)
            labels.append(SLdecoder(THK.sw[filtr][0]), Adict["SLKey"])
            ## handles for legend
            leg_handles.append(lh)
    fig.legend(leg_handles, labels, fontsize=9.5, numpoints=1, loc=6, 
               borderaxespad=12)
    bax = fig.add_axes([.72, .175, .26, .65])    
    ## no duplicate check boxes when multiple transects have the same name
    boxes = sorted(set(labels))
    ## initiate widget, all boxes start unchecked
    buttons = mpl.widgets.CheckButtons(bax, boxes, 
                                              [False for box in boxes])
                                              
    def hide_function(box):
        ## expand axes when plotting Constitucion
        if box == 'Constitucion':
            ax.axis([0, 80, 0, 80])
        else:
            ax.axis([0, 45, 0, 45])
        ## get each plot handle from the transect with name 'box' by looking up
        ## occurences of box in list labels
        hands = [handles[ii] for ii in range(len(labels)) if labels[ii] == box]
        ## set each plot handle to be visible or invisible        
        for h in hands:
            h.set_visible(not h.get_visible())
        plt.draw()
    
    ## call hide_function when clicked
    buttons.on_clicked(hide_function)
    print('Displaying figure...')
    plt.show()
    print('******************************************************************')

###############################################################################
def sublocation_plotter(Adict, *args, exclude=None):
    """
    plots a multipanel figure for each sublocation specified in *args
    """
    
    def make_plot(tnums, verbose=True):
        """
        makes figures with four subplots of deposit thickness, flow depth, 
        elevation, and mean grain size plotted against distance
        one figure for each transect
        tnums is a generator of transect numbers
        """
        for tnum in tnums:
            if tnum < 0:
                ## if tnums generator returns a negative value it is actually
                ## returning an SLCode at a sublocation where not enough data
                ## exists to create a transect. it does this as a courtesy to
                ## allow sublocation_plotter to report 'No data to plot'
                slname = SLdecoder(-tnum, Adict['SLKey'])
                print('** %s. No data to plot. No transect data.' % slname)
                continue
            filtr = THK.tnum == tnum
            ## get the sublocation code
            sloc = THK.sw[filtr][0]
            tsc = THK.st[filtr][0]
            slname = SLdecoder(sloc, Adict["SLKey"])
            fig_title = '{}, {}'.format(slname, tsc)                     
            gsfilenames, gsdts = get_values_on_transect_with_tuple(
                                                            (sloc, tsc),
                                                            Adict,
                                                            'GSFileUniform',
                                                            'Distance2shore'
                                                            )
            gsmeans = np.asarray([np.nan for _ in gsfilenames])
            gsstds = gsmeans.copy()
            for ii, gs in enumerate(gsfilenames):
                if gs:
                    try:
                        ## create TsuDBGSFile object
                        gsfile = TsuDBGSFile(gs)
                        ## get mean and std using TsuDBGSFile methods
                        gsmeans[ii] = gsfile.bulk_mean()
                        gsstds[ii] = gsfile.bulk_std()
                    ## FileNotFoundError or TypeError
                    except FileNotFoundError as fnfe:
                        print('%s - %s. Error reading GSFile. %s' 
                                % (tnum, fig_title, fnfe))
                        continue
                    except TypeError as te:
                        print('%s - %s. Error reading GSFile. %s' 
                                % (tnum, fig_title, te))
                        continue           
            if np.isnan(THK.sx[filtr]).all() and np.isnan(FLD.sx[filtr]).all()\
                and np.isnan(ELV.sx[filtr]).all() and np.isnan(gsmeans).all():
                if verbose:
                    print('** %s - %s. No data to plot.' % (tnum, fig_title))
                ## if no distance to shore data exists, no figure is created
                continue
            if verbose:
                print('%s - %s. Creating plot.' % (tnum, fig_title))
            ## get elevation datum. from the lookup dictionary made in csv2dic
            datum = nanmax(Adict['ElevationDatum'][Adict['SLCode'] == sloc])
            if np.isnan(datum):
                datum = 'unknown datum'
            else:
                datum = Adict['datum_lookup'][datum]
            ## plot
            fig, ax = plt.subplots(4, 1, sharex=True, figsize=(12, 12))
            plt.sca(ax[0])
            x, y = denan(THK.sds[filtr], THK.sx[filtr])
            plt.suptitle('\n\n'+fig_title, fontsize=18)
            plt.plot(x, y, 'r--', x, y, 'k.')
            plt.ylabel('Deposit Thickness (cm)', fontsize=12)
            plt.sca(ax[1])
            x, y = denan(FLD.sds[filtr], FLD.sx[filtr])
            plt.plot(x, y, 'b--', x, y, 'k.')
            plt.ylabel('Flow Depth (m)', fontsize=12)
            plt.sca(ax[2])
            x, y = denan(ELV.sds[filtr], ELV.sx[filtr])
            plt.plot(x, y, 'g--', x, y, 'k.')
            plt.ylabel('Elevation (m, '+datum+')', fontsize=12)
            plt.sca(ax[3])
            x, y = denan(gsdts.copy(), gsmeans)
            x2, y2 = denan(gsdts.copy(), gsstds)
            ind = np.argsort(x)
            x, y = x[ind], y[ind]
            _, p = plt.plot(x, y, 'm--', x, y, 'k.')
            _, p2 = plt.plot(x2, y2, 'y--', x2, y2, 'kx')
            plt.legend([p, p2], ['Mean', 'Standard Deviation'], numpoints=1, 
                       fontsize=10, frameon=False, loc=4)
            ax[3].invert_yaxis()
            plt.ylabel('Mean and Standard Deviation \n of Grain Size ' +\
                       r'($\mathsf{\phi}$)' + '\n\n', fontsize=12, ha='center')
            plt.xlabel('Distance to Shore (m)', fontsize=18)
            plt.xlim(left=THK.sds[filtr][0], right=THK.sds[filtr][-1])
    
    print('sublocation_plotter:')
    if exclude is True:
        exclude = Adict['incomplete_transect']
    out = denan(
                Adict['SLCode'],
                Adict['Transect'],
                Adict['Distance2shore'],
                Adict['Thickness'],
                Adict['MaxThickness'],
                Adict['ProjectedFlowDepth'],
                Adict['Elevation'],
                Adict['Modern'],
                n_rounds=3
                )
    out = runfilters(out, 1, verbose=False)
    THK = Transect((out[3]+out[4])/2., out[0], out[1], out[2], exclude=exclude, 
                    slKey=Adict["SLKey"])
    FLD = Transect(out[5], out[0], out[1], out[2], exclude=exclude, 
                   slKey=Adict["SLKey"])
    ELV = Transect(out[6], out[0], out[1], out[2], exclude=exclude, 
                   slKey=Adict["SLKey"])
    if not args:
        ## if no *args are passed, attempt to create figs for every transect
        tnums = (tnum for tnum in set(THK.tnum))
        make_plot(tnums)
    else:
        ## *args is a tuple of sublocations to plot
        for arg in args:
            ## find out what each element of the tuple is and make figs
            if isinstance(arg, str):
                ## lookup the tnum using the string name of the transect
                tnums = lookup_tnum(lookup_SLCode(arg, Adict['SLKey']), THK)
                make_plot(tnums)
            elif isinstance(arg, int):
                ## arg is the tnum
                tnums = [arg]
                make_plot(tnums)
    print('******************************************************************')
    print('Displaying figures...')
    plt.show()
    
###############################################################################
def meangs_distance_transect(Adict, transect_tuple, save_fig=False, 
                               scale=500, cmap='spectral', annotate=False):
    """
    plot all mean grain sizes from a given transect spatially
    y axis is depth, x axis is distance, size and color show grain size
    
    transect tuple is (sloc, tsc) like ('Sendai', 1)
    """
    file_names, dts = get_values_on_transect_with_tuple(transect_tuple,
                                                        Adict, 
                                                        'GSFileUniform', 
                                                        'Distance2shore')
    means = [None for x in file_names]
    depths = means.copy()
    gsfiles = means.copy()
    vmin = np.ones(len(depths))*np.nan
    vmax = vmin.copy()
    for ii, file in enumerate(file_names):
        if file:
            gs = TsuDBGSFile(file)
            gsfiles[ii] = gs
            means[ii] = gs.dist_means()
            depths[ii] = gs.mid_depth
            vmax[ii] = np.nanmax(means[ii])
            vmin[ii] = np.nanmin(means[ii])
        else:
            means[ii] = np.asarray([np.nan])
            depths[ii] = np.asarray([np.nan])
    vmin = np.nanmin(vmin)
    vmax = np.nanmax(vmax)
    
    def phi2mm(phi):
        return 2.**-phi
        
    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot(111)
    if np.isnan(vmin).all():
        plt.text(.5, .5, 'No grain-size data to display', ha='center')
        return fig
    for ii, d in enumerate(dts):
        if not np.isnan(d) and not np.isnan(depths[ii]).all():
            p = plt.scatter(np.ones(depths[ii].size)*d, depths[ii], cmap=cmap,
                            s=phi2mm(means[ii])*scale, c=means[ii], vmin=vmin, 
                            vmax=vmax, alpha=.6)
            if annotate:
                plt.text(d, np.nanmin(depths[ii]), gsfiles[ii].trench_name[0], 
                                      va='bottom', ha='center', rotation=90)
    fig_title = 'Grain size data at %s, %s - Transect %s' % (gs.sublocation, 
                                                             gs.location, 
                                                             transect_tuple[1])
    plt.title(fig_title)                                                     
    try:
        plt.legend([p], ['Relative grain-size'], scatterpoints=1, 
                   markerscale=None)
        cbar = plt.colorbar(orientation='vertical', fraction=.075, pad=.1, 
                                    aspect=30, shrink=.75)
        cbar.set_label(r'Grain-size ($\mathsf{\phi}$)')
        plt.ylabel('Depth below surface (%s)' % gs.depth_units)
        plt.xlabel('Distance from shore (m)')
        ax.invert_yaxis()
    except UnboundLocalError:
        plt.text(.5, .5, 
                 'No depth or no distance data with the grain-size data', 
                 ha='center')
    if save_fig:
        figsaver(fig, save_fig, fig_title.replace(' ', '_'))
    return fig
    
###############################################################################    
def plotall(menu, dic='Adict', kwargs='', show_figs=True, reverse=True):
    """
    run all plotting routines in dictionary 'menu'
    
    string 'kwargs' is used to pass kwargs to each plotting routine 
    eg (to specify a value for save_fig for all plots) kwargs="save_fig='png'"
    
    if reverse is True, flip the order in which the plots are created
    
    eg:    plotall({k: v for k, v in menu.items() if k < 10}, 
                   kwargs="save_fig='png'", show_figs=False)  
    """
    for key in sorted(menu.keys(), reverse=reverse):
        try:
            exec('fig%i = menu[key](%s, %s)' % (key, dic, kwargs))
        except Exception as e:
            print('failed %s' % menu[key])
            print(e)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    if show_figs:
        print('Displaying figures...')
        plt.show()

###############################################################################
class TsuDBGSFile(GSFile):
    """
    subclass of GSFile to define the directory holding Uniform GS data files
    """            
    project_directory = os.path.join(os.path.dirname(__file__),
                                     r'../TsuDepData/Uniform_GS_Data/')
                                     

###############################################################################
##  MAIN PROGRAM
###############################################################################
def main(
        xls_file_name='TsunamiSediments_AllData_BL_March2014_r6.xlsx', 
        dict_file_name = "TsuDB_Adict_2014-12-03.pkl",
        from_xls=True,
        save_dict=False,
        saveas_dict=True,
        TDB_DIR=None
    ):
    """
    load in Adict to make TsuDB data available for plotting routines
    """
    ## Settings
    if not TDB_DIR:
        ## if working directory is not specified, use directory of this file
        TDB_DIR = os.path.join(os.path.dirname(__file__), '..')
    xls_file_path = os.path.join(TDB_DIR, xls_file_name)

    np.set_printoptions(threshold=np.nan)
    plt.close('all')
    
    ## Main program to load database
    ## read data from master excel file
    if from_xls:
        Adict = xls2dic(xls_file_path)
        print('TsuDB data read in from xls file "%s"\n' % xls_file_name)
    ## save excel data into a python pickle file
    if save_dict and from_xls:
        dict_file_name = savedict(Adict, askfilename=saveas_dict)
        print('TsuDB data dictionary saved as "%s"\n' % dict_file_name)
    ## open data from a python pickle file (faster!)
    if not from_xls:
        Adict = opendict(dict_file_name)
        print('TsuDB data dictionary opened from "%s"\n' % dict_file_name) 

    Adict['TDB_DIR'] = TDB_DIR

    return Adict

############################################################################### 
if __name__ == '__main__':
    ## load Adict using main function
    Adict = main(from_xls=False, save_dict=True)
        
    ##--Plotting routines menu--##
    menu = {
            1: data_globe,
            2: flowdepth_thickness,
            3: maxthickness_maxflowdepth,
            4: localslope_thickness,
            5: meangs_flowdepth_thickness,
            6: meangs_flowdepth,
            7: meangs_thickness,
            8: flowdepth_thickness_semilog,
            9: sedimentconcentration_histogram,
            10: sedimentconcentration_distance,
            11: flowdepth_maxsuspensiongradedlayerthickness,
            12: averagethickness_flowdepth,
            13: volume_slope,
            14: distance_changeinthickness,
            }
    ##--Enter commands--##
#    plotall({k: v for k, v in menu.items() if k < 10}, 
#            kwargs="save_fig='png'", show_figs=False)    
#    mpl.rcParams['font.size'] = 20
#    sublocation_plotter(Adict, 26)
#    f = menu[4](Adict)
    fig = meangs_distance_transect(Adict, ('Kuala Merisi', 1))
    plt.show()