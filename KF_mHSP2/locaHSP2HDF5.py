# -*- coding: utf-8 -*-
"""
Routines and data structures for processing *HSPsqured* HDF5 file from 
the model input standpoint.

Use this module to isolate the reading of the input HSP2 HDF5 file and 
corresponding model setup. The reorganization of this program, relative
to *HSPsquared*, so that the main loop is the time loop means that we
do not want to keep the HDF5 file open for the entire simulation. 
Additionally are not providing full HSPF-functionality support at this
time and need to identify exactly what is supported and what is not
supported for the user.

This module contains customizations to work with two different HDF5
file formats. The original *HSPsquared* HDF5 format is for the 2.7
version that was the primary *HSPsquared* version prior to March 2020.
A 3.6+ version of *HSPsquared* was released in March-April 2020. This
updated version has different HDF5 file format. 

"""
# Copyright and License
"""
Copyright 2020 Nick Martin

Module Author: Nick Martin <nick.martin@stanfordalumni.org>

This file is part of mHSP2.

mHSP2 is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mHSP2 is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with mHSP2.  If not, see <https://www.gnu.org/licenses/>.

"""

# imports
import locaLogger as CL
from collections import defaultdict
import pandas as pd
import numpy as np

#----------------------------------------------------------------------
# MODULE wide globals
SEQUENCE = defaultdict( list )
"""Replaces sequence in original HSPsquared formulations"""
GENERAL = dict()
"""Replaces general in original HSPsquared formulations"""
MONTHLYS = defaultdict( dict )
"""Replaces monthlys, which are the dictionary of monthly tables.

Example: monthlys['PERLND', 'P001']['CEPSCM']
"""
UCS = dict()
"""Replaces ucs or user control.

Holds all default user control info in a dictionary
"""
nUCI = defaultdict( dict )
"""Replaces uci in new format HDF5 file.

Holds all default user control info in a dictionary
"""
TSDD = defaultdict( list )
"""Replaces tsdd which is the time series data structure.

Time series info default dictionary
"""
LINKDD = defaultdict( list )
"""Replaces linkdd which is the links database.

Data for LINK (combined NETWORK & SCHEMATIC) and MASSLINK information
"""
MLDD = defaultdict( list )
"""Replaces mldd or the mass links.

Data for mass links.
"""
XFLOWDD = dict( )
"""Replaces xflowdd.

This is not really used here.
"""
LOOKUP = defaultdict( list )
"""Replaces lookup.

Also not really used.
"""
KEY_GEN_START = "sim_start"
"""Key for simulation start time, in GENERAL dictionary"""
nKEY_START = "Start"
"""New format key for simulation start time, in GENERAL dictionary"""
KEY_GEN_END = "sim_end"
"""Key for simulation end time, in GENERAL dictionary"""
nKEY_END = "Stop"
"""New format key for simulation end time, in GENERAL dictionary"""
DFCOL_OPSEQ_DELT = "DELT"
"""Column name for time step in operational sequence DataFrame """
DFCOL_OPSEQ_ID = "ID"
"""Column name for ID in operational sequence DataFrame """
DFCOL_OPSEQ_TARG = "TARGET"
"""Column name for TARGET in operational sequence DataFrame """
DFCOL_OPSEQ_SDELT = "SDELT"
"""Column name for string time step in operational sequence DataFrame """
HSP2_TIME_FMT = "%Y-%m-%d %H:%M"
"""Time format for extraction from HSP2"""
AOS_DTYPE = np.dtype( [ ( DFCOL_OPSEQ_TARG, 'U6' ), 
                        ( DFCOL_OPSEQ_ID, 'U4' ), 
                        ( DFCOL_OPSEQ_SDELT, 'U4' ), 
                        ( DFCOL_OPSEQ_DELT, 'f4') ] )
"""Structured array specification type."""
ALLOPSEQ = None
"""Structured array or recarray to hold the operational sequence.

This is set off of the HDF5 file in locaHSP2HDF5.
"""
HDF_FMT = 0
"""HDF file format to read.

HSPsquared changed the HDF5 file format in 2020 with the release that 
was Python 3 compatible. Unfortunately, neither format is well 
documented. If this value is 0, then read in the original format. 
If > 0, then read in the new format.
"""


#----------------------------------------------------------------------
# methods
def detHDF5Format(hdfname):
    """Determine the HDF5 format.

    Need to know this to correctly read in the necessary values.

    Args:
        hdfname (str): HDF5 filename used for both input and output.

    Returns:
        int: HDF5 file format; 0 == original format; 1 == new format

    """
    # imports
    import h5py
    # globals
    global HDF_FMT
    # parameters
    GTABLE_PATH = '/CONTROL/GLOBAL/table'
    UNITS_STR = 'units'
    OldFmt = 0
    NewFmt = 1
    # locals
    FoundUnits = False
    # start
    chkFile = h5py.File(hdfname, 'r')
    globs = chkFile[GTABLE_PATH]
    NRows = globs.shape[0]
    for iI in range(NRows):
        testStr = globs[iI][0]
        utStr = testStr.decode()
        if utStr == UNITS_STR:
            FoundUnits = True
            break
        # end if
    # end for
    # close the file
    chkFile.close()
    # set the units
    if FoundUnits:
        return OldFmt
    # end if
    # return
    return NewFmt


def transform(ts, tindex, how):
    """Copy of transform method from HSP2squared.
    
    Because we include this function do not need to import the package

    Args:
        ts (pandas.DataFrame): time series in pandas DataFrame format
        tindex (pandas.DateTimeIndex): time index to use with the time series
        how (str): method for interpolation

    Returns:
        pandas.DataFrame: resampled time series

    """
    if ( ( len(ts) == len(tindex) ) and ( ts.index[0] == tindex[0] ) 
            and ( ts.index.freq == tindex.freq ) ):
        pass
    elif how == 'SAME' and 'M' in ts.index.freqstr:
        # possible Pandas bug
        ts = ts.reindex(tindex, method='bfill')
    elif  how in ['SAME', 'LAST']:
        if ts.index.freq > tindex.freq:
            ts = ts.reindex(tindex, method='PAD')
        elif ts.index.freq < tindex.freq:
            ts = ts.resample(tindex.freqstr).last()
    elif  how in ['MEAN']:
        if ts.index.freq > tindex.freq:
            ts = ts.reindex(tindex, method='PAD')
        elif ts.index.freq < tindex.freq:
            ts = ts.resample(tindex.freqstr).mean()
    elif how in ['DIV', 'SUM']:
        if ts.index.freq > tindex.freq:
            ratio = float(tindex.freq.nanos) / float(ts.index.freq.nanos)
            ts = ts.reindex(tindex, method='PAD') * ratio
        elif ts.index.freq < tindex.freq:
            ts = ts.resample(tindex.freqstr).sum()
    elif how in ['MONTHLY12', 'DAYVAL'] and len(ts) == 12:
        start = pd.to_datetime('01/01/' + str(tindex[0].year-1))
        stop  = pd.to_datetime('12/31/' + str(tindex[-1].year+1))
        tempindex = pd.date_range(start, stop, freq='MS')
        tiled = np.tile(ts, len(tempindex)/12)
        if how == 'DAYVAL':  # HSPF "interpolation"  interp to day, pad fill the day
            daily = pd.Series(tiled,index=tempindex).resample('D')
            ts = daily.interpolate(method='time').resample(tindex.freqstr).pad()
        else:
            ts = pd.Series(tiled, index=tempindex).resample(tindex.freqstr).pad()
    elif how in ['HOURLY24', 'LAPSE'] and  len(ts) == 24:
        start = pd.to_datetime('01/01/' + str(tindex[0].year-1))
        stop  = pd.to_datetime('12/31/' + str(tindex[-1].year+1) + ' 23:59')
        tempindex = pd.date_range(start, stop, freq='H')
        tile = np.tile(ts, len(tempindex)/24)
        ts = pd.Series(tile, index=tempindex)
        if tindex.freq > tempindex.freq:
            ts = ts.resample(tindex.freqstr).mean()
        elif tindex.freq < tempindex.freq:
            ts = ts.reindex(ts, method='PAD')
    elif how == 'SPARSE':
        x = pd.Series(np.NaN, tindex)
        for indx,value in ts.iteritems():
            iloc = tindex.get_loc(indx, method='nearest')
            x[x.index[iloc]] = value
            ts = x.fillna(method='pad')
    else:
        print('UNKNOWN AGG/DISAGG METHOD: ' + how)
    # end outer if
    # truncate to simulation [start,stop]
    ts = ts[tindex[0]: tindex[-1]]
    # shouldn't happen - debug
    if len(ts) != len(tindex):
        errMsg = "LENGTH mismatch - pd.Series %d and index %d" % \
                 ( len(ts), len(tindex))
        print("%s" % errMsg)
        CL.LOGR.error( errMsg )
    # end checking if
    # return
    return ts


def newHDFRead( hdfname ):
    """Logic to read the new format HDF file.
    
    Extraction from main to read everything that needed from 
    HDF5 file. Stores these main items now in module globals 
    rather than keeping the file open.

    Args:
        hdfname (str): HDF5 filename used for both input and output.

    Returns:
        int: function status, 0 == success

    """
    # imports
    # globals
    global nUCI, GENERAL, TSDD, LINKDD, MLDD, ALLOPSEQ, AOS_DTYPE
    global nKEY_START, nKEY_END
    # parameters
    goodReturn = 0
    badReturn = -1
    # start
    with pd.HDFStore( hdfname ) as store:
        # Extract some required initial simulation values.
        # general no longer has the units
        GENERAL = store['CONTROL/GLOBAL'].to_dict()
        # UCS or uci in new format terminology
        for path in store.keys():
            op, module, *other = path[1:].split(sep='/', maxsplit=3)
            s = '_'.join(other)
            if op in {'PERLND', 'IMPLND', 'RCHRES' }:
                for id, vdict in store[path].to_dict('index').items():
                    nUCI[(op, module, id)][s] = vdict
                # end for
            # end if
        # end for
        # end UCS
        # time series external sources
        for row in store['/CONTROL/EXT_SOURCES'].itertuples():
            # get timeseries' info
            TSDD[ row.TVOL, row.TVOLNO ].append( row )
        # end for row
        # get LINKS
        # get data for LINK (combined NETWORK & SCHEMATIC) and MASSLINK information
        for _, row in store[ 'CONTROL/LINKS' ].iterrows():
            LINKDD[ row.TVOL, row.TVOLNO ].append( row )
        # end of LINKDD for
        # get mass links
        for i,row in store[ 'CONTROL/MASS_LINKS' ].iterrows():
            MLDD[ row.MLNO ].append( row )
        # end of MLDD for
        # next get the operational sequence
        topSeq = store.get( key='/CONTROL/OP_SEQUENCE' )
    # end with
    # want to change this to a numpy structured array for possible future
    #   speed ups
    fullList = list()
    for row in topSeq.itertuples(index=False):
        try:
            btarg = str( row[ 0 ] )
            bid = str( row[ 1 ] )
            sdelt = str( row[ 2 ] )
            fdelt = float( sdelt )
        except:
            # this is an error
            errMsg = "Error converting row %s of operational sequence!!!" % \
                        str( row )
            CL.LOGR.error( errMsg )
            return badReturn
        # now append to our list
        rowList = ( btarg, bid, sdelt, fdelt )
        fullList.append( rowList )
    # end of for
    ALLOPSEQ = np.array( fullList, dtype=AOS_DTYPE )
    # end of with block and HDF5 file closed.
    # now add some checks on what we got
    # GENERAL
    genKeys = GENERAL['Info'].keys()
    if not nKEY_START in genKeys:
        # this is an error
        errMsg = "No %s key in GENERAL in %s!!!" % ( nKEY_START, hdfname )
        CL.LOGR.error( errMsg )
        return badReturn
    if not nKEY_END in genKeys:
        # this is an error
        errMsg = "No %s key in GENERAL in %s!!!" % ( nKEY_END, hdfname )
        CL.LOGR.error( errMsg )
        return badReturn
    # UCS
    if len( nUCI.keys() ) < 1:
        # this is an error
        errMsg = "No keys found in %s for UCS!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # TSDD
    if len( TSDD.keys() ) < 1:
        # this is an error
        errMsg = "No keys found in %s for TSDD!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # LINKDD
    if len( LINKDD.keys() ) < 1:
        # this maybe an error - means no RCHRES
        warnMsg = "No keys found in %s for LINKDD and no RCHRES " \
                  "defined !!!" % hdfname
        CL.LOGR.warning( warnMsg )
    # MLDD 
    if len( MLDD.keys() ) < 1:
        # this is an error
        errMsg = "No keys found in %s for MLDD!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # ALLOPSEQ
    if len( ALLOPSEQ ) < 1:
        # this is an error
        errMsg = "No operational sequence found in %s !!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # return
    return goodReturn


def origHDFRead( hdfname, reloadkeys ):
    """Logic to read the original HDF file format.
    
    Extraction from main to read everything that needed from 
    HDF5 file. Stores these main items now in module globals 
    rather than keeping the file open.

    Args:
        hdfname (str): HDF5 filename used for both input and output.
        reloadkeys (bool): Regenerates keys, used after adding new modules.

    Returns:
        int: function status, 0 == success

    """
    # imports
    # globals
    global SEQUENCE, GENERAL, MONTHLYS, UCS, TSDD, LINKDD
    global MLDD, XFLOWDD, LOOKUP, KEY_GEN_START, KEY_GEN_END
    global DFCOL_OPSEQ_DELT, DFCOL_OPSEQ_ID, DFCOL_OPSEQ_TARG
    global ALLOPSEQ, AOS_DTYPE
    # parameters
    goodReturn = 0
    badReturn = -1
    # start
    with pd.HDFStore( hdfname ) as store:
        # read our calculation sequence from the store
        for _,x in store['HSP2/CONFIGURATION'].sort_values(by=['Order']).iterrows():
            if x.Function and x.Function != 'noop':
                #importlib.import_module(x.Module)
                #x.Function = eval(x.Module + '.' + x.Function)
                SEQUENCE[x.Target].append(x)
            # end if
        # end for
        # now check for some required items and configuration in the store
        #  if these are not there then we can add them in.
        if 'TIMESERIES/LAPSE24' not in store:
            store['TIMESERIES/LAPSE24'] = store['HSP2/LAPSE24']
        # end if
        if 'TIMESERIES/SEASONS12' not in store:
            store['TIMESERIES/SEASONS12'] = store['HSP2/SEASONS12']
        # end if
        if 'TIMESERIES/SaturatedVaporPressureTable' not in store:
            store['TIMESERIES/SaturatedVaporPressureTable'] = store['HSP2/SaturatedVaporPressureTable']
        # end if
        # next extract some required initial simulation values.
        GENERAL = store['CONTROL/GLOBAL'].Data.to_dict()
        # general['msg'] = msg --- check this
        # now load the keys and reload if needed
        if reloadkeys:
            store['HSP2/KEYS'] = pd.Series(store.keys())
        # end if
        HKEYS = store['HSP2/KEYS']
        # create the monthlys tables
        for key in [key for key in HKEYS if 'MONTHLY' in key]:
            tokens = key.split('/')     # target=tokens[1], variable=tokens[-1]
            for indx,row in store[key].iterrows():
                MONTHLYS[(tokens[1], indx)][tokens[-1]] = tuple(row)
            # end of inner for
        # end of outer for
        # now get the UCS put this all here instead of having a separate function
        # start get UCS
        for x in ['PERLND', 'IMPLND', 'RCHRES']:
            for indx, row in store[x + '/GENERAL_INFO'].iterrows():
                UCS[x, 'GENERAL_INFO', indx] = row.to_dict()
            # end inner for
            for indx, row in store[x + '/ACTIVITY'].iterrows():
                UCS[x, 'ACTIVITY', indx] = row
            # end inner for
        # end of outer for
        getflag = dict()
        for x in store['HSP2/CONFIGURATION'].itertuples():
            getflag[x.Path[:-1] if x.Path.endswith('/') else x.Path] = x.Flag
        # end for
        HDATA = defaultdict(list)
        for key in HKEYS:
            tokens = key.split('/')
            if ( ( tokens[1] in ['PERLND', 'IMPLND', 'RCHRES'] ) and 
                 ( 'MONTHLY' not in key ) and ( 'ACTIVITY' not in key ) 
                 and ( 'SAVE' not in key ) and 
                 ( 'GENERAL_INFO' not in key ) ):
                indx = tokens[1] + '/' + tokens[2]
                HDATA[tokens[1], getflag[indx]].append(key)
            # end if
        # end for
        for x in HDATA:
            # NDM 3/4/2020: added sort=True to concat per Pandas warning
            temp = pd.concat( [store[path] for path in HDATA[x]], axis=1, 
                            sort=True )
            tempnames = temp.columns
            if x[0]=='RCHRES' and x[1]=='HYDRFG':
                for var in ['COLIN', 'OUTDG']:
                    names = [name for name in tempnames if var in name]
                    temp[var] = temp.apply( lambda x: tuple([x[name] for name in names]),
                                            axis=1 )
                    for name in names:
                        del temp[name]
                    # end inner for
                # end outer for
                for var in ['FUNCT', 'ODGTF', 'ODFVF']:
                    names = [name for name in tempnames if var in name]
                    for name in names:
                        temp[name] = temp[name].astype(int)
                    # end inner for
                    temp[var] = temp.apply(lambda x: tuple([x[name] for name in names]),axis=1)
                    for name in names:
                        del temp[name]
                    # end inner for
                # end of outer for
            # end if
            for indx,row in temp.iterrows():
                UCS[x[0], x[1], indx] = row
            # end for
            tokens = HDATA[x][0].split('/')
            for indx, row in store[tokens[1] + '/' + tokens[2] + '/' + 'SAVE'].iterrows():
                UCS[x[0], x[1], 'SAVE', indx] = row
            # end for
        # end for HDATA
        # end get UCS
        for row in store['/CONTROL/EXT_SOURCES'].itertuples():
            # get timeseries' info
            TSDD[ row.TVOL, row.TVOLNO ].append( row )
        # end for row
        # get data for LINK (combined NETWORK & SCHEMATIC) and MASSLINK information
        for _, row in store[ 'CONTROL/LINKS' ].iterrows():
            LINKDD[ row.TVOL, row.TVOLNO ].append( row )
        # end of LINKDD for
        for i,row in store[ 'CONTROL/MASS_LINK' ].iterrows():
            MLDD[ row.MLNO ].append( row )
        # end of MLDD for
        xflow = store[ '/HSP2/FLOWEXPANSION' ]
        for _, row in xflow.iterrows():
            XFLOWDD[ row.Flag, row.INFLOW ] = row
            LOOKUP[ row.Flag ].append( row.INFLOW )
        # end for
        # next get the operational sequence
        topSeq = store.get( key='/CONTROL/OP_SEQUENCE' )
        # want to change this to a numpy structured array for possible future
        #   speed ups
        fullList = list()
        for row in topSeq.itertuples(index=False):
            try:
                btarg = str( row[ 0 ] )
                bid = str( row[ 1 ] )
                sdelt = str( row[ 2 ] )
                fdelt = float( sdelt )
            except:
                # this is an error
                errMsg = "Error converting row %s of operational sequence!!!" % \
                         str( row )
                CL.LOGR.error( errMsg )
                return badReturn
            # now append to our list
            rowList = ( btarg, bid, sdelt, fdelt )
            fullList.append( rowList )
        # end of for
        ALLOPSEQ = np.array( fullList, dtype=AOS_DTYPE )
    # end of with block and HDF5 file closed.
    # now add some checks on what we got
    # start with SEQUENCE
    seqKeys = SEQUENCE.keys()
    nseqK = len( seqKeys )
    if nseqK < 1:
        # this is an error
        errMsg = "No keys found in %s for SEQUENCE!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # GENERAL
    genKeys = GENERAL.keys()
    if not KEY_GEN_START in genKeys:
        # this is an error
        errMsg = "No %s key in GENERAL in %s!!!" % ( KEY_GEN_START, hdfname )
        CL.LOGR.error( errMsg )
        return badReturn
    if not KEY_GEN_END in genKeys:
        # this is an error
        errMsg = "No %s key in GENERAL in %s!!!" % ( KEY_GEN_END, hdfname )
        CL.LOGR.error( errMsg )
        return badReturn
    # HKEYS
    if len( HKEYS.keys() ) < 1:
        # this is an error
        errMsg = "No keys found in %s for KEYS!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # UCS
    if len( UCS.keys() ) < 1:
        # this is an error
        errMsg = "No keys found in %s for UCS!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # TSDD
    if len( TSDD.keys() ) < 1:
        # this is an error
        errMsg = "No keys found in %s for TSDD!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # LINKDD
    if len( LINKDD.keys() ) < 1:
        # this maybe an error - means no RCHRES
        warnMsg = "No keys found in %s for LINKDD and no RCHRES " \
                  "defined !!!" % hdfname
        CL.LOGR.warning( warnMsg )
    # MLDD 
    if len( MLDD.keys() ) < 1:
        # this is an error
        errMsg = "No keys found in %s for MLDD!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # XFLOWDD
    if len( XFLOWDD.keys() ) < 1:
        # this is an error
        errMsg = "No keys found in %s for XFLOWDD!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # LOOKUP
    if len( LOOKUP.keys() ) < 1:
        # this is an error
        errMsg = "No keys found in %s for LOOKUP!!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # ALLOPSEQ
    if len( ALLOPSEQ ) < 1:
        # this is an error
        errMsg = "No operational sequence found in %s !!!" % hdfname
        CL.LOGR.error( errMsg )
        return badReturn
    # return
    return goodReturn


def initialHDFRead(hdfname, reloadkeys):
    """Determine the HDF5 file format and then call the method to read
    that format.

    Args:
        hdfname (str): HDF5 filename used for both input and output.
        reloadkeys (bool): Regenerates keys, used after adding new modules.
    
    Returns:
        int: function status, 0 == success

    """
    # imports
    # globals
    global HDF_FMT
    # before reading determine the format that expected
    HDF_FMT = detHDF5Format( hdfname )
    # now read our initial file
    if HDF_FMT == 0:
        retStat = origHDFRead( hdfname, reloadkeys )
    else:
        retStat = newHDFRead( hdfname )
    # end if
    # return
    return retStat


def setGTSDict( hdfname, simtimeinds, map_dict, gts ):
    """Set our global time series dictionary which contains each defined time series
    in the HDF5 file.
    
    The keys of the dictionary are the ts name which is SVOLNO.

    * Only SVOL == "*" is supported
    
    * Only MFACTOR as a number is supported

    **Note** that RCHRES COLIND and OUTDG have not been tested.

    Args:
        hdfname (str): FQDN path for the input HDF file
        simtimeinds (dict): SIMTIME_INDEXES from locaMain
        map_dict (dict): the mapping dictionary which will
                         be modified here
        gts (dict): the global time series dictionary which will
                    also be modified here.

    Returns:
        int: function status; success == 0

    """
    # import
    from locaHrchhyd import KEY_TS_COLIND, KEY_TS_OUTDGT
    from locaMain import DAILY_DELT_STR, TARG_RCHRES
    # globals
    global TSDD
    # parameter
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    allTSKeys = list( TSDD.keys() )
    tsIndex = simtimeinds[ DAILY_DELT_STR ]
    with pd.HDFStore( hdfname ) as store:
        for tKey in allTSKeys:
            cTargType = tKey[0]
            tsLister = list()
            tsdd = TSDD[tKey]
            for row in tsdd:
                try:
                    svolNum = str( row.SVOLNO )
                    tvolNum = str( row.TVOLNO )
                    tmemType = str( row.TMEMN )
                    tmemSB = str( row.TMEMSB )
                    svol = str( row.SVOL )
                    tran = str( row.TRAN )
                    mFact = float( row.MFACTOR )
                except:
                    # error
                    errMsg = "Issue parsing EXT_SOURCES line to TimeSeries\n" \
                             "Offending line %s !!!!" % str( row )
                    CL.LOGR.error( errMsg )
                    return badReturn
                # now set our path
                if svol != "*":
                    # this is an error
                    errMsg = "Expected EXT_SOURCES, SVOl to equal *\nIn this " \
                             "case SVOL is %s !!!" % svol
                    CL.LOGR.error( errMsg )
                    return badReturn
                # now make sure to set tran
                if len( tran ) < 3:
                    tran = "SAME"
                # end if
                # check for our key
                if not svolNum in gts.keys():
                    path = "%s%s" % ("TIMESERIES/", svolNum )
                    # now get the time series as a Pandas series
                    temp = store[path]
                    OurIndexer = temp.index.to_pydatetime()
                    OurIndexer = pd.DatetimeIndex( data=OurIndexer, freq='infer' )
                    OurVals = np.array( temp, dtype=np.float32 )
                    goodSeries = pd.Series( index=OurIndexer, data=OurVals )
                    goodSeries.fillna( value=0.0, inplace=True )
                    # now transform if needed
                    temp = transform(goodSeries, tsIndex, tran) * mFact
                    # then add it
                    gts[ svolNum ] = temp 
                # now are ready to populate our dictionaries
                # now check if RCHRES and add in extras if is
                if cTargType == TARG_RCHRES:
                    if tmemType in [ KEY_TS_COLIND, KEY_TS_OUTDGT ]:
                        # in this case try to process TMEMSB for the exit
                        if len( tmemSB ) < 1:
                            exitNo = -1
                        else:
                            intList = [ int(x) for x in tmemSB.split() ]
                            exitNo = intList[0]
                        # end if
                        tsLister.append( [ svolNum, tmemType, tvolNum, exitNo ] )
                    else:
                        tsLister.append( [ svolNum, tmemType, tvolNum ] )
                    # end if
                else:
                    tsLister.append( [ svolNum, tmemType, tvolNum ] )
                # end if
            # end of row for
            map_dict[tKey] = tsLister
        # end of key for 
    # end with and close the file
    # return
    return goodReturn


def setGFTabDict( hdfname, tdict, gftab ):
    """Set our global FTABLE dictionary which contains each defined FTABLE
    in the HDF5 file.
    
    The keys are the FTAB name which is SVOLNO.

    * Only SVOL == "*" is supported
    
    * Only MFACTOR as a number is supported

    **Requirements**: relies on TARG_DICT so must be called after checkOpsSpec

    Args:
        hdfname (str): FQDN path for the input HDF file
        tdict (dict): target dictionary from locaMain
        gftab (dict): FTAB dictionary from locaMain

    Returns:
        int: function status; success == 0
    
    """
    # import
    import re
    from locaMain import KEY_ACT_RRHYD, TARG_RCHRES, nKEY_ACT_RRHYD
    # globals
    global UCS, nUCI, HDF_FMT
    # parameter
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    allTargIDs = tdict[ TARG_RCHRES ]
    # open the hdfname file ...
    with pd.HDFStore( hdfname ) as store:
        # go through all of the RCHRES and collect FTABLES
        for tID in allTargIDs:
            # check the format before getting anything from UCS
            if HDF_FMT == 0:
                uc = UCS[ TARG_RCHRES, KEY_ACT_RRHYD, tID ]
            else:
                uc = nUCI[(TARG_RCHRES, nKEY_ACT_RRHYD, tID)]['PARAMETERS']
            # end if
            ftName = uc["FTBUCI"]
            # check for a valid name
            if not ftName:
                continue
            elif ( ftName == 0 ) or ( ftName == '0' ):
                continue
            # now check the length
            if len( ftName ) < 3:
                warnMsg = "Invalid FTABLE name of %s found for %s!!!" % \
                        ( ftName, tID )
                CL.LOGR.warning( warnMsg )
                continue
            # okay now parse the name
            intList = [ int(num) for num in re.findall( r'\d+', ftName ) ]
            if len( intList ) < 1:
                # this is an error
                errMsg = "Did not find an integer in FTABLE name %s!!!" % \
                        ftName 
                CL.LOGR.error( errMsg )
                return badReturn
            # now check the other way
            if len( intList ) > 1:
                warnMsg = "Found more than one integer in FTABLE name %s.\n" \
                        "Only use the first one." % ftName 
                CL.LOGR.warning( warnMsg )
            # now go
            ftNumber = intList[0]
            # check if already have it
            if ftNumber in gftab.keys():
                # then don't need to do anything else
                continue 
            # now process and add to our dictionary.
            pdFTable = store.get( "/FTABLES/%s" % ftName )
            raFTable = pdFTable.to_records( index=False )
            gftab[ftNumber] = raFTable
        # end for tID
    # end with and close the file
    # return
    return goodReturn


def getALLOPS( ):
    """Convenience function to return the module level global ALLOPSEQ

    Returns:
        numpy.recarray: ALLOPSEQ
    
    """
    global ALLOPSEQ
    return ALLOPSEQ


def getUCS( ):
    """Convenience function to return the module level global UCS
    """
    global UCS
    return UCS


def getGENERAL( ):
    """Convenience function to return the module level global GENERAL
    """
    global GENERAL
    return GENERAL


def getLINKDD( ):
    """Convenience function to return the module level global LINKDD
    """
    global LINKDD
    return LINKDD


def getMLDD( ):
    """Convenience function to return the module level global MLDD
    """
    global MLDD
    return MLDD


def getSEQUENCE():
    """Convenience function to get the module level global SEQUENCE
    """
    global SEQUENCE
    return SEQUENCE


def getUNITS():
    """Get the units from the GENERAL dictionary.

    Only works for the old format HDF5 file. Also, the units always
    have to be 1 because metric are not supported.

    Returns:
        int: integer telling which units are specified; 1 == standard;
            2 == metric

    """
    global HDF_FMT, GENERAL
    if HDF_FMT == 0:
        unitsInt = int( GENERAL['units'] )
    else:
        unitsInt = 1
    return unitsInt


def getnUCI():
    """Get the nUCI dictionary. For new format HDF5 files this replaces
    the UCS dictionary.

    Returns:
        defaultdict: nUCI, user control information from new format

    """
    global nUCI
    return nUCI


def getHDFFormat():
    """Get the integer format ID for this HDF5 file

    Returns:
        int: HDF5 file format; 0 == original; 1 == new
    
    """
    global HDF_FMT
    return HDF_FMT


def getMONTHLYs():
    """Get the dictionary of monthly values for the original or old
    HDF5 file format.

    Returns:
        defaultdict: MONTHLYS, key is (type, targID) which returns
                     a dictionary as the value. The sub-dictionary 
                     has keys that are parameter names and values 
                     that are a tuple of size 12.
    
    """
    global MONTHLYS
    return MONTHLYS

#EOF