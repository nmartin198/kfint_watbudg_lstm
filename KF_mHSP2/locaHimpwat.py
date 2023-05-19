# -*- coding: utf-8 -*-
"""
Replacement for *HSPsquared* himpwat that provides for water movement
and storage in impervious land segments, **IMPLND**.

Had to replace HSP2 himpwat so that can break into the main time loop
at the beginning and end of each day. This required fundamentally
restructuring the storage and memory allocation within HSP2.

locaHimpwat functions as a module handling storage for global
**IMPLND** variables as well as for parameter and constant 
definitions.

Internal time unit DELT60 is in hours for **IMPLND**.

"""
# Copyright and License
"""
Copyright 2023 Nick Martin

Module Author: Nick Martin <nick.martin@alumni.stanford.edu>

This file is part of a version of mHSP2 that was modified for
Kalman filter integration.

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
import numpy as np
import locaLogger as CL

# module wide parameters ------------------------------------
# original HSP2 parameters ++++++++++++++++++++++++++++++++++
MAXLOOPS  = int( 100 )
"""Newton method max loop iterations."""
TOLERANCE = float( 0.01 ) 
"""Tolerance for Newton method convergence"""
ERRMSG = ['IWATER: IROUTE Newton Method did not converge',    #ERRMSG0
]
"""Defined error messages from HSPF.

Used with errorsV for error handling. Currently, these messages 
written to the log file as errors when encountered.
"""
errorsV = np.zeros( len(ERRMSG), dtype=np.int32 )
"""Error tracking in liftedloop"""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# new module wide parameters
# want to keep the solution value and input value structures here

# some new control information
PARAM_GOOD = [ "LSUR", "NSUR", "PETMAX", "PETMIN", "RETSC", "SLSUR" ]
"""Input, non-state parameters that are used in mHSP2"""
PARAM_UNUSED = []
"""Input parameters that unused in mHSP2"""
FLAG_GOOD = [ "CSNOFG", "RTLIFG", "RTOPFG", "VNNFG", "VRSFG" ]
"""Flags that are at least referenced in mHSP2."""
FLAG_UNUSED = [ ]
"""Flags that are completely unused in mHSP2"""
MON_PARAMS = [ "NSURM", "RETSCM" ]
"""Parameter names for values that can be specified as monthly"""
MON_FLAGS = [ "VNNFG", "VRSFG" ]
"""Flag names for monthly flags"""
STATE_PARMS = [ "RETS", "SURS" ]
"""The list of state parameters"""
KEY_TS_PRECIP = "PREC"
"""External time series key for precipitation"""
KEY_TS_PET = "PETINP"
"""External time series key for input PET"""
INFLOW_TS_GOOD = [ KEY_TS_PRECIP, KEY_TS_PET, ]
"""All supported inflow time series in mHSP2"""
LAT_INFLOW_TS = [ "SURLI" ]
"""List of possible lateral inflow time series"""
INFLOW_TS_UNUSED = [ "AIRTMP", "GATMP", "DTMPG", "WINMOV", "SOLRAD", 
                     "CLOUD","SLSLD", "IQADFX", "IQADCN", "RAINF", 
                     "SNOCOV", "WYIELD", "PACKI" ]
"""Unsupported inflow time series"""
GOOD_OUTPUT_LIST = [ "IMPEV", "IMPS", "PET", "PETADJ", "RETS", "SUPY", 
                     "SURI", "SURO", "SURS" ]
"""List of currently supported outputs that can be written to the 
HDF5 file"""
BAD_OUTPUT_LIST = [  ]
"""Currently unsupported outputs """


# constants by simulation and location
DELT60 = None
"""Time step in hours to use in calculations"""
HOLD_MSUPY = None
"""Value of MSUPY to cover over between time steps"""
HOLD_DEC = None
"""Carry over calculation variable in case simulation is not daily """
HOLD_SRC = None
"""Carry over calculation variable in case simulation is not daily """

# data type specifications for making rec arrays
DEF_DT = None
"""The data type specification for time series structured array or 
record array"""
SPEC_DT = None
"""The data type specification for the calculation and input 
record arrays"""
FLAG_DT = None
"""The data type specification for flag record arrays"""

# Lateral inflow time series control
LATIN_CONTROL = None
"""Control structure telling which lateral inflows are active"""
OUTPUT_CONTROL = None
"""Control structure telling which time series are to be output"""
WS_AREAS = None
"""Areas of impervious segments in acres"""

# Holdover and carryover calculation variables
HR1FG = None
"""True at 1 am every day """

# Specified parameters that not monthly by pervious land target. These come from
#   the UCI file
# iwat-parm1
CSNOFG = None
"""Switch to turn on consideration of snow accumulation and melt.

SNOW is not currently supported."""
RTOPFG = None
"""Flag to select algorithm for overland flow.

RTOPFG == 1 then overland flow done as in predeccesor models - HSPX, 
ARM, and NPS. RTOPFG == 0 then a different algorithm is used."""
RTLIFG = None 
"""Flag for handling retention storage on lateral surface inflow.

If == 1, then lateral surface inflow is subject to retention. For 0,
not subject to retention
"""
VNNFG = None
"""Flag for using monthly variation in Manning's n for overland flow
"""
VRSFG = None
"""Flag for using monthly variation in retention storage capacity
"""
# iwat-parm2
LSUR = None
"""Length of the assumed overland flow plane in feet"""
NSUR = None
"""Manning's n for the assumed overland flow plane use English/Standard 
units versions from tables.
"""
RETSC = None
"""Retention, or interception, storage capacity of the surface in inches"""
SLSUR = None
"""Slope of the assumed overland flow plane, ft/ft """
# pwat-parm3
PETMAX = None
"""Air temperature below which ET will be arbitrarily reduced.

Only used if CSNOFG == 1. Units are degrees Fahrenheit. SNOW is not 
supported so this is not used.
"""
PETMIN = None
"""Air temperature below which ET will be set to zero.

Only used if CSNOFG == 1. Units are degrees Fahrenheit. SNOW is not 
supported and this is not used.
"""

#iwat-State1
I_RETS = None
"""Initial retention, or interception, storage in inches """
I_SURS = None
"""Initial surface or overland flow storage in inches """

# monthly arrays
NSURM = None 
"""Monthly overland flow Manning's n"""
RETSCM = None 
"""Monthly retention, or interception, storage capacity of the surface
in inches"""

# lateral inflow time series
SURLI = None 
"""Surface storage lateral inflow external time series, inches/ivld"""

# Data and simulated time series
IMPEV = None
"""Total simulated ET for impervious, inches/ivld """
IMPS = None
"""Total water stored in impervious lands, inches """
PET = None 
"""Potential evapotranspiration, inches/ivld"""
PETADJ = None
"""Adjusted PET from air temperature limits, inches/ivld """
RETS = None
"""Retention storage, inches"""
SUPY = None 
"""Moisture supplied to the land segment by precipitation, inches/ivld"""
SURI = None 
"""Surface storage inflow, inches/ivld"""
SURO = None 
"""Surface storage outflow, inches/ivld"""
SURS = None 
"""Surface storage, inches"""


def setDelT( sim_delt ):
    """Set the impervious land delt for calculations.

    The delt is stored as a module wide global

    Arguments:
        sim_delt (float): overall simulation time step in minutes

    """
    global DELT60
    # function
    DELT60 = sim_delt / 60.0
    # return
    return


def setupHR1FG( tIndex ):
    """Set the module-level global HR1FG

    Args:
        tIndex (pd.DateIndex): datetime index for the simulation output

    """
    # imports
    # globals
    global HR1FG
    # start
    HR1FG = np.where( tIndex.hour == 1, True, False )
    HR1FG[0] = True
    # return
    return


def setUpRecArrays( pwList, sim_len ):
    """ Create and initialize impervious land output arrays

    Args:
        pwList (list): list of IDs for this target type
        sim_len (int): number of output intervals in the simulation
    
    """
    # imports
    # globals
    global DEF_DT, SPEC_DT, FLAG_DT
    # time series
    global IMPEV, IMPS, PET, RETS, SUPY, SURI, SURS, SURO
    global PREC, PETINP, PETADJ
    # parameters
    global LSUR, NSUR, PETMAX, PETMIN, RETSC, SLSUR
    # initial states
    global I_RETS, I_SURS, WS_AREAS
    # flags
    global CSNOFG, RTLIFG, RTOPFG, VNNFG, VRSFG
    # lateral input control
    global LATIN_CONTROL, LAT_INFLOW_TS, SURLI
    # output control
    global OUTPUT_CONTROL, GOOD_OUTPUT_LIST
    # carry overs
    global HOLD_MSUPY, HOLD_DEC, HOLD_SRC
    # monthlys
    global NSURM, RETSCM
    # parameters
    # locals
    # start
    typeLister = list()
    tLister2 = list()
    tListerI = list()
    for tID in pwList:
        typeLister.append( ( tID, 'f4' ) )
        tLister2.append( ( tID, 'f8' ) )
        tListerI.append( (tID, 'i4') )
    # end for
    DEF_DT = np.dtype( typeLister )
    SPEC_DT = np.dtype( tLister2 )
    FLAG_DT = np.dtype( tListerI )
    # now go ahead and create all of our values
    # Time series
    IMPEV = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    IMPS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PET = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PETADJ = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    RETS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    SUPY = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    SURI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    SURS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    SURO = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PREC = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PETINP = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    # flags
    CSNOFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    RTLIFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    RTOPFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    VNNFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    VRSFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    LATIN_CONTROL = np.rec.array( np.zeros( len( LAT_INFLOW_TS ), 
                                  dtype=FLAG_DT ) )
    OUTPUT_CONTROL = np.rec.array( np.zeros( len( GOOD_OUTPUT_LIST), 
                                   dtype=FLAG_DT ) )
    # parameters
    LSUR = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    NSUR = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    PETMAX = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    PETMIN = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    RETSC = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    SLSUR = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    # initial states
    I_RETS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    I_SURS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) ) 
    # monthlys
    NSURM = np.rec.array( np.zeros( 12, dtype=SPEC_DT ) )
    RETSCM = np.rec.array( np.zeros( 12, dtype=SPEC_DT ) )
    # carryovers
    HOLD_MSUPY = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_DEC = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_SRC = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    WS_AREAS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    # return
    return


def setPrecipTS( targID, npTS ):
    """Set the precipitation time series from one data set
    to one target.

    SUPY is where precipitation is stored for calculations

    Args:
        targID (str): the target identifier - must be same as used
                        to create the rec array
        npTS (np.array): 1D array with the time series values

    """
    # imports
    # globals
    global SUPY
    # parameters
    # local
    # start
    SUPY[ targID ][:] += npTS 
    # return
    return


def setPETTS( targID, npTS ):
    """Set the PET time series from one data set
    to one target.

    PET is where pet is stored for calculations. Might be adjusted
    by various activities.

    Args:
        targID (str): the target identifier - must be same as used
                        to create the rec array
        npTS (np.array): 1D array with the time series values

    """
    # imports
    # globals
    global PET
    # parameters
    # local
    # start
    PET[ targID ][:] += npTS 
    # return
    return


def setWSAreas( targID, area ):
    """Set the watershed area to the global information structure.

    Area is in acres

    Args:
        targID (str): ID or recarray header to set
        area (float): area in acres

    """
    # globals
    global WS_AREAS
    # start
    WS_AREAS[targID][0] = area
    # return
    return


def setGoodFlag( targID, tFlag, fVal ):
    """Set the value for the specified flag structure

    Args:
        targID (str): ID or recarray header to set
        tFlag (str): flag string to identify the data structure
        fVal (int): flag value to set

    """
    # imports
    # globals
    global CSNOFG, RTOPFG, RTLIFG, VNNFG, VRSFG
    # parameters
    # locals
    # start
    if tFlag == "CSNOFG":
        CSNOFG[targID][0] = fVal
    elif tFlag == "RTOPFG":
        RTOPFG[targID][0] = fVal
    elif tFlag == "RTLIFG":
        RTLIFG[targID][0] = fVal
    elif tFlag == "VNNFG":
        VNNFG[targID][0] = fVal
    elif tFlag == "VRSFG":
        VRSFG[targID][0] = fVal
    # return
    return


def setGoodParam( targID, tParam, pVal ):
    """Set the value for the specified parameter structure

    Args:
        targID (str): ID or recarray header to set
        tParam (str): param string to identify the data structure
        pVal (float): parameter value to set

    """
    # imports
    # globals
    global LSUR, NSUR, PETMAX, PETMIN, RETSC, SLSUR
    # parameters
    # locals
    # start
    if tParam == "LSUR":
        LSUR[ targID ][0] = pVal
    elif tParam == "NSUR":
        NSUR[ targID ][0] = pVal
    elif tParam == "PETMAX":
        PETMAX[ targID ][0] = pVal
    elif tParam == "PETMIN":
        PETMIN[ targID ][0] = pVal
    elif tParam == "RETSC":
        RETSC[ targID ][0] = pVal
    elif tParam == "SLSUR":
        SLSUR[ targID ][0] = pVal
    # return
    return


def setMonthlyParams( targID, monName, monTuple ):
    """Set the value for the specified monthly parameter structures

    Args:
        targID (str): ID or recarray header to set
        monName (str): name for data structure to set the monthly values
        monTuple (tuple): tuple of 12 floats which are the values.

    """
    # imports
    # globals 
    global NSURM, RETSCM
    # parameters
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    if monName == "NSURM":
        NSURM[ targID ][:] = np.array( monTuple )
    elif monName == "RETSCM":
        RETSCM[ targID ][:] = np.array( monTuple )
    else:
        return badReturn
    # return
    return goodReturn


def setStateParams( targID, tParam, pVal ):
    """Set the value for the specified initial state structure

    Args:
        targID (str): ID or recarray header to set
        tParam (str): param string to identify the data structure
        pVal (float): parameter value to set

    """
    # imports
    # globals
    global I_RETS, I_SURS
    # parameters
    # locals
    # start
    if tParam == 'RETS':
        I_RETS[ targID ][0] = pVal
    elif tParam == 'SURS':
        I_SURS[ targID ][0] = pVal
    # return
    return


def configExternalTS( sim_len, TSMapList, AllTSDict ):
    """Transfer external time series from HDF5 input to module data
    structures

    Args:
        sim_len (int): the length of the simulation
        TSMapList (list): nested list with sublists, L, of time series 
            metadata for a particular target ID
            
            0. time series type

            1. time series ID

            2. target ID
        
        AllTSDict (dict): dictionary of time series by time series ID
    
    Returns:
        int: function status; 0 == success

    """
    # imports
    # global
    global INFLOW_TS_UNUSED, INFLOW_TS_GOOD, LAT_INFLOW_TS, KEY_TS_PRECIP
    global KEY_TS_PET
    # parameter
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    # set up a lateral dictionary for this target
    latDict = dict()
    # go throught the mapping list and allocate
    for mapList in TSMapList:
        tsType = mapList[1]
        tsID = mapList[0]
        tsTargID = mapList[2]
        # now check and see if the external type is supported
        if tsType in INFLOW_TS_UNUSED:
            # this means that cannot use this so provide 
            #   a warning
            warnMsg = "Currently external time series of type " \
                        "%s are unsupported. This time series " \
                        "will be ignored for %s - %s !!!" % \
                        ( tsType, "IMPLND", tsTargID )
            CL.LOGR.warning( warnMsg )
            continue
        # check if in the good list
        if tsType in INFLOW_TS_GOOD:
            # first check if a lateral inflows
            if tsType in LAT_INFLOW_TS:
                # add information to the lateral dictionary
                if tsTargID in latDict.keys():
                    latDict[ tsTargID ].append( [ tsType, tsID ] )
                else:
                    latDict[ tsTargID ] = [ [ tsType, tsID ] ]
            elif tsType == KEY_TS_PRECIP:
                # then this is a precip time series
                pVals = np.array( AllTSDict[tsID].values, dtype=np.float32 )
                setPrecipTS( tsTargID, pVals )
            elif tsType == KEY_TS_PET:
                # then this is a PET time series
                etVals = np.array( AllTSDict[tsID].values, dtype=np.float32 )
                setPETTS( tsTargID, etVals )
            else:
                # unsupported time series type
                errMsg = "Unsupported time series type %s " \
                         "found for %s - %s!!!" % \
                         ( tsType, "IMPLND", tsTargID )
                CL.LOGR.error( errMsg )
                return badReturn
            # end inner if
        else:
            # unknown and unsupported time series type
            warnMsg = "Unknown and unsupported time series type %s " \
                      "found for %s - %s. Time series will be " \
                      "ignored!!!"  % ( tsType, "IMPLND", tsTargID )
            CL.LOGR.warning( warnMsg )
            continue
        # end if
    # end for maplist
    # now process the lateral inflows
    LIKeys = list( latDict.keys() )
    # we expect there to be only one key but may be
    #  zero keys if no lateral inflows
    if len( LIKeys ) < 1:
        return goodReturn
    # now process the dictionary
    for tsTargID in LIKeys: 
        AllTypes = [ x[0] for x in latDict[tsTargID] ]
        AllTS = [ x[1] for x in latDict[tsTargID] ]
        UniqueTypes = set( AllTypes )
        numType = len( AllTypes )
        for cType in UniqueTypes:
            cInds = [ iI for  iI in range( numType ) if AllTypes[iI] == cType ]
            cTSIDs = [ AllTS[x] for x in cInds ]
            for tsID in cTSIDs:
                liVals = np.array( AllTSDict[tsID].values, dtype=np.float32 )
                setLatInflowTS( sim_len, tsTargID, cType, liVals )
            # end for tsID
        # end for cType
    # end for tsTargID
    # return
    return goodReturn


def setLatInflowTS( sim_len, targID, inflowType, tsVals ):
    """Method to setup lateral inflow time series

    Args:
        sim_len (int): the simulation length
        targID (str): target ID
        inflowType (str): type of lateral inflow
        tsVals (np.array): array of time series values
    
    """
    # import
    # globals
    global DEF_DT, SURLI, LATIN_CONTROL, LAT_INFLOW_TS
    # parameters
    # locals
    # start
    # set the control value to 1 for this type
    # find the index in the LAT_INFLOW_TS lists
    liIndex = LAT_INFLOW_TS.index( inflowType )
    LATIN_CONTROL[ targID ][ liIndex ] = 1
    # now setup the time series or assign the time series
    if inflowType == "SURLI":
        if SURLI is None:
            # initialize
            SURLI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
        else:
            SURLI[ targID ][:] = tsVals
    # return
    return


def setOutputControlFlags( targID, savetable, stTypes ):
    """Set the output control flags

    Args:
        targID (str): target id
        savetable (np.array or dict): boolean array of which outputs to save
        stTypes (list): keys or indexes to save

    Returns:
        int: function status; 0 == success

    """
    # imports
    # globals
    global GOOD_OUTPUT_LIST, BAD_OUTPUT_LIST, OUTPUT_CONTROL
    # parameters
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    for sType in stTypes:
        sVal = int( savetable[sType] )
        if sType in BAD_OUTPUT_LIST:
            if sVal > 0:
                # give a warning
                warnMsg = "Output type %s is not supported for IMPLND!!!" \
                        % sType
                CL.LOGR.warning( warnMsg )
            # now continue
            continue
        elif sType in GOOD_OUTPUT_LIST:
            stInd = GOOD_OUTPUT_LIST.index( sType )
            OUTPUT_CONTROL[targID][stInd] = sVal
        else:
            # this is an error because undefined type
            errMsg = "Undefined output type of %s for IMPLND!!!" % \
                     sType
            CL.LOGR.error( errMsg )
            return badReturn
        # end if
    # end for sType
    # return
    return goodReturn


def configFlagsParams( targID, cFlagVals, allIndexes ):
    """Set and configure flags and parameters for PERLND

    Args:
        targID (str): the target location ID
        cFlagVals (dict): collected flag values
        allIndexes (list): list of indexes for cFlagVals
    
    Returns:
        int: function status; 0 == success

    """
    # imports
    # globals
    global PARAM_GOOD, PARAM_UNUSED, MON_FLAGS, MON_PARAMS, FLAG_GOOD
    global FLAG_UNUSED, STATE_PARMS
    # parameters
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    # first do the good flags
    for fV in FLAG_GOOD:
        if not fV in allIndexes:
            # this is an error
            errMsg = "Did not find flag %s in run setup!!!" % \
                        fV
            CL.LOGR.error( errMsg )
            return badReturn
        # now update our values
        setGoodFlag( targID, fV, int( cFlagVals[fV] ) )
        # next look and see if the flag is in monthly
        if fV in MON_FLAGS:
            cFVal = int( cFlagVals[fV] )
            if cFVal > 0:
                # then set the monthly values
                checkIndex = MON_FLAGS.index( fV )
                monParamName = MON_PARAMS[ checkIndex ]
                if monParamName in allIndexes:
                    setMonthlyParams( targID, monParamName, cFlagVals[monParamName] )
                # end if
            # end if
        # end if
    # end for good flag
    # next give warnings for bad flags
    for fV in FLAG_UNUSED:
        if not fV in allIndexes:
            continue
        # if made it here write a warning
        cFVal = int( cFlagVals[fV] )
        if cFVal > 0:
            warnMsg = "Flag %s is set to %d\nThis functionality " \
                        "is not currently implemented!!!" % \
                        ( fV, cFVal )
            CL.LOGR.warning( warnMsg )
    # end for bad flag
    # now do the good parameters
    for fV in PARAM_GOOD:
        if not fV in allIndexes:
            # this is an error
            errMsg = "Did not find param %s in run setup!!!" % \
                        fV
            CL.LOGR.error( errMsg )
            return badReturn
        # now update our values
        setGoodParam( targID, fV, float( cFlagVals[fV] ) )
    # end for good param
    # give warnings for bad parameters
    for fV in PARAM_UNUSED:
        if not fV in allIndexes:
            continue
        # if made it here write a warning
        cFVal = float( cFlagVals[fV] )
        if cFVal > 0:
            warnMsg = "Param %s is set to %g\nThis functionality " \
                        "is not currently implemented!!!" % \
                        ( fV, cFVal )
            CL.LOGR.warning( warnMsg )
    # end for bad flag
    # finally do the state variables
    for fV in STATE_PARMS:
        if not fV in allIndexes:
            # this is an error
            errMsg = "Did not find state variable %s in run setup!!!" % \
                        fV
            CL.LOGR.error( errMsg )
            return badReturn
        # now update our values
        setStateParams( targID, fV, float( cFlagVals[fV] ) )
    # end for state
    # return
    return goodReturn


def getLatInflowByTypeTarget( targID, liType, iI ):
    """Get a lateral inflow by type and target

    Args:
        targID (str): target ID
        liType (str): lateral inflow type
        iI (int): current time index

    """
    global SURLI, LATIN_CONTROL, LAT_INFLOW_TS
    # start
    # set the control value to 1 for this type
    # find the index in the LAT_INFLOW_TS lists
    liIndex = LAT_INFLOW_TS.index( liType )
    flagVal = LATIN_CONTROL[ targID ][ liIndex ]
    if flagVal > 0:
        if liType == "SURLI":
            retVal = float( SURLI[ targID ][iI] )
        else:
            retVal = 0.0
        # end if
    else:
        retVal = 0.0
    # return
    return retVal


def iwater_liftedloop( iI, mon, targID ):
    """Modified version of liftedloop to do a single time step and
    return to the main time loop. 
    
    Module-wide recarrays are used to store all results and calculation
    variables between calls. Modified real number comparisons to be 
    more numerically reliable.

    Args:
        iI (int): index of current time step (0 to (sim_len-1))
        mon (int): current simulation month
        targID (str): ID for recarray columns

    Returns:
        int: count of the number of errors. Should be 0, but this
            provides a way to reference errorsV for error handling

    """
    # imports
    from copy import deepcopy
    from math import pow, sqrt, log, exp
    # globals
    # carry overs
    global HOLD_MSUPY, HOLD_DEC, HOLD_SRC
    # calculation constants
    global errorsV, DELT60, MAXLOOPS, TOLERANCE
    # switches, parameters, and flags
    global HR1FG, NSUR, NSURM, RETSC, RETSCM, VNNFG, VRSFG
    global RTLIFG, RTOPFG, LSUR, SLSUR, CSNOFG, PETMAX, PETMIN
    # data time series
    global SUPY, PET, PETINP, PETADJ
    # initial states
    global I_RETS, I_SURS
    # save time series only
    global RETS, SURS, IMPEV, IMPS, SURI, SURO
    # parameters
    smallVal = float( 1E-13 )
    # locals - put here in case want to Cython
    reto = float( 0.0 )     # retention storage outflow
    suri = float( 0.0 )     # surface storage inflow
    suro = float( 0.0 )     # surface storage outflow 
    dummy = float( 0.0 )    # calculation var
    sursm = float( 0.0 )    # mean surface detention storage
    surse = float( 0.0 )    # equilibrium surf detention storage
    tsuro = float( 0.0 )    # calc var
    d = float( 0.0 )        # calculation variable
    impev = float( 0.0 )    # et calclated
    ssupr = float( 0.0 )    # rate of moisture supply to overland flow
    sursnw = float( 0.0 )   # calc var
    ratio = float( 0.0 )    # calc var
    fact = float( 0.0 )     # calc var
    ffact = float( 0.0 )    # calc var
    fsuro = float( 0.0 )    # calc var
    dfsuro = float( 0.0 )   # calc var
    dsuro = float( 0.0 )    # calc var
    dfact = float( 0.0 )    # calc var
    errorCnt = int( 0 )     # error counter
    # initialize our calculation variables
    hr1fg = HR1FG[ iI ]
    # flags
    fl_csnofg = int( CSNOFG[targID][0] )
    # not supported so hardwire off
    fl_csnofg = int( 0 )
    # other flags are supported
    fl_rtlifg = int( RTLIFG[targID][0] )
    fl_rtopfg = int( RTOPFG[targID][0] )
    # parameters 
    lsur = float( LSUR[targID][0] )
    slsur = float( SLSUR[targID][0] )
    petmax = float( PETMAX[targID][0] )
    petmin = float( PETMAX[targID][0] )
    # check if using monthly
    if VNNFG[targID][0] > 0:
        nsur = NSURM[targID][(mon-1)]
    else:
        nsur = NSUR[targID]
    if VRSFG[targID][0] > 0:
        retsc = RETSCM[targID][(mon-1)]
    else:
        retsc = RETSC[targID]
    # get our time series values
    # include the PET adjustment values here
    if fl_csnofg == 1:
        # if sno turned on
        # initialize dummy variables for currently unsupported
        # time series
        ts_airtmp = float( 0.0 )
        ts_snocov = float( 0.0 )
        ts_rainf = float( 0.0 )
        ts_wyield = float( 0.0 )
        ts_petinp = float( PET[targID][iI] )
        # do supy first
        ts_supy = ts_rainf * ( 1.0 - ts_snocov ) + ts_wyield 
        petadj = 1.0 - ts_snocov
        #if (airtmp < PETMAX[loop])and petadj > 0.5:
        if ( ( ( petmax - ts_airtmp) >= smallVal ) and 
                        ( ( petadj - 0.5 ) >= smallVal ) ):
            petadj = 0.5
        # end if
        # if airtmp < PETMIN[loop]:
        if ( ( petmin - ts_airtmp ) >= smallVal ):
            petadj = 0.0
        # end if
        ts_pet = ts_petinp * petadj
    else:
        ts_supy = float( SUPY[targID][iI] )
        ts_pet = float( PET[targID][iI] )
    # set up calculation values for state
    if iI == 0:
        rets = float( I_RETS[targID][0] )
        surs = float( I_SURS[targID][0] )
    else:
        rets = float( RETS[targID][iI-1] )
        surs = float( SURS[targID][iI-1] )
    # get the lateral inflow values
    lits_surli = getLatInflowByTypeTarget( targID, "SURLI", iI )
    # set the previous msupy using a copy of the global
    oldmsupy = float( HOLD_MSUPY[targID][0] )
    dec = float( HOLD_DEC[targID][0] )
    src = float( HOLD_SRC[targID][0] )
    # set retiV
    if fl_rtlifg == 1:
        retiV = ts_supy + lits_surli
        # RETN
        rets = rets + retiV
        # if rets > retsc
        if ( ( rets - retsc ) > smallVal ):
            reto = rets - retsc
            rets = retsc
        else:
            reto = 0.0
        # end if
        suri = reto
    else:
        retiV = ts_supy
        # RETN
        rets = rets + retiV
        # if rets > retsc
        if ( ( rets - retsc ) > smallVal ):
            reto = rets - retsc
            rets = retsc
        else:
            reto = 0.0
        # end if
        suri = reto + lits_surli
    # end if
    msupy = suri + surs
    suro = 0.0
    # 2021 sometime, the structure of this calculation has changed as HSP2 has
    #  evolved. Modify this to follow the new structure.
    if ( ( msupy - 0.0002 ) > smallVal ):
        if fl_rtopfg == 1:
            # IROUTE for RTOPFG==True, the way it is done in arm, nps, and hspx
            if ( abs( oldmsupy - 0.0 ) < smallVal ) or hr1fg:
                # Time to recompute
                dummy  = nsur * lsur
                dec = 0.00982 * pow( ( dummy / sqrt( slsur ) ), 0.6 )
                src = 1020.0 * ( sqrt( slsur ) / dummy )
            # end if
            sursm = ( surs + msupy ) * 0.5
            dummy = sursm * 1.6
            if ( ( suri - 0.0 ) > smallVal ):
                d = dec * pow( suri, 0.6 )
                #if d > sursm:
                if ( ( d - sursm ) > smallVal ):
                    surse = d
                    dummy = sursm * ( 1.0 + 0.6 * 
                                        pow( (sursm / surse), 3.0 ) )
                # end inner if
            # end outer if
            tsuro = DELT60 * src * pow( dummy, 1.67 )
            if ( ( tsuro - msupy ) > smallVal ):
                suro = msupy
                surs = 0.0
            else:
                suro = tsuro
                surs = msupy - suro
            # end if
        else:
            # IROUTE for RTOPFG==False
            #if oldmsupy == 0.0 or HR1FG[loop]:
            if ( abs( oldmsupy - 0.0 ) <= smallVal ) or hr1fg:
                # Time to recompute
                dummy = nsur * lsur
                dec = 0.00982 * pow( ( dummy / sqrt(slsur) ), 0.6 )
                src = 1020.0 * sqrt( slsur ) / dummy
            # end if
            ssupr  = suri / DELT60
            # if ssupr > 0.0
            if ( ( ssupr - 0.0 ) > smallVal ):
                surse = dec * pow( ssupr, 0.6 )
            else:
                surse = 0.0
            # end if
            sursnw = msupy
            suro = 0.0
            # now loop
            for count in range(MAXLOOPS):
                # if ssupr > 0.0:
                if ( ( ssupr - 0.0 ) > smallVal ):
                    ratio = sursnw / surse
                    # if ratio <= 1.0
                    if ( ( 1.0 - ratio ) > ( -1.0 * smallVal) ):
                        fact = 1.0 + 0.6 * pow( ratio, 3.0 )
                    else:
                        fact = 1.6
                    # end inner if
                else:
                    fact  = 1.6
                    ratio = 1e30
                # end if
                ffact  = ( ( DELT60 * src * pow( fact, 1.667 ) ) * 
                                pow( sursnw, 1.667 ) )
                fsuro  = ffact - suro
                dfact  = -1.667 * ffact
                dfsuro = ( dfact / sursnw ) - 1.0
                #if ratio <= 1.0:
                if ( ( 1.0 - ratio ) > ( -1.0 * smallVal ) ):
                    dfsuro += ( ( dfact / ( fact * surse ) ) * 1.8 * 
                                    pow( ratio, 2.0 ) )
                # end if
                dsuro = fsuro / dfsuro
                suro = suro - dsuro
                sursnw = msupy - suro
                # check if meet tolerance
                #if abs(dsuro / suro) < TOLERANCE:
                if ( abs( dsuro / suro ) < TOLERANCE ):
                    break
                # end if
            else:
                # IROUTE did not converge
                errorsV[0] = errorsV[0] + 1
                errorCnt += 1
                errMsg = "iwater_liftedloop - %s " % ERRMSG[0]
                CL.LOGR.error( errMsg )
            # end for else
        # end routing method if
    else:
        # msupy essentially zero
        suro = msupy
        surs = 0.0
    # end if
    # this section replaces EVRETN
    #if rets > 0.0:
    if ( ( rets - 0.0 ) > smallVal ):
        # if pet > rets
        if ( ( ts_pet - rets ) > smallVal ):
            impev = rets
            rets = 0.0
        else:
            impev = ts_pet
            rets = rets - impev
        # end inner if
    else:
        impev = 0.0
        rets = 0.0
    # end if rets
    #save results
    # update the carry overs
    HOLD_MSUPY[targID][0] = msupy
    HOLD_DEC[targID][0] = dec
    HOLD_SRC[targID][0] = src
    # update the time series
    RETS[targID][iI] = rets
    IMPEV[targID][iI] = impev
    SURI[targID][iI] = suri
    SURO[targID][iI] = suro
    SURS[targID][iI] = surs
    IMPS[targID][iI] = rets + surs
    if fl_csnofg == 1:
        PETADJ[targID][iI] = petadj
    # return
    return errorCnt


def writeOutputs( store, tIndex ):
    """Write the outputs to the hdf file at the end of the simulation

    Args:
        store (pd.HDFStore): hdf5 file store to write to
        tIndex (pd.DateIndex): time index for the simulation

    Returns:
        int: function status; 0 == success

    """
    # imports
    import pandas as pd
    # globals
    global GOOD_OUTPUT_LIST, OUTPUT_CONTROL
    global IMPEV, IMPS, PET, PETADJ, RETS, SUPY, SURI, SURS, SURO
    # parameters
    goodReturn = 0
    badReturn = -1
    pathStart = "/RESULTS/IMPLND_"
    pathEnd = "/IWATER"
    # locals
    # start
    # get the columns list for recarrays
    colsList = list( OUTPUT_CONTROL.dtype.names )
    # go through by target and output
    for tCol in colsList:
        # get the path
        path = "%s%s%s" % ( pathStart, tCol, pathEnd )
        # create an empty DataFrame with a time index
        df = pd.DataFrame(index=tIndex)
        iCnt = 0
        for cOut in GOOD_OUTPUT_LIST:
            # first check our output control
            if OUTPUT_CONTROL[tCol][iCnt] == 0:
                # skip this output
                iCnt += 1
                continue
            # end if
            if cOut == "IMPEV":
                outView = IMPEV[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "IMPS":
                outView = IMPS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "PET":
                outView = PET[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "PETADJ":
                outView = PETADJ[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "RETS":
                outView = RETS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "SUPY": 
                outView = SUPY[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "SURI":
                outView = SURI[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "SURO":
                outView = SURO[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "SURS":
                outView = SURS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            else:
                # this is an error - unsupported output
                errMsg = "Unsupported output type of %s!!!!" % cOut
                CL.LOGR.error( errMsg )
                return badReturn
            # end if
            # increment our counter
            iCnt += 1
        # end for output type
        # have the dataframe to save to our path
        store.put( path, df.astype( np.float32 ) )
    # end of column for
    # return
    return goodReturn


def getWatershedAreabyTarg( targID ):
    """Get WS_AREA or watershed area by target id

    Args:
        targID (str): current PERLND target
    
    Returns:
        float: watershed areas in acres

    """
    # globals
    global WS_AREAS
    # get
    warea = float( WS_AREAS[targID][0] )
    # return
    return warea


def getSURObyTargTS( iI, targID ):
    """Get the total outflow from impervious land, inches/day

    Args:
        iI (int): time step index to extract the storage values
        targID (str): current PERLND target

    Return:
        float: suro, total outflow in inches/day

    """
    # global
    global SURO
    #
    suro = float( SURO[targID][iI] )
    # return
    return suro


# EOF