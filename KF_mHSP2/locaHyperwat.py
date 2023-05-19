# -*- coding: utf-8 -*-
"""
Replacement for *HSPsquared* hperwat which provides for water movement 
and storage in pervious land segments, i.e. **PERLND**.

Had to replace HSPsquared hyperwat so that can break into the main time 
loop at the beginning and end of each day. This required fundamentally
restructuring the storage and memory allocation within HSP2.

*locaHyperwat* functions as a module handling storage for global
**PERLND** variables as well as for parameter and constant 
definitions.

Internal time unit, DELT60, is in hours for **PERLND**.

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
"""Newton method max loops"""
ERRMSG = ['PWATER: Sum of irrtgt is not one',             #ERRMSG0
          'PWATER: UZRAA exceeds UZRA array bounds',      #ERRMSG1
          'PWATER: INTGB exceeds INTGRL array bounds',    #ERRMSG2
          'PWATER: Reduced AGWO value to available',      #ERRMSG3
          'PWATER: Reduced GWVS to AGWS',                 #ERRMSG4
          'PWATER: GWVS < -0.02, set to zero',            #ERRMSG5
          'PWATER: Proute runoff did not converge',       #ERRMSG6
          'PWATER: UZI highly negative',                  #ERRMSG7
          'PWATER: Reset AGWS to zero']                   #ERRMSG8
"""Defined error messages - can be used with errorsV for error handling.

Currently, these messages written to the log file as errors when 
encountered.
"""
errorsV = np.zeros( len(ERRMSG), dtype=np.int32 )
"""Error handling in liftedloop from original HSPF formulation."""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# new module wide parameters
# want to keep the solution value and input value structures here

# some new control information
PARAM_GOOD = [ "AGWETP", "AGWRC", "BASETP", "CEPSC", "DEEPFR", 
               "FOREST", "FZG", "FZGL", "INFEXP", "INFILD", "INFILT",
               "INTFW", "IRC", "KVARY", "LSUR", "LZETP", "LZSN", "NSUR",
               "PETMAX", "PETMIN", "SLSUR", "UZSN" ]
"""Input, non-state parameters that are used in mHSP2"""
PARAM_UNUSED = [ "BELV", "DELTA", "GWDATM", "IFWSC", "LELFAC", "MELEV",
                 "PCW", "PGW", "SREXP", "SRRC", "STABNO", "UELFAC", 
                 "UPGW" ]
"""Input parameters that are unused or unsupported in mHSP2"""
FLAG_GOOD = [ "CSNOFG", "ICEFG", "RTOPFG", "UZFG", "VCSFG", "VIFWFG", "VIRCFG",
              "VLEFG", "VNNFG", "VUZFG" ]
"""Flags that are at least referenced in mHSP2"""
FLAG_UNUSED = [ "HWTFG", "IFFCFG", "IFRDFG", "IRRGFG" ]
"""Flags that are completely unused and unsupported in mHSP2"""
MON_PARAMS = [ "LZETPM", "CEPSCM", "INTFWM", "IRCM", "NSURM", "UZSNM" ]
"""Parameter names for values that can be specified as monthly"""
MON_FLAGS = [ "VLEFG", "VCSFG", "VIFWFG", "VIRCFG", "VNNFG", "VUZFG" ]
"""Flag names for monthly flags"""
STATE_PARMS = [ 'CEPS', 'SURS', 'UZS', 'IFWS', 'LZS', 'AGWS', 'GWVS' ]
"""The list of state parameters"""
KEY_TS_PRECIP = "PREC"
"""External time series key for precipitation"""
KEY_TS_PET = "PETINP"
"""External time series key for input PET"""
INFLOW_TS_GOOD = [ KEY_TS_PRECIP, KEY_TS_PET, "AGWLI", "IFWLI", "LZLI", 
                   "SURLI", "UZLI" ]
"""All supported inflow time series"""
LAT_INFLOW_TS = [ "AGWLI", "IFWLI", "LZLI", "SURLI", "UZLI" ]
"""List of possible lateral inflow time series"""
INFLOW_TS_UNUSED = [ "GATMP", "DTMPG", "WINMOV", "SOLRAD", "CLOUD",
                     "SLSED", "PQADFX", "PQADCN", "PEADFX", "PEADCN",
                     "NIADFX", "NIADCN", "PHADFX", "PHADCN", "TRADFX",
                     "TRADCN", "RAINF", "SNOCOV", "WYIELD", "PACKI",
                     "LGTMP" ]
"""Unsupported inflow time series names"""
GOOD_OUTPUT_LIST = [ "AGWET", "AGWI", "AGWO", "AGWS", "BASET", "CEPE", 
                     "CEPS", "GWVS", "IFWI", "IFWO", "IFWS", "IGWI",
                     "INFFAC", "INFIL", "LZET", "LZI", "LZS", "RPARM", 
                     "PERC", "PERO", "PERS", "PET", "PETADJ", "SUPY", 
                     "SURI", "SURO", "SURS", "TAET", "TGWS", "UZET", 
                     "UZI", "UZS" ]
"""List of currently supported time series outputs.

These can be written to the HDF5 file at the end of simulation time.
"""
BAD_OUTPUT_LIST = [ "GWEL", "IRDRAW", "IRRAPP", "IRRDEM", "IRSHRT",
                    "RZWS", "SURET" ]
"""Currently unsupported outputs """

# constants by simulation and location
DELT60 = None
"""Time step in hours to use in calculations"""
IRRAPPV = np.zeros( 7, dtype=np.float64 )
"""Irrigation paramater? - set to zeros so does not matter for logic."""
IRRCEP = float( 0.0 )
"""Unknonwn parameter - set to zero so does not matter"""
UZRA = np.array( [ 0.0, 1.25, 1.50, 1.75, 2.00, 2.10, 2.20, 2.25, 2.5, 4.0 ], 
                 dtype=np.float64 )
"""Function coordinates for evaluating upper zone behavior"""
INTGRL = np.array( [ 0.0, 1.29, 1.58, 1.92, 2.36, 2.81, 3.41, 3.8,  7.1, 3478. ],
                   dtype=np.float64 )
"""Function coordinates for evaluating upper zone behavior"""

# data type specifications for making rec arrays
DEF_DT = None
"""The data type specification for time series structured array or record array"""
SPEC_DT = None
"""The data type specification for the calculation and input record arrays"""
FLAG_DT = None
"""The data type specification for flag record arrays"""

# Control structures
LATIN_CONTROL = None
"""Control structure telling which lateral inflows are active"""
OUTPUT_CONTROL = None
"""Control structure telling which time series are to be output"""
WS_AREAS = None
"""Watershed areas for pervious segments in acres"""

# Holdover and carryover calculation variables
DAYFG = None
"""Calculation control to determine when day starts/ends """
HOLD_MSUPY = None
"""Value of MSUPY to cover over between time steps"""
HOLD_RLZRAT = None
"""Carry overs between calculations """
HOLD_LZFRAC = None
"""Carry overs between calculations"""
HOLD_DEC = None
"""Carry over calculation variable in case is not daily """
HOLD_SRC = None
"""Carry over calculation variable in case is not daily """
HOLD_IFWK1 = None
"""Carry over calculation variable in case is not daily"""
HOLD_IFWK2 = None
"""Carry over calculation variable in case is not daily"""

# Specified parameters that not monthly by pervious land target. These come from
#   the UCI file
# pwat-parm1
CSNOFG = None
"""Switch to turn on consideration of snow accumulation and melt.

SNOW calculations are currently unsupported and so this switch is 
hard coded to off.
"""
RTOPFG = None
"""Flag to select the algorithm for overland flow.

RTOPFG == 1 then overland flow done as in predeccesor models - HSPX, 
ARM, and NPS. RTOPFG == 0 then a different algorithm is used."""
UZFG = None
"""UZFG selects the method for computing inflow to the upper zone.

If UZFG is 1, upper zone inflow is computed in the same way as in 
the predecessor models HSPX, ARM and NPS. A value of 0 results in 
the use of a different algorithm, which is less sensitive to changes 
in DELT
"""
VLEFG = None
"""Flag for monthly variation in lower zone ET parameter.

If 1 then the flag is on.
"""
VCSFG = None
"""Flag for monthly variation in interception storage capacity."""
VUZFG = None
"""Flag for monthly variation in upper zone nominal storage."""
VNNFG = None
"""Flag for monthly variation in Manning's n for overland flow."""
VIFWFG = None
"""Flag for monthly variation in interflow inflow parameter."""
VIRCFG = None
"""Flag for monthly variation in interflow recession paramater."""
# pwat-parm2
FOREST = None
"""FOREST is the fraction of the PERLND which is covered by forest.

This is only relevant if CSNOFG == 1 and then forest fraction will
continue to transpire in winter. As SNOW is not supported, this is
never relevant in the current mHSP2 implementation.
"""
LZSN = None
"""Lower soil zone nominal storage depth in inches."""
INFILT = None
"""Index to infiltration capacity of the soil, inches/invl"""
LSUR = None
"""Length of the assumed overland flow plane in feet"""
SLSUR = None
"""Slope of the assumed overland flow plane, ft/ft """
KVARY = None
"""Parameter that affects behavior of groundwater recession flow.

Purpose is to allow recession flow to be non-exponential in its
decay time. Units are 1/in."""
KGWV = None
"""Groundwater outflow recession parameter 1/day.

Calculated internally from AGWRC."""
AGWRC = None
"""Basic groundwater recession rate if KVARY is 0 and there is
no inflow to groundwater.

Defined as the rate of flow today divided by the rate of flow 
yesterday. Units are 1/day."""
# pwat-parm3
PETMAX = None
"""Air temperature below which ET will be arbitrarily reduced.

Only used if CSNOFG == 1. Units are degrees Fahrenheit. Not 
supported in this implementation because SNOW is not available.
"""
PETMIN = None
"""Air temperature below which ET will be set to zero.

Only used if CSNOFG == 1. Units are degrees Fahrenheit. 
Not supported in this implementation because SNOW is not available.
"""
INFEXP = None
"""Exponent in infiltration equation, dimensionless.
"""
INFILD = None
"""Ratio between maximum and mean infiltration capacities, dimensionless
"""
DEEPFR = None
"""Fraction of groundwater inflow which will enter deep and inactive 
groundwater.

This water is lost from the HSPF representation and is a model outflow. 
In coupled mode simulations, this discharge is applied as specified 
infiltration to the unsaturated zone flow (UZF) package in MODFLOW 6.
"""
BASETP = None
"""Fraction of remaining potential ET which can be satisfied from baseflow
or groundwater outflow.
"""
AGWETP = None
"""Fraction of remaining potential ET which can be satistifed from 
active groundwater storage if enough is available.
"""
#pwat-parm4
CEPSC = None
"""Interception storage capacity in inches.
"""
UZSN = None
"""Upper zone nominal storage in inches
"""
NSUR = None
"""Manning's n for the assumed overland flow plane.

Use English/Standard units versions from tables.
"""
INTFW = None
"""Interflow inflow parameter, dimensionless.
"""
IRC = None
"""Interflow recession parameter.

Under zero inflow, the ratio of todays interflow outflow rate to 
yesterday's rate. Units are 1/day
"""
LZETP = None
"""Lower zone ET parameter.

Index to the density of deep-rooted vegetation, dimensionless.
"""
#pwat-parm5
FZG = None
"""Parameter that adjusts for the effect of ice in the snow pack on
infiltration when IFFCFG is 1.

It is not used if IFFCFG is 2. Units are 1/inch. Not used in this 
implementation because SNOW is not supported.
"""
FZGL = None
"""Lower limit of INFFAC as adjusted by ice in the snow pack when IFFCFG
is 1.

If IFFCFG is 2, FZGL is the value of INFFAC when the lower layer temperature
is at or below freezing. Dimensionless parameter, not used in this 
implementation because SNOW is not supported.
"""
#pwat-State1
I_CEPS = None
"""Initial interception storage in inches """
I_SURS = None
"""Initial surface or overland flow storage in inches """
I_UZS = None
"""Initial upper zone storage in inches """
I_IFWS = None
"""initial interflow storage in inches """
I_LZS = None
"""Initial lower zone storage in inches """
I_AGWS = None
"""Initial active groundwater storage in inches """
I_GWVS = None
"""Initial index to groundwater slope in inches """

# monthly arrays
LZEPTM = None 
"""Monthly lower zone ET parameter"""
CEPSCM = None 
"""Monthly interception storage ET parameter"""
INTFWM = None 
"""Monthly interflow inflow parameter"""
IRCM = None 
"""Monthly interflow recession parameter"""
NSURM = None 
"""Monthly overland flow Manning's n"""
UZSNM = None 
"""Monthly upper soil zone nominal storage"""

# Data and simulated time series
AGWET = None
"""AET from active groundwater, inches/ivld"""
AGWI = None 
"""Active groundwater inflow, inches/ivld"""
AGWO = None 
"""Active groundwater outflow, inches/ivld"""
AGWS = None 
"""Active groundwater storage, inches"""
BASET = None 
"""AET from baseflow, inches/invld"""
CEPE = None 
"""AET from interception storage, inches/ivld"""
CEPS = None 
"""Interception storage, inches"""
GWVS = None 
"""Index to available groundwater slope, inches"""
IFWI = None 
"""Interflow inflow, inches/ivld"""
IFWO = None 
"""Interflow outflow, inches/ivld"""
IFWS = None 
"""Interflow storage, inches"""
IGWI = None 
"""Inflow to inactive groundwater, inches/ivld"""
INFFAC = None 
"""Factor to account for frozen ground.

Not currently implemented because SNOW is not implemented and always 
set to 1."""
INFIL = None 
"""Infiltration to soil, inches/ivld"""
LZET = None 
"""Lower soil zone AET, inches/ivld"""
LZI = None 
"""Lower soil zone inflow, inches/ivld"""
LZS = None 
"""Lower soil zone storage, inches"""
PERC = None 
"""Percolation from upper to lower soil zones, inches/ivld"""
RPARM = None
"""Maximum ET opportunity, inches/ivld"""
SURI = None 
"""Surface storage inflow, inches/ivld"""
SURO = None 
"""Surface storage outflow, inches/ivld"""
SURS = None 
"""Surface storage, inches"""
TAET = None 
"""Total PERLND AET, inches/ivld"""
TGWS = None 
"""Total groundwater storage, should be equal to active groundwater 
storage prior to ET, inches/ivld"""
UZET = None 
"""Upper soil zone AET, inches/ivld"""
UZI = None 
"""Upper soil zone inflow, inches/ivld"""
UZS = None 
"""Upper soil zone storage, inches"""
PERO = None 
"""Total outflow from pervious land, inches/ivld"""
PERS = None 
"""Total water stored in pervious land, inches"""
AGWLI = None 
"""Active groundwater lateral inflow external time series, inches/ivld"""
IFWLI = None 
"""Interflow lateral inflow external time series, inches/ivld"""
LZLI = None 
"""Lower soil zone lateral inflow external time series, inches/ivld"""
SURLI = None 
"""Surface storage lateral inflow external time series, inches/ivld"""
UZLI = None 
"""Upper soil zone lateral inflow external time series, inches/ivld"""
SUPY = None 
"""Moisture supplied to the land segment by precipitation, inches/ivld"""
PET = None 
"""Potential evapotranspiration, inches/ivld"""
PETADJ = None
"""Adjusted PET for temperature restrictions, inches/ivld """


def setDelT( sim_delt ):
    """Set the pervious land delt for calculations.

    The delt is stored as a module wide global. Also sets DELT60 
    module-wide global. Once have this adjust the other module-wides 
    that depend on this value to go to internal units

    Args:
        sim_delt (float): overall simulation time step in minutes

    """
    global DELT60, KGWV, AGWRC, INFILT, AGWRC, SPEC_DT
    # function
    DELT60 = sim_delt / 60.0
    if AGWRC is None:
        warnMsg = "No PERLND operators defined in model."
        CL.LOGR.warning( warnMsg )
    else:
        calcHelp = AGWRC.view( ( np.float64, len( AGWRC.dtype.names ) ) )
        calcHelp2 = ( 1.0 - np.power( calcHelp[0,:], ( DELT60 / 24.0 ) ) ) 
        KGWV[0] = np.array( tuple(calcHelp2), dtype=SPEC_DT )
    if INFILT is None:
        warnMsg = "No PERLND operators defined in model."
        CL.LOGR.warning( warnMsg )
    else:
        calcHelp = INFILT.view( ( np.float64, len( INFILT.dtype.names ) ) )
        calcHelp2 = calcHelp[0,:] * DELT60 
        INFILT[0] = np.array( tuple(calcHelp2), dtype=SPEC_DT )
    # return
    return


def setupDAYFG( tIndex ):
    """Set the module-level global DAYFG

    Args:
        tIndex (pd.DateIndex): datetime index for the simulation output

    """
    # imports
    # globals
    global DAYFG
    # start
    DAYFG = np.where( tIndex.hour == 0, True, False )
    DAYFG[0] = True
    # return
    return


def setUpRecArrays( pwList, sim_len ):
    """ Create and initialize pervious land output arrays

    Args:
        pwList (list): list of IDs for this target type
        sim_len (int): number of output intervals in the simulation
    
    """
    # imports
    # globals
    global DEF_DT, SPEC_DT, FLAG_DT
    # time series
    global AGWET, AGWI, AGWO, AGWS, BASET, CEPE, CEPS, GWVS
    global IFWI, IFWO, IFWS, IGWI, INFFAC, INFIL, LZET, LZI
    global LZS, PERC, SURI, SURO, SURS, TAET, TGWS
    global UZET, UZI, UZS, PERO, PERS, SUPY, PET, RPARM, PETADJ
    # parameters
    global LZSN, INFILT, LSUR, SLSUR, KVARY, KGWV, AGWRC, PETMAX
    global PETMIN, INFEXP, INFILD, DEEPFR, BASETP, AGWETP, CEPSC
    global FOREST, UZSN, NSUR, INTFW, IRC, LZETP, FZG, FZGL
    global WS_AREAS
    # initial states
    global I_CEPS, I_SURS
    global I_UZS, I_IFWS, I_LZS, I_AGWS, I_GWVS
    # flags
    global VLEFG, VCSFG, VUZFG, VNNFG, VIFWFG, VIRCFG
    global CSNOFG, RTOPFG, UZFG, ICEFG
    # lateral input control
    global LATIN_CONTROL, LAT_INFLOW_TS 
    # output control
    global OUTPUT_CONTROL, GOOD_OUTPUT_LIST
    # carry overs
    global HOLD_MSUPY, HOLD_RLZRAT, HOLD_LZFRAC, HOLD_DEC, HOLD_SRC
    global HOLD_IFWK1, HOLD_IFWK2
    # monthlys
    global LZEPTM, CEPSCM, INTFWM, IRCM, NSURM, UZSNM
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
    AGWET = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    AGWI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    AGWO = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    AGWS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    BASET = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    CEPE = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    CEPS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    GWVS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    IFWI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    IFWO = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    IFWS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    IGWI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    INFFAC = np.rec.array( np.ones( sim_len, dtype=DEF_DT ) )
    INFIL = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    LZET = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    LZI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    LZS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PERC = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    RPARM = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    SURI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    SURO = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    SURS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    TAET = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    TGWS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    UZET = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    UZI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    UZS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PERO = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PERS = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    SUPY = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PET = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PETADJ = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    # flags by perlnd
    CSNOFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ICEFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    RTOPFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    UZFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    VLEFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    VCSFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    VUZFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    VNNFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    VIFWFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    VIRCFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    LATIN_CONTROL = np.rec.array( np.zeros( len( LAT_INFLOW_TS ), 
                                  dtype=FLAG_DT ) )
    OUTPUT_CONTROL = np.rec.array( np.zeros( len( GOOD_OUTPUT_LIST), 
                                   dtype=FLAG_DT ) )
    # parameters by perlnd
    FOREST = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    LZSN = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    INFILT = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    LSUR = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    SLSUR = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    KVARY = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    KGWV = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    AGWRC = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    PETMAX = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    PETMIN = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    INFEXP = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    INFILD = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    DEEPFR = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    BASETP = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    AGWETP = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    CEPSC = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    UZSN = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    NSUR = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    INTFW = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    IRC = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    LZETP = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    FZG = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    FZGL = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    I_CEPS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    I_SURS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    I_UZS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    I_IFWS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    I_LZS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    I_AGWS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    I_GWVS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    WS_AREAS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    # carry overs
    HOLD_MSUPY = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_DEC = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_SRC = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_IFWK1 = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_IFWK2 = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    # the last two carryovers want a-non default creation value
    # so need to create lists of arrays
    list1 = list()
    list2 = list()
    for tNam in SPEC_DT.names:
        nar1 = -1.0E30 * np.ones(1, dtype=np.float64 )
        nar2 = -1.0E30 * np.ones(1, dtype=np.float64 )
        list1.append( nar1 )
        list2.append( nar2 )
    # end for
    HOLD_RLZRAT = np.core.records.fromarrays( list1, dtype=SPEC_DT )
    HOLD_LZFRAC = np.core.records.fromarrays( list2, dtype=SPEC_DT )
    # monthlys
    LZEPTM = np.rec.array( np.zeros( 12, dtype=SPEC_DT ) )
    CEPSCM = np.rec.array( np.zeros( 12, dtype=SPEC_DT ) )
    INTFWM = np.rec.array( np.zeros( 12, dtype=SPEC_DT ) )
    IRCM = np.rec.array( np.zeros( 12, dtype=SPEC_DT ) )
    NSURM = np.rec.array( np.zeros( 12, dtype=SPEC_DT ) )
    UZSNM = np.rec.array( np.zeros( 12, dtype=SPEC_DT ) )
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
    """Set the PET time series from one data set to one target.

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
    global CSNOFG, RTOPFG, UZFG, VLEFG, VCSFG, VUZFG, VNNFG
    global VIFWFG, VIRCFG, ICEFG
    # parameters
    # locals
    # start
    if tFlag == "CSNOFG":
        CSNOFG[targID][0] = fVal
    elif tFlag == "ICEFG":
        ICEFG[targID][0] = fVal
    elif tFlag == "RTOPFG":
        RTOPFG[targID][0] = fVal
    elif tFlag == "UZFG":
        UZFG[targID][0] = fVal
    elif tFlag == "VCSFG":
        VCSFG[targID][0] = fVal
    elif tFlag == "VIFWFG":
        VIFWFG[targID][0] = fVal
    elif tFlag == "VIRCFG":
        VIRCFG[targID][0] = fVal
    elif tFlag ==  "VLEFG":
        VLEFG[targID][0] = fVal   
    elif tFlag == "VNNFG":
        VNNFG[targID][0] = fVal   
    elif tFlag == "VUZFG":
        VUZFG[targID][0] = fVal
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
    global AGWETP, AGWRC, BASETP, CEPSC, DEEPFR, FOREST, FZG, FZGL
    global INFEXP, INFILD, INFILT, INTFW, IRC, KVARY, LSUR, LZETP
    global LZSN, NSUR, PETMAX, PETMIN, SLSUR, UZSN 
    # parameters
    # locals
    # start
    if tParam == "AGWETP":
        AGWETP[ targID ][0] = pVal
    elif tParam == "AGWRC":
        AGWRC[ targID ][0] = pVal
    elif tParam == "BASETP":
        BASETP[ targID ][0] = pVal
    elif tParam == "CEPSC":
        CEPSC[ targID ][0] = pVal
    elif tParam == "DEEPFR":
        DEEPFR[ targID ][0] = pVal
    elif tParam == "FOREST":
        FOREST[ targID ][0] = pVal
    elif tParam == "FZG":
        FZG[ targID ][0] = pVal
    elif tParam == "FZGL":
        FZGL[ targID ][0] = pVal
    elif tParam == "INFEXP":
        INFEXP[ targID ][0] = pVal
    elif tParam == "INFILD":
        INFILD[ targID ][0] = pVal
    elif tParam == "INFILT":
        INFILT[ targID ][0] = pVal
    elif tParam == "INTFW":
        INTFW[ targID ][0] = pVal
    elif tParam == "IRC":
        IRC[ targID ][0] = pVal
    elif tParam == "KVARY":
        KVARY[ targID ][0] = pVal
    elif tParam == "LSUR":
        LSUR[ targID ][0] = pVal
    elif tParam == "LZETP":
        LZETP[ targID ][0] = pVal
    elif tParam == "LZSN":
        LZSN[ targID ][0] = pVal
    elif tParam == "NSUR":
        NSUR[ targID ][0] = pVal
    elif tParam == "PETMAX":
        PETMAX[ targID ][0] = pVal
    elif tParam == "PETMIN":
        PETMIN[ targID ][0] = pVal
    elif tParam == "SLSUR":
        SLSUR[ targID ][0] = pVal
    elif tParam == "UZSN":
        UZSN[ targID ][0] = pVal
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
    global LZEPTM, CEPSCM, INTFWM, IRCM, NSURM, UZSNM
    # parameters
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    if monName == "LZETPM":
        LZEPTM[ targID ][:] = np.array( monTuple )
    elif monName == "ZETPM":
        LZEPTM[ targID ][:] = np.array( monTuple )
    elif monName == "CEPSCM":
        CEPSCM[ targID ][:] = np.array( monTuple )
    elif monName == "INTFWM":
        INTFWM[ targID ][:] = np.array( monTuple )
    elif monName == "IRCM":
        IRCM[ targID ][:] = np.array( monTuple )
    elif monName == "NSURM":
        NSURM[ targID ][:] = np.array( monTuple )
    elif monName == "UZSNM":
        UZSNM[ targID ][:] = np.array( monTuple )
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
    global I_CEPS, I_SURS, I_UZS, I_IFWS, I_LZS, I_AGWS, I_GWVS
    # parameters
    # locals
    # start
    if tParam == 'CEPS':
        I_CEPS[ targID ][0] = pVal
    elif tParam == 'SURS':
        I_SURS[ targID ][0] = pVal
    elif tParam == 'UZS': 
        I_UZS[ targID ][0] = pVal
    elif tParam == 'IFWS':
        I_IFWS[ targID ][0] = pVal
    elif tParam == 'LZS':
        I_LZS[ targID ][0] = pVal
    elif tParam == 'AGWS':
        I_AGWS[ targID ][0] = pVal
    elif tParam == 'GWVS':
        I_GWVS[ targID ][0] = pVal
    # return
    return


def configExternalTS( sim_len, TSMapList, AllTSDict ):
    """Transfer external time series from HDF5 input to module data
    structures.

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
                        ( tsType, "PERLND", tsTargID )
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
                         ( tsType, "PERLND", tsTargID )
                CL.LOGR.error( errMsg )
                return badReturn
            # end inner if
        else:
            # unknown and unsupported time series type
            warnMsg = "Unknown and unsupported time series type %s " \
                      "found for %s - %s. Time series will be " \
                      "ignored!!!"  % ( tsType, "PERLND", tsTargID )
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
    global DEF_DT, AGWLI, IFWLI, LZLI, SURLI, UZLI, LATIN_CONTROL
    global LAT_INFLOW_TS
    # parameters
    # locals
    # start
    # set the control value to 1 for this type
    # find the index in the LAT_INFLOW_TS lists
    liIndex = LAT_INFLOW_TS.index( inflowType )
    LATIN_CONTROL[ targID ][ liIndex ] = 1
    # now setup the time series or assign the time series
    if inflowType == "AGWLI":
        if AGWLI is None:
            # initialize
            AGWLI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
        else:
            AGWLI[ targID ][:] = tsVals
    elif inflowType == "IFWLI":
        if IFWLI is None:
            # initialize
            IFWLI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
        else:
            IFWLI[ targID ][:] = tsVals
    elif inflowType == "LZLI":
        if LZLI is None:
            # initialize
            LZLI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
        else:
            LZLI[ targID ][:] = tsVals
    elif inflowType == "SURLI":
        if SURLI is None:
            # initialize
            SURLI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
        else:
            SURLI[ targID ][:] = tsVals
    elif inflowType == "UZLI":
        if UZLI is None:
            # initialize
            UZLI = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
        else:
            UZLI[ targID ][:] = tsVals
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
    global GOOD_OUTPUT_LIST, BAD_OUTPUT_LIST, OUTPUT_CONTROL, INFLOW_TS_GOOD
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
                warnMsg = "Output type %s is not supported for PERLND!!!" \
                        % sType
                CL.LOGR.warning( warnMsg )
            # now continue
            continue
        elif sType in GOOD_OUTPUT_LIST:
            stInd = GOOD_OUTPUT_LIST.index( sType )
            OUTPUT_CONTROL[targID][stInd] = sVal
        elif sType in INFLOW_TS_GOOD:
            # this are not included in outputs because are specified inputs
            # info message
            infoMsg = "Output type %s is a specified inflow and so not " \
                      "included in outputs." % sType 
            CL.LOGR.info( infoMsg )
            continue
        else:
            # this is an error because undefined type
            errMsg = "Undefined output type of %s for PERLND!!!" % \
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
    
    Return:
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
            if fV == "ICEFG":
                # this one will only be there if a
                #  snow section
                continue
            # end if
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
    """Get a lateral inflow by type and target for the specified time 
    interval.

    Args:
        targID (str): target ID
        liType (str): lateral inflow type
        iI (int): current time index

    Returns:
        float: inflow rate for time interval
    
    """
    global AGWLI, IFWLI, LZLI, SURLI, UZLI, LATIN_CONTROL
    global LAT_INFLOW_TS
    # start
    # set the control value to 1 for this type
    # find the index in the LAT_INFLOW_TS lists
    liIndex = LAT_INFLOW_TS.index( liType )
    flagVal = LATIN_CONTROL[ targID ][ liIndex ]
    if flagVal > 0:
        if liType == "AGWLI":
            retVal = float( AGWLI[ targID ][iI] )
        elif liType == "IFWLI":
            retVal = float( IFWLI[ targID ][iI] )
        elif liType == "LZLI":
            retVal = float( LZLI[ targID ][iI] )
        elif liType == "SURLI":
            retVal = float( SURLI[ targID ][iI] )
        elif liType == "UZLI":
            retVal = float( UZLI[ targID ][iI] )
        else:
            retVal = 0.0
        # end if
    else:
        retVal = 0.0
    # return
    return retVal


def pwater_liftedloop( iI, mon, targID ):
    """Modified version of liftedloop to do a single time step and
    return to the main time loop.
    
    Module-wide recarrays are used to store all results and calculation 
    variables between calls. Modified real number comparisons to be more 
    numerically reliable.

    Water supply to the ground surface, MSUPY, is the starting point 
    for the calculation.

    1. MSUPY is divided into infiltration (infilt) to the Lower Zone 
        and potential direct runoff (PDRO)
    
    2. Calculated Upper Zone Inflow (UZI) is taken out of PDRO and
        the remainder is available for interflow inflow (ifwi) and 
        potential surface storage (psur)
    
    3. Interflow storage does not have a nominal capacity. Like
        active groundwater storage, interflow storage is adjusted
        by outflow and inflow but does not overflow. The recession
        in interflow outflow which is controlled by IRC which is
        the interflow recession parameter (1/day).
    
    4. Percolation (perc) for Upper Zone to Lower Zone is calculated
        based on Upper Zone storage plus UZI and the uzs/uzsn
        being larger than lzs/lzsn
    
    5. Inflow to the lower zone and active groundwater is infilt + 
        perc + lateral inflows (iperc). This is partitioned to Lower Zone
        percolation (lperc) which provides the Lower Zone inflow (lzi)
    
    6. The remainder of total percolation (iperc) after removing
        lzi is groundwater water inflow (gwi) and is split between 
        active groundwater inflow (agwi) and inflow to inactive 
        groundwater (igwi)
    
    7. IGWI is the DEEPFR percentage of gwi and leaves the soil 
        column and HSPF calculations.
    
    8. There is no nominal storage limitation on active groundwater.
        Active groundwater outflow to baseflow is calculated using
        AGWRC and KVARY which essentially provide for delayed
        outflow rates and spread the outflow over time.

    * INTFW is the interflow inflow parameter (dimensionless)
    
    * ratio = max( 1.0001, ( intfw * pow( 2.0, lzrat ) ) )

    * lzrat = lzs / lzsn 

    Args:
        iI (int): index of current time step (0 to (sim_len-1))
        mon (int): current simulation month
        targID (str): ID for recarray columns

    Returns:
        int: count of the number of errors. Should generally be 0 
            but used to reference errorsV for error handling
    
    """
    # imports
    from copy import deepcopy
    from math import pow, sqrt, log, exp
    # globals
    # carry overs
    global HOLD_MSUPY, HOLD_RLZRAT, HOLD_LZFRAC, HOLD_DEC, HOLD_SRC
    global HOLD_IFWK1, HOLD_IFWK2, ERRMSG
    # calculation constants
    global IRRCEP, IRRAPPV, errorsV, UZRA, INTGRL, DELT60
    # switches, parameters, and flags
    global DAYFG, INFFAC, INFILT, KVARY, LZSN, BASETP
    global LZEPTM, LZETP, CEPSCM, CEPSC, INTFWM, INTFW, IRCM, IRC 
    global NSURM, NSUR, UZSNM, UZSN, RTOPFG, FOREST
    global VLEFG, VCSFG, VIFWFG, VIRCFG, VNNFG, VUZFG, UZFG
    global INFEXP, INFILD, LSUR, SLSUR, DEEPFR, AGWETP
    global PETMAX, PETMIN, ICEFG, CSNOFG, FZG, FZGL
    # data time series
    global SUPY, PET, PETADJ
    # initial states
    global I_CEPS, I_SURS, I_LZS, I_UZS, I_IFWS, I_AGWS, I_GWVS
    # storage ts for states
    global CEPS, SURS, LZS, UZS, IFWS, AGWS, GWVS, TGWS
    # save time series only
    global AGWET, AGWI, AGWO, BASET, CEPE, IFWI, IFWO, IGWI, INFIL
    global LZET, LZI, PERC, PERO, PERS, SURI, SURO, TAET, UZET, UZI
    global RPARM
    # parameters
    smallVal = float( 1E-13 )
    # locals - put here in case want to Cython
    cepo = float( 0.0 )
    msupy = float( 0.0 )
    suri = float( 0.0 )
    suro = float( 0.0 )
    ifwi = float( 0.0 )
    uzi = float( 0.0 )
    lzrat = float( 0.0 )
    ibar = float( 0.0 )
    imax = float( 0.0 )
    imin = float( 0.0 )
    iimax = float( 0.0 )
    iimin = float( 0.0 )
    dummy = float( 0.0 )
    dec = float( 0.0 )
    src = float( 0.0 )
    ratio = float( 0.0 )
    under = float( 0.0 )
    over = float( 0.0 )
    over2 = float( 0.0 )
    infil = float( 0.0 )
    pdro = float( 0.0 )
    uzrat = float( 0.0 )
    k1 = float( 0.0 )
    k2 = float( 0.0 )
    uzfrac = float( 0.0 )
    uzraa = float( 0.0 )
    intga = float( 0.0 )
    intgb = float( 0.0 )
    uzrab = float( 0.0 )
    psur = float( 0.0 )
    pifwi = float( 0.0 )
    kifw = float( 0.0 )
    ifwk2 = float( 0.0 )
    ifwk1 = float( 0.0 )
    inflo = float( 0.0 )
    value = float( 0.0 )
    ifwo = float( 0.0 )
    perc = float( 0.0 )
    iperc = float( 0.0 )
    lperc = float( 0.0 )
    lzi = float( 0.0 )
    indx = float( 0.0 )
    gwi = float( 0.0 )
    igwi = float( 0.0 )
    agwi = float( 0.0 )
    ainflo = float( 0.0 )
    agwo = float( 0.0 )
    avail = float( 0.0 )
    rempet = float( 0.0 )
    taet = float( 0.0 )
    baset = float( 0.0 )
    baspet = float( 0.0 )
    cepe = float( 0.0 )
    uzet = float( 0.0 )
    uzpet = float( 0.0 )
    agwet = float( 0.0 )
    gwpet = float( 0.0 )
    lzet = float( 0.0 )     # calculated lower zone et
    lzpet = float( 0.0 )    # lower zone pet
    kk = int( 0 )
    errorCnt = int( 0 )     # error counter
    ans = np.zeros( 3, dtype=np.float64 )   # return from proute
    # 
    # IFRDG is not currently supported so always false
    IFRDFG = False
    # initialize our calculation variables
    dayfg = DAYFG[ iI ]
    # flags
    fl_uzfg = int( UZFG[targID][0] )
    fl_vlefg = int( VLEFG[targID][0] )
    fl_rtopfg = int( RTOPFG[targID][0] )
    # CSNOFG also not supported so set to false
    fl_csnofg = int( 0.0 )
    #fl_csnofg = int( CSNOFG[targID][0] )
    # ICEFG also not supported so set to fals
    fl_icefg = int( 0.0 )
    #fl_icefg = int( ICEFG[targID][0] )
    # parameters 
    kgw = float( KGWV[targID][0] )
    infexp = float( INFEXP[targID][0] )
    infilt = float( INFILT[targID][0] )
    infild = float( INFILD[targID][0] )
    lsur = float( LSUR[targID][0] )
    slsur = float( SLSUR[targID][0] )
    deepfr = float( DEEPFR[targID][0] )
    kvary = float( KVARY[targID][0] )
    lzsn = float( LZSN[targID][0] )
    basetp = float( BASETP[targID][0] )
    agwetp = float( AGWETP[targID][0] )
    forest = float( FOREST[targID][0] )
    petmax = float( PETMAX[targID][0] )
    petmin = float( PETMIN[targID][0] )
    fzg = float( FZG[targID][0] )
    fzgl = float( FZGL[targID][0] )
    # check if using monthly
    if fl_vlefg == 1:
        lzetp = LZEPTM[targID][(mon-1)]
    else:
        lzetp = LZETP[targID][0]
    if VCSFG[targID][0] > 0:
        cepsc = CEPSCM[targID][(mon-1)]
    else:
        cepsc = CEPSC[targID]
    if VUZFG[targID][0] > 0:
        uzsn = UZSNM[targID][(mon-1)]
    else:
        uzsn = UZSN[targID]
    if VIFWFG[targID][0] > 0:
        intfw = INTFWM[targID][(mon-1)]
    else:
        intfw = INTFW[targID]
    if VNNFG[targID][0] > 0:
        nsur = NSURM[targID][(mon-1)]
    else:
        nsur = NSUR[targID]
    if VIRCFG[targID][0] > 0:
        irc = IRCM[targID][(mon-1)]
    else:
        irc = IRC[targID]
    # get our time series values
    # set changes if CSNOFG active
    if fl_csnofg == 1:
        """ snow is being considered - allow for it find the moisture supplied
        to interception storage.  rainf is rainfall in inches/ivl. adjust for
        fraction of land segment covered by snow. wyield is the water yielded by
        the snowpack in inches/ivl. it has already been adjusted to an effective
        yield over the entire land segment """
        # this is not currently supported but maintained here
        # for future expansion
        # initialize dummy variables for currently unsupported
        # time series
        ts_airtmp = float( 0.0 )
        ts_snocov = float( 0.0 )
        ts_rainf = float( 0.0 )
        ts_wyield = float( 0.0 )
        ts_packi = float( 0.0 )
        ts_petinp = float( PET[targID][iI] )
        # do supy first
        ts_supy = ts_rainf * ( 1.0 - ts_snocov ) + ts_wyield 
        # then do PET
        petadj = ( 1.0 - forest ) * ( 1.0 - ts_snocov ) + forest
        #if (airtmp < PETMAX[loop])and petadj > 0.5:
        if ( ( ( petmax - ts_airtmp) >= smallVal ) and 
                        ( ( petadj - 0.5 ) >= smallVal ) ):
            petadj = 0.5
        # end if
        # if airtmp < PETMIN[loop]:
        if ( ( petmin - ts_airtmp ) >= smallVal ):
            petadj = 0.0
        # end if
        # check the iceflag
        if fl_icefg == 1:
            # calculate factor to reduce infiltration and percolation 
            # to account for frozen ground
            inffac = 1.0 - fzg * ts_packi
            if ( ( fzgl - inffac ) >= smallVal ):
                inffac = fzgl 
            # end if
        # end if
        ts_pet = ts_petinp * petadj
    else:
        # inffac should be 1.0 unless fl_icefg == 1
        inffac = float( 1.0 )
        ts_supy = float( SUPY[targID][iI] )
        ts_pet = float( PET[targID][iI] )
    # end if
    # set up calculation values for state
    if iI == 0:
        ceps = float( I_CEPS[targID][0] )
        surs = float( I_SURS[targID][0] )
        lzs = float( I_LZS[targID][0] )
        uzs = float( I_UZS[targID][0] )
        ifws = float( I_IFWS[targID][0] )
        gwvs = float( I_GWVS[targID][0] )
        agws = float( I_AGWS[targID][0] )
        ceps = float( I_CEPS[targID][0] )
        rparm = float( 0.0 )
    else:
        ceps = float( CEPS[targID][iI-1] )
        surs = float( SURS[targID][iI-1] )
        lzs = float( LZS[targID][iI-1] )
        uzs = float( UZS[targID][iI-1] )
        ifws = float( IFWS[targID][iI-1] )
        gwvs = float( GWVS[targID][iI-1] )
        agws = float( AGWS[targID][iI-1] )
        ceps = float( CEPS[targID][iI-1] )
        rparm = float( RPARM[targID][iI-1] )
    # get the lateral inflow values
    lits_agwli = getLatInflowByTypeTarget( targID, "AGWLI", iI )
    lits_ifwli = getLatInflowByTypeTarget( targID, "IFWLI", iI )
    lits_lzli = getLatInflowByTypeTarget( targID, "LZLI", iI )
    lits_surli = getLatInflowByTypeTarget( targID, "SURLI", iI )
    lits_uzli = getLatInflowByTypeTarget( targID, "UZLI", iI )
    # set the previous msupy using a copy of the global
    oldmsupy = float( HOLD_MSUPY[targID][0] )
    rlzrat = float( HOLD_RLZRAT[targID][0] )
    lzfrac = float( HOLD_LZFRAC[targID][0] )
    dec = float( HOLD_DEC[targID][0] )
    src = float( HOLD_SRC[targID][0] )
    ifwk1 = float( HOLD_IFWK1[targID][0] )
    ifwk2 = float( HOLD_IFWK2[targID][0] )
    #
    # INTERCEPTION Start ----------------------------------------------
    # Start - ICEPT
    #    ICEPT == Simulate the interception of moisture by vegetal or 
    #              other ground cover
    ceps = ceps + ts_supy + IRRCEP
    #if ( ceps > cepsc ):
    if ( ( ceps - cepsc ) >= smallVal ):
        cepo = ceps - cepsc
        ceps = cepsc
    # END ICEPT
    # INTERCEPTION End ------------------------------------------------
    #
    # SURFACE Start ---------------------------------------------------
    # Start - PWATRX
    suri  = cepo + lits_surli
    msupy = suri + surs + IRRAPPV[2]
    # determine the current value of the lower zone storage ratio
    lzrat = lzs / lzsn
    #if msupy <= 0.0:
    if ( ( msupy - 0.0 ) < smallVal ):
        surs = 0.0
        suro = 0.0
        ifwi = 0.0
        infil = 0.0
        uzi = 0.0
    else:
        # Start - SURFAC
        """Distribute the water available for infiltration and runoff - 
        units of fluxes are in./ivl establish locations of sloping 
        lines on infiltration/inflow/sur runoff figure.  prefix "i" 
        refers to "infiltration" line, ibar is the mean infiltration 
        capacity over the segment, internal units of infilt are inches/ivl"""
        ibar = infilt / ( pow( lzrat, infexp ) )
        # following should never happend until snow and ice 
        # are incorporated
        #if inffac < 1.0 :
        if ( ( 1.0 - inffac ) >= smallVal ):
            ibar = ibar * inffac
        # end if
        # infild is an input parameter - ratio of maximum to mean 
        # infiltration capacity
        imax = ibar * infild
        imin = ibar - (imax - ibar)
        #if ( dayfg ) or ( oldmsupy == 0.0):
        if ( dayfg ) or ( abs( oldmsupy - 0.0 ) < smallVal ):
            dummy = nsur * lsur
            dec = 0.00982 * pow( ( dummy / sqrt( slsur )), 0.6 )
            src = 1020.0  * ( sqrt( slsur ) / dummy )
        # end if
        # INTFW is the interflow inflow parameter
        ratio = max( 1.0001, ( intfw * pow( 2.0, lzrat ) ) )
        # DISPOSE
        # DIVISN
        #if ( msupy <= imin ):
        if ( ( imin - msupy ) > ( -1.0 * smallVal ) ):
            # msupy line is entirely below other line
            under = msupy
            over = float( 0.0 )
            #elif ( msupy > imax ):
        elif ( ( msupy - imax ) >= smallVal ):
            # msupy line is entirely above other line
            under = (imin + imax) * 0.5
            over = msupy - under
        else:
            # msupy  line crosses other line
            over = ( ( ( pow( (msupy - imin), 2.0 ) ) * 0.5 ) / 
                        ( imax - imin ) )
            under = msupy - over
        #END DIVISN
        infil = under
        #if over <= 0.0:
        if ( ( over - 0.0 ) < smallVal ):
            surs = 0.0
            suro = 0.0
            ifwi = 0.0
            uzi  = 0.0
        else:
            # start DISPOS
            # there is some potential interflow inflow and maybe 
            # surface detention/outflow -- the sum of these is 
            # potential direct runoff
            pdro = over
            # determine how much of this potential direct runoff will 
            # be taken by the upper zone
            if ( fl_uzfg > 0 ):
                # UZINF2 -- HSPX, ARM, NPS type calculation
                """Compute inflow to upper zone during this interval, 
                using "fully forward" type algorithm  as used in HSPX, 
                ARM and NPS.  Note:  although this method should give results 
                closer to those produced by HSPX, etc., its output will
                be more sensitive to delt than that given by subroutine uzinf"""
                uzrat = uzs / uzsn
                # if ( uzrat < 2.0 ):
                if ( ( 2.0 - uzrat ) >= smallVal ):
                    k1 = 3.0 - uzrat
                    uzfrac = 1.0 - (uzrat * 0.5) * pow( ( 1.0 / (1.0 + k1) ), k1 )
                else:
                    k2 = (2.0 * uzrat) - 3.0
                    uzfrac = pow( ( 1.0 / (1.0 +  k2) ), k2 )
                # end inner if
                uzi = pdro * uzfrac
                # END if UZINF2
            else:
                # UZINF
                """Compute the inflow to the upper zone during this time 
                interval. Do this using a table look-up to handle the 
                non-analytic integral given in supporting documentation."""
                # find the value of the integral at initial uzra
                uzraa = uzs / uzsn
                # uzra[kk] < uzraa <= uzra[kk+1]
                kk = np.argmax( uzraa < UZRA ) - 1
                if ( kk == -1 ):
                    kk = 8
                    # ERRMSG1: UZRAA exceeds UZRA array bounds
                    errorsV[1] += 1
                    errorCnt += 1
                    errMsg = "pwater_liftedloop - %s " % ERRMSG[1]
                    CL.LOGR.error( errMsg )
                # end of error if
                intga = ( INTGRL[kk] + ( INTGRL[kk+1] - INTGRL[kk] ) * ( uzraa - 
                            UZRA[kk] ) / ( UZRA[kk+1] - UZRA[kk] ) )
                intgb = ( pdro / uzsn ) + intga
                # intgrl[kk] <= intgb < intgrl[kk+1]
                kk = np.argmax( intgb < INTGRL ) - 1
                if ( kk == -1 ) :
                    # ERRMSG2: INTGB exceeds INTGRL array bounds
                    errorsV[2] += 1
                    kk = 8
                    errorCnt += 1
                    errMsg = "pwater_liftedloop - %s " % ERRMSG[2]
                    CL.LOGR.error( errMsg )
                # end if
                uzrab = ( UZRA[kk] + ( UZRA[kk+1] - UZRA[kk] )  * ( intgb - 
                            INTGRL[kk] ) / ( INTGRL[kk+1] - INTGRL[kk] ) )
                uzi = ( uzrab - uzraa ) * uzsn
                #if uzi < 0.0:
                if ( abs( uzi - 0.0 ) < smallVal ):
                    # check if zero
                    uzi = 0.0
                elif ( ( 0.0 - uzi ) >= smallVal ):
                    # check if negative
                    uzi = 0.0
                # end uzi if
            # END if UZINF
            #if uzi > pdro:
            if ( ( uzi - pdro ) >= smallVal ):
                uzi = pdro
            uzfrac = uzi / pdro
            # the prefix "ii" is used on variables on second divisn
            iimin = imin * ratio
            iimax = imax * ratio
            # DIVISN
            #if ( msupy <= iimin ):
            if ( ( iimin - msupy ) > ( -1.0 * smallVal ) ):
                over2 = 0.0
            #elif ( msupy > iimax ):
            elif ( ( msupy - iimax ) >= smallVal ):
                over2 = msupy - (iimin + iimax) * 0.5
            else:
                # msupy  line crosses other line
                over2 =  ( ( pow( (msupy - iimin), 2.0 ) * 0.5 ) / 
                          ( iimax - iimin ) )
            # END if DIVISN
            # psur is potential surface detention/runoff
            psur = over2
            # pifwi is potential interflow inflow
            pifwi = pdro - psur
            ifwi  = pifwi * ( 1.0 - uzfrac )
            #if psur <= 0.0:
            if ( abs( psur - 0.0 ) < smallVal ):
                surs = 0.0
                suro = 0.0
            elif ( ( 0.0 - psur ) >= smallVal ):
                surs = 0.0
                suro = 0.0
            else:
                # there will be something on or running off the 
                # surface reduce it to account for the upper 
                # zone's share
                psur = psur * ( 1.0 - uzfrac )
                # determine how much of this potential surface 
                # detention/outflow will run off in this time interval
                ans = proute( psur, fl_rtopfg, DELT60, dec, src, surs )
                suro = float( ans[0] )
                surs = float( ans[1] )
                retErrCnt = int( ans[2] )
                errorsV[6] += retErrCnt
                if retErrCnt > 0:
                    errorCnt += 1
                # end if
            # end if
        # END if DISPOS
    # END SURFAC
    # SURFACE End -----------------------------------------------------
    #
    # INTERFLOW Start -------------------------------------------------
    # INTFLW  to simulate interflow
    if dayfg:
        kifw  = -log( irc ) / ( 24.0 / DELT60 )
        ifwk2 = 1.0 - exp( -1.0 * kifw )
        ifwk1 = 1.0 - ( ifwk2 / kifw )
    # end if
    # surface and near-surface zones of the land segment have not
    # been subdivided into blocks
    inflo = ifwi + lits_ifwli
    value = inflo + ifws
    #if value > 0.00002:
    if ( ( value - 0.00002 ) >= smallVal ):
        ifwo = ( ifwk1 * inflo ) + ( ifwk2 * ifws )
        ifws = value - ifwo
    else:
        ifwo = 0.0
        ifws = 0.0
         #nothing worth routing-dump back to uzs
        uzs = uzs + value
    # END if INTFLW
    # INTERFLOW End ---------------------------------------------------
    #
    # UPPER ZONE Start ------------------------------------------------
    # UZONE
    uzrat = uzs / uzsn
    # add inflow to uzs
    uzs = uzs + uzi + lits_uzli + IRRAPPV[3]
    perc = 0.0
    #if ( ( uzrat - lzrat ) > 0.01 ):
    if ( ( ( uzrat - lzrat ) - 0.01 ) >= smallVal ):
        # simulate percolation
        perc = 0.1 * infilt * inffac * uzsn * pow( (uzrat - lzrat), 3.0) 
        #if perc > uzs:
        if ( ( perc - uzs ) >= smallVal ):
            # computed value is too high so merely empty storage
            perc = uzs
            uzs = 0.0
        else:
            uzs = uzs - perc
        # end of inner if
    # END if UZONE
    # collect inflows to lower zone and groundwater
    iperc = perc + infil + lits_lzli
    # UPPER ZONE End ------------------------------------------------- 
    # 
    # LOWER ZONE Start ------------------------------------------------
    # LZONE simulate lower zone behavior
    lperc = iperc + IRRAPPV[4]
    lzi = 0.0
    #if lperc > 0.0:
    if ( ( lperc - 0.0 ) >= smallVal ):
        #  if necessary, recalculate the fraction of infiltration plus 
        # percolation which will be taken by lower zone
        #if abs( lzrat - rlzrat ) > 0.02 or IFRDFG:
        if ( ( ( abs( lzrat - rlzrat ) - 0.02 ) >= smallVal ) or IFRDFG ):
            #  it is time to recalculate
            rlzrat = lzrat
            #if lzrat <= 1.0:
            if ( ( 1.0 - lzrat ) > ( -1.0 * smallVal ) ):
                indx = 2.5 - ( 1.5 * lzrat )
                if IFRDFG:
                    lzfrac = 1.0
                else:
                    lzfrac = 1.0 - lzrat * pow( ( 1.0 / (1.0 + indx) ), indx )
                # end inner if
            else:
                indx = ( 1.5 * lzrat ) - 0.5
                exfact = -1.0 * int( IFRDFG )
                if IFRDFG:
                    lzfrac = exp( exfact * ( lzrat - 1.0 ) )
                else:
                    lzfrac = pow( ( 1.0 / ( 1.0 + indx ) ), indx )
                # end inner if
            # end if lzrat
        # end if abs
        lzi = lzfrac * lperc
        lzs = lzs + lzi
    # END if LZONE
    # LOWER ZONE End --------------------------------------------------
    #
    # GROUNDWATER Start -----------------------------------------------
    # simulate groundwater behavior - first account for the fact 
    # that iperc doesn't include lzirr
    gwi = ( iperc + IRRAPPV[4] ) - lzi
    # Start GWATER
    igwi = 0.0
    agwi = 0.0
    #if gwi > 0.0:
    if ( ( gwi - 0.0 ) >= smallVal ):
        igwi = deepfr * gwi
        agwi = gwi - igwi
    # end if
    ainflo = agwi + lits_agwli + IRRAPPV[5]
    agwo = 0.0
    # evaluate groundwater recharge parameter
    #if abs( kvary ) > 0.0:
    if ( ( abs(kvary) - 0.0 ) >= smallVal ):
        # update the index to variable groundwater slope
        gwvs = gwvs + ainflo
        if dayfg:
            #if gwvs > 0.0001:
            if ( ( gwvs - 0.0001 ) >= smallVal ):
                gwvs = gwvs * 0.97
            else:
                gwvs = 0.0
            # end inner if
        # end dayfg if
        # groundwater outflow(baseflow)
        #if agws > 1.0e-20:
        if ( ( agws - 0.0 ) >= smallVal ):
            # enough water to have outflow
            agwo = kgw * (1.0 + kvary * gwvs) * agws
            avail = ainflo + agws
            #if agwo > avail:
            if ( ( agwo - avail ) >= smallVal ):
                # ERRMSG3: Reduced AGWO value to available
                errorsV[3] += 1
                agwo = avail
                errorCnt += 1
                errMsg = "pwater_liftedloop - %s " % ERRMSG[3]
                CL.LOGR.error( errMsg )
            # end inner if
        # end agws if
        #elif agws > 1.0e-20:
    elif ( ( agws - 0.0 ) >= smallVal ):
        # enough water to have outflow
        agwo = kgw * agws
    # end kvary if
    #if agwo < 0.0:
    if ( ( 0.0 - agwo ) >= smallVal ):
        agwo = 0.0
    # end if agwo
    # no remaining water - this should happen only with hwtfg=1 it may
    # happen from lateral inflows, which is a bug, in which case negative
    # values for agws should show up in the output timeseries
    agws = agws + ( ainflo - agwo )
    # check removed to fix PERLND segments 101, 185
    #if agws < 0.0:
    if ( abs( agws - 0.0 ) < smallVal ):
        agws = 0.0
    elif ( ( 0.0 - agws ) >= smallVal ):
        #ERRMSG8: Reset AGWS to zero
        errorsV[8] += 1
        agws = 0.0
        errorCnt += 1
        errMsg = "pwater_liftedloop - %s " % ERRMSG[8]
        CL.LOGR.error( errMsg )
    # end if
    """ # check removed - now total PERLND agreement with HSPF
        if abs(kvary) > 0.0 and gwvs > agws:
            errorsV[4] += 1  # ERRMSG4: Reduced GWVS to AGWS
            gwvs = agws
    """
    # set the total groundwater storage value
    TGWS[targID][iI] = agws
    # END GWATER
    # GROUNDWATER End -------------------------------------------------
    #
    # EVAPORATION Start -----------------------------------------------
    # EVAPT to simulate evapotranspiration
    # rempet is remaining potential et - inches/ivl
    rempet = ts_pet
    # taet is total actual et - inches/ivlc
    taet  = 0.0
    baset = 0.0
    # interesting that et comes out of baseflow before interception 
    # storage and upper zone ??
    #if ( rempet > 0.0 ) and ( basetp > 0.0 ):
    if ( ( ( rempet - 0.0 ) >= smallVal ) and 
            ( ( basetp - 0.0 ) >= smallVal ) ):
        # ETBASE et from baseflow
        baspet = basetp * rempet
        #if baspet > agwo:
        if ( ( baspet - agwo ) >= smallVal ):
            baset = agwo
            agwo = 0.0
        else:
            baset = baspet
            agwo = agwo - baset
        # end inner if
        taet = taet + baset
        rempet = rempet - baset
    # END if ETBASE
    cepe  = 0.0
    #if ( rempet > 0.0 ) and ( ceps > 0.0 ):
    if ( ( ( rempet - 0.0 ) >= smallVal ) and 
            ( ( ceps - 0.0 ) >= smallVal ) ):
        # EVICEP
        #if rempet > ceps:
        if ( ( rempet - ceps ) >= smallVal ):
            cepe = ceps
            ceps = 0.0
        else:
            cepe = rempet
            ceps = ceps - cepe
        # end inner if
        taet = taet + cepe
        rempet = rempet - cepe
    # END if EVICEP
    uzet  = 0.0
    #if rempet > 0.0:
    if ( ( rempet - 0.0 ) >= smallVal ):
        # ETUZON
        # ETUZS
        #if ( uzs > 0.001 ):
        if ( ( uzs - 0.001 ) >= smallVal ):
            # there is et from the upper zone estimate the uzet opportunity
            uzrat = uzs / uzsn
            # if ( uzrat > 2.0 ):
            if ( ( uzrat - 2.0 ) >= smallVal ):
                uzpet = rempet
            else:
                uzpet = 0.5 * uzrat * rempet
            # end if
            #if uzpet > uzs:
            if ( ( uzpet - uzs ) >= smallVal ):
                uzet = uzs
                uzs = 0.0
            else:
                uzet = uzpet
                uzs = uzs - uzet
            # end if
        # END if UTUZA
        # these lines return to ETUZON
        taet = taet + uzet
        rempet = rempet - uzet
    # END if ETUZON
    agwet = 0.0
    #if ( rempet > 0.0 )  and ( agwetp > 0.0 ):
    if ( ( ( rempet - 0.0 ) >= smallVal ) and 
            ( ( agwetp - 0.0 ) >= smallVal ) ):
        # ETAGW et from groundwater determine remaining capacity
        gwpet = rempet * agwetp
        #if gwpet > agws:
        if ( ( gwpet - agws ) >= smallVal ):
            agwet = agws
            agws  = 0.0
        else:
            agwet = gwpet
            agws = agws - agwet
        # end inner if
        #if abs( kvary ) > 0.0:
        if ( ( abs( kvary ) - 0.0 ) >= smallVal ):
            # update variable storage
            gwvs = gwvs - agwet
            #if ( gwvs < -0.02 ):
            if ( abs( gwvs - 0.0 ) < smallVal ):
                # is zero
                gwvs = 0.0 
            elif ( ( 0.0 - gwvs ) >= smallVal ):
                # then have a negative value
                # ERRMSG5: GWVS < -0.02, set to zero
                errorsV[5] += 1
                gwvs = 0.0
                errorCnt += 1
                errMsg = "pwater_liftedloop - %s " % ERRMSG[5]
                CL.LOGR.error( errMsg )
            # end if
        # end if kvary
        taet = taet + agwet
        rempet = rempet - agwet
    # END if ETAGW
    # et from lower zone is handled here because it must be called every 
    # interval to make sure that seasonal variation in parameter lzetp 
    # and recalculation of rparm are correctly done ; simulate et from 
    # the lower zone note: thj made changes in some releae to the 
    # original HSPF, check carefully
    #ETLZON
    if dayfg:
        # it is time to recalculate et opportunity parameter rparm is 
        # max et opportunity - inches/ivl
        lzrat = lzs / lzsn
        #if ( lzetp <= 0.99999 ):
        if ( ( 0.99999 - lzetp ) > ( -1.0 * smallVal ) ):
            rparm =  ( 0.25 / ( 1.0 - lzetp ) ) * lzrat * ( DELT60 / 24.0 )
        else:
            rparm = 1.0e10
    # end dayfg if
    lzet = 0.0
    #if ( rempet > 0.0 ) and ( lzs > 0.02 ):
    if ( ( ( rempet - 0.0 ) >= smallVal ) and 
                ( ( lzs - 0.02 ) >= smallVal ) ):
        # assume that et can take place
        #if lzetp >= 0.99999:
        if ( ( lzetp - 0.99999 ) > ( -1.0 * smallVal ) ):
            # special case - will try to draw et from whole land 
            # segment at remaining potential rate
            lzpet = rempet * lzetp
        elif fl_vlefg <= 1:
            # usual case - desired et will vary over the whole 
            # land seg
            #if ( rempet > rparm ):
            if ( ( rempet - rparm ) >= smallVal ):
                lzpet = 0.5 * rparm
            else:
                #if ( rparm > 0.000001 ):
                if ( ( rparm - 0.0 ) >= smallVal ):
                    lzpet = rempet * ( 1.0 - rempet / ( 2.0 * rparm ) )
                else:
                    lzpet = rempet * ( 1.0 - rempet / 2.0 )
            # end if
            #if ( lzetp < 0.5 ):
            if ( ( 0.5 - lzetp ) >= smallVal ):
                # reduce the et to account for area devoid of 
                # vegetation
                lzpet = lzpet * 2.0 * lzetp
            # end if
        else:
            #  VLEFG >= 2:   # et constant over whole land seg
            #if ( lzrat < 1.0 ):
            if ( ( 1.0 - lzrat ) >= smallVal ):
                lzpet = lzetp * lzrat * rempet
            else:
                lzpet = lzetp * rempet
            # end inner if
        # if lzetp
        #if ( lzpet < ( lzs - 0.02 ) ):
        if ( ( ( lzs - 0.02 ) - lzpet ) >= smallVal ):
            lzet = lzpet
        else:
            lzet = lzs - 0.02
        # end if
        lzs = lzs - lzet
        taet = taet + lzet
        rempet = rempet - lzet
    # END if ETLZON
    # EVAPORATION end -------------------------------------------------
    #
    # UPDATE start ----------------------------------------------------
    # update the carry overs
    HOLD_MSUPY[targID][0] = msupy
    HOLD_RLZRAT[targID][0] = rlzrat
    HOLD_LZFRAC[targID][0] = lzfrac
    HOLD_DEC[targID][0] = dec
    HOLD_SRC[targID][0] = src
    HOLD_IFWK1[targID][0] = ifwk1
    HOLD_IFWK2[targID][0] = ifwk2
    # update the solution time series
    INFFAC[targID][iI] = inffac
    TGWS[targID][iI] = agws
    AGWET[targID][iI] = agwet
    AGWI[targID][iI] = agwi
    AGWO[targID][iI] = agwo
    AGWS[targID][iI] = agws
    BASET[targID][iI] = baset
    CEPE[targID][iI] = cepe
    CEPS[targID][iI] = ceps
    GWVS[targID][iI] = gwvs
    IFWI[targID][iI] = ifwi
    IFWO[targID][iI] = ifwo
    IFWS[targID][iI] = ifws
    IGWI[targID][iI] = igwi
    INFIL[targID][iI] = infil
    LZET[targID][iI] = lzet
    LZI[targID][iI] = lzi
    LZS[targID][iI] = lzs
    PERC[targID][iI] = perc
    RPARM[targID][iI] = rparm
    SURI[targID][iI] = suri
    SURO[targID][iI] = suro
    SURS[targID][iI] = surs
    TAET[targID][iI] = taet
    UZET[targID][iI] = uzet
    UZI[targID][iI] = uzi
    UZS[targID][iI] = uzs
    PERO[targID][iI] = suro + ifwo + agwo
    PERS[targID][iI] = ceps + surs + ifws + uzs + lzs + agws
    if fl_csnofg == 1:
        PETADJ[targID][iI] = petadj
    # UPDATE end ------------------------------------------------------
    # now return
    return errorCnt


def proute( psur, rtopfg, delt60, dec, src, surs ):
    """Determine how much potential surface detention (PSUR) 
    runs off in one simulation interval.

    Modified real number comparisons to be more numerically reliable.

    Args:
        psur (float): potential surface detention
        rtopfg (int): calculation flag to tell how to calculate
        delt60 (float): model time step adjusted to hours from minutes
        dec (float): calculated routing variable
        src (float): calculated routing variable
        surs (float): surface or overland flow storage
        ans (np.array): calculation vector

    Returns:
        np.array: three item array, ans, of return values
        
            0. suro ; surface outflow
            
            1. surs ; adjusted surface storage

            2. err ; counter for errors encountered
    
    """
    # import
    from math import pow
    # globals
    global MAXLOOPS, ERRMSG
    # parameters
    smallVal = float( 1E-13 )
    largVal = float( 2E-4 )
    # locals - put here in case want to do Cython
    ssupr = float( 0.0 )    # rate of moisture supply to overland flow surface
    surse = float( 0.0 )    # equilibrium detention storage for current 
                            # supply rate
    sursnw = float( 0.0 )
    suro = float( 0.0 )     # surface outflow in/interval
    ratio = float( 0.0 )    # ratio
    fact = float( 0.0 )     # factor
    ffact = float( 0.0 )    # coefficient
    fsuro = float( 0.0 )    # coefficient
    dfact = float( 0.0 )    # coefficient
    dfsuro = float( 0.0 )   # 
    dterm = float( 0.0 )    # additional term 
    dsuro = float( 0.0 )    #
    change = float( 0.0 )   # convergence 
    sursm = float( 0.0 )    # mean surface detention
    dummy = float( 0.0 )    # calculation variable
    tsuro = float( 0.0 )    # calculation variable
    err = int( 0 )
    # the return
    ans = np.zeros( 3, dtype=np.float64 )
    #count = int( 0 )
    # start
    # if psur > 0.0002
    if ( ( psur - 0.0 ) >= largVal ):
        # something is worth routing on the surface
        if rtopfg != 1:
            # do routing the new way, estimate the rate of supply to the 
            # overland flow surface - inches/hour
            ssupr = ( psur - surs ) / delt60
            # if ssupr > 0.0
            if ( ( ssupr - 0.0 ) > smallVal ):
                # determine equilibrium depth for this supply rate
                surse = dec * pow( ssupr, 0.6 )
            else:
                surse = 0.0
            # end if
            # determine runoff by iteration - newton's method,  estimate 
            #   the new surface storage
            sursnw = psur
            suro = 0.0 
            for count in range( MAXLOOPS ):
                # if ssupr > 0.0:
                if ( ( ssupr - 0.0 ) > smallVal ):
                    ratio = sursnw / surse
                    # if ratio <= 1.0
                    if ( ( ratio - 1.0 ) < smallVal ):
                        fact = 1.0 + 0.6 * pow( ratio, 3.0 )
                    else:
                        fact = 1.6
                    # end if
                else:
                    ratio = 1.0e30
                    fact = 1.6
                # end outer if
                # coefficient in outflow equation
                ffact = ( ( delt60 * src * pow( fact, 1.667 ) ) * 
                            pow( sursnw, 1.667 ) )
                fsuro = ffact - suro
                dfact = -1.667 * ffact
                dfsuro = ( dfact / sursnw ) - 1.0
                # if ratio <= 1.0: 
                if ( ( ratio - 1.0 ) < smallVal ):
                    #  additional term required in derivative wrt suro
                    dterm = dfact / ( fact * surse ) * 1.8 * pow( ratio, 2.0 )
                    dfsuro = dfsuro + dterm
                # end if
                dsuro = fsuro / dfsuro
                suro = suro - dsuro
                # if suro <= 1.0e-10:
                # boundary condition- don't let suro go negative
                if ( abs( suro - 0.0 ) < smallVal ):
                    suro = 0.0
                elif ( ( suro - 0.0 ) <= ( -1.0 * smallVal ) ):
                    # extra switch for if is really negative in
                    #   case want to add error checking in the future
                    suro = 0.0
                # end if
                sursnw = psur - suro
                change = 0.0
                # if abs(suro) > 0.0:
                if ( ( abs(suro) - 0.0 ) >= smallVal ):
                    change = abs( dsuro / suro )
                # end if
                # if change < 0.01:
                if ( ( change - 0.01 ) < smallVal ):
                    # then are ready to be done
                    break
                # end if
            else:   # no break
                # ERRMSG6: Proute runoff did not converge
                errMsg = "proute - %s " % ERRMSG[6]
                CL.LOGR.error( errMsg )
                err += 1
            surs = sursnw
        else:
            # do routing the way it is done in arm, nps, and hspx estimate
            # the rate of supply to the overland flow surface - inches/ivl
            ssupr = psur - surs
             # estimate the mean surface detention
            sursm = ( surs + psur ) * 0.5
            # estimate the equilibrium detention depth for this supply 
            #    rate - surse
            # if ssupr > 0.0:
            if ( ( ssupr - 0.0 ) >= smallVal ):
                # preliminary estimate of surse
                dummy = dec * pow( ssupr, 0.6 )
                # if dummy > sursm:
                if ( ( dummy - sursm ) >= smallVal ):
                    # flow is increasing
                    surse = dummy
                    dummy = sursm * ( 1.0 + 0.6 * 
                                        pow( ( sursm / surse ), 3.0 ) )
                else:
                    # flow on surface is at equilibrium or receding
                    dummy = sursm * 1.6
                # end inner if
            else:
                # flow on the surface is receding - equilibrium detention 
                # is assumed equal to actual detention
                dummy = sursm * 1.6
            # end outer if
            # check the temporary calculation of surface outflow
            tsuro = delt60 * src * pow( dummy, 1.667 )
            # if tsuro > psur
            if ( ( tsuro - psur ) >= smallVal ):
                suro = psur
                surs = 0.0
            else:
                suro = tsuro
                surs = psur - suro
            # end if
        # end if calc method
    else:
        # send what is on the overland flow plane straight to the channel
        suro = psur
        surs = 0.0
    # end if psur threshold
    #if suro <= 1.0e-10:
    if ( abs( suro - 0.0 ) < smallVal ):
        # the case of effectively 0
        suro = 0.0
    elif ( ( suro - 0.0 ) <= ( -1.0 * smallVal ) ):
        # the case of actually negative
        suro = 0.0 
    # end suro check if
    #return suro, surs, err in a numpy array
    ans[0] = suro
    ans[1] = surs
    ans[2] = float( err )
    return ans


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
    global AGWET, AGWI, AGWO, AGWS, BASET, CEPE, CEPS, GWVS, IFWI, IFWO
    global IFWS, IGWI, INFFAC, INFIL, LZET, LZI, LZS, RPARM, PERC, PERO
    global PERS, PET, SUPY, SURI, SURO, SURS, TAET, TGWS, UZET, UZI, UZS
    # parameters
    goodReturn = 0
    badReturn = -1
    pathStart = "/RESULTS/PERLND_"
    pathEnd = "/PWATER"
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
            if cOut == "AGWET":
                outView = AGWET[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "AGWI":
                outView = AGWI[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "AGWO":
                outView = AGWO[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "AGWS":
                outView = AGWS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "BASET":
                outView = BASET[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "CEPE": 
                outView = CEPE[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "CEPS":
                outView = CEPS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "GWVS":
                outView = GWVS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "IFWI":
                outView = IFWI[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "IFWO":
                outView = IFWO[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "IFWS":
                outView = IFWS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "IGWI":
                outView = IGWI[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "INFFAC":
                outView = INFFAC[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "INFIL":
                outView = INFIL[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "LZET":
                outView = LZET[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "LZI":
                outView = LZI[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "LZS":
                outView = LZS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "RPARM":
                outView = RPARM[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "PERC":
                outView = PERC[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "PERO":
                outView = PERO[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "PERS":
                outView = PERS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "PET":
                outView = PET[tCol].view( dtype=np.float32 )
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
            elif cOut == "TAET":
                outView = TAET[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "TGWS":
                outView = TGWS[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "UZET":
                outView = UZET[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "UZI":
                outView = UZI[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "UZS":
                outView = UZS[tCol].view( dtype=np.float32 )
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


def getIGWIbyTargTS( iI, targID ):
    """Get IGWI, outflow to inactive groundwater for a specified target
    and time step index.

    Args:
        iI (int): current simulation day index, 0-based
        targID (str): current PERLND target
    
    Returns:
        float: ovol, outflow to inactive groundwater in acre-ft/day

    """
    # globals
    global IGWI
    # get
    ovol = float( IGWI[targID][iI] )
    # return
    return ovol


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


def getNominalStorages( targID ):
    """Get the nominal storage values for this watershed in
    inches

    Args:
        targID (str): current PERLND target
    
    Returns:
        float: uzsn, upper zone nominal storage in inches
        float: lzsn, lower zone nominal storage in inches

    """
    # globals
    global LZSN, UZSN
    #
    uzsn = float( UZSN[targID][0] )
    lzsn = float( LZSN[targID][0] )
    # return
    return uzsn, lzsn


def getCurrentStorages( iI, targID ):
    """Get the current storage values for this watershed in
    inches.

    Args:
        iI (int): time step index to extract the storage values
        targID (str): current PERLND target
    
    Returns:
        float: uzs, upper zone current storage in inches
        float: lzs, lower zone currernt storage in inches

    """
    # globals
    global LZS, UZS
    #
    uzs = float( UZS[targID][iI] )
    lzs = float( LZS[targID][iI] )
    # return
    return uzs, lzs


def setCurrentStorages( iI, targID, uzs, lzs ):
    """Set the current storage values for this watershed in
    inches.

    Args:
        iI (int): time step index to extract the storage values
        targID (str): current PERLND target
        uzs (float): upper zone current storage in inches
        lzs (float): lower zone currernt storage in inches

    """
    # globals
    global LZS, UZS
    #
    UZS[targID][iI] = uzs
    LZS[targID][iI] = lzs
    # return
    return


def getPERObyTargTS( iI, targID ):
    """Get the total outflow from pervious land, inches/day

    Args:
        iI (int): time step index to extract the storage values
        targID (str): current PERLND target

    Return:
        float: pero, total outflow in inches/day

    """
    # global
    global PERO
    #
    pero = float( PERO[targID][iI] )
    # return
    return pero


# EOF