# -*- coding: utf-8 -*-
"""
Replacement for *HSPsquared* hrchhyd that represents water movement and
storage in stream segments and well mixed reservoirs, or **RCHRES**.

Had to replace *HSPsquared* hrchhyd so that can break into the main time
loop at the beginning and end of each day.This required fundamentally
restructuring the storage and memory allocation within *HSPsquared*.

locaHrchhyd functions as a module handling storage for global
**RCHRES** variables as well as for parameter and constant
definitions.

Internal time units are in seconds, DELTS. Internal length units
are feet and all areas, volumes, and lengths are converted to
feet for internal calculations and then reconverted back to
acres and acre-ft for areas and volumes respectively.

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
TOLERANCE = 0.001
"""Newton method convergence closure criterion.

Only used in the auxil function which has been replaced with npAuxil."""
MAXLOOPS  = 100
"""Newton method maximum number of iterations.

Only used in the auxil function which has been replaced with npAuxil."""
ERRMSG = ['HYDR: SOLVE equations are indeterminate',             #ERRMSG0
          'HYDR: extrapolation of rchtab will take place',       #ERRMSG1
          'HYDR: SOLVE trapped with an oscillating condition',   #ERRMSG2
          'HYDR: Solve did not converge',                        #ERRMSG3
          'HYDR: Solve converged to point outside valid range']  #ERRMSG4
"""Defined error messages.

Used with errorsV for error handling. Currently these errors, if occur,
are written to the log file.
"""
errorsV = np.zeros( len(ERRMSG), dtype=np.int32 )
"""Error handling in liftedloop.

This has been largely replaced but is maintained for backwards
compatibility/tracing."""
# units conversion constants, 1 ACRE is 43560 sq ft. assumes input in acre-ft
VFACT = 43560.0
"""Acre-ft to cubic feet conversion """
AFACT = 43560.0
"""Acres to square feet conversion """
VFACTA = 1.0/VFACT
"""Volume conversion factor"""
LFACTA = 1.0
"""Length conversion factor"""
AFACTA = 1.0/AFACT
"""Area conversion factor """
SFACTA = 1.0
"""Conversion factor"""
TFACTA = 1.0
"""Conversion factor"""
# physical constants (English units)
GAM = 62.4
"""Unit weight or density of water in lb/ft3"""
GRAV = 32.2
"""Accleration due to earth's gravity ft/s2"""
AKAPPA = 0.4
"""Von Karmann's constant"""
ORG_SSA_CALC = False
"""Switch to determine lookup table calculation metod.

If true, use the original auxil function to calculate surface area
and depth from volume. If false use the np array lookup function
interp."""

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# new module wide parameters
# want to keep the solution value and input value structures here
# some new control information
DELTS = 0.0
"""RCHRES calculation time step in seconds"""
MAX_EXITS = 5
"""Maximum number of exits for a RCHRES"""
PARAM_GOOD = [ "DB50", "DELTH", "FTBUCI", "KS", "LEN", "STCOR",  ]
"""Input, non-state parameters that are used in mHSP2"""
PARAM_UNUSED = [ "CONVFM", "FTBW", "IREXIT", "IRMINV" ]
"""Input parameters that unused in mHSP2"""
RR_TGRPN_SUPP = "INFLOW"
"""Only supported mass link target or destination group name"""
RR_TMEMN_SUPP = "IVOL"
"""Only supported mass link target destination group member"""
KEY_TS_PRECIP = "PREC"
"""External time series key for precipitation"""
KEY_TS_PET = "POTEV"
"""External time series key for PET """
KEY_TS_COLIND = "COLIND"
"""External time series key for COLIND """
KEY_TS_OUTDGT = "OUTDGT"
"""External time series key for OUTDGT"""
FLAG_GOOD = [ "AUX1FG", "AUX2FG", "AUX3FG", "FUNCT", "ODFVF", "ODGTF" ]
"""Flags that are at least referenced in mHSP2"""
nFUNCT_FLAG = [ "FUNCT1", "FUNCT2", "FUNCT3", "FUNCT4", "FUNCT5" ]
"""New HDF5 format, FUNCT flags; One for each exit rather than a column"""
nODFVF_FLAG = [ "ODFVF1", "ODFVF2", "ODFVF3", "ODFVF4", "ODFVF5" ]
"""New HDF5 format, ODFVF flags; One for each exit rather than a column"""
nODGTF_FLAG = [ "ODGTF1", "ODGTF2", "ODGTF3", "ODGTF4", "ODGTF5" ]
"""New HDF5 format, ODGTF flags; One for each exit rather than a column"""
nFLAG_GOOD = [ "AUX1FG", "AUX2FG", "AUX3FG" ]
"""New HDF5 format, flags that at least referenced in mHSP2"""
nFLAG_GOOD.extend( nFUNCT_FLAG )
nFLAG_GOOD.extend( nODFVF_FLAG )
nFLAG_GOOD.extend( nODGTF_FLAG )

FLAG_UNUSED = [ "ICAT", "VCONFG" ]
"""Unused flags in mHSP2"""
INIT_PARMS = [ "COLIN", "OUTDG", "VOL" ]
"""The list of initial parameters that suppported"""
nCOLIN_STATE = [ "COLIN1", "COLIN2", "COLIN3", "COLIN4", "COLIN5" ]
"""New HDF5 format, COLIN state parameters """
nOUTDG_STATE = [ "OUTDG1", "OUTDG2", "OUTDG3", "OUTDG4", "OUTDG5" ]
"""New HDF5 format, COLIN state parameters """
nINIT_PARMS = [ "VOL" ]
"""New HDF5 format, list of initial parameters that suppported"""
nINIT_PARMS.extend( nCOLIN_STATE )
nINIT_PARMS.extend( nOUTDG_STATE )

INIT_PARMS_UNUSED = [ "CAT", "ICAT "]
"""Unused initial parameters in mHSP2"""
EXTERNAL_TS_GOOD = [ KEY_TS_PRECIP, KEY_TS_PET, RR_TMEMN_SUPP,
                     KEY_TS_COLIND, KEY_TS_OUTDGT ]
"""All supported external time series"""
EXTERNAL_TS_UNUSED = [ "COTDGT", "CIVOL", "ICON", "GATMP",
                       "DEWTMP", "SOLRAD", "CLOUD", "WIND", "TGRND", "PHVAL",
                       "ROC", "BIO", "COADFX", "COADCN",  "GQADFX", "GQADCN",
                       "NUADFX", "NUADCN", "PLADFX", "PLADCN" ]
"""Unsupported external time series"""
GOOD_OUTPUT_LIST = [ "AVDEP", "AVVEL", "DEP", "HRAD", "IVOL", "O1",
                     "O2", "O3", "O4", "O5", "OVOL1", "OVOL2",
                     "OVOL3", "OVOL4", "OVOL5", "PRSUPY", "RO",
                     "ROVOL", "SAREA", "STAGE", "VOL", "TAU",
                     "TWID", "USTAR", "VOLEV" ]
"""List of currently supported outputs and thus calculated values"""
BAD_OUTPUT_LIST = [ "AVSECT", 'CDFVOL1', 'CDFVOL2', 'CDFVOL3',
                    'CDFVOL4', 'CDFVOL5', 'CIVOL', 'CO1', 'CO2',
                    'CO3', 'CO4', 'CO5', 'COVOL1', 'COVOL2', 'COVOL3',
                    'COVOL4', 'COVOL5', "CRO", "CROVOL", "CVOL",
                    'RIRDEM', 'RIRSHT' ]
"""Currently unsupported outputs """
NEXITS = None
"""Data structure to hold number of exits for each RCHRES"""
SCHEMATIC_MAP = dict()
""" Schematic map for identifying inflow locations to each RCHRES.

Keys are target IDs and values are list, enumerated below.

    0. (str): source volume type

    1. (str): source volume ID

    2. (str): source volume output time series

    3. (list): integer exit ID

    4. (float): AFACTOR, area conversion multiplier

    5. (float): MFACTOR, conversion multiplier

"""
OUTPUT_CONTROL = None
"""Control structure telling which time series are to be output"""

# Holdover and carryover calculation variables
HOLD_RO = None
"""Value of RO to cover over between time steps"""
HOLD_OS1 = None
"""Value of OS for exit 1 to hold over """
HOLD_OS2 = None
"""Value of OS for exit 1 to hold over """
HOLD_OS3 = None
"""Value of OS for exit 1 to hold over """
HOLD_OS4 = None
"""Value of OS for exit 1 to hold over """
HOLD_OS5 = None
"""Value of OS for exit 1 to hold over """

# data type specifications for making rec arrays
DEF_DT = None
"""The data type specification for time series structured array or
record array"""
SPEC_DT = None
"""The data type specification for the calculation and input
record arrays"""
FLAG_DT = None
"""The data type specification for flag record arrays"""

# Specified parameters that not monthly by rchres. These come from
#   the UCI file
# HYDR-PARM1
AUX1FG = None
"""Flag identifying if certain flow outputs will be calculated"""
AUX2FG = None
"""Flag identifying if certain flow outputs will be calculated"""
AUX3FG = None
"""Flag identifying if certain flow outputs will be calculated"""
ODFVFG1 = None
"""Flag identifying FTAB output for each RCHRES for exit 1"""
ODFVFG2 = None
"""Flag identifying FTAB output for each RCHRES for exit 2"""
ODFVFG3 = None
"""Flag identifying FTAB output for each RCHRES for exit 3"""
ODFVFG4 = None
"""Flag identifying FTAB output for each RCHRES for exit 4"""
ODFVFG5 = None
"""Flag identifying FTAB output for each RCHRES for exit 5"""
ODGTFG1 = None
"""Flag identifying time series output for each RCHRES for exit 1"""
ODGTFG2 = None
"""Flag identifying time series output for each RCHRES for exit 2"""
ODGTFG3 = None
"""Flag identifying time series output for each RCHRES for exit 3"""
ODGTFG4 = None
"""Flag identifying time series output for each RCHRES for exit 4"""
ODGTFG5 = None
"""Flag identifying time series output for each RCHRES for exit 5"""
FUNCT1 = None
"""Flag identifying functional relationship for combining time and
vol for exit 1"""
FUNCT2 = None
"""Flag identifying functional relationship for combining time and
vol for exit 2"""
FUNCT3 = None
"""Flag identifying functional relationship for combining time and
vol for exit 3"""
FUNCT4 = None
"""Flag identifying functional relationship for combining time and
vol for exit 4"""
FUNCT5 = None
"""Flag identifying functional relationship for combining time and
vol for exit 5"""

# HYDR-PARM2
FTABNO = None
"""FTABLE id for each RCHRES """
LEN = None
"""Length for each RCHRES"""
DELTH = None
"""Drop in water elevation from upstream to downstream"""
STCOR = None
"""Correction to RCHRES depth to calculate stage"""
KS = None
"""Weighting factor for hydraulic routing"""
DB50 = None
"""Sediment median grain diameter
Specified in inches in inputs and converted to feet for calcs """
# HYDR-INIT
I_VOL = None
"""Initial volume of water in the RCHRES """
COLIN1 = None
"""Initial value of COLIND for exit 1"""
COLIN2 = None
"""Initial value of COLIND for exit 1"""
COLIN3 = None
"""Initial value of COLIND for exit 1"""
COLIN4 = None
"""Initial value of COLIND for exit 1"""
COLIN5 = None
"""Initial value of COLIND for exit 1"""
LKFG = None
"""Lake flag; 0 == stream and 1 == lake """
OUTDG1 = None
"""Initial demand for time dependenent outflow exit 1"""
OUTDG2 = None
"""Initial demand for time dependenent outflow exit 2"""
OUTDG3 = None
"""Initial demand for time dependenent outflow exit 3"""
OUTDG4 = None
"""Initial demand for time dependenent outflow exit 4"""
OUTDG5 = None
"""Initial demand for time dependenent outflow exit 5"""


# Data and simulated time series
AVDEP = None
"""Average depth in ft"""
AVVEL = None
""" Average velocity in ft/s"""
COLIND1 = None
"""COLIND external time series index ratio for exit 1"""
COLIND2 = None
"""COLIND external time series index ratio for exit 2"""
COLIND3 = None
"""COLIND external time series index ratio for exit 3"""
COLIND4 = None
"""COLIND external time series index ratio for exit 4"""
COLIND5 = None
"""COLIND external time series index ratio for exit 5"""
DEP = None
"""Dpeth in feet in rchres"""
EXIVOL = None
"""External time series inflow, acre-feet per time interval"""
HRAD = None
"""Hydraulic radius in ft"""
IVOL = None
"""Inflow to RCHRES, acre-feet per time interval"""
O1 = None
"""Rate of outflow through exit 1, ft3/s """
O2 = None
"""Rate of outflow through exit 2, ft3/s """
O3 = None
"""Rate of outflow through exit 3, ft3/s """
O4 = None
"""Rate of outflow through exit 4, ft3/s """
O5 = None
"""Rate of outflow through exit 5, ft3/s """
OUTDGT1 = None
"""OUTDGT time varying external time series for exit 1"""
OUTDGT2 = None
"""OUTDGT time varying external time series for exit 2"""
OUTDGT3 = None
"""OUTDGT time varying external time series for exit 3"""
OUTDGT4 = None
"""OUTDGT time varying external time series for exit 4"""
OUTDGT5 = None
"""OUTDGT time varying external time series for exit 5"""
OVOL1 = None
"""Volume of outflow through exit 1, af/day """
OVOL2 = None
"""Volume of outflow through exit 2, af/day """
OVOL3 = None
"""Volume of outflow through exit 3, af/day """
OVOL4 = None
"""Volume of outflow through exit 4, af/day """
OVOL5 = None
"""Volume of outflow through exit 5, af/day """
POTEV = None
"""Input time series of potential evaporation from reservoir surface,
inches per interval """
PREC = None
"""Input time series of precipitation to RCHRES, inches per interval"""
PRSUPY = None
"""Volume of water contributed by precipitation to the surface,
af/day """
RO = None
"""Total rate of outflow from RCHRES, ft3/s"""
ROVOL = None
"""Total volume of outflow from RCHRES, af/day """
SAREA = None
"""Surface area of RCHRES in acres """
STAGE = None
"""Stage of RCHRES = DEP + STCOR, ft"""
TAU = None
"""Bed shear stress"""
TWID = None
"""Stream top width, ft"""
USTAR = None
"""Shear velocity"""
VOL = None
"""Volume of water in the RCHRES, af  """
VOLEV = None
"""Volume of water lost by evaporation, af/day """


def setDelT( sim_delt ):
    """Set the pervious land delt for calculations.

    The delt is stored as a module wide global. Sets DELTS
    module-wide global.

    Arguments:
        sim_delt (float): overall simulation time step in minutes

    """
    global DELTS
    # function
    DELTS = sim_delt * 60.0
    # return
    return


def setNExits( targID, numExit ):
    """Set the number of exits for each RCHRES

    Args:
        targID (str): the target iD
        numExit (int): the number of exits

    """
    # imports
    # globals
    global NEXITS
    # parameters
    # locals
    NEXITS[targID][0] = numExit
    # return
    return


def setLakeFlag( targID, lFlag ):
    """Set the lake flag for each RCHRES

    Args:
        targID (str): the target iD
        lFlag (int): the number of exits

    """
    # imports
    # globals
    global LKFG
    # parameters
    # locals
    LKFG[targID][0] = lFlag
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
    # control parameters
    global NEXITS, ODFVFG1, ODFVFG2, ODFVFG3, ODFVFG4, ODFVFG5
    global ODGTFG1, ODGTFG2, ODGTFG3, ODGTFG4, ODGTFG5
    global FUNCT1, FUNCT2, FUNCT3, FUNCT4, FUNCT5
    global OUTDG1, OUTDG2, OUTDG3, OUTDG4, OUTDG5
    global OUTPUT_CONTROL, GOOD_OUTPUT_LIST, LKFG, AUX1FG
    global AUX2FG, AUX3FG
    # parameters
    global FTABNO, LEN, DELTH, STCOR, KS, DB50
    # initial values
    global I_VOL, COLIN1, COLIN2, COLIN3, COLIN4, COLIN5
    # time series
    global DEP, IVOL, O1, O2, O3, O4, O5, OVOL1, OVOL2, OVOL3
    global OVOL4, OVOL5, POTEV, PREC, PRSUPY, RO, ROVOL, SAREA
    global STAGE, VOL, VOLEV, EXIVOL
    global COLIND1, COLIND2, COLIND3, COLIND4, COLIND5
    global OUTDGT1, OUTDGT2, OUTDGT3, OUTDGT4, OUTDGT5
    global TAU, USTAR, AVVEL, AVDEP, HRAD, TWID
    # carryovers
    global HOLD_RO, HOLD_OS1, HOLD_OS2, HOLD_OS3, HOLD_OS4, HOLD_OS5
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
    # no initialize and allocate
    # control parameters
    LKFG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    NEXITS = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODFVFG1 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODFVFG2 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODFVFG3 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODFVFG4 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODFVFG5 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODGTFG1 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODGTFG2 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODGTFG3 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODGTFG4 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    ODGTFG5 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    FUNCT1 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    FUNCT2 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    FUNCT3 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    FUNCT4 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    FUNCT5 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    OUTPUT_CONTROL = np.rec.array( np.zeros( len( GOOD_OUTPUT_LIST),
                                   dtype=FLAG_DT ) )
    AUX1FG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    AUX2FG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    AUX3FG = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    # parameters
    FTABNO = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    LEN = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    DB50 = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    DELTH = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    STCOR = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    KS = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    # initial values
    I_VOL = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    COLIN1 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    COLIN2 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    COLIN3 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    COLIN4 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    COLIN5 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    OUTDG1 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    OUTDG2 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    OUTDG3 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    OUTDG4 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    OUTDG5 = np.rec.array( np.zeros( 1, dtype=FLAG_DT ) )
    # time series
    AVDEP = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    AVVEL = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    COLIND1 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    COLIND2 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    COLIND3 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    COLIND4 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    COLIND5 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    DEP = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    EXIVOL = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    HRAD = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    IVOL = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    O1 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    O2 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    O3 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    O4 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    O5 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OUTDGT1 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OUTDGT2 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OUTDGT3 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OUTDGT4 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OUTDGT5 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OVOL1 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OVOL2 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OVOL3 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OVOL4 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    OVOL5 = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    POTEV = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PREC = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    PRSUPY = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    RO = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    ROVOL = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    SAREA = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    STAGE = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    TAU = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    TWID = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    USTAR = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    VOL = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    VOLEV = np.rec.array( np.zeros( sim_len, dtype=DEF_DT ) )
    # carryovers
    HOLD_RO = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_OS1 = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_OS2 = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_OS3 = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_OS4 = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    HOLD_OS5 = np.rec.array( np.zeros( 1, dtype=SPEC_DT ) )
    # return
    return


def setPrecipTS( targID, npTS ):
    """Set the precipitation time series from one data set
    to one target. SUPY is where precipitation is stored for calculations

    Args:
        targID (str): the target identifier - must be same as used
                        to create the rec array
        npTS (np.array): 1D array with the time series values

    """
    # imports
    # globals
    global PREC
    # parameters
    # local
    # start
    PREC[ targID ][:] += npTS
    # return
    return


def setPETTS( targID, npTS ):
    """Set the PET time series from one data set
    to one target. PET is where pet is stored for calculations.
    Might be adjusted by various activities.

    Args:
        targID (str): the target identifier - must be same as used
                        to create the rec array
        npTS (np.array): 1D array with the time series values

    """
    # imports
    # globals
    global POTEV
    # parameters
    # local
    # start
    POTEV[ targID ][:] += npTS
    # return
    return


def setCOLINDTS( targID, npTS, nExit ):
    """Set the COLIND time series from one data set
    to one target and exit number.

    Args:
        targID (str): the target identifier - must be same as used
                        to create the rec array
        npTS (np.array): 1D array with the time series values
        nExit (int): the COLIND index number for this RR

    """
    # imports
    # globals
    global COLIND1, COLIND2, COLIND3, COLIND4, COLIND5, MAX_EXITS, NEXITS
    # parameters
    # local
    # start
    # first need to get our exit number
    totEx = int( NEXITS[targID][0] )
    ofExArr = getODFVFG( targID, totEx )
    retArray = np.argwhere( ofExArr == (-1.0 * nExit ) )
    if retArray.size <= 0:
        # then there was an issue
        errMsg = "Did not find a negative exit denoting COLIND use. " \
                 "Expected to find a %d value for ODFVFG for %s!!!" % \
                 ( (-1.0 * nExit ), targID )
        CL.LOGR.error( errMsg )
        return
    # end if
    setExit = int( retArray[0] )
    if ( setExit >= 0 ) and ( setExit < MAX_EXITS ):
        # then we have a good exit value
        if setExit == 0:
            COLIND1[targID][:] += npTS
        elif setExit == 1:
            COLIND2[targID][:] += npTS
        elif setExit == 2:
            COLIND3[targID][:] += npTS
        elif setExit == 3:
            COLIND4[targID][:] += npTS
        elif setExit == 4:
            COLIND5[targID][:] += npTS
        # end if
    else:
        # this is an error
        errMsg = "COLIND external time series has a index value of %d.\n" \
                 "This functionality is not supported and will be ignored!!!" \
                 % nExit
        CL.LOGR.error( errMsg )
    # return
    return


def setOUTDGTTS( targID, npTS, nExit ):
    """Set the OUTDGT time series from one data set
    to one target and exit number.

    Args:
        targID (str): the target identifier - must be same as used
                        to create the rec array
        npTS (np.array): 1D array with the time series values
        nExit (int): the ODGTFG index number for this RR

    """
    # imports
    # globals
    global OUTDGT1, OUTDGT2, OUTDGT3, OUTDGT4, OUTDGT5, MAX_EXITS, NEXITS
    # parameters
    # local
    # start
    # first need to get our exit number
    totEx = int( NEXITS[targID][0] )
    odExArr = getODGTFG( targID, totEx )
    retArray = np.argwhere( odExArr == nExit )
    if retArray.size <= 0:
        # then there was an issue
        errMsg = "Did not find a matching index for OUTDGT use. " \
                 "Expected to find a %d value for OUTDGT for %s!!!" % \
                 ( nExit, targID )
        CL.LOGR.error( errMsg )
        return
    # end if
    setExit = int( retArray[0] )
    if ( setExit >= 0 ) and ( setExit < MAX_EXITS ):
        # then we have a good exit value
        if setExit == 0:
            OUTDGT1[targID][:] += npTS
        elif setExit == 1:
            OUTDGT2[targID][:] += npTS
        elif setExit == 2:
            OUTDGT3[targID][:] += npTS
        elif setExit == 3:
            OUTDGT4[targID][:] += npTS
        elif setExit == 4:
            OUTDGT5[targID][:] += npTS
        # end if
    else:
        # this is an error
        errMsg = "OUTDGT external time series has a index value of %d.\n" \
                 "This functionality is not supported and will be ignored!!!" \
                 % nExit
        CL.LOGR.error( errMsg )
    # return
    return


def setExInTS( targID, npTS ):
    """Set the external inflow time series from one data set
    to one target. PET is where pet is stored for calculations. Might be
    adjusted by various activities.

    Args:
        targID (str): the target identifier - must be same as used
                        to create the rec array
        npTS (np.array): 1D array with the time series values

    """
    # imports
    # globals
    global EXIVOL
    # parameters
    # local
    # start
    EXIVOL[ targID ][:] += npTS
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
    global EXTERNAL_TS_UNUSED, EXTERNAL_TS_GOOD, KEY_TS_PRECIP, KEY_TS_PET
    global KEY_TS_COLIND, KEY_TS_OUTDGT
    # parameter
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    # go throught the mapping list and allocate
    for mapList in TSMapList:
        tsType = mapList[1]
        tsID = mapList[0]
        tsTargID = mapList[2]
        # now check and see if the external type is supported
        if tsType in EXTERNAL_TS_UNUSED:
            # this means that cannot use this so provide
            #   a warning
            warnMsg = "Currently external time series of type " \
                        "%s are unsupported. This time series " \
                        "will be ignored for %s - %s !!!" % \
                        ( tsType, "RCHRES", tsTargID )
            CL.LOGR.warning( warnMsg )
            continue
        # check if in the good list
        if tsType in EXTERNAL_TS_GOOD:
            # first check if a lateral inflows
            if tsType == KEY_TS_PRECIP:
                # then this is a precip time series
                pVals = np.array( AllTSDict[tsID].values, dtype=np.float32 )
                setPrecipTS( tsTargID, pVals )
            elif tsType == KEY_TS_PET:
                # then this is a PET time series
                etVals = np.array( AllTSDict[tsID].values, dtype=np.float32 )
                setPETTS( tsTargID, etVals )
            elif tsType == RR_TMEMN_SUPP:
                # then this is an inflow time series
                extInVals = np.array( AllTSDict[tsID].values, dtype=np.float32 )
                setExInTS( tsTargID, extInVals )
            elif tsType == KEY_TS_COLIND:
                if len( mapList ) >= 4:
                    nExit = mapList[3]
                else:
                    nExit = -1
                colInVals = np.array( AllTSDict[tsID].values, dtype=np.float32 )
                setCOLINDTS( tsTargID, colInVals, nExit )
            elif tsType == KEY_TS_OUTDGT:
                if len( mapList ) >= 4:
                    nExit = mapList[3]
                else:
                    nExit = -1
                outdVals = np.array( AllTSDict[tsID].values, dtype=np.float32 )
                setOUTDGTTS( tsTargID, outdVals, nExit )
            else:
                # unsupported time series type
                errMsg = "Unsupported time series type %s " \
                         "found for %s - %s!!!" % \
                         ( tsType, "RCHRES", tsTargID )
                CL.LOGR.error( errMsg )
                return badReturn
            # end inner if
        else:
            # unknown and unsupported time series type
            warnMsg = "Unknown and unsupported time series type %s " \
                      "found for %s - %s. Time series will be " \
                      "ignored!!!"  % ( tsType, "RCHRES", tsTargID )
            CL.LOGR.warning( warnMsg )
            continue
        # end if
    # end for maplist
    # return
    return goodReturn


def setGoodFlag( targID, tFlag, tfVal ):
    """Set the value for the specified flag structure

    Args:
        targID (str): ID or recarray header to set
        tFlag (str): flag string to identify the data structure
        tfVal (tuple or int): flag value to set

    """
    # imports
    # globals
    global ODFVFG1, ODFVFG2, ODFVFG3, ODFVFG4, ODFVFG5, MAX_EXITS
    global ODGTFG1, ODGTFG2, ODGTFG3, ODGTFG4, ODGTFG5
    global FUNCT1, FUNCT2, FUNCT3, FUNCT4, FUNCT5
    global AUX1FG, AUX2FG, AUX3FG
    # parameters
    # locals
    # start
    if tFlag == "ODFVF":
        for iI in range( MAX_EXITS ):
            if iI == 0:
                ODFVFG1[targID][0] = int( tfVal[iI] )
            elif iI == 1:
                ODFVFG2[targID][0] = int( tfVal[iI] )
            elif iI == 2:
                ODFVFG3[targID][0] = int( tfVal[iI] )
            elif iI == 3:
                ODFVFG4[targID][0] = int( tfVal[iI] )
            elif iI == 4:
                ODFVFG5[targID][0] = int( tfVal[iI] )
            # end if
        # end for exits
    elif tFlag == "ODFVF1":
        ODFVFG1[targID][0] = int( tfVal )
    elif tFlag == "ODFVF2":
        ODFVFG2[targID][0] = int( tfVal )
    elif tFlag == "ODFVF3":
        ODFVFG3[targID][0] = int( tfVal )
    elif tFlag == "ODFVF4":
        ODFVFG4[targID][0] = int( tfVal )
    elif tFlag == "ODFVF5":
        ODFVFG5[targID][0] = int( tfVal )
    elif tFlag == "ODGTFG":
        for iI in range( MAX_EXITS ):
            if iI == 0:
                ODGTFG1[targID][0] = int( tfVal[iI] )
            elif iI == 1:
                ODGTFG2[targID][0] = int( tfVal[iI] )
            elif iI == 2:
                ODGTFG3[targID][0] = int( tfVal[iI] )
            elif iI == 3:
                ODGTFG4[targID][0] = int( tfVal[iI] )
            elif iI == 4:
                ODGTFG5[targID][0] = int( tfVal[iI] )
            # end if
        # end for exits
    elif tFlag == "ODGTF1":
        ODGTFG1[targID][0] = int( tfVal )
    elif tFlag == "ODGTF2":
        ODGTFG2[targID][0] = int( tfVal )
    elif tFlag == "ODGTF3":
        ODGTFG3[targID][0] = int( tfVal )
    elif tFlag == "ODGTF4":
        ODGTFG4[targID][0] = int( tfVal )
    elif tFlag == "ODGTF5":
        ODGTFG5[targID][0] = int( tfVal )
    elif tFlag == "FUNCT":
        for iI in range( MAX_EXITS ):
            if iI == 0:
                FUNCT1[targID][0] = int( tfVal[iI] )
            elif iI == 1:
                FUNCT2[targID][0] = int( tfVal[iI] )
            elif iI == 2:
                FUNCT3[targID][0] = int( tfVal[iI] )
            elif iI == 3:
                FUNCT4[targID][0] = int( tfVal[iI] )
            elif iI == 4:
                FUNCT5[targID][0] = int( tfVal[iI] )
            # end if
        # end for exits
    elif tFlag == "FUNCT1":
        FUNCT1[targID][0] = int( tfVal )
    elif tFlag == "FUNCT2":
        FUNCT2[targID][0] = int( tfVal )
    elif tFlag == "FUNCT3":
        FUNCT3[targID][0] = int( tfVal )
    elif tFlag == "FUNCT4":
        FUNCT4[targID][0] = int( tfVal )
    elif tFlag == "FUNCT5":
        FUNCT5[targID][0] = int( tfVal )
    elif tFlag == "AUX1FG":
        AUX1FG[targID][0] = int( tfVal )
    elif tFlag == "AUX2FG":
        AUX2FG[targID][0] = int( tfVal )
    elif tFlag == "AUX3FG":
        AUX3FG[targID][0] = int( tfVal )
    # end defined flags
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
    import re
    # globals
    global DELTH, FTABNO, KS, LEN, STCOR, DB50
    # parameters
    # locals
    # start
    if tParam == "DELTH":
        DELTH[ targID ][0] = float( pVal )
    elif tParam == "DB50":
        if pVal > 1.0E-9:
            assVal = pVal
        else:
            assVal = 0.01
        # end if
        DB50[ targID ][0] = float( assVal )
    elif tParam == "FTBUCI":
        intList = [ int(num) for num in re.findall( r'\d+', pVal ) ]
        if len(intList) >= 1:
            ftInt = intList[0]
        else:
            ftInt = -1
        FTABNO[ targID ][0] = ftInt
    elif tParam == "KS":
        KS[ targID ][0] = float( pVal )
    elif tParam == "LEN":
        if pVal > 0.01:
            assVal = pVal
        else:
            assVal = 0.1
        LEN[ targID ][0] = float( assVal )
    elif tParam == "STCOR":
        STCOR[ targID ][0] = float( pVal )
    # return
    return


def setInitialParams( targID, tParam, pVal ):
    """Set the value for the specified initial values

    Args:
        targID (str): ID or recarray header to set
        tParam (str): param string to identify the data structure
        pVal (float or tuple): parameter value(s) to set

    """
    # imports
    # globals
    global COLIN1, COLIN2, COLIN3, COLIN4, COLIN5, MAX_EXITS
    global OUTDG1, OUTDG2, OUTDG3, OUTDG4, OUTDG5
    global I_VOL
    # parameters
    # locals
    # start
    if tParam == 'VOL':
        I_VOL[ targID ][0] = float( pVal )
    elif tParam == "COLIN":
        for iI in range( MAX_EXITS ):
            if iI == 0:
                COLIN1[targID][0] = int( pVal[iI] )
            elif iI == 1:
                COLIN2[targID][0] = int( pVal[iI] )
            elif iI == 2:
                COLIN3[targID][0] = int( pVal[iI] )
            elif iI == 3:
                COLIN4[targID][0] = int( pVal[iI] )
            elif iI == 4:
                COLIN5[targID][0] = int( pVal[iI] )
            # end if
        # end for exits
    elif tParam == "COLIN1":
        COLIN1[targID][0] = int( pVal )
    elif tParam == "COLIN2":
        COLIN2[targID][0] = int( pVal )
    elif tParam == "COLIN3":
        COLIN3[targID][0] = int( pVal )
    elif tParam == "COLIN4":
        COLIN4[targID][0] = int( pVal )
    elif tParam == "COLIN5":
        COLIN5[targID][0] = int( pVal )
    elif tParam == "OUTDG":
        for iI in range( MAX_EXITS ):
            if iI == 0:
                OUTDG1[targID][0] = float( pVal[iI] )
            elif iI == 1:
                OUTDG2[targID][0] = float( pVal[iI] )
            elif iI == 2:
                OUTDG3[targID][0] = float( pVal[iI] )
            elif iI == 3:
                OUTDG4[targID][0] = float( pVal[iI] )
            elif iI == 4:
                OUTDG5[targID][0] = float( pVal[iI] )
            # end if
        # end for exits
    elif tParam == "OUTDG1":
        OUTDG1[targID][0] = float( pVal )
    elif tParam == "OUTDG2":
        OUTDG2[targID][0] = float( pVal )
    elif tParam == "OUTDG3":
        OUTDG3[targID][0] = float( pVal )
    elif tParam == "OUTDG4":
        OUTDG4[targID][0] = float( pVal )
    elif tParam == "OUTDG5":
        OUTDG5[targID][0] = float( pVal )
    # return
    return


def configFlagsParams( targID, cFlagVals, allIndexes, hdfType ):
    """Set and configure flags and parameters for RCHRES.

    The new HDF5 file format contains numerous differences in the
    way that various flags and states are represented relative to
    the original.

    Args:
        targID (str): the target location ID
        cFlagVals (dict): collected flag values
        allIndexes (list): list of indexes for cFlagVals
        hdfType (int): type of HDF5 file; 0 == original format;
            1 == new format

    """
    # imports
    # globals
    global PARAM_GOOD, PARAM_UNUSED, FLAG_GOOD, FLAG_UNUSED, INIT_PARMS
    global INIT_PARMS_UNUSED, nFLAG_GOOD, nINIT_PARMS
    # parameters
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    # first do the good flags; this is different depending on the
    #    HDF5 file format
    if hdfType == 0:
        for fV in FLAG_GOOD:
            if not fV in allIndexes:
                # this is an error
                errMsg = "Did not find flag %s in run setup!!!" % \
                            fV
                CL.LOGR.error( errMsg )
                return badReturn
            # now update our values
            setGoodFlag( targID, fV, cFlagVals[fV] )
            # no monthly flags are currently supported.
            #   if/when then these go here
        # end for good flag
    else:
        for fV in nFLAG_GOOD:
            if not fV in allIndexes:
                 # this is an error
                errMsg = "Did not find flag %s in run setup!!!" % \
                            fV
                CL.LOGR.error( errMsg )
                return badReturn
            # now update our values
            setGoodFlag( targID, fV, cFlagVals[fV] )
            # no monthly flags are currently supported.
            #   if/when then these go here
        # end for good flag
    # next give warnings for bad flags
    for fV in FLAG_UNUSED:
        if not fV in allIndexes:
            continue
        # if made it here write a warning
        if type( cFlagVals[fV] ) is tuple:
            cFVal = str( cFlagVals[fV] )
            warnMsg = "Flag %s is set to %s\nThis functionality " \
                        "is not currently implemented!!!" % \
                        ( fV, cFVal )
            CL.LOGR.warning( warnMsg )
        else:
            try:
                cFVal = int( cFlagVals[fV] )
            except:
                warnMsg = "Flag %s had a value of %s. Set to 0!!!" % \
                          (fV, str( cFlagVals[fV] ) )
                CL.LOGR.warning( warnMsg )
                cFVal = 0
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
        setGoodParam( targID, fV, cFlagVals[fV] )
    # end for good param
    # give warnings for bad parameters
    for fV in PARAM_UNUSED:
        if not fV in allIndexes:
            continue
        # if made it here write a warning
        if type( cFlagVals[fV] ) is tuple:
            cFVal = str( cFlagVals[fV] )
            warnMsg = "Param %s is set to %s\nThis functionality " \
                        "is not currently implemented!!!" % \
                        ( fV, cFVal )
            CL.LOGR.warning( warnMsg )
        else:
            cFVal = float( cFlagVals[fV] )
            if cFVal > 0:
                warnMsg = "Param %s is set to %g\nThis functionality " \
                            "is not currently implemented!!!" % \
                            ( fV, cFVal )
                CL.LOGR.warning( warnMsg )
    # end for bad flag
    # finally do the state variables, this is different depending on the
    #    HDF5 file format
    if hdfType == 0:
        for fV in INIT_PARMS:
            if not fV in allIndexes:
                # this is an error
                errMsg = "Did not find state variable %s in run setup!!!" % \
                            fV
                CL.LOGR.error( errMsg )
                return badReturn
            # now update our values
            setInitialParams( targID, fV, cFlagVals[fV] )
        # end for state
    else:
        for fV in nINIT_PARMS:
            if not fV in allIndexes:
                # this is an error
                errMsg = "Did not find state variable %s in run setup!!!" % \
                            fV
                CL.LOGR.error( errMsg )
                return badReturn
            # now update our values
            setInitialParams( targID, fV, cFlagVals[fV] )
        # end for state
    # end if
    # give warnings for unused initial values
    for fV in INIT_PARMS_UNUSED:
        if not fV in allIndexes:
            continue
        # if made it here write a warning
        if type( cFlagVals[fV] ) is tuple:
            cFVal = str( cFlagVals[fV] )
            warnMsg = "Init param %s is set to %s\nThis functionality " \
                        "is not currently implemented!!!" % \
                        ( fV, cFVal )
            CL.LOGR.warning( warnMsg )
        else:
            cFVal = float( cFlagVals[fV] )
            if cFVal > 0:
                warnMsg = "Init param %s is set to %g\nThis functionality " \
                            "is not currently implemented!!!" % \
                            ( fV, cFVal )
                CL.LOGR.warning( warnMsg )
        # end if type
    # end for bad flag
    # return
    return goodReturn


def setOutputControlFlags( targID, savetable, stTypes ):
    """Set the output control flags

    Args:
        targID (str): target id
        savetable (np.array or dict): which outputs to save, bools
        stTypes (list): keys or indexes to save

    Returns:
        int: function status; 0 == success

    """
    # imports
    # globals
    global GOOD_OUTPUT_LIST, BAD_OUTPUT_LIST, OUTPUT_CONTROL
    # parameters
    goodReturn = 0
    # locals
    # start
    for sType in stTypes:
        sVal = int( savetable[sType] )
        if sType in BAD_OUTPUT_LIST:
            if sVal > 0:
                # give a warning
                warnMsg = "Output type %s is not supported for RCHRES!!!" \
                        % sType
                CL.LOGR.warning( warnMsg )
            # now continue
            continue
        elif sType in GOOD_OUTPUT_LIST:
            stInd = GOOD_OUTPUT_LIST.index( sType )
            OUTPUT_CONTROL[targID][stInd] = sVal
        else:
            # issue a warning because is undefined type
            warnMsg = "Undefined output type of %s for RCHRES!!!" % \
                     sType
            CL.LOGR.warning( warnMsg )
            continue
        # end if
    # end for sType
    # return
    return goodReturn


def addInflowMap( targID, sVolType, sVolID, aFactor, massLink ):
    """Add an inflow mapping to the schematic map.

    This tells a RCHRES where to collect inflows from that
    are part of the internal routing.

    Args:
        targID (str): target ID
        sVolType (str): source type (PERLND, IMPLND, RCHRES)
        sVolID (str): source ID
        aFactor (float): area factor for value adjustment
        massLink (list): parsed MASS LINK definition list.

            0. (str): destination type
            1. (str): destination category
            2. (str): destination sub category
            3. (float): mfactor
            4. (str): source type
            5. (str): source category
            6. (str): source sub category
            7. (list): source exit

    Returns:
        int: function status; success == 0

    """
    # imports
    from locaMain import TARG_RCHRES
    # globals
    global SCHEMATIC_MAP, RR_TGRPN_SUPP, RR_TMEMN_SUPP
    # parameters
    goodReturn = 0
    badReturn = -1
    # locals
    # start
    # do some checks
    if massLink[0] != TARG_RCHRES:
        errMsg = "Link target type is %s and only type %s is supported!!!" \
                 % ( massLink[0], TARG_RCHRES )
        CL.LOGR.error( errMsg )
        return badReturn
    if massLink[1] != RR_TGRPN_SUPP:
        errMsg = "Link target category is %s and only category %s is " \
                 "supported!!!" % ( massLink[1], RR_TGRPN_SUPP )
        CL.LOGR.error( errMsg )
        return badReturn
    if massLink[2] != RR_TMEMN_SUPP:
        errMsg = "Link target sub category is %s and only sub %s is " \
                 "supported!!!" % ( massLink[2], RR_TMEMN_SUPP )
        CL.LOGR.error( errMsg )
        return badReturn
    # check to make sure that have defined source outflow type
    if massLink[5] == "ROFLOW":
        sOutType = "OVOL"
    else:
        sOutType = massLink[6]
    # end if
    # make our link list
    schemeLL = [ sVolType, sVolID, sOutType, massLink[7], aFactor, massLink[3] ]
    if targID in SCHEMATIC_MAP.keys():
        # then append
        SCHEMATIC_MAP[targID].append( schemeLL )
    else:
        SCHEMATIC_MAP[targID] = [ schemeLL ]
    # end if
    # return
    return goodReturn


def makeRowFT( vol, volumeFT, depthFT, sareaFT, dischList ):
    """Make a row array representing the interpolated values
    from the FTAB for this volume

    Args:
        vol (float) : current volume in rchres
        volumeFT (np.array) : volume vector from FTABLE
        depthFT (np.array) : depth vector from FTABLE
        sareaFT (np.array) : surface area vector from FTABLE
        dischList (list): list of np.array that have discharge
                          values

    Returns:
        np.array: interpolated FTAB row

    """
    # imports
    # globals
    # parameters
    smallVal = float( 1E-10 )
    # locals - list for future Cython
    nExits = len( dischList )
    rowFT = np.zeros( 3 + nExits, dtype=np.float64 )
    # start
    if ( ( vol - 0.0 ) > smallVal ):
        # then need to calculate
        rowFT[0] = np.interp( vol, volumeFT, depthFT )
        rowFT[1] = np.interp( vol, volumeFT, sareaFT )
        rowFT[2] = vol
        for jJ in range(3, ( 3 + nExits ), 1):
            rowFT[jJ] = np.interp( vol, volumeFT, dischList[jJ - 3] )
        # end for
    # end main if
    # return
    return rowFT


def makeRowFTbyIndx( indx, volumeFT, depthFT, sareaFT, dischList ):
    """Make a row array representing the extracted values
    from the FTAB for this index

    Args:
        indx (int) : index or row for FTAB
        volumeFT (np.array) : volume vector from FTABLE
        depthFT (np.array) : depth vector from FTABLE
        sareaFT (np.array) : surface area vector from FTABLE
        dischList (list): list of np.array that have discharge
                          values

    Returns:
        np.array: extracted FTAB row

    """
    # imports
    # globals
    # parameters
    # locals - list for future Cython
    nExits = len( dischList )
    rowFT = np.zeros( 3 + nExits, dtype=np.float64 )
    # start
    rowFT[0] = depthFT[indx]
    rowFT[1] = sareaFT[indx]
    rowFT[2] = volumeFT[indx]
    for jJ in range(3, ( 3 + nExits ), 1):
        rowFT[jJ] = dischList[jJ - 3][indx]
    # end for
    # return
    return rowFT


def createTSIVOL( targID, iI ):
    """Set the IVOL input to a RCHRES target by simulation time step
    This needs to account for external time series IVOL entries and
    the internal routing between structures.

    Args:
        targID (str): target RCHRES ID
        iI (int): current time step

    Returns:
        float: input ivol in ft3/day

    """
    # imports
    from locaMain import TARG_RCHRES, TARG_PERVLND, TARG_IMPLND
    from locaHyperwat import getPERObyTargTS
    from locaHimpwat import getSURObyTargTS
    # globals
    global SCHEMATIC_MAP, EXIVOL, VFACT
    global OVOL1, OVOL2, OVOL3, OVOL4, OVOL5
    # parameters
    # locals - declare here in case Cythonize
    ivol = float( 0.0 )     # return inflow vol
    ovol = float ( 0.0 )    # rchres outflow
    # start
    # external is in acre-ft/day convert to ft3/day
    ivol = float( EXIVOL[targID][iI] ) * VFACT
    # now go through schematic loop
    if not targID in SCHEMATIC_MAP.keys():
        return ivol
    # now process
    usLinks = SCHEMATIC_MAP[ targID ]
    for uL in usLinks:
        sType = uL[0]
        sID = uL[1]
        sOut = uL[2]
        oEList = uL[3]
        afact = uL[4]
        mfact = uL[5]
        if sType == TARG_PERVLND:
            if sOut == "PERO":
                peroIn = getPERObyTargTS( iI, sID )
                peroVol = peroIn * afact * mfact * VFACT
                ivol = ivol + peroVol
            else:
                errMsg = "Unsupported %s outflow type of %s. Only PERO is " \
                         "currently supported !!!" % ( TARG_PERVLND, sOut )
                CL.LOGR.error( errMsg )
            # end inner if
        elif sType == TARG_IMPLND:
            if sOut == "SURO":
                suroIn = getSURObyTargTS( iI, sID )
                suroVol = suroIn * afact * mfact * VFACT
                ivol = ivol + suroVol
            else:
                errMsg = "Unsupported %s outflow type of %s. Only SURO is " \
                         "currently supported !!!" % ( TARG_IMPLND, sOut )
                CL.LOGR.error( errMsg )
            # end inner if
        elif sType == TARG_RCHRES:
            if sOut == "OVOL":
                if oEList[0] == 1:
                    ovol = float( OVOL1[sID][iI] )
                elif oEList[0] == 2:
                    ovol = float( OVOL2[sID][iI] )
                elif oEList[0] == 3:
                    ovol = float( OVOL3[sID][iI] )
                elif oEList[0] == 4:
                    ovol = float( OVOL4[sID][iI] )
                elif oEList[0] == 5:
                    ovol = float( OVOL5[sID][iI] )
                else:
                    errMsg = "Unsupported exit number of %d for %s in " \
                             "link to %s. Inflow ignored!!!" % \
                             ( oEList[0], sID, targID )
                    CL.LOGR.error( errMsg )
                    ovol = float(0.0)
                # end inner if
                ivol = ivol + ( ovol * afact * mfact * VFACT )
            else:
                errMsg = "Unsupported %s outflow type of %s. Only OVOL is " \
                         "currently supported !!!" % ( TARG_RCHRES, sOut )
                CL.LOGR.error( errMsg )
            # end if
        else:
            errMsg = "Unsupported target type of %s found for %s!!!" % \
                     ( sType, targID )
            CL.LOGR.error( errMsg )
        # end if type
    # end for link
    # return
    return ivol


def getODGTFG( targID, nexits ):
    """Get the ODGTF flags by exit for a particular target

    Args:
        targID (str): target RCHRES ID
        nexits (int): number of exits

    Returns:
        np.array: flag values by exit

    """
    # imports
    # globals
    global MAX_EXITS, ODGTFG1, ODGTFG2, ODGTFG3, ODGTFG4, ODGTFG5
    # parameter
    # locals
    odgtf = np.zeros( MAX_EXITS, dtype=np.int32 )
    retodgtf = np.zeros( nexits, dtype=np.int32 )
    # start
    odgtf[0] = int( ODGTFG1[targID][0] )
    odgtf[1] = int( ODGTFG2[targID][0] )
    odgtf[2] = int( ODGTFG3[targID][0] )
    odgtf[3] = int( ODGTFG4[targID][0] )
    odgtf[4] = int( ODGTFG5[targID][0] )
    retodgtf[:] = odgtf[:nexits]
    # return
    return retodgtf


def getODFVFG( targID, nexits ):
    """Get the ODFVF flags by exit for a particular target

    Args:
        targID (str): target RCHRES ID
        nexits (int): number of exits

    Returns:
        np.array: flag values by exit

    """
    # imports
    # globals
    global MAX_EXITS, ODFVFG1, ODFVFG2, ODFVFG3, ODFVFG4, ODFVFG5
    # parameter
    # locals
    odfvf = np.zeros( MAX_EXITS, dtype=np.int32 )
    retodfvf = np.zeros( nexits, dtype=np.int32 )
    # start
    odfvf[0] = int( ODFVFG1[targID][0] )
    odfvf[1] = int( ODFVFG2[targID][0] )
    odfvf[2] = int( ODFVFG3[targID][0] )
    odfvf[3] = int( ODFVFG4[targID][0] )
    odfvf[4] = int( ODFVFG5[targID][0] )
    retodfvf[:] = odfvf[:nexits]
    # return
    return retodfvf


def getFUNCT( targID, nexits ):
    """Get the FUNCT flags by exit for a particular target

    Args:
        targID (str): target RCHRES ID
        nexits (int): number of exits

    Returns:
        np.array: flag values by exit

    """
    # imports
    # globals
    global MAX_EXITS, FUNCT1, FUNCT2, FUNCT3, FUNCT4, FUNCT5
    # parameter
    # locals
    funct = np.zeros( MAX_EXITS, dtype=np.int32 )
    retfunct = np.zeros( nexits, dtype=np.int32 )
    # start
    funct[0] = int( FUNCT1[targID][0] )
    funct[1] = int( FUNCT2[targID][0] )
    funct[2] = int( FUNCT3[targID][0] )
    funct[3] = int( FUNCT4[targID][0] )
    funct[4] = int( FUNCT5[targID][0] )
    retfunct[:] = funct[:nexits]
    # return
    return retfunct


def getCOLIND( targID, nexits, iI, odfvf ):
    """Get the COLIND array for active exits and this time
    interval

    Args:
        targID (str): target RCHRES ID
        nexits (int): number of exits
        iI (int): current time step
        odfvf (np.array): array of ODVFG flags by exit

    Returns:
        float: input ivol in ft3/day

    """
    # imports
    # globals
    global MAX_EXITS, COLIN1, COLIN2, COLIN3, COLIN4, COLIN5
    global COLIND1, COLIND2, COLIND3, COLIND4, COLIND5
    # parameters
    # locals
    colind = np.zeros( MAX_EXITS, dtype=np.float64 )
    retcolind = np.zeros( nexits, dtype=np.float64 )
    # start
    # for this need to just go through the odfvf flag values
    #  and fill accordingly.
    for jJ in range( nexits ):
        cOVFlag = odfvf[jJ]
        if cOVFlag == 0:
            continue
        # end if
        if cOVFlag < 0:
            if jJ == 0:
                colind[jJ] = float(  COLIND1[targID][iI] )
            elif jJ == 1:
                colind[jJ] = float(  COLIND2[targID][iI] )
            elif jJ == 2:
                colind[jJ] = float(  COLIND3[targID][iI] )
            elif jJ == 3:
                colind[jJ] = float(  COLIND4[targID][iI] )
            elif jJ == 4:
                colind[jJ] = float( COLIND5[targID][iI] )
            # end if
        else:
            colind[jJ] = cOVFlag
        # end if
    # end for
    retcolind[:] = colind[:nexits]
    # return
    return retcolind


def getOUTDGT( targID, nexits, iI, odgtf ):
    """Get the OUTDGT array for active exits and this time
    interval

    Args:
        targID (str): target RCHRES ID
        nexits (int): number of exits
        iI (int): current time step
        odgtf (np.array): array of ODGTF flags by exit

    Returns:
        float: input ivol in ft3/day

    """
    # imports
    # globals
    global MAX_EXITS, OUTDG1, OUTDG2, OUTDG3, OUTDG4, OUTDG5
    global OUTDGT1, OUTDGT2, OUTDGT3, OUTDGT4, OUTDGT5
    # parameters
    # locals
    outdgt = np.zeros( MAX_EXITS, dtype=np.float64 )
    retoutdgt = np.zeros( nexits, dtype=np.float64 )
    # start
    # first check if need to do anything
    totalOVF = odgtf.sum()
    if totalOVF <= 0.0:
        return retoutdgt
    # end if
    outdgt[0] = float( OUTDGT1[targID][iI] )
    outdgt[1] = float( OUTDGT2[targID][iI] )
    outdgt[2] = float( OUTDGT3[targID][iI] )
    outdgt[3] = float( OUTDGT4[targID][iI] )
    outdgt[4] = float( OUTDGT5[targID][iI] )
    # assign for return
    retoutdgt[:] = outdgt[:nexits]
    # return
    return retoutdgt


def getOSEffHO( targID, nexits ):
    """ Get the oseff hold over array

    Args:
        targID (str): target RCHRES ID
        nexits (int): number of exits

    Returns:
        np.array: hold overs by exit

    """
    # imports
    # globals
    global MAX_EXITS, HOLD_OS1, HOLD_OS2, HOLD_OS3, HOLD_OS4, HOLD_OS5
    # parameter
    # locals
    oseff = np.zeros( MAX_EXITS, dtype=np.float64 )
    retoseff = np.zeros( nexits, dtype=np.float64 )
    # start
    oseff[0] = float( HOLD_OS1[targID][0] )
    oseff[1] = float( HOLD_OS2[targID][0] )
    oseff[2] = float( HOLD_OS3[targID][0] )
    oseff[3] = float( HOLD_OS4[targID][0] )
    oseff[4] = float( HOLD_OS5[targID][0] )
    retoseff[:] = oseff[:nexits]
    # return
    return retoseff


def setOSEffHO( o, targID, nexits ):
    """ Set the oseff hold over array

    Args:
        o (np.array): calculated outflow array
        targID (str): target RCHRES ID
        nexits (int): number of exits

    """
    # imports
    # globals
    global MAX_EXITS, HOLD_OS1, HOLD_OS2, HOLD_OS3, HOLD_OS4, HOLD_OS5
    # parameter
    # locals
    # start
    for jJ in range( nexits ):
        if jJ == 0:
            HOLD_OS1[targID][0] = o[jJ]
        elif jJ == 1:
            HOLD_OS2[targID][0] = o[jJ]
        elif jJ == 2:
            HOLD_OS3[targID][0] = o[jJ]
        elif jJ == 3:
            HOLD_OS4[targID][0] = o[jJ]
        elif jJ == 4:
            HOLD_OS5[targID][0] = o[jJ]
        # end if
    # end for
    # return
    return


def hydr_liftedloop( iI, targID, fTabDict ):
    """Modified version of liftedloop to do a single time step and
    return to the main time loop.

    Module-wide recarrays are used to store all results and calculation
    variables between calls. Modified real number comparisons to be more
    numerically reliable.

    Args:
        iI (int): index of current time step (0 to (sim_len-1))
        targID (str): ID for recarray columns
        fTabDict (dict): dictionary of FTABLE recarrays

    Returns:
        int: count of the number of errors. Should generally be 0 but
                used this to reference errorsV for error handling

    """
    # imports
    from math import sqrt, log10, pow
    # globals
    # carry overs
    # calc constants
    global errorsV, SFACTA, LFACTA, AFACTA, VFACTA, TFACTA, ERRMSG
    global DELTS, VFACT, AFACT, GAM, GRAV, AKAPPA, MAX_EXITS, NEXITS
    # switches, parameters, and flags
    global AUX1FG, AUX2FG, AUX3FG, ORG_SSA_CALC
    global LKFG, FTABNO, LEN, DELTH, STCOR, KS, DB50
    # data time series
    # initial states
    global I_VOL
    # storage ts for states
    global VOL
    # save time series only
    global DEP, IVOL, O1, O2, O3, O4, O5, OVOL1, OVOL2, OVOL3
    global OVOL4, OVOL5, POTEV, PREC, PRSUPY, RO, ROVOL, SAREA
    global STAGE, TAU, USTAR, VOLEV, AVVEL, AVDEP
    global HRAD, TWID
    # carryovers
    global HOLD_RO, HOLD_OS1, HOLD_OS2, HOLD_OS3, HOLD_OS4, HOLD_OS5
    # parameters
    smallVal = float( 1E-10 )   # numeric compare threshold
    smallVol = float( 1E-5 )    # volume threshold
    ConvInTFt = float( 12.0 )
    nexits = int( NEXITS[targID][0] )
    ftabno = int( FTABNO[targID][0] )
    # locals - put here in case want to Cython
    coks = float( 0.0 )     # ks complement
    facta1 = float( 0.0 )   # calculation factor
    prsupy = float( 0.0 )   # precipitaton volumetric rate on surface
    dep = float( 0.0 )      # calculated depth in feet
    sarea = float( 0.0 )    # calculated surface area in ft2
    avvel = float( 0.0 )    # average velocity in ft/s
    o = np.zeros( nexits, dtype=np.float64 )  # calculated outflow by exit
    od1 = np.zeros( nexits, dtype=np.float64 ) # outflow demand lower
    od2 = np.zeros( nexits, dtype=np.float64 ) # outflow demand upper
    odz = np.zeros( nexits, dtype=np.float64 ) # outflow demand calc var
    ovol = np.zeros( nexits, dtype=np.float64 ) # outflow volume by exit
    oseff = np.zeros( nexits, dtype=np.float64 ) # oseff
    roseff = float( 0.0 )   # calc variable
    indx = int( 0 )         # FTAB row
    ro = float( 0.0 )       # total discharge
    a1 = float( 0.0 )       # calc variable
    rod1 = float( 0.0 )     # calc variable
    rod2 = float( 0.0 )     # calc variable
    v1 = float( 0.0 )       # calc variable
    v2 = float( 0.0 )       # calc variable
    volt = float( 0.0 )     # total volume
    volev = float( 0.0 )    # evaporation from surface
    volpev = float( 0.0 )   # potential et
    volint = float( 0.0 )   # interpolated volume
    rovol = float( 0.0 )    # calculated total outflow
    oint = float( 0.0 )     # outflow initial
    rodz = float( 0.0 )     # calculation variable
    premov = int( -20 )     # calc var
    move = int( 10 )        # calc var
    vv1 = float( 0.0 )      # calc var
    vv2 = float( 0.0 )      # calc var
    facta2 = float( 0.0 )   # calc var
    factb2 = float( 0.0 )   # calc var
    factc2 = float( 0.0 )   # calc var
    det = float( 0.0 )      # calc var
    factr = float( 0.0 )    # calc var
    diff = float( 0.0 )     # calc var
    tro = float( 0.0 )      # calc var
    aa1 = float( 0.0 )      # calc var
    length = float( 0.0 )   # length in ft
    twid = float( 0.0 )     # calc var
    avdep = float( 0.0 )    # calc var
    ustar = float( 0.0 )    # calculated shear velocity
    tau = float( 0.0 )      # calculated bed shear stress
    hrad = float( 0.0 )     # calculated hydraulic radius
    slope = float( 0.0 )    # calculated slope
    errorCnt = int( 0 )     # error counter
    outRO = float( 0.0 )    # output RO
    outO = np.zeros( nexits, dtype=np.float64 ) # output O
    outOVol = np.zeros( nexits, dtype=np.float64 ) # output ovol
    outROVol = float( 0.0 )  # output rovol
    # VCONF is not currently supported so always 1.0 so the factor
    # has no impact
    convf = float( 1.0 )
    # irrigation not supported
    irexit = int( -1 )
    irrdem = float( 0.0 )
    rirwdl = float( 0.0 )   # calc var
    irminv = float( 0.0 )   # calc var
    # flags
    fl_aux1fg = int( AUX1FG[targID][0] )
    fl_aux2fg = int( AUX2FG[targID][0] )
    fl_aux3fg = int( AUX3FG[targID][0] )
    fl_lkfg = int( LKFG[targID][0] )
    # get our exit flags
    funct = getFUNCT( targID, nexits )
    odfvf = getODFVFG( targID, nexits )
    odgtf = getODGTFG( targID, nexits )
    nodfv  = np.bool( np.any( odfvf ) )
    # HSPF parameters
    ks = float( KS[targID][0] )
    stcor = float( STCOR[targID][0] )
    delth = float( DELTH[targID][0] )
    rlength = float( LEN[targID][0] )
    db50 = float( DB50[targID][0] ) / ConvInTFt
    # get our time series values
    ts_prec = float( PREC[targID][iI] ) / ConvInTFt
    ts_pet = float( POTEV[targID][iI] ) / ConvInTFt
    ts_inflow = createTSIVOL( targID, iI )
    colind = getCOLIND( targID, nexits, iI, odfvf )
    outdgt = getOUTDGT( targID, nexits, iI, odgtf )
    # get our ftable arrays
    cRArray = fTabDict[ftabno]
    numRows = len( cRArray )
    depthFT = cRArray['Depth'].view(dtype=np.float64)
    volume = cRArray['Volume'].view(dtype=np.float64)
    surfarea = cRArray['Area'].view(dtype=np.float64)
    volumeFT = volume * VFACT
    sareaFT = surfarea * AFACT
    # get out the discharge array(s)
    dischList = list()
    ftabcols = list( cRArray.dtype.names )
    nftCols = len( ftabcols )
    ndisCols = nftCols - 3
    for jJ in range( 1, ndisCols + 1, 1 ):
        cKey = "Disch%d" % jJ
        dischList.append( cRArray[cKey].view(dtype=np.float64) )
    # end for
    # initial parameter calcs
    coks = 1.0 - ks
    facta1 = 1.0 / ( coks * DELTS )
    # get the top volume
    topvolume = volumeFT[ numRows - 1 ]
    # set up calculation values for state
    if iI == 0:
        vol = float( I_VOL[targID][0] ) * VFACT
        if vol >= topvolume:
            errorsV[1] += 1
            errorCnt += 1
        # need to do a series of calculations here to get values
        indx = fndrow( vol, volumeFT )
        if nodfv:
            # need to interpolate from our initial volume
            v1 = volumeFT[indx]
            v2 = volumeFT[indx+1]
            rowsFT1 = makeRowFTbyIndx( indx, volumeFT, depthFT, sareaFT,
                                       dischList )
            rowsFT2 = makeRowFTbyIndx( indx+1, volumeFT, depthFT, sareaFT,
                                       dischList )
            rod1, od1[:] = demand( v1, rowsFT1, funct, nexits, DELTS,
                                   convf, colind, outdgt, odgtf )
            rod2, od2[:] = demand( v2, rowsFT2, funct, nexits, DELTS,
                                   convf, colind, outdgt, odgtf )
            a1 = ( v2 - vol ) / ( v2 - v1 )
            o[:] = ( a1 * od1 )  + ( (1.0 - a1) * od2)
            ro = ( a1 * rod1 ) + ( (1.0 - a1) * rod2)
        else:
            # no outflow demands have an f(vol) component
            rowsFT1 = makeRowFTbyIndx( indx, volumeFT, depthFT, sareaFT,
                                       dischList )
            ro, o[:] = demand( vol, rowsFT1, funct, nexits, DELTS, convf,
                               colind, outdgt, odgtf )
        # end if
        # initial depth and surface area
        if ORG_SSA_CALC:
            dep, sarea = auxil( volumeFT, depthFT, sareaFT, indx,
                                vol, fl_aux1fg )
        else:
            dep, sarea = npAuxil( volumeFT, depthFT, sareaFT, vol )
        # end if
        # initial setup
        roseff = ro
        oseff[:] = o
    else:
        vol = float( VOL[targID][iI-1] ) * VFACT
        indx = fndrow( vol, volumeFT )
        # carry overs
        roseff = float( HOLD_RO[targID][0] )
        oseff = getOSEffHO( targID, nexits )
        # initial depth and surface area
        if ORG_SSA_CALC:
            dep, sarea = auxil( volumeFT, depthFT, sareaFT, indx,
                                vol, fl_aux1fg )
        else:
            dep, sarea = npAuxil( volumeFT, depthFT, sareaFT, vol )
        # end if ORG_SSA_CALC
    # end if
    # start
    # irrigation exit and other considerations not supported
    #   irexit is set so this will not be called but the code block is
    #   maintained for future expansion
    #   check if irrigation exit is set
    if irexit >= 0:
        # rirwdl equivalent to OVOL for the irrigation exit
        #if rirwdl > 0.0:
        if ( ( rirwdl - 0.0 ) > smallVal ):
            if ( ( irminv - ( vol - rirwdl ) ) > smallVal ):
                vol = irminv
            else:
                vol = vol - rirwdl
            # end inner if
            # if vol >= volumeFT[-1]:
            if ( ( vol - topvolume ) > smallVal ):
                # ERRMSG1: extrapolation of rchtab will take place
                errorsV[1] += 1
                errorCnt += 1
                errMsg = " hydr_liftedloop - %s" % ERRMSG[1]
                CL.LOGR.error( errMsg )
            # end error check if
            # DISCH with hydrologic routing
            # find row index that brackets the VOL per comment in lines
            # already done
            # indx = fndrow(vol, volumeFT)
            vv1 = volumeFT[indx]
            vv2 = volumeFT[indx+1]
            rowsFT1 = makeRowFTbyIndx( indx, volumeFT, depthFT, sareaFT,
                                        dischList )
            rowsFT2 = makeRowFTbyIndx( indx+1, volumeFT, depthFT, sareaFT,
                                        dischList )
            rod1, od1[:] = demand( vv1, rowsFT1, funct, nexits, DELTS,
                                    convf, colind, outdgt, odgtf )
            rod2, od2[:] = demand( vv2, rowsFT2, funct, nexits, DELTS,
                                    convf, colind, outdgt, odgtf )
            aa1 = (vv2 - vol) / (vv2 - vv1)
            ro = ( aa1 * rod1 ) + ( ( 1.0 - aa1 ) * rod2 )
            o[:] = ( aa1 * od1 )  + ( ( 1.0 - aa1 ) * od2 )
            # back to HYDR
            # recompute surface area and depth
            if ORG_SSA_CALC:
                indx = fndrow( vol, volumeFT )
                dep, sarea = auxil( volumeFT, depthFT, sareaFT, indx,
                                    vol, fl_aux1fg )
            else:
                dep, sarea = npAuxil( volumeFT, depthFT, sareaFT, vol )
            # end if
            # end aux1fg if
        else:
            irrdem = 0.0
        # end if rirwdl
        #o[irexit] = 0.0
    # end if irrigation
    prsupy = ts_prec * sarea
    volt = vol + ts_inflow + prsupy
    # if volt < 1E-10:
    if ( ( volt - 0.0 ) < smallVal ):
        volt = float( 0.0 )
    # end check if
    # can only do evaporation if calculate surface area
    if fl_aux1fg > 0:
        volpev = ts_pet * sarea
        # NDM LOCA debug 03/11/2020 - correct negative outflows
        # if volpev >= volt:
        if ( ( volpev - volt ) > ( -1.0 * smallVal ) ):
            volev = volt
            volt = float( 0.0 )
        else:
            volev = volpev
            volt -= volev
        # end inner if
    # end of evap calc if
    # ROUTE/NOROUT calls
    # common code
    # find intercept of eq 4 on vol axis
    volint = volt - ( ks * roseff * DELTS )
    # if volint < (volt * 1.0e-5):
    if ( ( ( volt * smallVol ) - volint ) >= smallVal ):
        volint = float( 0.0 )
    # end if
    # if volint <= 0.0:
    if ( ( volint - 0.0 ) < smallVal ):
        #  case 3 -- no solution to simultaneous equations
        indx = 0
        vol = 0.0
        ro = 0.0
        o[:] = 0.0
        rovol = volt
        # if roseff > 0.0
        if ( ( roseff - 0.0 ) > smallVal ):
            ovol[:] = ( rovol / roseff)  * oseff
        else:
            ovol[:] = rovol / float( nexits )
        # end inner if
    else:
        # case 1 or 2
        oint = volint * facta1
        if nodfv:
            # ROUTE
            rowsFT0 = makeRowFTbyIndx( 0, volumeFT, depthFT, sareaFT,
                                       dischList )
            rodz, odz[:] = demand( 0.0, rowsFT0, funct, nexits, DELTS,
                                   convf, colind, outdgt, odgtf )
            # if oint > rodz:
            if ( ( oint - rodz ) > smallVal ):
                # SOLVE,  Solve the simultaneous equations for case 1
                # outflow demands can be met in full
                # premov will be used to check whether we are in a
                # trap, arbitrary value
                vv1 = volumeFT[ indx ]
                vv2 = volumeFT[ indx+1 ]
                rowsFT1 = makeRowFTbyIndx( indx, volumeFT, depthFT, sareaFT,
                                           dischList )
                rowsFT2 = makeRowFTbyIndx( indx+1, volumeFT, depthFT, sareaFT,
                                           dischList )
                rod1, od1[:] = demand( vv1, rowsFT1, funct, nexits, DELTS,
                                       convf, colind, outdgt, odgtf )
                rod2, od2[:] = demand( vv2, rowsFT2, funct, nexits, DELTS,
                                       convf, colind, outdgt, odgtf )
                # iterative solve
                while move != 0:
                    facta2 = rod1 - rod2
                    factb2 = vv2 - vv1
                    factc2 = ( vv2 * rod1 ) - ( vv1 * rod2 )
                    det = ( facta1 * factb2 ) - facta2
                    # if det == 0.0:
                    if ( abs( det - 0.0 ) <= smallVal ):
                        det = 0.0001
                        # ERRMSG0: SOLVE is indeterminate
                        errorsV[0] += 1
                        errorCnt += 1
                        errMsg = " hydr_liftedloop - %s" % ERRMSG[0]
                        CL.LOGR.error( errMsg )
                    # end if check
                    vol = max( 0.0,  ( ( ( oint * factb2 ) - factc2 ) / det ) )
                    # if vol > vv2:
                    if ( ( vol - vv2 ) >= smallVal ):
                        # if volume greater than plus 1 index
                        if ( indx >= ( numRows - 2 ) ):
                            # if vol > topvolume:
                            if ( ( vol - topvolume ) > smallVal ):
                                # ERRMSG1: extrapolation of rchtab will take place
                                errorsV[1] += 1
                                errorCnt += 1
                                errMsg = " hydr_liftedloop - %s" % ERRMSG[1]
                                CL.LOGR.error( errMsg )
                            # end error if
                            move = 0
                        else:
                            move = 1
                            indx += 1
                            vv1 = vv2
                            od1[:] = od2
                            rod1 = rod2
                            vv2 = volumeFT[ indx+1 ]
                            rowsFT1 = rowsFT2.copy()
                            rowsFT2 = makeRowFTbyIndx( indx+1, volumeFT,
                                                       depthFT, sareaFT,
                                                       dischList )
                            rod2, od2[:] = demand( vv2, rowsFT2, funct,
                                                   nexits, DELTS,
                                                   convf, colind, outdgt,
                                                   odgtf )
                        # end if index check
                        # elif vol < vv1:
                    elif ( ( vv1 - vol ) >= smallVal ):
                        # if volume is less than the indx volume
                        indx -= 1
                        move = -1
                        vv2 = vv1
                        od2[:] = od1
                        rod2 = rod1
                        vv1 = volumeFT[indx]
                        rowsFT2 = rowsFT1.copy()
                        rowsFT1 = makeRowFTbyIndx( indx, volumeFT,
                                                   depthFT, sareaFT,
                                                   dischList )
                        rod1, od1[:] = demand( vv1, rowsFT1, funct,
                                               nexits, DELTS, convf,
                                               colind, outdgt, odgtf )
                    else:
                        # volume in between index and index plus 1
                        move = 0
                    # check whether algorithm is in a trap, yo-yoing back and forth
                    if move + premov == 0:
                        # ERRMSG2: oscillating trap
                        errorsV[2] += 1
                        errorCnt += 1
                        move = 0
                        errMsg = " hydr_liftedloop - %s" % ERRMSG[2]
                        CL.LOGR.error( errMsg )
                    # end trap check if
                    premov = move
                # end while
                ro = oint - ( facta1 * vol )
                # now do some checks
                # if  vol < 1.0e-5:
                if ( ( vol - 0.0 ) < smallVol ):
                    ro = oint
                    vol = 0.0
                # end vol == 0 check
                # if ro < 1.0e-10:
                if ( abs( ro - 0.0 ) <= smallVal ):
                    # case of ro == 0
                    ro = 0.0
                    # add this because if total ro is zero then
                    # all o should be zero also
                    o[:] = 0.0
                elif ( ( ro - 0.0 ) < ( -1.0 * smallVal ) ):
                    # if ro <= 0.0:
                    # this is case of negative ro
                    ro = 0.0
                    o[:] = 0.0
                else:
                    diff = vol - vv1
                    if ( ( diff - 0.0 ) < 0.01 ):
                        factr = 0.0
                    else:
                        factr = diff / ( vv2 - vv1 )
                    # end inner if
                    o[:]  = od1 + ( od2 - od1 ) * factr
                # end if ro check
            else:
                # case 2 -- outflow demands cannot be met in full
                ro  = 0.0
                for i in range( nexits ):
                    tro  = ro + odz[i]
                    # if tro <= oint:
                    if ( ( tro - oint ) < smallVal ):
                        o[i] = odz[i]
                        ro = tro
                    else:
                        o[i] = oint - ro
                        ro = oint
                    # end if
                # end for
                vol = 0.0
                indx = 0
        else:
            # NOROUT
            rowsFT1 = makeRowFTbyIndx( indx, volumeFT, depthFT, sareaFT,
                                       dischList )
            rod1, od1[:] = demand( vol, rowsFT1, funct, nexits, DELTS,
                                   convf, colind, outdgt, odgtf )
            # if oint >= rod1:
            if ( ( oint - rod1 ) > ( -1.0 * smallVal ) ):
                #case 1 -outflow demands are met in full
                ro = rod1
                vol  = volint - ( coks * ro * DELTS )
                # if vol < 1.0e-5:
                if ( ( vol - 0.0 ) < smallVol ):
                    vol = 0.0
                # end if vol check
                o[:] = od1
            else:
                # case 2 -outflow demands cannot be met in full
                ro  = 0.0
                for i in range( nexits ):
                    tro = ro + odz[i]
                    # if tro <= oint:
                    if ( ( tro - oint ) < smallVal ):
                        o[i] = odz[i]
                        ro = tro
                    else:
                        o[i] = oint - ro
                        ro = oint
                    # end inner if
                # end for
                vol = 0.0
                indx = 0
            # end case if
        # end route / no route if
        # common  ROUTE/NOROUT code
        # this is left in for future expansion but is not supported
        # and so will not be called without code modifications
        #  an irrigation demand was made before routing
        if  (irexit >= 0) and (irrdem > 0.0):
            oseff[irexit] = irrdem
            o[irexit] = irrdem
            roseff += irrdem
            ro += irrdem
            # irrdemV[loop] = irrdem
        # end unsupported if
        # estimate the volumes of outflow
        ovol[:] = ( ( ks * oseff ) + ( coks * o ) ) * DELTS
        rovol = ( ( ks * roseff ) + ( coks * ro ) ) * DELTS
    # end if volint
    # HYDR
    if fl_aux1fg:
        # compute final depth, surface area
        #if vol >= topvolume:
        if ( ( vol - topvolume ) > smallVal ):
            # ERRMSG1: extrapolation of rchtab
            errorsV[1] += 1
            errorCnt += 1
            errMsg = " hydr_liftedloop - %s" % ERRMSG[1]
            CL.LOGR.error( errMsg )
        # end if error
        # find the index again
        indx = fndrow( vol, volumeFT )
        if ORG_SSA_CALC:
            dep, sarea = auxil( volumeFT, depthFT, sareaFT, indx,
                                vol, fl_aux1fg )
        else:
            dep, sarea = npAuxil( volumeFT, depthFT, sareaFT, vol )
        # end if
        # now assign these
        DEP[targID][iI] = dep
        SAREA[targID][iI] = sarea * AFACTA
    # do our other assigments
    PRSUPY[targID][iI] = prsupy * AFACTA
    # NDM Debug 03/11/2020 - get rid of negative outflows
    outRO = ro * SFACTA * LFACTA
    outO = o * SFACTA * LFACTA
    outOVol = ovol * VFACTA
    outROVol = rovol * VFACTA
    if ( ( outRO - 0.0 ) > smallVal ):
        RO[targID][iI] = outRO
        HOLD_RO[targID][0] = ro
        setOSEffHO( o, targID, nexits )
        ROVOL[targID][iI] = outROVol
    else:
        RO[targID][iI] = 0.0
        HOLD_RO[targID][0] = 0.0
        setOSEffHO( np.zeros( nexits, dtype=np.float64 ), targID, nexits )
        ROVOL[targID][iI] = 0.0
        outO[:] = 0.0
        outOVol[:] = 0.0
    # end if
    #    also assign by exit
    for jJ in range( nexits ):
        if jJ == 0:
            O1[targID][iI] = outO[jJ]
            OVOL1[targID][iI] = outOVol[jJ]
        elif jJ == 1:
            O2[targID][iI] = outO[jJ]
            OVOL2[targID][iI] = outOVol[jJ]
        elif jJ == 2:
            O3[targID][iI] = outO[jJ]
            OVOL3[targID][iI] = outOVol[jJ]
        elif jJ == 3:
            O4[targID][iI] = outO[jJ]
            OVOL4[targID][iI] = outOVol[jJ]
        elif jJ == 4:
            O5[targID][iI] = outO[jJ]
            OVOL5[targID][iI] = outOVol[jJ]
        # end if
    # end for
    VOLEV[targID][iI] = volev * VFACTA
    VOL[targID][iI] = vol * VFACTA
    IVOL[targID][iI] = ts_inflow * VFACTA
    # calculate optionals
    length = rlength * 5280.0  # length of reach converted to feet, hydr-parm2
    # if vol > 0.0 and sarea > 0.0:
    if fl_aux1fg > 0:
        if ( ( ( vol - 0.0 ) >= smallVol ) and ( ( sarea - 0.0 ) > smallVal ) ):
            twid  = sarea / length
            avdep = vol / sarea
        # end inner if
    # end outer if
    # end if
    if fl_aux1fg == 2:
        twid = sarea / length
        avdep = 0.0
    # end if
    if fl_aux2fg > 0:
        if ( ( vol - 0.0 ) >= smallVol ):
            avvel = ( ( length * ro ) / vol )
        else:
            avvel = 0.0
        # end if vol
    # end aux2 flag
    if ( ( fl_aux3fg > 0 ) and ( fl_aux2fg > 0 ) and ( fl_aux1fg > 0 ) ):
        #if avdep > 0.0:
        if ( ( avdep - 0.0 ) >= smallVal ):
            # these lines replace SHEAR; ustar (bed shear velocity), tau (bed shear stress)
            if fl_lkfg > 0:
                # lake calculations
                diff = ( 17.66 + ( log10( avdep / ( 96.5 * db50 ) ) )
                         * ( 2.3 / AKAPPA ) )
                if ( ( diff - 0.0 ) > smallVal ):
                    ustar = avvel / diff
                else:
                    ustar = 0.0
                # end if denominator
                tau = ( GAM / GRAV ) * pow( ustar, 2.0 )
            else:
                # stream calculations
                if ( ( avdep - 0.0 ) >= smallVal ):
                    diff = ( 2.0 * avdep ) + twid
                    if ( ( diff - 0.0 ) > smallVal ):
                        hrad = ( avdep * twid ) / diff
                    else:
                        hrad = 0.0
                    # end inner if
                else:
                    hrad = 0.0
                # calculate slope
                slope = delth / length
                ustar = sqrt( GRAV * slope * hrad )
                tau = ( GAM * slope ) * hrad
            # end if LKFG
        else:
            ustar = 0.0
            tau = 0.0
        # end if avvel
    # end if aux3flg
    # now assign based on our flags
    if fl_aux1fg > 0:
        TWID[targID][iI] = twid * LFACTA
        AVDEP[targID][iI] = avdep * LFACTA
        STAGE[targID][iI] = ( dep + stcor ) * LFACTA
    # end aux1 if
    if fl_aux2fg > 0:
        AVVEL[targID][iI] = avvel * SFACTA * LFACTA
    # end aux2 if
    if fl_aux3fg > 0:
        USTAR[targID][iI] = ustar * SFACTA * LFACTA
        TAU[targID][iI] = tau * TFACTA
        if fl_lkfg == 0:
            HRAD[targID][iI] = hrad * LFACTA
        # end if LKFG
    # end if aux3
    # return
    return errorCnt


def fndrow(v, volFT):
    """ Finds highest index in FTable volume column whose volume  < v.

    Modified to use numpy argmax

    Args:
        v (float) : volume to check
        volFT (np.array) : volume table vector

    """
    # locals in case Cython
    vLen = int( 0 )
    indx = int( 0 )
    # start
    vLen = volFT.shape[0]
    indx = np.argmax( volFT > v )
    # in the case that 0 is returned then now that
    # are beyond the array in some fashion
    if indx == 0:
        if v >= volFT[vLen - 1]:
            # minus 2 so that plus 1 works
            indx = vLen - 2
    else:
        # this is the standard case of within bounds
        indx = indx - 1
    # return
    return indx


def demand(vol, rowFT, funct, nexits, delts, convf, colind, outdgt, ODGTF):
    """ Calculate outflow demand for one FTAB row.

    Interpolate the row from the volume and pass as rowFT. Modified real number
    comparisons to be more numerically reliable.

    Args:
        vol (float): current volume

        rowFT (np.array): array of FTAB row

        funct (int): the value for combined type calculation switch

        nexits (int): number of exits

        delts (float): time step in seconds

        convf (int): float multiplier - flag and values different from 1.0 not
            supported

        colind (np.array): Array for number of exits this RR that has a number
            identifying the ODFVFG calculation rule to use for this exit. If colind is
            0 there is no fN(Vol) component. If colind is positive integer then it is
            the value of column index in the FTAB table to use for fN(Vol) interpolation.
            If colind is negative then it needs to have the form of X.Y where this X.Y
            value is taken from a time series specified by the user. X denotes the first
            FTAB column to use and 0.Y dentoes the proportion for this X column. The fN(Vol)
            discharge is calculated, in this case, as (X column value * (1.0 - 0.Y ) ) +
            ( X + 1 column value * 0.Y ).

        outdgt (np.array): Array for number of exits this RR that has COLIND time series
            for this time

        ODGTF (np.array): flag for whether to use outdgt which should be
            MAX_EXITS

    Returns:
        tuple: calculated outflow demands.

        0. (float): ro, or total, demand

        1. (np.array): o, demand by exit, array

    """
    # imports
    # globals
    # parameters
    # locals in case Cythonize
    od = np.zeros(nexits)   # return o demand array
    odSum = float( 0.0 )    # return ro demand
    col = float( 0.0 )      # FTAB column for the exit may be float for
                            #   multiplier
    icol = int( 0 )         # FTAB integer column for the exit
    diff = float( 0.0 )     # calculation var
    _od1 = float( 0.0 )     # calculation var
    a = float( 0.0 )        # calculation var
    b = float( 0.0 )        # calculation var
    c = float( 0.0 )        # calculation var
    # start
    for i in range(nexits):
        col = colind[i]
        icol = int(col)
        if icol != 0:
            diff = col - float(icol)
            if ( ( diff - 0.0 ) >= 1.0E-6 ):
                # this is the case where COLIND has the multiplier
                # and exit
                _od1 = rowFT[icol-1]
                # this calculation seems completely wrong. It is
                # commented out and adjusted in the following line.
                # It seems the correct equation should be Eq. 13 in
                #  the HSPF user manual.
                # Note that this functionality has never been tested
                # in mHSP2 and is unsupported.
                #od[i] = _od1 + diff * ( _od1 - rowFT[icol] ) * convf
                od[i] = ( ( ( _od1 * ( 1.0 - diff ) ) +
                            ( rowFT[icol] * diff ) ) * convf )
            else:
                # this is the case where just use the designated column
                # from the FTAB
                od[i] = rowFT[icol-1] * convf
            # end if
        # end if icol != 0
        # now see if need to do time varying time series
        icol = int(ODGTF[i])
        if icol > 0:
            if ( int(col) > 0 ):
                # both f(time) and f(vol)
                a = od[i]
                b = outdgt[i]
                c = (vol - b) / delts
                # need to use funct to determine relationship
                #  between f(time) and f(vol)
                if ( funct[i] == 1 ):
                    od[i] = min(a,b)
                elif ( funct[i] == 2 ):
                    od[i] = max(a,b)
                elif ( funct[i] == 3 ):
                    od[i] = a+b
                elif ( funct[i] == 4 ):
                    od[i] = max(a,c)
                # end funct if
            else:
                # pbd added for f(time) only
                od[i] = outdgt[i]
            # end if both or only f(time)
        # end if f(time)
    # end for exits
    # get our sum
    odSum = od.sum()
    # return
    return ( odSum, od )


def auxil( volumeFT, depthFT, sareaFT, indx, vol, AUX1FG ):
    """Compute depth and surface area

    Modified real number comparisons to be more numerically reliable.

    Args:
        volumeFT (np.array) : volume vector from FTABLE
        depthFT (np.array) : depth vector from FTABLE
        sareaFT (np.array) : surface area vector from FTABLE
        indx (int) : current index for volume bigger than current value
        vol (float) : current volume in rchres
        AUX1FG (int): flag for auxiliary calculations

    Returns:
        tuple: calculated hydraulic characteristics from FTAB interp

            0. (float): depth in feet

            1. (float): surface area in sq. ft.

    """
    # imports
    from math import pow
    # globals
    global MAXLOOPS, TOLERANCE, errorsV
    # parameters
    smallVal = float( 1E-10 )   # numeric comparison threshold
    smallVol = float( 1E-5 )    # volume threshold
    # locals - list for future Cython
    dep = float( 0.0 )     # return depth
    sarea = float( 0.0 )   # return surface area
    sa1 = float( 0.0 )     # surface area at indx
    a = float( 0.0 )       # surface area at indx + 1
    b = float( 0.0 )       # calculation variable
    vol1 = float( 0.0 )    # calculation variable
    vol2 = float( 0.0 )    # calculation variable
    c = float( 0.0 )       # calculation variable
    rdep2 = float( 0.5 )   # intitial guess
    rdep1 = float( 0.0 )   # calculation variable
    dep1 = float( 0.0 )    # calculation variable
    dep2 = float( 0.0 )    # calcualtion variable
    # start
    if ( ( vol - 0.0 ) >= smallVol ):
        sa1 = sareaFT[ indx ]
        a = sareaFT[ indx+1 ] - sa1
        b = 2.0 * sa1
        vol1 = volumeFT[ indx ]
        vol2 = volumeFT[ indx+1 ]
        c = -1.0 * ( (vol - vol1) / ( vol2 - vol1) ) * ( b + a )
        for i in range( MAXLOOPS ):
            rdep1 = rdep2
            rdep2 = rdep1 - ( ( a * pow( rdep1, 2.0 ) + ( b * rdep1 ) + c )
                                / ( ( 2.0 * a * rdep1 ) + b ) )
            if ( abs( rdep2 - rdep1 ) < TOLERANCE ):
                break
            # end if
        else:
            # convergence failure error message
            errorsV[3] += 1
        # end for MAXLOOPS
        if ( ( ( rdep2 - 1.0 ) >= smallVal ) or
                    ( ( 0.0 - rdep2 ) >= smallVal ) ):
            # converged outside valid range error message
            errorsV[4] += 1
        # end error check if
        dep1 = depthFT[indx]
        dep2 = depthFT[indx+1]
        # manual eq (36)
        dep = dep1 + ( rdep2 * ( dep2 - dep1 ) )
        sarea = sa1 + ( a * rdep2 )
    elif AUX1FG == 2:
        # removed in HSPF 12.4
        dep = depthFT[indx]
        sarea = sareaFT[indx]
    # end main if
    # return
    return dep, sarea


def npAuxil( volumeFT, depthFT, sareaFT, vol ):
    """Compute depth and surface area using built in numpy methods

    Args:
        volumeFT (np.array) : volume vector from FTABLE
        depthFT (np.array) : depth vector from FTABLE
        sareaFT (np.array) : surface area vector from FTABLE
        vol (float) : current volume in rchres

    Returns:
        tuple: calculated hydraulic characteristics from FTAB interp

            0. (float): depth in feet

            1. (float): surface area in sq. ft.

    """
    # imports
    # globals
    # parameters
    smallVal = float( 1E-10 )   # numeric comparison threshold
    smallVol = float( 1E-5 )    # volume threshold
    # locals - list for future Cython
    dep = float( 0.0 )     # return depth
    sarea = float( 0.0 )   # return surface area
    # start
    if ( ( vol - 0.0 ) >= smallVol ):
        # then need to calculate
        dep = float( np.interp( vol, volumeFT, depthFT ) )
        sarea = float( np.interp( vol, volumeFT, sareaFT ) )
    # end main if
    # return
    return dep, sarea


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
    global AVDEP, AVVEL, DEP, HRAD, IVOL, O1, O2, O3, O4, O5, OVOL1
    global OVOL2, OVOL3, OVOL4, OVOL5, PRSUPY, RO, ROVOL, SAREA, STAGE
    global VOL, TAU, TWID, USTAR, VOLEV
    # parameters
    goodReturn = 0
    badReturn = -1
    pathStart = "/RESULTS/RCHRES_"
    pathEnd = "/HYDR"
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
            if cOut == "AVDEP":
                outView = AVDEP[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "AVVEL":
                outView = AVVEL[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "DEP":
                outView = DEP[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "HRAD":
                outView = HRAD[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "IVOL":
                outView = IVOL[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "O1":
                outView = O1[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "O2":
                outView = O2[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "O3":
                outView = O3[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "O4":
                outView = O4[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "O5":
                outView = O5[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "OVOL1":
                outView = OVOL1[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "OVOL2":
                outView = OVOL2[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "OVOL3":
                outView = OVOL3[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "OVOL4":
                outView = OVOL4[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "OVOL5":
                outView = OVOL5[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "PRSUPY":
                outView = PRSUPY[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "RO":
                outView = RO[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "ROVOL":
                outView = ROVOL[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "SAREA":
                outView = SAREA[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "STAGE":
                outView = STAGE[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "VOL":
                outView = VOL[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "TAU":
                outView = TAU[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "TWID":
                outView = TWID[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "USTAR":
                outView = USTAR[tCol].view( dtype=np.float32 )
                df[cOut] = outView
            elif cOut == "VOLEV":
                outView = VOLEV[tCol].view( dtype=np.float32 )
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


def getOVOLbyExit( nExit ):
    """Return the OVOL data structure for a specified exit

    Args:
        nExit (int): the exit number

    Returns:
        np.recarray: OVOL, outflow volume data storage structure

    """
    # globals
    global OVOL1, OVOL2, OVOL3, OVOL4, OVOL5, MAX_EXITS
    # start
    if nExit == 1:
        return OVOL1
    elif nExit == 2:
        return OVOL2
    elif nExit == 3:
        return OVOL3
    elif nExit == 4:
        return OVOL4
    elif nExit == 5:
        return OVOL5
    else:
        # error
        errMsg = "Specified RCHRES exit %d exceeds maximum exit of %d.\n" \
                 "OVOL1 returned." % ( nExit, MAX_EXITS )
        CL.LOGR.error( errMsg )
        return OVOL1
    # end if


#EOF