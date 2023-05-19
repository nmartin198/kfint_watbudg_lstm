# -*- coding: utf-8 -*-
"""
mHSP2 custom logger leveraging Python logging

Provides specification and configuration of Python's logging API to use for
debugging and informational purposes. This module is for mHSP2.

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
import logging
import os

# parameters
LOGNAME = "mHSP2-SA_Log.txt"
"""Log file name"""
LOG_LEVEL = logging.INFO
"""Logging level """
START_TIME = None
"""mHSP2 model start time"""

# set up the logger
LOGR = logging.getLogger('mHSP2')
"""Custom logging object"""
LOGR.setLevel( LOG_LEVEL )

FH = None
"""File handler"""
FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
"""Custom formatter"""

START_MSG = """Only standard units (acres, feet, inches) are supported.
Even if you set the units to 2, which is suppossed to denote metric, 
standard units and conversions are hard-coded into HSPsquared and into
this revised version.

List of Inputs with Expected Units:

PERLND

LZSN: Lower soil zone nominal storage depth in inches
INFILT: Index to infiltration capacity of the soil, inches/day
LSUR: Length of the assumed overland flow plane in feet
SLSUR: Slope of the assumed overland flow plane, ft/ft
KVARY: Parameter that affects behavior of groundwater recession flow
       Purpose is to allow recession flow to be non-exponential in its
       decay time. Units are 1/in
AGWRC: Basic groundwater recession rate if KVARY is 0 and there is
       no inflow to groundwater. Defined as the rate of flow today 
       divided by the rate of flow yesterday. Units are 1/day
PETMAX: Air temperature below which ET will be arbitrarily reduced
       Only used if CSNOFG == 1. Units are degrees Fahrenheit.
PETMIN: Air temperature below which ET will be set to zero
        Only used if CSNOFG == 1. Units are degrees Fahrenheit.
INFEXP: Exponent in infiltration equation, dimensionless.
INFILD: Ratio between maximum and mean infiltration capacities, 
        dimensionless
DEEPFR: Fraction of groundwater inflow which will enter deep and 
        inactive groundwater. Lost from the HSPF system.
BASETP: Fraction of remaining potential ET which can be satisfied 
        from baseflow or groundwater outflow
AGWETP: Fraction of remaining potential ET which can be satistifed 
        from active groundwater storage if enough is available.
CEPSC: Interception storage capacity in inches.
UZSN: Upper zone nominal storage in inches
NSUR: Manning's n for the assumed overland flow plane use 
        English/Standard units versions from tables.
INTFW: Interflow inflow parameter, dimensionless.
IRC: Interflow recession parameter; Under zero inflow, the ratio 
     of todays interflow outflow rate to yesterday's rate. Units 
     are 1/day
LZETP: Lower zone ET parameter; index to the density of deep-rooted 
       vegetation, dimensionless.
FZG: Parameter that adjusts for the effect of ice in the snow pack on
     infiltration when IFFCFG is 1. It is not used if IFFCFG is 2. Units
     are 1/inch
FZGL: Lower limit of INFFAC as adjusted by ice in the snow pack when 
      IFFCFG is 1. If IFFCFG is 2, FZGL is the value of INFFAC when 
      the lower layer temperature is at or below freezing. Dimensionless 
      parameter
CEPS: Initial interception storage in inches
SURS: Initial surface or overland flow storage in inches
UZS: Initial upper zone storage in inches
IFWS: Initial interflow storage in inches
LZS: Initial lower zone storage in inches
AGWS: Initial active groundwater storage in inches
GWVS: Initial index to groundwater slope in inches
AGWLI: Active groundwater lateral inflow external time series, inches/day
IFWLI: Interflow lateral inflow external time series, inches/day
LZLI: Lower soil zone lateral inflow external time series, inches/day
SURLI: Surface storage lateral inflow external time series, inches/day
UZLI: Upper soil zone lateral inflow external time series, inches/day

IMPLND:

LSUR: Length of the assumed overland flow plane in feet
NSUR: Manning's n for the assumed overland flow plane use 
      English/Standard units versions from tables.
RETSC: Retention, or interception, storage capacity of the surface 
      in inches
SLSUR: Slope of the assumed overland flow plane, ft/ft
PETMAX: Air temperature below which ET will be arbitrarily reduced
        Only used if CSNOFG == 1. Units are degrees Fahrenheit.
PETMIN: Air temperature below which ET will be set to zero
        Only used if CSNOFG == 1. Units are degrees Fahrenheit.
RETS: Initial retention, or interception, storage in inches
SURS: Initial surface or overland flow storage in inches
SURLI: Surface storage lateral inflow external time series, inches/day

RCHRES:

FTABNO: FTABLE id for each RCHRES, dimensionless
LEN: Length for each RCHRES in miles
DELTH: Drop in water elevation from upstream to downstream, in feet
STCOR: Correction to RCHRES depth to calculate stage, in feet
KS: Weighting factor for hydraulic routing
DB50: Sediment median grain diameter; specified in inches in inputs 
      and converted to feet for calcs
VOL: Initial volume of water in the RCHRES, in acre-ft
EXIVOL: External time series inflow, acre-feet per day
PREC: Input time series of precipitation to RCHRES, inches per day
POTEV: Input time series of potential evaporation from reservoir 
       surface, inches per day
"""
"""mHSP2 start up user message.

Provides implemented inputs to the program along with required units.
"""

END_MSG = """Only standard units (acres, feet, inches) are supported.
Even if you set the units to 2, which is suppossed to denote metric, 
standard units and conversions are hard-coded into HSPsquared and into
this revised version.

List of Outputs with Units:

PERLND

AGWET: AET from active groundwater, inches/day
AGWI: Active groundwater inflow, inches/day
AGWO: Active groundwater outflow, inches/day
AGWS: Active groundwater storage, inches
BASET: AET from baseflow, inches/day
CEPE: AET from interception storage, inches/day
CEPS: Interception storage, inches
GWVS: Index to available groundwater slope, inches
IFWI: Interflow inflow, inches/day
IFWO: Interflow outflow, inches/day
IFWS: Interflow storage, inches
IGWI: Inflow to inactive groundwater, inches/day
INFFAC: Factor to account for frozen ground Not currently implemented 
        and always set to 1
INFIL: Infiltration to soil, inches/day
LZET: Lower soil zone AET, inches/day
LZI: Lower soil zone inflow, inches/day
LZS: Lower soil zone storage, inches
PERC: Percolation from upper to lower soil zones, inches/day
RPARM: Maximum ET opportunity, inches/day
SURI: Surface storage inflow, inches/day
SURO: Surface storage outflow, inches/day
SURS: Surface storage, inches
TAET: Total PERLND AET, inches/day
TGWS: Total groundwater storage, should be equal to active groundwater
      storage prior to ET, inches/day
UZET: Upper soil zone AET, inches/day
UZI: Upper soil zone inflow, inches/day
UZS: Upper soil zone storage, inches
PERO: Total outflow from pervious land, inches/day
PERS: Total water stored in pervious land, inches
SUPY: Moisture supplied to the land segment by precipitation, inches/day
PET: Potential evapotranspiration, inches/day
PETADJ: Adjusted PET for temperature restrictions, inches/day

IMPLND

IMPEV: Total simulated ET for impervious, inches/day
IMPS: Total water stored in impervious lands, inches
PET: Potential evapotranspiration, inches/day
PETADJ: Adjusted PET from air temperature limits, inches/day
RETS: Retention storage, inches
SUPY: Moisture supplied to the land segment by precipitation, inches/day
SURI: Surface storage inflow, inches/day
SURO: Surface storage outflow, inches/day
SURS: Surface storage, inches

RCHRES

AVDEP: Average depth in ft
AVVEL: Average velocity in ft/s
DEP: Depth in feet in RCHRES
HRAD: Hydraulic radius in ft
IVOL: Inflow to RCHRES, acre-feet per day
O1: Rate of outflow through exit 1, ft3/s
O2: Rate of outflow through exit 2, ft3/s
O3: Rate of outflow through exit 3, ft3/s
O4: Rate of outflow through exit 4, ft3/s
O5: Rate of outflow through exit 5, ft3/s
OVOL1: Volume of outflow through exit 1, af/day
OVOL2: Volume of outflow through exit 2, af/day
OVOL3: Volume of outflow through exit 3, af/day
OVOL4: Volume of outflow through exit 4, af/day
OVOL5: Volume of outflow through exit 5, af/day
PRSUPY: Volume of water contributed by precipitation to the 
        surface, af/day
RO: Total rate of outflow from RCHRES, ft3/s
ROVOL: Total volume of outflow from RCHRES, af/day
SAREA: Surface area of RCHRES in acres
STAGE: Stage of RCHRES = DEP + STCOR, ft
TAU: Bed shear stress
TWID: Stream top width, ft
USTAR: Shear velocity ft/s
VOL: Volume of water in the RCHRES, af
VOLEV: Volume of water lost by evaporation, af/day
"""
"""mHSP2 wrap-up user message.

Provides implemented outputs from the program along with required units.
"""


def loggerStart( LFPath ):
    """Start the logger to use with mHSP2

    Args:
        LFPath (str): FQDN path for the log file

    """
    # local imports
    import datetime as dt
    # globals
    global LOGR, START_TIME, FH, LOGNAME, FORMATTER, LOGR, START_MSG
    # start the configuration
    # get the full log file path
    LFPath = os.path.normpath( os.path.join( LFPath, LOGNAME ) )
    FH = logging.FileHandler( LFPath, mode='w' )
    FH.setLevel( LOG_LEVEL )
    FH.setFormatter( FORMATTER )
    LOGR.addHandler( FH )
    # now write the first entry
    START_TIME = dt.datetime.now()
    LOGR.info( "Start mHSP2 model at %s\n" % 
                  START_TIME.strftime( "%Y-%m-%d %H:%M:%S" ) )
    LOGR.info( "%s" % START_MSG )
    # return
    return


def loggerEnd():
    """End the mHSP2 logger
    """
    # imports
    import datetime as dt
    # globals
    global LOGR, START_TIME, END_MSG
    # set the end time
    LOGR.info( "%s" % END_MSG )
    END_TIME = dt.datetime.now()
    eTimerS = ( END_TIME - START_TIME ).total_seconds()
    eTimerM = eTimerS/60.0
    LOGR.info( "End of mHSP2 at %s - elapsed time - %10.2f min\n" % 
               ( END_TIME.strftime( "%Y-%m-%d %H:%M:%S" ), eTimerM ) )
    # return
    return


#EOF