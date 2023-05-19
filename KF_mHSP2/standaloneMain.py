#!/usr/bin/python3
"""
Main program statement for running mHSP2 as standalone only.

Command line argument options

    * *modelDir* (str): path for model directory with input files

Typical usage example

    python C:\Repositories\mHSP2\standaloneMain.py C:\\Working\\Test_Models\\WG_mHSP2 -f Uvalde14.h5

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
import sys
import os
import argparse
# local package imports. Can use the standard import approach because are not
#   run as independent processes
import locaMain as HSP2
import locaLogger as CL


#standalone execution block
# assumes that this module is executed within the same current directory
# as the input file
if __name__ == "__main__":
    # do the argument processing stuff first
    apUsage = "%(prog)s <model directory> -f <mHSP2 HDF5 file>"
    apDesc = "Execute standalone mHSP2"
    parser = argparse.ArgumentParser( usage=apUsage, description=apDesc )
    parser.add_argument( action='store', nargs=1,
                         dest='modelDir', type=str,
                         help='Model directory with input file(s)',
                         metavar="model directory" )
    parser.add_argument( '-f', '--file', action='store', nargs=1,
                         dest="inFile", type=str,
                         metavar="Input file",
                         help="Main input file", required=True )
    # parse the command line arguments received and set the simulation directory
    args = parser.parse_args()
    Sim_Dir = os.path.normpath( args.modelDir[0] )
    # check that our directory exists
    if not os.path.isdir( Sim_Dir ):
        # this is an error
        errMsg = "Model directory %s does not exist!!!" % Sim_Dir
        sys.exit( errMsg )
    # get the current directory
    CWD = os.getcwd()
    if CWD != Sim_Dir:
        os.chdir( Sim_Dir )
    # get and check the model input hdf5 file
    hsp5File = args.inFile[0]
    if hsp5File is None:
        errMsg = "No HDF5 file name specified for HSP2 !!!"
        sys.exit( errMsg )
    elif len( hsp5File ) < 4:
        errMsg = "Receivied invalid HDF5 file name of %s" % hsp5File
        sys.exit( errMsg )
    # now can continue
    hsp2InputFile = os.path.normpath( os.path.join( Sim_Dir, hsp5File ) )
    if not os.path.isfile( hsp2InputFile ):
        # this is an error
        errMsg = "HSP2 input file %s does not exist !!!" % hsp2InputFile
        sys.exit( errMsg )
    # end if
    # now set-up the logger
    CL.loggerStart( Sim_Dir )
    # run the model
    retStat = HSP2.salocaMain( Sim_Dir, hsp2InputFile )
    if retStat != 0:
        errMsg = "Error in mHSP2 simulation. Please see the log file."
        CL.LOGR.error( errMsg )
        sys.exit( errMsg )
    # close the logger
    CL.loggerEnd()
    # return to the current directory
    if CWD != Sim_Dir:
        os.chdir( CWD )
    # end if
    print( "Successful termination of mHSP2" )
    # end


#EOF