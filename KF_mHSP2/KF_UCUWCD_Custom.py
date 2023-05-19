# -*- coding: utf-8 -*-
"""
Custom functions and parameters for Kalman Filter UCUWCD implementation.

This module provide everything needed to implement the Kalman filter 
integration of the Uvalde County Underground Water Conservation 
District (UCUWCD) Water Balance Model with a custom deep learning (DL)
predictor for aquifer segment water levels. 

"""
# Copyright and License
"""
Copyright 2023 Southwest Research Institute (SwRI)

Module Author: Nick Martin <nick.martin@alumni.stanford.edu>

This file contains modifications and extensions to provide a Kalman filter-
based data assimilation for the mHSP2 variant of the Hydrological 
Simulation Program–FORTRAN (HSPF). These modifications and extensions
were developed and implemented as part of SwRI Internal Research and
Development Grant 15-R6209.

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
import os
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------
# parameters
NUM_MEAS = 5
"""Number of measurements"""
NUM_DELSTOR = 5
"""Number of state variables derived from measurements"""
NUM_MESTATE = NUM_MEAS + NUM_DELSTOR
"""Number of measurements and states that solve for"""
# standard deviations from EA_LSTM predictions
# 33130.7, 3342.6633, 38418.734, 1711.9792, 1183.4873,
#   8129.666, 2132.6372, 23981.258, 1178.5802, 764.8653
# standard deviations from data
# 5166.09, 5617.6187, 50523.406, 2538.7422, 1917.0648,
#   2528.6833, 1393.2394, 17771.164, 1141.1305, 524.88116
# --------------------------------------------------------------------------
# Iteration 1 - Q is standard deviations of data; var-cov of data used also
#tITER = 1
#"""Model Iteration ID"""
#npQMat = np.array( [ [1.0*5166.09, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 1.0*5617.6187, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 1.0*50523.406, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 1.0*2538.7422, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 1.0*1917.0648, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 2528.6833, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1393.2394, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 17771.164, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1141.1305, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 524.88116,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npDVolCov.pkl"
# Iteration 2 - Q is standard deviation of LSTM preds; var-cov of predictions used too
#tITER = 2
#"""Model Iteration ID"""
#npQMat = np.array( [ [1.0*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 1.0*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 1.0*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 1.0*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 1.0*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npPVolCov.pkl"
# Iteration 3 - Q is standard deviations of data; var-cov of predictions
#tITER = 3
#"""Model Iteration ID"""
#npQMat = np.array( [ [1.0*5166.09, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 1.0*5617.6187, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 1.0*50523.406, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 1.0*2538.7422, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 1.0*1917.0648, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 2528.6833, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1393.2394, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 17771.164, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1141.1305, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 524.88116,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npPVolCov.pkl"
# Iteration 4 - Q std of predictions; abs var-cov of predictions
#tITER = 4
#"""Model Iteration ID"""
#npQMat = np.array( [ [1.0*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 1.0*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 1.0*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 1.0*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 1.0*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 1.0*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1.0*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 1.0*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1.0*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npRedVolCov.pkl"
#"""Variance-covariance matrix for measurements, 10 x 10"""
#SSE_AdjM = np.array( [1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32 ).flatten()
#"""Adjust variance to contribute moving Kalman Gain"""
# Iteration 5 - Q std of predictions adjusted by 0.001; abs var-cov of predictions
#tITER = 5
#"""Model Iteration ID"""
#npQMat = np.array( [ [0.001*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0.001*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0.001*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0.001*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0.001*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0.001*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0.001*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0.001*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0.001*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npAbsPVolCov.pkl"
# Iteration 6 - Q std of predictions adjusted by 1000.0; abs var-cov of predictions
#tITER = 6
#"""Model Iteration ID"""
#npQMat = np.array( [ [1000.0*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 1000.0*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 1000.0*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 1000.0*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 1000.0*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 1000.0*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1000.0*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 1000.0*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1000.0*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000.0*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npAbsPVolCov.pkl"
# Iteration 7 - Q is standard deviation of LSTM preds * 0.001; var-cov of predictions
#tITER = 7
#"""Model Iteration ID"""
#npQMat = np.array( [ [0.001*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0.001*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0.001*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0.001*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0.001*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0.001*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0.001*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0.001*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0.001*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npPVolCov.pkl"
# Iteration 8 - Q is standard deviation of LSTM preds * 1000.0; var-cov of predictions
#tITER = 8
#"""Model Iteration ID"""
#npQMat = np.array( [ [1000*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 1000*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 1000*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 1000*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 1000*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 1000*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1000*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 1000*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1000*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npPVolCov.pkl"
# Iteration 9 - Q std of predictions with custom mult; abs var-cov of predictions
#tITER = 9
#"""Model Iteration ID"""
#npQMat = np.array( [ [10000*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 10000*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 10000*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 10*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0.1*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 10000*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 10000*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 10000*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 10*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npAbsPVolCov.pkl"
# Iteration 10 - Q std of predictions with custom mult; abs var-cov of predictions
#tITER = 10
#"""Model Iteration ID"""
#npQMat = np.array( [ [1000.0*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 500.0*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 1000.0*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 500.0*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 1.0*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 1000.0*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1000.0*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 1000.0*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1000.0*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000.0*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npAbsPVolCov.pkl"
# Iteration 11 - Q std of predictions with custom mult; abs var-cov of predictions
#tITER = 11
#"""Model Iteration ID"""
#npQMat = np.array( [ [1000.0*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 250.0*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 1000.0*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 250.0*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0.75*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 1000.0*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1000.0*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 1000.0*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1000.0*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000.0*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npAbsPVolCov.pkl"
# Iteration 12 - Q std of predictions with custom mult; abs var-cov of predictions
#tITER = 12
#"""Model Iteration ID"""
#npQMat = np.array( [ [1000.0*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 50.0*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 500.0*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 50.0*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0.25*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 1000.0*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1000.0*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 1000.0*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1000.0*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000.0*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npAbsPVolCov.pkl"
# Iteration 13 - Q std of predictions; abs var-cov of predictions
#tITER = 13
#"""Model Iteration ID"""
#npQMat = np.array( [ [0.01*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0.1*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 20.0*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0.01*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0.01*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 1.0*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1.0*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 1.0*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1.0*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npRedVolCov.pkl"
#"""Variance-covariance matrix for measurements, 10 x 10"""
# Iteration 14 - Q std of predictions; abs var-cov of predictions
#tITER = 14
#"""Model Iteration ID"""
#npQMat = np.array( [ [1.0*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 50.0*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 100.0*38418.734, 0, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 1.0*1711.9792, 0, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 1.0*1183.4873, 0, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 1.0*8129.666, 0, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 1.0*2132.6372, 0, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 1.0*23981.258, 0, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 1.0*1178.5802, 0,],
#                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0*764.8653,],], dtype=np.float32 )
#"""Specification of the noise matrix or Q matrix, use sigma values"""
#Rk_File_Name = "npRedVolCov.pkl"
#"""Variance-covariance matrix for measurements, 10 x 10"""
#SSE_AdjM = np.array( [2000.0, 0.001, 0.00001, 10000000.0, 100000000.0], dtype=np.float32 ).flatten()
#"""Adjust variance to contribute moving Kalman Gain"""
# Iteration 15 - Q std of predictions; abs var-cov of predictions
tITER = 15
"""Model Iteration ID. This iteration is the final model and the Kalman gain related 
matrices specified in the following show the gain tuning parameters and final values."""
npQMat = np.array( [ [10.0*33130.7, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
                     [0, 1.0*3342.6633, 0, 0, 0, 0, 0, 0, 0, 0,],
                     [0, 0, 10.0*38418.734, 0, 0, 0, 0, 0, 0, 0,],
                     [0, 0, 0, 10.0*1711.9792, 0, 0, 0, 0, 0, 0,],
                     [0, 0, 0, 0, 10.0*1183.4873, 0, 0, 0, 0, 0,],
                     [0, 0, 0, 0, 0, 1.0*8129.666, 0, 0, 0, 0,],
                     [0, 0, 0, 0, 0, 0, 1.0*2132.6372, 0, 0, 0,],
                     [0, 0, 0, 0, 0, 0, 0, 1.0*23981.258, 0, 0,],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1.0*1178.5802, 0,],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0*764.8653,],], dtype=np.float32 )
"""Specification of the noise matrix or Q matrix, use sigma values"""
Rk_File_Name = "npRedVolCov.pkl"
"""Variance-covariance matrix for measurements, 10 x 10"""
SSE_AdjM = np.array( [1000.0, 1.0, 0.01, 0.01, 0.01], dtype=np.float32 ).flatten()
"""Adjust variance to contribute moving Kalman Gain"""
INIT_STATE_UNC = 8.0E+9
"""Initial state uncertainty"""
Aquifers_Dict = { "R001" : [ "FT001", "Edwards", ["J-27", "J-27_ft"]],
                  "R002" : [ "FT002", "Buda_1", ["Willoughby",
                                                 "Willoughby_ft"]],
                  "R003" : ["FT003", "Austin", ["McBride", "McB1_ft"]],
                  "R009" : ["FT009", "LeonaGrav", ["Ehler", "Ehler_ft"]],
                  "R010" : ["FT010", "Buda_2", ["McBride", "McB3_ft"]], }
"""Aquifers metadata dictionary"""
Extracts_Dict = { "R001" : [ 5, "OUTDGT5"],
                  "R002" : [ 5, "OUTDGT5"],
                  "R003" : [ 3, "OUTDGT3"],
                  "R009" : [ 2, "OUTDGT2"],
                  "R010" : [ 3, "OUTDGT3"],}
"""Dictionary identifying where to find the pumping time series"""
FTAB_File_Name = "FTabDict.pkl"
"""File name for precompiled and serialized FTABLE dictionary"""
AMat_File_Name = "npAmat.pkl"
"""File name for pre-created, state-transition or A matrix"""
HMat_File_Name = "npHmat.pkl"
"""File name for pre-created, state to measurement or H matrix"""
Meas_File_Name = "All_Predictions-Processed.xlsx"
"""Spreadsheet with all water level measurements"""
Meas_Dir = "Meas"
"""Sub-directory where measurement spreadsheet is stored"""
Mon_Meas_Sheet_Name = "All_Monthly_Predicted"
"""Sheet name for sheet with monthly values"""


#-----------------------------------------------------------------------
# module level data structures
DAILY_INDEX = None
"""Daily simulation time index"""
MONTHLY_INDEX = None
"""Monthly version of the daily simulation index"""
NUM_DAYS = None
"""Number of days that will be simulated"""
NUM_MONTHS = None
"""Number of months that will be simulated"""
npR_kMat = None
"""R_k matrix which is variance-covariance for measurements"""
FTAB_DICT = None
"""Dictionary with FTABLE interpolation tables"""
npAMat = None
"""A matrix which is the state transition matrix, A"""
npHMat = None
"""A matrix which is the state to measurement matrix, H"""
CUR_MONTH = 0
"""Index or counter for the current month"""
DFMeasures = None
"""DataFrame with the time series of measurements in volume"""
MEAS = np.zeros( NUM_MEAS, dtype=np.float32 )
"""Numpy array of measured values for current month"""
PRED = np.zeros( NUM_MEAS, dtype=np.float32 )
"""Numpy array of predicted values, by HSPF, for current month"""
MEAS_PREV = np.zeros( NUM_MEAS, dtype=np.float32 )
"""Measured values from previous month"""
PRED_PREV = np.zeros( NUM_MEAS, dtype=np.float32 )
"""Predicted values from pervious month"""
PRED_VOL_DICT = dict()
"""Dictionary that holds the originally predicted volumes for each month"""
SVCmat = np.zeros( (NUM_MESTATE, NUM_MESTATE), dtype=np.float32 )
"""State-variance, covariance matrix, P_k, for current month"""
SVCmat_PREV = np.zeros( (NUM_MESTATE, NUM_MESTATE), dtype=np.float32 )
"""State-variance, covariance matrix, P_k-1, for previous month"""
H = np.zeros( (1,NUM_MESTATE), dtype=np.float32 )
"""State to measurement transition matrix"""
H[0,:NUM_MEAS] = 1
HT = np.zeros( (NUM_MESTATE,1), dtype=np.float32 )
"""Transposed state to measurement transition matrix"""
HT[:NUM_MEAS,0] = 1
FullHdrColList = list()
"""List of column names for NUM_MESTATE x NUM_MESTATE tracking"""
for iI in range(1,NUM_MESTATE + 1):
    for jJ in range(1,NUM_MESTATE+1):
        cColN = "r%dc%d" % ( iI, jJ )
        FullHdrColList.append( cColN )
    # end col for
# end row for
PMHdrColList = [ "v1", "v2", "v3", "v4", "v5", "delv1", "delv2", "delv3",
                 "delv4", "delv5" ]
"""List of column names for data structures that track predict/measured."""
KGainTrackDF = None
"""Kalman Gain tracking DataFrame"""
SSECovTrackDF = None
"""System state error covariance matrix, P_{k}, tracking (NUM_MESTATE,NUM_MESTATE)"""
PredInitDF = None
"""Initial prediction tracking DataFrame (NUM_MESTATE)"""
PredAdjDF = None
"""Adjusted prediction, x_prime, tracking DataFrame (NUM_MESTATE)"""
PredCorrectDF = None
"""Corrected prediction, PRED_PREV, tracking DataFrame (NUM_MEAS)"""
ResidsDF = None
"""Residuals tracking (NUM_MESTATE)"""
PumpTrackDF = None
"""Tracking corrected pumping"""
ExInflowTrackDF = None
"""Tracking adjusted external inflow"""
# lambdas
ConvAF2FT2 = lambda af: af * 43560.0
ConvAFD2CFS = lambda afd: afd * (1.0/(24.0*60.0*60.0)) * (43560.0/1.0)
ConvCFS2AFD = lambda cfs: cfs * ((24.0*60.0*60.0)/1.0) * (1.0/43560.0)


#-----------------------------------------------------------------------
# functions
def preSimSetup( simdir, tIndex ):
    """Setup and load data structures for calculations

    Parameters
    ----------
    simdir : STR
        Verified model simulaton directory
    tIndex : pd.DateTimeIndex
        Daily simulation time index.

    For testing:
        tIndex = pd.date_range(start=pd.Timestamp( 2016, 1, 1, 0),
                               end=pd.Timestamp( 2019, 10, 31, 23, 59, 0 ),
                               freq="D")
        simdir = "D:\\SwRI_Projects\\LSTM_IRD\\NumModels\\Kalman_Filter\\SG_Data"

    Returns
    -------
    None.

    """
    # imports
    import pickle
    from locaHrchhyd import OUTDGT1, OUTDGT2, OUTDGT3, OUTDGT4, OUTDGT5
    from locaHrchhyd import EXIVOL
    # globals
    global DAILY_INDEX, MONTHLY_INDEX, NUM_DAYS, NUM_MONTHS
    global npR_kMat, Rk_File_Name, FTAB_DICT, FTAB_File_Name
    global AMat_File_Name, npAMat, KGainTrackDF, FullHdrColList
    global HMat_File_Name, npHMat, PMHdrColList, SSECovTrackDF
    global PredInitDF, PredAdjDF, PredCorrectDF, ResidsDF, tITER
    global PumpTrackDF, ExInflowTrackDF, Aquifers_Dict, Extracts_Dict
    global NUM_MESTATE, NUM_MEAS, INIT_STATE_UNC, SSE_AdjM
    # parameters
    # locals
    # start
    # copy
    DAILY_INDEX = tIndex.copy()
    NUM_DAYS = len( DAILY_INDEX )
    # resample
    lastTS = DAILY_INDEX[(NUM_DAYS-1)]
    startTS = DAILY_INDEX[0]
    if lastTS.day == 1:
        adjLastTS = lastTS - pd.Timedelta(days=0.8)
        useLastTS = pd.Timestamp( adjLastTS.year, adjLastTS.month,
                                  adjLastTS.day, 23, 59, )
    else:
        useLastTS = lastTS
    # end if
    MONTHLY_INDEX = pd.date_range( start=startTS, end=useLastTS, freq='MS' )
    NUM_MONTHS = len( MONTHLY_INDEX )
    # load the calculation matrices
    InFiler = os.path.normpath( os.path.join( simdir, Rk_File_Name ) )
    with open( InFiler, 'rb' ) as IF:
        npR_kMat = pickle.load( IF )
    # end with
    # now fill the unobserved state
    for iI in range(NUM_MESTATE):
        if iI < NUM_MEAS:
            npR_kMat[iI,:] = SSE_AdjM[iI] * npR_kMat[iI,:]
        else:
            npR_kMat[iI,iI] = INIT_STATE_UNC
        # end if
    # end for
    InFiler = os.path.normpath( os.path.join( simdir, FTAB_File_Name ) )
    with open( InFiler, 'rb' ) as IF:
        FTAB_DICT = pickle.load( IF )
    # end with
    InFiler = os.path.normpath( os.path.join( simdir, AMat_File_Name ) )
    with open( InFiler, 'rb' ) as IF:
        npAMat = pickle.load( IF )
    # end with
    InFiler = os.path.normpath( os.path.join( simdir, HMat_File_Name ) )
    with open( InFiler, 'rb' ) as IF:
        npHMat = pickle.load( IF )
    # end with
    # get the measurements
    getMeasDF( simdir, MONTHLY_INDEX[0], MONTHLY_INDEX[NUM_MONTHS-1] )
    # setup the Kalman Gain tracking DF
    KGainTrackDF = pd.DataFrame( index=MONTHLY_INDEX, columns=FullHdrColList,
                                 dtype=np.float32 )
    # setup the other tracking DataFrames
    SSECovTrackDF = pd.DataFrame( index=MONTHLY_INDEX, columns=FullHdrColList,
                                  dtype=np.float32 )
    PredInitDF = pd.DataFrame( index=MONTHLY_INDEX, columns=PMHdrColList,
                               dtype=np.float32 )
    PredAdjDF = pd.DataFrame( index=MONTHLY_INDEX, columns=PMHdrColList,
                              dtype=np.float32 )
    PredCorrectDF = pd.DataFrame( index=MONTHLY_INDEX,
                                  columns=PMHdrColList[:NUM_MEAS],
                                  dtype=np.float32 )
    ResidsDF = pd.DataFrame( index=MONTHLY_INDEX, columns=PMHdrColList,
                             dtype=np.float32 )
    # set up the forcing tracking
    dfCols = sorted( Aquifers_Dict.keys() )
    PumpTrackDF = pd.DataFrame( 0.0, index=DAILY_INDEX, columns=dfCols,
                                dtype=np.float32 )
    ExInflowTrackDF = pd.DataFrame( 0.0, index=DAILY_INDEX, columns=dfCols,
                                    dtype=np.float32 )
    # now output the original/initial pumping and external inflow values
    OrgPumpDF = pd.DataFrame( 0.0, index=DAILY_INDEX, columns=dfCols,
                              dtype=np.float32 )
    OrgExInflowDF = pd.DataFrame( 0.0, index=DAILY_INDEX, columns=dfCols,
                                  dtype=np.float32 )
    # go through and fill
    for rId in dfCols:
        pInfo = Extracts_Dict[rId]
        curPExit = pInfo[0]
        if curPExit == 1:
            OrgPumpDF[rId] = (OUTDGT1[:][rId] ).flatten()
            PumpTrackDF[rId] = (OUTDGT1[:][rId] ).flatten()
        elif curPExit == 2:
            OrgPumpDF[rId] = (OUTDGT2[:][rId] ).flatten()
            PumpTrackDF[rId] = (OUTDGT2[:][rId] ).flatten()
        elif curPExit == 3:
            OrgPumpDF[rId] = (OUTDGT3[:][rId] ).flatten()
            PumpTrackDF[rId] = (OUTDGT3[:][rId] ).flatten()
        elif curPExit == 4:
            OrgPumpDF[rId] = (OUTDGT4[:][rId] ).flatten()
            PumpTrackDF[rId] = (OUTDGT4[:][rId] ).flatten()
        elif curPExit == 5:
            OrgPumpDF[rId] = (OUTDGT5[:][rId] ).flatten()
            PumpTrackDF[rId] = (OUTDGT5[:][rId] ).flatten()
        # end if
        # now get the external inflow
        OrgExInflowDF[rId] = ( EXIVOL[:][rId] ).flatten()
        ExInflowTrackDF[rId] = ( EXIVOL[:][rId] ).flatten()
    # end reservoir for
    # now output
    OutFiler = os.path.normpath( os.path.join( simdir, "OrigPumpingDF_%d.pkl" % tITER) )
    OrgPumpDF.to_pickle( OutFiler )
    OutFiler = os.path.normpath( os.path.join( simdir, "OrigExInflowDF_%d.pkl" % tITER ) )
    OrgExInflowDF.to_pickle( OutFiler )
    # return
    return


def getMeasDF( simdir, monStartTS, monStopTS ):
    """Get a DataFrame of measurements from the spreadsheet

    Parameters
    ----------
    simdir : STR
        Verified model simulaton directory
    monStartTS : pd.Timestamp
        Starting time stamp for simulation.
    monStopTS : pd.Timestamp
        Last monthly time step for simulation.

    Returns
    -------
    None.

    """
    # imports
    # globals
    global Aquifers_Dict, Meas_File_Name, Meas_Dir, Mon_Meas_Sheet_Name
    global DFMeasures, FTAB_DICT
    # parameters
    # locals
    endTS = monStopTS + pd.Timedelta(days=15)
    # get the extraction and DataFrame Columns
    extractCols = list()
    dfCols = sorted( Aquifers_Dict.keys() )
    for tKey in dfCols:
        aqVals = Aquifers_Dict[tKey]
        extractCols.append( aqVals[2][1] )
    # end for
    # load the file
    InFiler = os.path.normpath( os.path.join( simdir, Meas_Dir, Meas_File_Name ) )
    curDF = pd.read_excel( InFiler, sheet_name=Mon_Meas_Sheet_Name,
                           header=0, index_col=0, parse_dates=True )
    curDF = curDF.loc[monStartTS:endTS].copy()
    wlDF = curDF[extractCols].copy()
    # now need to convert the monthly average water levels to volumes.
    DataDict = dict()
    cCnt = 0
    for tCol in dfCols:
        curFTAB = FTAB_DICT[tCol]
        DataDict[tCol] = np.interp( wlDF[extractCols[cCnt]].to_numpy(dtype=np.float32),
                                    curFTAB["Depth_ft"].to_numpy(dtype=np.float32),
                                    curFTAB["Volume_af"].to_numpy(dtype=np.float32), )
        cCnt += 1
    # end for
    DFMeasures = pd.DataFrame( index=wlDF.index, data=DataDict, dtype=np.float32 )
    # return
    return


def predCorrectWrapper( tInd, updateMonth, newMonth ):
    """Kalman filter prediction calculations then calc correction calcs.

    Testing:
        tInd = 60
        updateMonth = 2
        newMonth = 3
        CUR_MONTH = 1

    Parameters
    ----------
    tInd : TYPE
        DESCRIPTION.
    updateMonth : TYPE
        DESCRIPTION.
    newMonth : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # imports
    # globals
    global CUR_MONTH, npQMat, INIT_STATE_UNC, Aquifers_Dict, NUM_MESTATE
    global DAILY_INDEX, MONTHLY_INDEX, npR_kMat, FTAB_DICT, npAMat, npHMat
    global NUM_MEAS, NUM_DELSTOR, MEAS, PRED, MEAS_PREV, PRED_PREV
    global SVCmat, SVCmat_PREV, KGainTrackDF, SSECovTrackDF
    global PredInitDF, PredAdjDF, PredCorrectDF, ResidsDF
    # parameters
    # locals
    npAllMeas = np.zeros( NUM_MESTATE, dtype=np.float32 )
    npAllPred = np.zeros( NUM_MESTATE, dtype=np.float32 )
    # start
    lastInd = tInd - 1
    endTS = DAILY_INDEX[lastInd] + pd.Timedelta(days=0.75)
    startTS = MONTHLY_INDEX[CUR_MONTH]
    curNumDays = startTS.days_in_month
    startInd = (lastInd+1) - curNumDays
    useDeltaT = 1.0
    # now need to set our A matrix
    AMat = npAMat.copy()
    for iI in range(NUM_MEAS):
        AMat[iI,(iI+NUM_MEAS)] = useDeltaT
    # end for
    # do calculations based on value of CUR_MONTH
    if CUR_MONTH == 0:
        # initialization step for k == 1
        # end row for
        exMeas = extractCurMonMeas( startTS )
        np.copyto( MEAS, exMeas )
        exPred = extractCurMonPred( startTS, endTS, CUR_MONTH )
        np.copyto( PRED, exPred )
        npAllMeas[:NUM_MEAS] = MEAS
        npAllPred[:NUM_MEAS] = PRED
        PredInitDF.loc[startTS] = npAllPred.flatten()
        PredAdjDF.loc[startTS] = 0.0
        ResidsDF.loc[startTS] = 0.0
        SVCmat = npR_kMat.copy()
        SSECovTrackDF.loc[startTS] = SVCmat.flatten()
        KGainTrackDF.loc[startTS] = 0.0
        np.copyto( SVCmat_PREV, SVCmat )
        doCorrect = False
    elif CUR_MONTH == 1:
        # this is the reinitialization step
        exMeas = extractCurMonMeas( startTS )
        np.copyto( MEAS, exMeas )
        exPred = extractCurMonPred( startTS, endTS, CUR_MONTH )
        np.copyto( PRED, exPred )
        npAllMeas[:NUM_MEAS] = MEAS
        npAllMeas[NUM_MEAS:] = ( MEAS - MEAS_PREV ) / useDeltaT
        npAllPred[:NUM_MEAS] = PRED
        npAllPred[NUM_MEAS:] = ( PRED - PRED_PREV ) / useDeltaT
        PredInitDF.loc[startTS] = npAllPred.flatten()
        PredAdjDF.loc[startTS] = 0.0
        ResidsDF.loc[startTS] = 0.0
        SVCmat = npR_kMat.copy()
        np.copyto( SVCmat_PREV, SVCmat )
        SSECovTrackDF.loc[startTS] = SVCmat.flatten()
        KGainTrackDF.loc[startTS] = 0.0
        np.copyto( PRED_PREV, PRED )
        doCorrect = False
    else:
        # standard calculation step.
        # we get the predicted values just like before from HSPF.
        exMeas = extractCurMonMeas( startTS )
        np.copyto( MEAS, exMeas )
        exPred = extractCurMonPred( startTS, endTS, CUR_MONTH )
        np.copyto( PRED, exPred )
        npAllMeas[:NUM_MEAS] = MEAS
        npAllMeas[NUM_MEAS:] = ( MEAS - MEAS_PREV ) / useDeltaT
        npAllPred[:NUM_MEAS] = PRED
        npAllPred[NUM_MEAS:] = ( PRED - PRED_PREV ) / useDeltaT
        PredInitDF.loc[startTS] = npAllPred.flatten()
        # we also now predict the state-variance covariance matrix
        SVCpred = np.dot( np.dot( AMat, SVCmat_PREV ), AMat.T ) + npQMat
        # next calculate the Kalman Gain for this interval
        SMat = np.dot( np.dot( npHMat, SVCpred ), npHMat.T ) + npR_kMat
        curKG = np.dot( np.dot( SVCpred, npHMat.T ), np.linalg.inv(SMat) )
        # add Kalman gain to tracking DF
        KGainTrackDF.loc[startTS] = curKG.flatten()
        # calculate the residual term
        resid = npAllMeas - np.dot( npHMat, npAllPred )
        ResidsDF.loc[startTS] = resid.flatten()
        # calculate our new prediction accounting for KG and the difference
        #   between measured and HSPF predicted.
        xPrime = npAllPred + np.dot( curKG, resid )
        PredAdjDF.loc[startTS] = xPrime.flatten()
        PPrime = SVCpred - np.dot( np.dot( curKG, npHMat ), SVCpred )
        SSECovTrackDF.loc[startTS] = PPrime.flatten()
        np.copyto( SVCmat_PREV, PPrime )
        np.copyto( PRED_PREV, xPrime[:NUM_MEAS] )
        # get the adjusted prediction residuals. These are in acre-ft
        AdjPred = xPrime[:NUM_MEAS] - PRED
        # adjust the previous month for these residuals
        detMakeTSAdjusts( AdjPred, startInd, lastInd, startTS, endTS )
        # call the adjustments
        doCorrect = True
    # end if
    # update scurrent month before return
    np.copyto( MEAS_PREV, MEAS )
    CUR_MONTH += 1
    # return
    return ( doCorrect, startInd, lastInd )


def extractCurMonMeas( monTS ):
    """Extract EA-LSTM predictions, which are measurements in this app

    Parameters
    ----------
    monTS : pd.Timestamp
        DateTime index for current month.

    Returns
    -------
    npCurMeas : np.array( NUM_MEAS, dtype=np.float32)
        Has the measured volumes for the current month.

    """
    # imports
    # globals
    global NUM_MEAS, DFMeasures
    # parameters
    # locals
    npCurMeas = np.array( NUM_MEAS, dtype=np.float32 )
    # get
    npCurMeas = DFMeasures.loc[monTS].to_numpy(dtype=np.float32).flatten()
    # return
    return npCurMeas


def extractCurMonPred( monTS, endTS, monInd ):
    """Extract predicted values by HSPF

    Parameters
    ----------
    monTS : pd.Timestamp
        DateTime index for the start of the current month.
    endTS : pd.Timestamp
        Timestamp for about 18 hours into the last day of the month
    monInd : int
        Current month index in the simulation.

    Returns
    -------
    npCurPred : np.array( NUM_MEAS, dtype=np.float32)
        Has the predicted volumes for the current month.

    """
    # imports
    from locaHrchhyd import VOL
    # globals
    global DAILY_INDEX, Aquifers_Dict, NUM_MEAS, PRED_VOL_DICT
    # parameters
    # locals
    npCurPred = np.zeros( NUM_MEAS, dtype=np.float32 )
    # start
    tfCols = sorted( Aquifers_Dict.keys() )
    # get the current VOL recarray as a DataFrame with a DateTimeIndex
    curDF = pd.DataFrame( VOL )
    dtDF = curDF[tfCols].copy()
    dtDF["dayind"] = DAILY_INDEX
    dtDF.set_index("dayind", inplace=True )
    dtDF = dtDF.rename_axis(None)
    # now extract just the current month
    cMonDF = dtDF.loc[monTS:endTS].copy()
    cMeanSeries = cMonDF.mean()
    PRED_VOL_DICT[monInd] = cMonDF
    npCurPred = cMeanSeries.to_numpy(dtype=np.float32).flatten()
    # return
    return npCurPred


def detMakeTSAdjusts( AdjResids, startInd, lastInd, StartTS, EndTS ):
    """Make 'corrector' adjustments to past month time series.

    All internal HSPF calculations are done in terms of, and internal
    calculation variables use, length == feet and time == seconds.
    OUTDGT1, OUTDGT2, OUTDGT3, OUTDGT4, and OUTDGT5 are in cfs.
    EXIVOL is special and stores in acre-ft per day.

    Parameters
    ----------
    AdjResids : np.ndarray( NUM_MEAS )
        Residuals between adjusted prediction and initial prediction.
    startInd : int
        index for the first day in the adjustment month
    lastInd : int
        index for the last day in the adjustment month
    StartTS: pd.Timestamp
        Starting time stamp for the adjustment month
    EndTS: pd.Timestamp
        Ending time stamp for the adjustment month

    Returns
    -------
    None.

    """
    # imports
    from locaHrchhyd import OUTDGT1, OUTDGT2, OUTDGT3, OUTDGT4, OUTDGT5
    from locaHrchhyd import EXIVOL
    # globals
    global Extracts_Dict, DAILY_INDEX, ConvAFD2CFS, ConvCFS2AFD
    global PumpTrackDF, ExInflowTrackDF
    # parameters
    # locals
    # start
    curNumDays = float( StartTS.days_in_month )
    tfCols = sorted( Aquifers_Dict.keys() )
    rCnt = 0
    for rId in tfCols:
        curResid = float( AdjResids[rCnt] )
        pInfo = Extracts_Dict[rId]
        curPExit = pInfo[0]
        if curResid < 0.0:
            # this means that need to increase pumping to decrease volume
            addPRateAFD = abs( curResid ) / curNumDays
            addPRateCFS = ConvAFD2CFS(addPRateAFD)
            if curPExit == 1:
                OUTDGT1[startInd:(lastInd+1)][rId] += addPRateCFS
                PumpTrackDF[rId].loc[StartTS:EndTS] = (
                    OUTDGT1[startInd:(lastInd+1)][rId] ).flatten()
            elif curPExit == 2:
                OUTDGT2[startInd:(lastInd+1)][rId] += addPRateCFS
                PumpTrackDF[rId].loc[StartTS:EndTS] = (
                    OUTDGT2[startInd:(lastInd+1)][rId] ).flatten()
            elif curPExit == 3:
                OUTDGT3[startInd:(lastInd+1)][rId] += addPRateCFS
                PumpTrackDF[rId].loc[StartTS:EndTS] = (
                    OUTDGT3[startInd:(lastInd+1)][rId] ).flatten()
            elif curPExit == 4:
                OUTDGT4[startInd:(lastInd+1)][rId] += addPRateCFS
                PumpTrackDF[rId].loc[StartTS:EndTS] = (
                    OUTDGT4[startInd:(lastInd+1)][rId] ).flatten()
            elif curPExit == 5:
                OUTDGT5[startInd:(lastInd+1)][rId] += addPRateCFS
                PumpTrackDF[rId].loc[StartTS:EndTS] = (
                    OUTDGT5[startInd:(lastInd+1)][rId] ).flatten()
            # end if
        elif curResid > 0.0:
            # this means to reduce pumping to increase volume
            # if there is not enough pumping to reduce then need to add
            #   external inflow to compensate.
            residPRateAFD = curResid / curNumDays
            if curPExit == 1:
                curDF = pd.DataFrame( OUTDGT1 )
                curDF = curDF[[rId]].copy()
            elif curPExit == 2:
                curDF = pd.DataFrame( OUTDGT2 )
                curDF = curDF[[rId]].copy()
            elif curPExit == 3:
                curDF = pd.DataFrame( OUTDGT3 )
                curDF = curDF[[rId]].copy()
            elif curPExit == 4:
                curDF = pd.DataFrame( OUTDGT4 )
                curDF = curDF[[rId]].copy()
            else:
                curDF = pd.DataFrame( OUTDGT5 )
                curDF = curDF[[rId]].copy()
            # end if
            curDF["dayind"] = DAILY_INDEX
            curDF.set_index("dayind", inplace=True )
            curDF = curDF.rename_axis(None)
            truncDF = curDF.loc[StartTS:EndTS].copy()
            avePumpCFS = float( truncDF[rId].mean() )
            avePumpAFD = ConvCFS2AFD( avePumpCFS )
            if ( avePumpAFD - residPRateAFD ) > 0.1:
                # in this case can just subtract
                residPRateCFS = ConvAFD2CFS(residPRateAFD)
                if curPExit == 1:
                    OUTDGT1[startInd:(lastInd+1)][rId] -= residPRateCFS
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT1[startInd:(lastInd+1)][rId] ).flatten()
                elif curPExit == 2:
                    OUTDGT2[startInd:(lastInd+1)][rId] -= residPRateCFS
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT2[startInd:(lastInd+1)][rId] ).flatten()
                elif curPExit == 3:
                    OUTDGT3[startInd:(lastInd+1)][rId] -= residPRateCFS
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT3[startInd:(lastInd+1)][rId] ).flatten()
                elif curPExit == 4:
                    OUTDGT4[startInd:(lastInd+1)][rId] -= residPRateCFS
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT4[startInd:(lastInd+1)][rId] ).flatten()
                else:
                    OUTDGT5[startInd:(lastInd+1)][rId] -= residPRateCFS
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT5[startInd:(lastInd+1)][rId] ).flatten()
                # end if
            else:
                # this is the case of reducing pumping and adding inflow
                remResidAFD = max( residPRateAFD - avePumpAFD, 0.0 )
                if remResidAFD > 0.1:
                    EXIVOL[startInd:(lastInd+1)][rId] += remResidAFD
                    ExInflowTrackDF[rId].loc[StartTS:EndTS] = (
                        EXIVOL[startInd:(lastInd+1)][rId] ).flatten()
                # end if
                if curPExit == 1:
                    OUTDGT1[startInd:(lastInd+1)][rId] = 0.0
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT1[startInd:(lastInd+1)][rId] ).flatten()
                elif curPExit == 2:
                    OUTDGT2[startInd:(lastInd+1)][rId] = 0.0
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT2[startInd:(lastInd+1)][rId] ).flatten()
                elif curPExit == 3:
                    OUTDGT3[startInd:(lastInd+1)][rId] = 0.0
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT3[startInd:(lastInd+1)][rId] ).flatten()
                elif curPExit == 4:
                    OUTDGT4[startInd:(lastInd+1)][rId] = 0.0
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT4[startInd:(lastInd+1)][rId] ).flatten()
                else:
                    OUTDGT5[startInd:(lastInd+1)][rId] = 0.0
                    PumpTrackDF[rId].loc[StartTS:EndTS] = (
                        OUTDGT5[startInd:(lastInd+1)][rId] ).flatten()
                # end if
            # end if
        # end if
        rCnt += 1
    # end reservoir for
    # all adjusted
    # return
    return


def updatePrevPrediction( lastInd ):
    """

    Parameters
    ----------
    lastInd: INT
        index for the last day in the adjustment month

    Returns
    -------
    None.

    """
    # imports
    # globals
    global PRED_PREV, CUR_MONTH, DAILY_INDEX, MONTHLY_INDEX, PredCorrectDF
    # parameters
    # locals
    # start
    endTS = DAILY_INDEX[lastInd] + pd.Timedelta(days=0.75)
    startTS = MONTHLY_INDEX[CUR_MONTH-1]
    corPred = extractCurMonPred( startTS, endTS, CUR_MONTH-1 )
    PredCorrectDF.loc[startTS] = corPred
    np.copyto( PRED_PREV, corPred )
    # return
    return


def serializeTrackingDFs( out_dir ):
    """Serialize the tracking DataFrames

    Parameters
    ----------
    out_dir : str
        Output directory.

    Returns
    -------
    None.

    """
    # imports
    # globals
    global PredInitDF, PredAdjDF, PredCorrectDF, ResidsDF, SSECovTrackDF
    global PumpTrackDF, ExInflowTrackDF, KGainTrackDF, tITER
    # start
    OutFiler = os.path.normpath( os.path.join( out_dir, "KalmanGainDF_%d.pkl" % tITER) )
    KGainTrackDF.to_pickle( OutFiler )
    OutFiler = os.path.normpath( os.path.join( out_dir, "PStateErrCovDF_%d.pkl" % tITER ) )
    SSECovTrackDF.to_pickle( OutFiler )
    OutFiler = os.path.normpath( os.path.join( out_dir, "PredInitDF_%d.pkl" % tITER ) )
    PredInitDF.to_pickle( OutFiler )
    OutFiler = os.path.normpath( os.path.join( out_dir, "PredAdjDF_%d.pkl" % tITER ) )
    PredAdjDF.to_pickle( OutFiler )
    OutFiler = os.path.normpath( os.path.join( out_dir, "PredCorrectDF_%d.pkl" % tITER ) )
    PredCorrectDF.to_pickle( OutFiler )
    OutFiler = os.path.normpath( os.path.join( out_dir, "ResidualsDF_%d.pkl" % tITER ) )
    ResidsDF.to_pickle( OutFiler )
    OutFiler = os.path.normpath( os.path.join( out_dir, "AdjustedPumpingDF_%d.pkl" % tITER ) )
    PumpTrackDF.to_pickle( OutFiler )
    OutFiler = os.path.normpath( os.path.join( out_dir, "AdjustExInflowDF_%d.pkl" % tITER ) )
    ExInflowTrackDF.to_pickle( OutFiler )
    # return
    return


def extractAllPredfromH5( hStore, tfCols ):
    """Convenience function for testing - extracts all predicted values

    Parameters
    ----------
    hStore : str
        Fully qualified path and filename for H5 file.
    tfCols : list
        List of reservoir IDs to extract.

    For testing:
        hStore = os.path.normpath( os.path.join( simdir, "UCUWCD_AC.h5" ) )

    Returns
    -------
    RRPredMonDF : pd.DataFrame
        Time series of monthly predictions for the reservoirs.

    """
    # imports
    # globals
    # parameters
    # locals
    RRList = list()
    # start
    with pd.HDFStore( hStore ) as store:
        for RId in tfCols:
            cLabel = RId
            rKey = "/RESULTS/RCHRES_%s/HYDR/" % cLabel
            cResDF = store.get(key=rKey)
            cResDF = cResDF[["VOL"]]
            cResDF.columns = [cLabel]
            RRList.append( cResDF )
        # end for
    # end with and close the file
    RRPredDF = pd.concat( RRList, axis=1, join='inner')
    # convert to monthly
    RRPredMonDF = RRPredDF.resample( "MS" ).mean()
    # return
    return RRPredMonDF

