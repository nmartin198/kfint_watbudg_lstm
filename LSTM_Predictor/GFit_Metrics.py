#!/usr/bin/python3
""" Standard numpy array 'goodness-of-fit' metrics

These are developed using primarily numpy objects and methods

* acalc_nse: Calculates NSE across all outputs, i.e., average NSE
* acalc_kge: Calculates KGE across all outputs, i.e., average KGE
* unscale_hzst: Unscales predictions for targets that have hydrologic scaling
                with zero threshold support
* vcalc_nse: Calculates NSE for each output and returns a vector of NSE
* vcalc_kge_alpha: Calculates KGE, alpha subcomponent for each output
* vcalc_kge_beta: Calculates KGE, Beta subcomponent for each output
* vcalc_kge_r: Calculates KGE, r subcomponent for each output
* vcalc_kge: Calculates KGE for each output

"""
# Copyright and License
"""
Copyright 2022 Southwest Research Institute

Module Author: Nick Martin <nick.martin@alumni.stanford.edu>

This file, ealstm_cells.py, is a collection of custom Keras extension
methods.

ealstm_cells.py is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ealstm_cells.py is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with ealstm_cells.py.  If not, see <https://www.gnu.org/licenses/>.

"""
import numpy as np
from scipy import stats as sstats


def acalc_nse( obs, predict ):
    """Calculate NSE from observed, obs, and predicted, predict, arrays.

    Calculates a single NSE from all outputs

    Args:
        obs (np.array): observed time series
        predict (np.array): predicted, or simulated, time series
    
    Returns:
        NSE (float): Nash-Sutcliffe-Efficiency value which can range from
                        negative infinity to 1.0
    """
    # check type
    if not isinstance( obs, np.ndarray ):
        raise TypeError("Input `obs` must be numpy arrays!!!")
    # end if
    if not isinstance( predict, np.ndarray ):
        raise TypeError("Input `predict` must be numpy array!!!")
    # end if
    if predict.shape[0] != obs.shape[0]:
        raise ValueError( "Length of `predict` and `obs` must be equal!!!" )
    # end if
    obs = obs.flatten()
    predict = predict.flatten()
    # now are ready to do our calcs.
    resids = obs - predict
    sq_resids = np.square( resids )
    sum_sqresides =sq_resids.sum()
    obs_mean = obs.mean()
    diff_mean = obs - obs_mean
    sq_diffm = np.square( diff_mean )
    sum_diffsq = sq_diffm.sum()
    if abs( sum_diffsq - 0.0 ) > 1e-6:
        mult_denom = 1.0 / sum_diffsq
        right_side = sum_sqresides * mult_denom
    else:
        right_side = -5000.0
    # end if
    # calc
    NSE = 1.0 - right_side
    # return
    return NSE


def acalc_kge( obs, predict ):
    """Calculate KGE from observed, obs, and predicted, predict, arrays.

    Args:
        obs (np.array): observed time series
        predict (np.array): predicted, or simulated, time series
    
    Returns:
        KGE (float): Kling-Gupta Efficiency value which can range from
                        negative infinity to 1.0
    """
    from math import sqrt, pow
    # check type
    if not isinstance( obs, np.ndarray ):
        raise TypeError("Input `obs` must be numpy arrays!!!")
    # end if
    if not isinstance( predict, np.ndarray ):
        raise TypeError("Input `predict` must be numpy array!!!")
    # end if
    if predict.shape[0] != obs.shape[0]:
        raise ValueError( "Length of `predict` and `obs` must be equal!!!" )
    # end if
    obs = obs.flatten()
    predict = predict.flatten()
    # now are ready to do our calcs.
    # alpha term
    N = predict.shape[0]
    obs_mean = obs.mean()
    sim_mean = predict.mean()
    obs_diff = obs - obs_mean
    obs_sqdiff = np.square( obs_diff )
    obs_sum_sqdiff = obs_sqdiff.sum()
    sim_diff = predict - sim_mean
    sim_sqdiff = np.square( sim_diff )
    sim_sum_sqdiff = sim_sqdiff.sum()
    obs_std = sqrt( obs_sum_sqdiff / N )
    sim_std = sqrt( sim_sum_sqdiff / N )
    alpha = obs_std / sim_std
    # beta term
    Beta = sim_mean / obs_mean 
    # r term
    mdiff_simXobs = obs_diff * sim_diff
    sum_mdiff = mdiff_simXobs.sum()
    denom = sqrt( sim_sum_sqdiff ) * sqrt( obs_sum_sqdiff )
    if abs( denom - 0.0 ) > 1.0e-6:
        mult_denom = 1.0 / denom
    else:
        mult_denom = 0.0
    # end if
    r = sum_mdiff * mult_denom
    # calculate ED term
    ED = sqrt( pow( r - 1.0, 2.0 ) + pow( alpha - 1.0, 2.0 ) + 
                pow( Beta - 1.0, 2.0 ) )
    KGE = 1.0 - ED
    # return
    return KGE


def unscale_hzst( preds, ecdf_list ):
    """Unscale predictions whose training and validation targets have 
    hydrologic scaling with zero threshold support (hszt).

    Args:
        preds (np.ndarray): predicted values (Time intervals, Num targets/outputs)
        ecdf_list (list): list of empirical CDFs that used for hszt

    Returns:
        unscaled (np.ndarray): unscaled predictions (Time intervals, Num targets/outputs)
    
    """
    # parameters
    ZMu = 0.0
    ZStd = 1.0
    # start
    if not isinstance( preds, np.ndarray ):
        raise TypeError("Predictions must be a Numpy array!!!")
    # end if
    if not isinstance( ecdf_list, list ):
        raise TypeError("ECDFs must be provided in a list!!!")
    # end if
    NumOuts1 = preds.shape[1]
    NumOuts2 = len( ecdf_list )
    if NumOuts1 != NumOuts2:
        raise ValueError("Unequal outputs of %d and %d provided!!!" % 
                          ( NumOuts1, NumOuts2 ) )
    # end if
    NumOuts = NumOuts1
    NumInts = preds.shape[0]
    unscaled = np.zeros( (NumInts, NumOuts), dtype=np.float32 )
    for jJ in range( NumOuts ):
        curECDF = ecdf_list[jJ]
        cCumZProbs = sstats.norm.cdf( preds[:,jJ], loc=ZMu, scale=ZStd )
        cCumZPercs = cCumZProbs * 100.0
        for kK in range(NumInts):
            curPerc = float( cCumZPercs[kK] )
            unscaled[kK,jJ] = curECDF.retValueForPerc( curPerc )
        # end inner for
    # end outer for
    # return
    return unscaled


def vcalc_nse( obs, predict ):
    """Calculate NSE from observed, obs, and predicted, predict, arrays.

    Calculates an array of NSE values, one for each output

    Args:
        obs (np.array): observed time series (#Num Time Intervals, #Outputs)
        predict (np.array): predicted, or simulated, time series
                            (#Num Time Intervals, #Outputs)
    
    Returns:
        NSE (np.array): Nash-Sutcliffe-Efficiency value which can range from
                        negative infinity to 1.0 (#Outputs)
    """
    # check type
    if not isinstance( obs, np.ndarray ):
        raise TypeError("Input `obs` must be numpy arrays!!!")
    # end if
    if not isinstance( predict, np.ndarray ):
        raise TypeError("Input `predict` must be numpy array!!!")
    # end if
    if predict.shape[0] != obs.shape[0]:
        raise ValueError( "Length of `predict` and `obs` must be equal!!!" )
    # end if
    if len( predict.shape ) != len( obs.shape ):
        raise TypeError("Inputs `obs` and `predict` must have the same rank!!!")
    # end if
    if len( obs.shape ) > 1:
        if predict.shape[1] != obs.shape[1]:
            raise ValueError("Dimensions of `predict` and `obs` must be equal!!!" )
        # end if
    # end if
    # now are ready to do our calcs.
    resids = obs - predict
    sq_resids = np.square( resids )
    sum_sqresides =sq_resids.sum( axis=0 )
    obs_mean = obs.mean( axis=0 )
    diff_mean = obs - obs_mean
    sq_diffm = np.square( diff_mean )
    sum_diffsq = sq_diffm.sum( axis=0 )
    denom = np.where( np.abs( sum_diffsq ) < 0.00001, 0.00001, sum_diffsq )
    right_side = sum_sqresides / denom 
    # calc
    NSE = 1.0 - right_side
    # return
    return NSE


def vcalc_kge_alpha( obs, predict ):
    """Calculate KGE, alpha subcomponent from unscaled predictions.

    Returns an array of alpha values (one for each output)

    Args:
        obs (np.array): observed time series (#time intervals, #outputs)
        predict (np.array): predicted, or simulated, time series (#time intervals, #outputs)
    
    Returns:
        alpha (np.array): KGE alpha subcomponent (#outputs)
    """
    # check type
    if not isinstance( obs, np.ndarray ):
        raise TypeError("Input `obs` must be numpy arrays!!!")
    # end if
    if not isinstance( predict, np.ndarray ):
        raise TypeError("Input `predict` must be numpy array!!!")
    # end if
    if predict.shape[0] != obs.shape[0]:
        raise ValueError( "Length of `predict` and `obs` must be equal!!!" )
    # end if
    if len( predict.shape ) != len( obs.shape ):
        raise TypeError("Inputs `obs` and `predict` must have the same rank!!!")
    # end if
    if len( obs.shape ) > 1:
        if predict.shape[1] != obs.shape[1]:
            raise ValueError("Dimensions of `predict` and `obs` must be equal!!!" )
        # end if
    # end if
    # now are ready to do our calcs.
    # alpha term
    N = obs.shape[0]
    obs_mean = obs.mean( axis=0 )
    sim_mean = predict.mean( axis=0 )
    obs_diff = obs - obs_mean
    obs_sqdiff = np.square( obs_diff )
    obs_sum_sqdiff = obs_sqdiff.sum( axis=0 )
    sim_diff = predict - sim_mean
    sim_sqdiff = np.square( sim_diff )
    sim_sum_sqdiff = sim_sqdiff.sum( axis=0 )
    obs_std = np.sqrt( obs_sum_sqdiff / N )
    sim_std = np.sqrt( sim_sum_sqdiff / N )
    denom = np.where( np.abs( sim_std ) < 0.00001, 0.00001, sim_std )
    alpha = obs_std / denom
    alpha = np.where( (np.abs(obs_std) <= 0.01) & (np.abs(sim_std) <= 0.01), 
                        1.0, alpha )
    return alpha


def vcalc_kge_beta( obs, predict ):
    """Calculate KGE, Beta subcomponent from unscaled predictions.

    Returns an array of Beta values (one for each output)

    Args:
        obs (np.array): observed time series (#time intervals, #outputs)
        predict (np.array): predicted, or simulated, time series (#time intervals, #outputs)
    
    Returns:
        Beta (np.array): KGE Beta subcomponent (#outputs)
    """
    # check type
    if not isinstance( obs, np.ndarray ):
        raise TypeError("Input `obs` must be numpy arrays!!!")
    # end if
    if not isinstance( predict, np.ndarray ):
        raise TypeError("Input `predict` must be numpy array!!!")
    # end if
    if predict.shape[0] != obs.shape[0]:
        raise ValueError( "Length of `predict` and `obs` must be equal!!!" )
    # end if
    if len( predict.shape ) != len( obs.shape ):
        raise TypeError("Inputs `obs` and `predict` must have the same rank!!!")
    # end if
    if len( obs.shape ) > 1:
        if predict.shape[1] != obs.shape[1]:
            raise ValueError("Dimensions of `predict` and `obs` must be equal!!!" )
        # end if
    # end if
    # now are ready to do our calcs.
    # Beta term
    obs_mean = obs.mean( axis=0 )
    sim_mean = predict.mean( axis=0 )
    denom = np.where( np.abs( obs_mean ) < 0.00001, 0.00001, obs_mean )
    Beta = sim_mean / denom
    Beta = np.where( (np.abs(obs_mean) <= 0.01) & (np.abs(sim_mean) <= 0.01),
                      1.0, Beta )
    # return
    return Beta


def vcalc_kge_r( obs, predict ):
    """Calculate KGE, r subcomponent from unscaled predictions.

    Returns an array of r values (one for each output)

    Args:
        obs (np.array): observed time series (#time intervals, #outputs)
        predict (np.array): predicted, or simulated, time series (#time intervals, #outputs)
    
    Returns:
        r (np.array): KGE r subcomponent (#outputs)
    """
    # check type
    if not isinstance( obs, np.ndarray ):
        raise TypeError("Input `obs` must be numpy arrays!!!")
    # end if
    if not isinstance( predict, np.ndarray ):
        raise TypeError("Input `predict` must be numpy array!!!")
    # end if
    if predict.shape[0] != obs.shape[0]:
        raise ValueError( "Length of `predict` and `obs` must be equal!!!" )
    # end if
    if len( predict.shape ) != len( obs.shape ):
        raise TypeError("Inputs `obs` and `predict` must have the same rank!!!")
    # end if
    if len( obs.shape ) > 1:
        if predict.shape[1] != obs.shape[1]:
            raise ValueError("Dimensions of `predict` and `obs` must be equal!!!" )
        # end if
    # end if
    # now are ready to do our calcs.
    # r term
    N = predict.shape[0]
    obs_mean = obs.mean( axis=0 )
    sim_mean = predict.mean( axis=0 )
    obs_diff = obs - obs_mean
    obs_sqdiff = np.square( obs_diff )
    obs_sum_sqdiff = obs_sqdiff.sum( axis=0 )
    sim_diff = predict - sim_mean
    sim_sqdiff = np.square( sim_diff )
    sim_sum_sqdiff = sim_sqdiff.sum( axis=0 )
    mdiff_simXobs = obs_diff * sim_diff
    sum_mdiff = mdiff_simXobs.sum( axis=0 )
    denom1 = np.sqrt( sim_sum_sqdiff ) * np.sqrt( obs_sum_sqdiff )
    denom = np.where( np.abs( denom1 ) < 0.00001, 0.00001, denom1 )
    r = sum_mdiff / denom
    # return
    return r


def vcalc_kge( obs, predict ):
    """Calculate KGE from unscaled predictions.

    Returns an array

    Args:
        obs (np.array): observed time series (#time intervals, #outputs)
        predict (np.array): predicted, or simulated, time series (#time intervals, #outputs)
    
    Returns:
        KGE (np.array): KGE (#outputs)
    """
    # check type
    if not isinstance( obs, np.ndarray ):
        raise TypeError("Input `obs` must be numpy arrays!!!")
    # end if
    if not isinstance( predict, np.ndarray ):
        raise TypeError("Input `predict` must be numpy array!!!")
    # end if
    if predict.shape[0] != obs.shape[0]:
        raise ValueError( "Length of `predict` and `obs` must be equal!!!" )
    # end if
    if len( predict.shape ) != len( obs.shape ):
        raise TypeError("Inputs `obs` and `predict` must have the same rank!!!")
    # end if
    if len( obs.shape ) > 1:
        if predict.shape[1] != obs.shape[1]:
            raise ValueError("Dimensions of `predict` and `obs` must be equal!!!" )
        # end if
    # end if
    # now are ready to do our calcs.
    alpha = vcalc_kge_alpha( obs, predict )
    Beta = vcalc_kge_beta( obs, predict )
    r = vcalc_kge_r( obs, predict )
    # calculate ED term
    ED = np.sqrt( np.power( r - 1.0, 2.0 ) + 
                  np.power( alpha - 1.0, 2.0 ) + 
                  np.power( Beta - 1.0, 2.0 ) )
    KGE = 1.0 - ED
    # return
    return KGE


#EOF