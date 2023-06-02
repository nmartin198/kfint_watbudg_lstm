# Dynamic Assimilation of Deep Learning Predictions to a Process-Based Water Budget

Kalman filter integration of an LSTM predictor to a process-based water budget. This is a custom, 
proof-of-concept integration. A Kalman filter algorithm integrates a process-based water budget 
calculation with a deep learning predictor. The process-based water budget model is the "forward
model." The deep learning predictor is a trained LSTM network model that predicts water levels
in aquifer segments; LSTM predicted water levels are used as "measurements" by the Kalman filter
algorithm.  

## Journal Article

The purpose of this GitHub is to provide the archive of model files and source code for the 
associated journal article.  

* TBD - Citation here  


## Process-based Water Budget Model

An existing, process-based water budget model is modified for use as the "forward model" in this
study. The existing model, which was modified for use here, is the Uvalde County Underground Water
Conservation District (UCUWCD) Water Balance Model.  

The modified, forward model is an **mHSP2** model. **mHSP2** is a port of 
[HSPsquared](https://github.com/respec/HSPsquared) created specifically for coupled simulation 
with MODFLOW 6. [HSPsquared](https://github.com/respec/HSPsquared) is an HSPF variant that 
was rewritten in pure Python. **mHSP2** only provides the water movement and storage 
capabilities of [HSPsquared](https://github.com/respec/HSPsquared).  

The main difference between **mHSP2** and [HSPsquared](https://github.com/respec/HSPsquared) 
is that **mHSP2** was created with the simulation time loop as 
the main simulation loop to facilitate dynamic coupling to MODFLOW 6. 
HSPF-variants traditionally use an operating module instance loop that 
is executed in routing order as the main simulation loop. This approach 
requires that the time simulation loop be executed for each operating 
module instance.  

**mHSP2** was employed as a standalone HSPF variant for the UCUWCD Water Balance Model because
an issues was uncovered with regular HSPF for representation of time-dependent outflows from
RCHRES. **mHSP2** was then used in this Kalman filter integration because the source code was
available and could be modified to include the Kalman filter update.  

[**mHSP2 Source Code**](https://github.com/nmartin198/mHSP2)

[**Additional details on mHSP2**](https://nmartin198.github.io/pyHS2MF6/mHSP2.html)

[**Modified mHSP2 forward model, InitPBased_rpNR.h5**](https://github.com/nmartin198/kfint_watbudg_lstm/tree/main/Model_Files/Original)


## LSTM Predictor

An EA-LSTM predictor for aquifer segment water level elevation was created and trained. This predictor provides water level "measurements" when equipment is malfunctioning or otherwise unavailable.

[**Trained, complex graph LSTM predictor**](https://github.com/nmartin198/kfint_watbudg_lstm/tree/main/LSTM_Predictor)


## Kalman Filter Integration Source Code

The integration code is a modified version of **mHSP2**. Modifications to **mHSP2** are minor and consist of insertion of calls to functions and routines in the file KF_UCUWCD_Custom.py. This module contains the Kalman filter calculations are routines for outputting and tracking important Kalman filter implementation parameters.

[**Kalman Filter Integration Source Code**](https://github.com/nmartin198/kfint_watbudg_lstm/tree/main/KF_mHSP2)

[**KF_UCUWCD_Custom.py**](https://github.com/nmartin198/kfint_watbudg_lstm/tree/main/KF_mHSP2/KF_UCUWCD_Custom.py)


## Model Files

All model input files and example output files are available in the **Model Files** subdirectory.

[**Model Files**](https://github.com/nmartin198/kfint_watbudg_lstm/tree/main/Model_Files)


## Author

* **Nick Martin** nmartin@swri.org

## License

This project is licensed under the GNU Affero General Public License v.3.0 - see the [LICENSE](LICENSE) file for details.
