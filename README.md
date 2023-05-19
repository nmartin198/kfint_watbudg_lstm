# kfint_watbudg_lstm
Kalman filter integration of an LSTM predictor to a process-based water budget. This is a custom, 
proof-of-concept integration. A Kalman filter algorithm integrates a process-based water budget 
calculation with a deep learning predictor. The process-based water budget model is the "forward
model." The deep learning predictor is a trained LSTM network model that predicts water levels
in aquifer segments; LSTM predicted water levels are used as "measurements" by the Kalman filter
algorithm.  

## Journal Article

The purpose of this GitHub is to provide the archive of model files and source code for the 
associated journal article.  

* Citation here  


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

[**Additional details on mHSP2**](https://nmartin198.github.io/pyHS2MF6/mHSP2.html)

[**Modified mHSP2 forward model, InitPBased_rpNR.h5**](https://github.com/nmartin198/kfint_watbudg_lstm/tree/main/Model_Files/Original)


## LSTM Predictor

An EA-LSTM predictor for aquifer segment water level elevation was created and trained. This predictor provides water level "measurements" when equipment is malfunctioning or otherwise unavailable.

[**Trained, complex graph LSTM predictor**](https://github.com/nmartin198/kfint_watbudg_lstm/tree/main/LSTM_Predictor)


## Kalman Filter Integration Source Code


**mHSP2** is the HSPF variant used in the [**pyHS2MF6**](https://nmartin198.github.io/pyHS2MF6/)
integrated hydrologic model. It has been separated out from **pyHS2MF6** 
to facilitate standalone use of this program.

**mHSP2** is a port of [HSPsquared](https://github.com/respec/HSPsquared) created 
specifically for coupled simulation with MODFLOW 6.
[HSPsquared](https://github.com/respec/HSPsquared) is an HSPF variant 
that was rewritten in pure Python. **mHSP2** only provides the water 
movement and storage capabilities of [HSPsquared](https://github.com/respec/HSPsquared).

The main difference between **mHSP2** and [HSPsquared](https://github.com/respec/HSPsquared) 
is that **mHSP2** was created with the simulation time loop as 
the main simulation loop to facilitate dynamic coupling to MODFLOW 6. 
HSPF-variants traditionally use an operating module instance loop that 
is executed in routing order as the main simulation loop. This approach 
requires that the time simulation loop be executed for each operating 
module instance.

[**Additional details**](https://nmartin198.github.io/pyHS2MF6/mHSP2.html)


## Getting Started

The Python source code for **mHSP2** is in the [*src*](https://github.com/nmartin198/mHSP2/tree/main/src) 
directory. The best way to start is to go through the 
[HSPF Standalone Example Model](https://nmartin198.github.io/pyHS2MF6/cs_sa_HSPF.html) 
in the **pyHS2MF6** documentation and run that model and verify that obtain 
similar results.

The next step would be to use the [example model](https://github.com/nmartin198/mHSP2/tree/main/example_input) 
included in this repository as a second example case.

There is also a utility [Jupyter Notebook](https://github.com/nmartin198/mHSP2/tree/main/jupyter_notebooks/Conv_pyHSPF_HSP2.ipynb) 
included in this repository that provides an example of going 
from the traditional HSPF inputs of UCI and WDM files to the HDF5 
input file used by **mHSP2**.


## Author

* **Nick Martin** nmartin@swri.org

## License

This project is licensed under the GNU Affero General Public License v.3.0 - see the [LICENSE](LICENSE) file for details.
