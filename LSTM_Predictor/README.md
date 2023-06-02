# Complex Graph LSTM Predictor

This subdirectory holds the complex graph, LSTM predictor model and associated custom source code.

[**Complex Graph LSTM Predictor**](https://github.com/nmartin198/kfint_watbudg_lstm/tree/main/LSTM_Predictor/CG3BranchSGlog)


## How to load the LSTM Predictor

The LSTM Predictor was developed in Keras in a Conda Python Tensorflow environment.

To load the model and prepare it for predictions:

    import tensorflow as tf
    from tensorflow.keras import layers
    import tensorflow.keras as keras
    import ealstm_cells as EA
    LRATE = 0.001
    NUM_INT_CELLS = 50
    DO_RATE = 0.35
    SEQ_LEN = 183
    BATCH_SIZE = 25
    uMSE = keras.losses.MeanSquaredError( reduction="auto", name="cMSE")
    uMetMAE = keras.metrics.MeanAbsoluteError( name="cMAE" )
    uMetRMSE = keras.metrics.RootMeanSquaredError( name="cRMSE" )
    uAdam = keras.optimizers.Adam( learning_rate=LRATE, name="cAdam" )
    CUST_OBJS = {"cEALSTM" : EA.cEALSTM, "cEALSTMCell" : EA.cEALSTMCell }
    cgModel = keras.models.load_model( *CG3BranchSGLog*, compile=False, custom_objects=CUST_OBJS )
    cgModel.compile( optimizer=uAdam, loss=uMSE, metrics=[uMetMAE, uMetRMSE,], )

