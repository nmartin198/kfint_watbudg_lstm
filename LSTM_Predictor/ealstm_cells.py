#!/usr/bin/python3
""" Module with EA-LSTM extensions for Keras/Tensorflow

This module requires keras and tensorflow. It provides classes for extending
the basic Keras RNN to be an Entity-Aware (EA), Long Short-Term Memomory (LSTM)
representation. EA-LSTM allows for use of static and dynamic inputs, where
the dynamic inputs are the standard RNN sequences or time series.

Need to a custom layer for EALSTM cells and custom cells that are EALSTM

Follow the examples at:

https://keras.io/guides/making_new_layers_and_models_via_subclassing/
https://keras.io/guides/working_with_rnns/#working-with-rnns

The 'Input' layer will not accept shapes that are not a tuple and using 
multiple 'Input' layers (one for each input type) contains a hidden 
requirement that both 'Input's use the same batch size. As a result, 
static features were made a property of cEALSTMCell and are assigned
after the cell is created. This allows use of all standard keras and
TensorFlow model building, trainining, and fitting functionality.

In addition to EA-LSTM, this module also includes Keras custom metrics
and loss functions.

Metrics
  Class Form
    * NashSutcliffeEfficiency: Nash–Sutcliffe Efficiency (NSE)
    * KGBeta: Kling–Gupta Efficiency, Beta subcomponent
    * KGAlpha: Kling–Gupta Efficiency, Alpha subcomponent
    * KGr: Kling–Gupta Efficiency, r subcomponent
    * KGEff: Kling–Gupta Efficiency (KGE)
    * NashSutcliffeEfficiency_HSZT: NSE with hydrologic scaling and zero threshold support (hszt)
    * KGBeta_HSZT: Kling–Gupta Efficiency, Beta subcomponent with hszt
    * KGAlpha_HSZT: Kling–Gupta Efficiency, Alpha subcomponent with hszt
    * KGr_HSZT: Kling–Gupta Efficiency, r subcomponent with hszt
    * KGEff_HSZT: Kling–Gupta Efficiency (KGE) with hszt


Losses
  Functional Form
    * nse: NSE
    * nse_star: NSE modified for multiple subbasins after Kratzert et al. (2018)
    * kg_Beta: KGE, Beta subcomponent
    * kg_alpha: KGE, Alpha subcomponent
    * kg_r: KGE, r subcomponent
    * kg_eff: KGE
    * mse_hszt: mean square error (MSE) with hydrologic scaling and zero threshold support
    * nse_hszt: NSE with hydrologic scaling and zero threshold support
    * kg_beta_hszt: KGE, Beta subcomponent with hydrologic scaling and zero threshold
    * kg_alpha_hszt: KGE, alpha subcomponent with hydrologic scaling and zero threshold
    * kg_r_hszt: KGE, r subcomponent with hydrologic scaling and zero threshold support
    * kge_hszt: KGE with hydrologic scaling and zero threshold support
  
  Class Form
    * NSEStar: NSE modified for multiple subbasins using nse_star
    * NSE: NSE using nse
    * KEffLoss: KGE using kg_eff
    * MSE_HSZT: MSE with support for hydrologic scaling and zero threshold

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
# imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.backend import ContextValueCache
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.util import dispatch
from tensorflow.python.framework import ops

# need to make a custom layer and custom LSTM cell to implement Entity-Aware LSTM
#  in keras/TF. Both the custom layer class and the custom LSTM cell class need
#  inherent from layers.Layer.
#
# In Keras - The Layer class: the combination of state (weights) and some computation. 
# A layer encapsulates both a state (the layer's "weights") and a transformation from 
# inputs to outputs (a "call", the layer's forward pass).
# weights are automatically tracked by the layer upon being set as layer attributes with
#   'trainable=True'. The add_weight() method is the shortcut approach for adding
#   weights as layer attributes.
# In Keras API, recommend creating layer weights in the build(self, inputs_shape) method 
#    of your layer so that can create the size/shape of the weights once this is known.
# 
# For generic Layer customizations -
# Implement a get_config() method to enable serialization
# Can add losses (add_loss()) and metrics (add_metric()) to layers
#
# Look at the -
#   Privileged training and mask arguments in the call() method - will likely need these
#     for training to work correctly.
# For the custom cell class extend the functionality from LSTMCell but need to
#    haver Layer as the super ...
class DropoutRNNCellMixin:
  """Object that hold dropout related fields for RNN Cell from keras.

  This class is not a standalone RNN cell. It is supposed to be used with a 
  RNN cell by multiple inheritance. Any cell that mix with class should have 
  following fields:
    dropout: a float number within range [0, 1). The ratio that the input
      tensor need to dropout.
    recurrent_dropout: a float number within range [0, 1). The ratio that the
      recurrent state weights need to dropout.
  This object will create and cache created dropout masks, and reuse them for
  the incoming data, so that the same mask is used for every batch input.
  """

  def __init__(self, *args, **kwargs):
    self._create_non_trackable_mask_cache()
    super(DropoutRNNCellMixin, self).__init__(*args, **kwargs)

  def _create_non_trackable_mask_cache(self):
    """Create the cache for dropout and recurrent dropout mask.

    Note that the following two masks will be used in "graph function" mode,
    e.g. these masks are symbolic tensors. In eager mode, the `eager_*_mask`
    tensors will be generated differently than in the "graph function" case,
    and they will be cached.

    Also note that in graph mode, we still cache those masks only because the
    RNN could be created with `unroll=True`. In that case, the `cell.call()`
    function will be invoked multiple times, and we want to ensure same mask
    is used every time.

    Also the caches are created without tracking. Since they are not picklable
    by python when deepcopy, we don't want `layer._obj_reference_counts_dict`
    to track it by default.
    """
    self._dropout_mask_cache = ContextValueCache(self._create_dropout_mask)
    self._recurrent_dropout_mask_cache = ContextValueCache( 
                                        self._create_recurrent_dropout_mask)

  def reset_dropout_mask(self):
    """Reset the cached dropout masks if any.

    This is important for the RNN layer to invoke this in it `call()` method so
    that the cached mask is cleared before calling the `cell.call()`. The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._dropout_mask_cache.clear()

  def reset_recurrent_dropout_mask(self):
    """Reset the cached recurrent dropout masks if any.

    This is important for the RNN layer to invoke this in it call() method so
    that the cached mask is cleared before calling the cell.call(). The mask
    should be cached across the timestep within the same batch, but shouldn't
    be cached between batches. Otherwise it will introduce unreasonable bias
    against certain index of data within the batch.
    """
    self._recurrent_dropout_mask_cache.clear()

  def _create_dropout_mask(self, inputs, training, count=1):
    return _generate_dropout_mask( tf.ones_like(inputs),
                                   self.dropout,
                                   training=training,
                                   count=count )

  def _create_recurrent_dropout_mask(self, inputs, training, count=1):
    return _generate_dropout_mask( tf.ones_like(inputs),
                                   self.recurrent_dropout,
                                   training=training,
                                   count=count)

  def get_dropout_mask_for_cell(self, inputs, training, count=1):
    """Get the dropout mask for RNN cell's input.

    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.

    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.dropout == 0:
      return None
    init_kwargs = dict(inputs=inputs, training=training, count=count)
    return self._dropout_mask_cache.setdefault(kwargs=init_kwargs)

  def get_recurrent_dropout_mask_for_cell(self, inputs, training, count=1):
    """Get the recurrent dropout mask for RNN cell.

    It will create mask based on context if there isn't any existing cached
    mask. If a new mask is generated, it will update the cache in the cell.

    Args:
      inputs: The input tensor whose shape will be used to generate dropout
        mask.
      training: Boolean tensor, whether its in training mode, dropout will be
        ignored in non-training mode.
      count: Int, how many dropout mask will be generated. It is useful for cell
        that has internal weights fused together.
    Returns:
      List of mask tensor, generated or cached mask based on context.
    """
    if self.recurrent_dropout == 0:
      return None
    init_kwargs = dict(inputs=inputs, training=training, count=count)
    return self._recurrent_dropout_mask_cache.setdefault(kwargs=init_kwargs)

  def __getstate__(self):
    # Used for deepcopy. The caching can't be pickled by python, since it will
    # contain tensor and graph.
    state = super(DropoutRNNCellMixin, self).__getstate__()
    state.pop('_dropout_mask_cache', None)
    state.pop('_recurrent_dropout_mask_cache', None)
    return state

  def __setstate__(self, state):
    state['_dropout_mask_cache'] = ContextValueCache(self._create_dropout_mask)
    state['_recurrent_dropout_mask_cache'] = ContextValueCache(
                                        self._create_recurrent_dropout_mask)
    super(DropoutRNNCellMixin, self).__setstate__(state)


class cEALSTMCell(DropoutRNNCellMixin, keras.layers.Layer):
  """Custom cell class for the EA-LSTM layer.

  Args:
    units: Positive integer, dimensionality of the output space; number of internal cells
    num_static: Number of static features
    num_dynamic: Number of dynamic features
    static_features: np.array of static features with shape (1, num_static)
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    recurrent_initializer: Initializer for the `recurrent_kernel`
      weights matrix,
      used for the linear transformation of the recurrent state.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et al., 2015](
        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    recurrent_regularizer: Regularizer function applied to
      the `recurrent_kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    recurrent_constraint: Constraint function applied to
      the `recurrent_kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.

  Call arguments:
    inputs: A 2D tensor that is dynamic inputs only
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               num_static,
               num_dynamic,
               static_features=None,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    # keep the check for acceptable number of outputs
    if units < 0:
      raise ValueError(f'Received an invalid value for argument `units`, '
                       f'expected a positive integer, got {units}.')
    # end if
    # do the initialization of super
    super(cEALSTMCell, self).__init__(**kwargs)
    # now set attributes from the init pass; keep most of these from LSTM
    self.units = units
    self.num_static = num_static
    self.num_dynamic = num_dynamic
    self.activation = keras.activations.get(activation)
    self.recurrent_activation = keras.activations.get(recurrent_activation)
    self.use_bias = use_bias
    self.kernel_initializer = keras.initializers.get(kernel_initializer)
    self.recurrent_initializer = keras.initializers.get(recurrent_initializer)
    self.bias_initializer = keras.initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias
    self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
    self.bias_regularizer = keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = keras.constraints.get(kernel_constraint)
    self.recurrent_constraint = keras.constraints.get(recurrent_constraint)
    self.bias_constraint = keras.constraints.get(bias_constraint)
    # now apply the dropout(s)
    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    # set the sizes that can now - this is the size of the outputs. A list is
    #   used because have two state tensors - 0: memory state, h; 1: carry state, c
    self.state_size = tuple( [self.units, self.units] )
    self.output_size = self.units
    # now initialize the static_features and sequence length to None
    if static_features is None:
      self.static_features = static_features
    else:
      # this should only happen when load from a saved model.
      if isinstance( static_features, list ):
        # static features get serialized to a nested list.
        static_features = np.array( static_features, dtype=np.float32 )
      # use our assign method
      self.assign_static( static_features )
    # return
    return
  
  def assign_static( self, npArray ):
    """Assign the static features as a property of the cells.
    
    Args:
      npArray (np.array): must have shape(1, self.num_static)

    """
    # make sure that got a numpy array
    if not isinstance( npArray, np.ndarray ):
      raise TypeError('cEALSTMCell assign_static method must receive a numpy '
                       f'array as the argument!!! Received {type(npArray)}')
    # end if
    # verify the array shape
    if len( npArray.shape ) != 2:
      raise ValueError('cEALSTMCell assign_static method must receive a numpy '
                       'array with shape length of 2 (rank 2). Received length '
                       '%d' % len( npArray.shape ) )
    # end if
    dim0 = npArray.shape[0]
    dim1 = npArray.shape[1]
    if dim0 != 1:
      raise ValueError('cEALSTMCell assign_static should receive an array that '
                       'has length 1 for dimension 0. Received length %d' % dim0 )
    # end if
    if dim1 != self.num_static:
      raise ValueError('cEALSTMCell assign_static should receive an array that '
                       'has length %d for dimension 1, equal to num_static. '
                       'Instead received dimension 1 length of %d' % 
                       ( self.num_static, dim1 ) )
    # end if
    # now that the checks are done. Assign a tensor to our attribute using the 
    #   numpy array.
    self.static_features = npArray
    # return
    return

  def build(self, input_shape):
    """Dedicated and required state creation method for RNN.
    
    Custom implementation of the required 'build' method. This is where
    we create the weights for the cell so that can use flexible sizing
    in the class.

    Because we are doing EALSTM, have dynamic and static inputs. Dynamic 
    inputs correspond to the input_shape arg. Static inputs need to be 
    assigned to class property/attribute using the *assign_static* method.

    Args:
      input_shape (tf.TensorShape): shape for dynamic inputs

    """
    dynamic_dims = input_shape
    num_dynamic = dynamic_dims[-1]
    # do some checks
    if self.num_dynamic != num_dynamic:
      raise ValueError( "cEALSTMCell number of dynamic features is unequal between "
                        "initialization, %d, and build, %d " % 
                        (self.num_dynamic, num_dynamic) )
    # end if
    seq_len = dynamic_dims[1]
    if seq_len is None:
      # this is an error
      raise ValueError( "cEALSTMCell build method must receive sequence length from "
                        "dynamic Inputs layer." )
    # end if
    # now that have our dimensions create our weights. Use add_weight() to add
    #    trainable weights for static, dynamic, and recurrent dynamic.
    self.W_s = self.add_weight( shape=(self.num_static, self.units),
                  name='W_static',
                  initializer=self.kernel_initializer,
                  regularizer=self.kernel_regularizer,
                  constraint=self.kernel_constraint,
                  trainable=True )
    self.W_dynamic = self.add_weight( shape=(self.num_dynamic, self.units * 3),
                                      name='W_dynamic',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True )
    self.W_recdyn = self.add_weight( shape=(self.units, self.units * 3),
                                      name='W_re_dynamic',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True )
    # keep the stuff for implementing bias
    if self.use_bias:
      if self.unit_forget_bias:
        # set the bias initialization function
        def bias_initializer(_, *args, **kwargs):
          return keras.backend.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              keras.initializers.get('ones')((self.units,), *args, **kwargs),
              self.bias_initializer((self.units,), *args, **kwargs),
          ])
      else:
        # set the bias intialization function
        bias_initializer = self.bias_initializer
      # end if
      self.bias = self.add_weight( shape=(self.units * 3,),
                                   name='bias_dynamic',
                                   initializer=bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint,
                                   trainable=True )
      self.bias_static = self.add_weight( shape=(self.units,),
                                          name='bias_static',
                                          initializer=self.bias_initializer,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          trainable=True )
    else:
      self.bias = None
      self.bias_static = None
    # end if
    self.built = True
    # return
    return

  def call(self, inputs, states, training=None):
    """Dedicated and required call method for RNN

    This method must be customized/overridden for a custom RNN
    cell. In this case, our 'inputs' and 'states' will be somewhat
    custom because we have 'static' and 'dynamic' inputs and we have
    cell memory states (h) and carryover states (c).

    Args:
      inputs (tf.Tensor): providing one time step for batch and
                          and dynamic features size..    
    """
    # extract the different types of state
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state
    # process inputs
    input_dyn = inputs
    batch_size = input_dyn.shape[0]
    # get the masks that we need
    dp_mask = self.get_dropout_mask_for_cell( input_dyn, training, count=3 )
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell( h_tm1, training, 
                                                            count=3 )
    # calculate our x_s terms; this is here in case try to do 'dropout' with
    #   static inputs
    inputs_i = keras.backend.dot( tf.ones( (batch_size, 1), dtype=keras.backend.floatx() ),
                                  tf.convert_to_tensor( self.static_features, keras.backend.floatx() ) )
      # calculate our dynamic inputs taking into consideration dropout
    if ( 0 < self.dropout < 1.0 ) and training:
      inputs_f = input_dyn * dp_mask[0]
      inputs_g = input_dyn * dp_mask[1]
      inputs_o = input_dyn * dp_mask[2]
    else:
      inputs_f = input_dyn
      inputs_g = input_dyn
      inputs_o = input_dyn
    # end if
    # get our static weights
    k_i = self.W_s
    # get our dynamic weights by splitting our structure
    k_f, k_g, k_o = tf.split( self.W_dynamic, num_or_size_splits=3, 
                              axis=1 )
    # calculate our dot products
    x_i = keras.backend.dot( inputs_i, k_i )
    x_f = keras.backend.dot( inputs_f, k_f )
    x_g = keras.backend.dot( inputs_g, k_g )
    x_o = keras.backend.dot( inputs_o, k_o )
    # get our recurrent weights
    u_f, u_g, u_o = tf.split( self.W_recdyn, num_or_size_splits=3, 
                              axis=1 )
    # calculate the U terms, get the h-1 terms
    if ( 0 < self.recurrent_dropout < 1.0 ) and training:
      h_tm1_f = h_tm1 * rec_dp_mask[0]
      h_tm1_g = h_tm1 * rec_dp_mask[1]
      h_tm1_o = h_tm1 * rec_dp_mask[2]
    else:
      h_tm1_f = h_tm1
      h_tm1_g = h_tm1
      h_tm1_o = h_tm1
    # end if
    # calculate U terms
    U_f = keras.backend.dot( h_tm1_f, u_f )
    U_g = keras.backend.dot( h_tm1_g, u_g )
    U_o = keras.backend.dot( h_tm1_o, u_o )
    # add our terms together
    x_f = x_f + U_f
    x_g = x_g + U_g
    x_o = x_o + U_o
    # add in the bias terms
    if self.use_bias:
      b_i = self.bias_static
      b_f, b_g, b_o = tf.split( self.bias, num_or_size_splits=3, axis=0 )
      x_i = keras.backend.bias_add(x_i, b_i)
      x_f = keras.backend.bias_add(x_f, b_f)
      x_g = keras.backend.bias_add(x_g, b_g)
      x_o = keras.backend.bias_add(x_o, b_o)
    # end if
    i_stat = self.recurrent_activation( x_i )
    f_t = self.recurrent_activation( x_f )
    g_t = self.activation( x_g )
    o_t = self.recurrent_activation( x_o )
    # calculate our state values
    c_t = ( f_t * c_tm1 ) + ( i_stat * g_t )
    h_t = o_t * self.activation( c_t )
    # return
    return h_t, [h_t, c_t]

  def get_config(self):
    """Required method to implement serialization.
    
    """
    config = {
        'units':
            self.units,
        'num_static':
            self.num_static,
        'num_dynamic':
            self.num_dynamic,
        'static_features':
            self.static_features,
        'activation':
            keras.activations.serialize(self.activation),
        'recurrent_activation':
            keras.activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            keras.initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            keras.initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            keras.initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            keras.regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            keras.regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            keras.regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            keras.constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            keras.constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            keras.constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
    }
    config.update({})
    base_config = super(cEALSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class cEALSTM(keras.layers.Layer):
  """Custom, Entity Aware Long Short-Term Memory (EA-LSTM) layer.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  Args:
    cell: Must be an instance of cEALSTMCell; units or hidden size is specified
      as part of 'cell'creation
    return_sequences: Boolean. Whether to return the last output. in the output
      sequence, or the full sequence. Default: `False`.
    stateful: Boolean (default `False`). If True, the last state for each sample
      at index i in a batch will be used as initial state for the sample of
      index i in the following batch.

  Call arguments:
    inputs: A 3D tensor with shape `[batch, timesteps, feature]`.
    mask: Will be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used (optional, defaults to `None`).
    initial_state: List of initial state tensors to be passed to the first
      call of the cell (optional, defaults to `None` which causes creation
      of zero-filled initial state tensors).
  """
  def __init__(self,
               cell,
               return_sequences=False,
               stateful=False,
               **kwargs):
    # Check our cell
    if not isinstance( cell, cEALSTMCell ):
      raise TypeError( 'Only cEALSTMCell cells work with the cEALSTM layer!!!')
    # end if
    # do super
    super(cEALSTM, self).__init__(**kwargs)
    # now set instance attributes. Only keep those that might need to use.
    self.cell = cell
    self.stateful = stateful
    self.return_sequences = return_sequences
    # initialize states and batch size to zeros
    self.states = None
    self.batch_size = None
    self.def_outputs = None
    # end __init__
  
  @property
  def units(self):
    return self.cell.units
  
  @property
  def num_static(self):
    return self.cell.num_static

  @property
  def num_dynamic(self):
    return self.cell.num_dynamic
  
  @property
  def state_size(self):
    return self.cell.state_size
  
  @property
  def output_size(self):
    return self.cell.output_size
  
  @property
  def static_features(self):
    return self.cell.static_features

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  def get_config(self):
    config = {
        'cell':
            self.cell,
        'return_sequences':
            self.return_sequences,
        'stateful':
            self.stateful,
        'batch_size':
            self.batch_size,
    }
    config.update()
    base_config = super(cEALSTM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  
  def build(self, input_shape):
    """Customization of required build method for EALSTM layer.
    
    Required that will use 1 keras.Input layer to handle dynamic inputs.
    The static inputs are given to the cEALSTMCell instance and become
    a property or attribute of the cells. The Input layer specification
    also must contain the optional batch_size= specification so that
    do not have to add all of the extra TypeSpec and so forth.

    input_dyn = keras.Input(shape=(Sample_Len, dyn_features), 
                            batch_size=batch_size)
    outputs = ealstm( input_dyn )

    Args:
      input_shape (tf.TensorShape): TensorShape for dynamic inputs.
        Time major is not supported so the Tensorshape should always
        be ( batch_size, sequence_length, number_dynamic_features )
    
    """
    # confirm that we receive the formats that are expecting.
    if not isinstance( input_shape, tf.TensorShape ):
      raise TypeError( "cEALSTM, ea lstm layer only accepts inputs created using "
                       "keras.Input layer which will provide a TensorShape. Received %s " %
                       type(input_shape) )
    # end if
    # confirm that have a batch size specified ...
    if input_shape[0] is None:
      raise ValueError("cEALSTM, EA LSTM layer only accepts inputs created using "
                       "keras.Input layer with batch_size specified. Please add " 
                       "batch_size=batch_size to the Input layer specification.")
    # end if
    batch_size = input_shape[0]
    self.batch_size = batch_size
    req_dynamic_TS = tf.TensorShape( [ self.batch_size, input_shape[1], input_shape[2] ] )
    # end if
    adj_input_shape = req_dynamic_TS
    # allow cell (if layer) to build now the available shape information.
    if isinstance(self.cell, keras.layers.Layer) and not self.cell.built:
      with keras.backend.name_scope(self.cell.name):
        self.cell.build(adj_input_shape)
        self.cell.built = True
      # end with
    # end if
    # check if stateful and need to reset_states
    if self.stateful:
      self.reset_states()
    # end if
    self.built = True

  def call(self, inputs, mask=None, training=None, initial_state=None):
    """Required call function borrowed from v2 LSTM.

    Modified to remove everything that unneeded and unused.

    Args:
      inputs (tf.Tensor): dynamic inputs should have shape (batch_size, 
        sequence_len, num_dynamic_features )
      mask (None): not used 
      training (boolean): should layer behave in training mode or in 
        inference mode. This argument is passed to the cell when calling 
        it. This is only relevant if `dropout` or `recurrent_dropout` is 
        used (optional, defaults to `None`).

    """
    # mask is not supported here because will always do based on the dropout
    if mask:
      mask = None
    # end if
    # The input should be dense, padded with zeros. So no ragged input
    #      considerations
    # Basic input processing to determine where we are in execution ...
    dynamic_inputs = tf.nest.flatten(inputs)[0]
    dynamic_shape = dynamic_inputs.shape
    batch_size = dynamic_shape[0]
    timesteps = dynamic_shape[1]
    # check the batch size and timesteps
    if batch_size != self.batch_size:
      raise ValueError("Batch size must be specified as an argument to "
                       "keras.Input(shape= , batch_size= ). Batch size "
                       "passed with Input is %d. Batch size in EALSTM "
                       "call is %d." % (self.batch_size, batch_size) )
    # end if
    # check if this is the first time that call has been called using
    #   self.def_outputs
    if self.def_outputs is None:
      self.def_outputs = self.makeDefaultOutputs( self.cell.output_size, 
                                                  timesteps, batch_size=self.batch_size )
      if self.states is None:
        self.states = _generate_zero_filled_state( self.batch_size, 
                                                   self.cell.state_size,
                                                   keras.backend.floatx() )
      # end inner if
      return self.def_outputs
    # end if
    # this check is probably not needed.
    if self.states is None:
      self.states = _generate_zero_filled_state( self.batch_size, 
                                                 self.cell.state_size,
                                                 keras.backend.floatx() )
    # end if
    # Do the _process_inputs functionality to set the state values
    if self.stateful:
      if initial_state is not None:
        # When layer is stateful and initial_state is provided, check if the
        # recorded state is same as the default value (zeros). Use the recorded
        # state if it is not same as the default.
        non_zero_count = tf.add_n([tf.math.count_nonzero(s)
                                   for s in tf.nest.flatten(self.states)])
        # Set strict = True to keep the original structure of the state.
        initial_state = tf.compat.v1.cond(non_zero_count > 0,
                                          true_fn=lambda: self.states,
                                          false_fn=lambda: initial_state,
                                          strict=True)
      else:
        initial_state = self.states
      # end inner if
      initial_state = tf.nest.map_structure(
          # When the layer has a inferred dtype, use the dtype from the cell.
          lambda v: tf.cast(v, keras.backend.floatx() ), initial_state )
    elif initial_state is None:
      initial_state = _generate_zero_filled_state( self.batch_size,
                                                   self.cell.state_size, 
                                                   keras.backend.floatx() )
    # end if
    # have the cells reset their drop out masks before moving on to entire batch calculations
    self.cell.reset_dropout_mask()
    self.cell.reset_recurrent_dropout_mask()
    # start of backend.rnn functionality
    # Unstack the batch dimension. Switch indices in the dynamic inputs and
    #   unstack in the sequence length/time step direcion
    # local function to swap ..
    def swap_batch_timestep(input_t):
      # Swap the batch and timestep dim for the incoming tensor.
      axes = list(range(len(input_t.shape)))
      axes[0], axes[1] = 1, 0
      return tf.compat.v1.transpose(input_t, axes)
    # now swap
    inputs = tf.nest.map_structure(swap_batch_timestep, dynamic_inputs)
    states = tuple( initial_state )
    # input_t provides list with length, seq_len, of tensors with shape 
    #     (batch_size, num_dynamic_features)
    input_t = tf.unstack( inputs )
    # now are ready to go through timesteps loop
    # do the unroll time steps here
    successive_states = []
    successive_outputs = []
    # loop across time steps
    for i in range(timesteps):
      inp = input_t[i]
      output, new_states = self.cell.call( inp, states, 
                                           training=training )
      states = tuple( new_states )
      successive_outputs.append(output)
      successive_states.append(states)
    # end for
    last_output = successive_outputs[-1]
    new_states = successive_states[-1]
    outputs = tf.stack(successive_outputs)
    # now remap shape
    outputs = tf.nest.map_structure(swap_batch_timestep, outputs)
    #   backend.rnn returns ( last_output, outputs, new_states )
    # end of backend.rnn functionality
    # now do the lstm
    if self.stateful:
      updates = [
          tf.compat.v1.assign(self_state, tf.cast(state, self_state.dtype))
          for self_state, state in zip(self.states, new_states) ]
      self.add_update(updates)
    # end if
    # check if need to return sequences ..
    if self.return_sequences:
      output = outputs
    else:
      output = last_output
    # end if
    return output
  
  def reset_states(self, states=None):
    """Reset the recorded states for a stateful RNN layer from keras.

    Can only be used when RNN layer is constructed with `stateful` = `True`.

    Args:
      states: Numpy arrays that contains the value for the initial state, which
        will be feed to cell at the first time step. When the value is None,
        zero filled numpy array will be created based on the cell state size.

    Raises:
      AttributeError: When the layer is not stateful.
      ValueError: When the batch size of the layer is unknown.
      
    """
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    # end if
    if self.batch_size is None:
      raise ValueError( "Use the functional API and specify the batch size by passing a "
                        "`batch_shape` argument to your dynamic Input layer." )
    # end if
    batch_size = self.batch_size
    if self.states is None:
      # In this case need to do initialization
      flat_init_state_values = tf.nest.flatten(_generate_zero_filled_state(
                                  batch_size, self.cell.state_size, keras.backend.floatx()))
      # end if
      flat_states_variables = tf.nest.map_structure( keras.backend.variable, 
                                                     flat_init_state_values )
      self.states = tf.nest.pack_sequence_as( self.cell.state_size,
                                              flat_states_variables )
    elif states is None:
      # this happens when passed from build
      for state, size in zip(tf.nest.flatten(self.states),
                             tf.nest.flatten(self.cell.state_size)):
        keras.backend.set_value( state,
                np.zeros([batch_size] + tf.TensorShape(size).as_list()))
      # end for
    else:
      # this is what happens during training/eval loops
      flat_states = tf.nest.flatten(self.states)
      flat_input_states = tf.nest.flatten(states)
      if len(flat_input_states) != len(flat_states):
        raise ValueError(f'Layer {self.name} expects {len(flat_states)} '
                         f'states, but it received {len(flat_input_states)} '
                         f'state values. States received: {states}')
      # end if
      set_value_tuples = []
      for i, (value, state) in enumerate(zip(flat_input_states,
                                             flat_states)):
        if value.shape != state.shape:
          raise ValueError(
              f'State {i} is incompatible with layer {self.name}: '
              f'expected shape={(batch_size, state)} '
              f'but found shape={value.shape}')
        # end if
        set_value_tuples.append((state, value))
      # end for
      keras.backend.batch_set_value(set_value_tuples)
    # end if
  
  def makeDefaultOutputs(self, num_states, timesteps, batch_size=None ):
    """Use this method to provide the Tensor output size.

    Args:
      num_states (int): hidden size of LSTM layer
      timesteps (int): sequence length for inputs
      batch_size (int): batch size for this simulation.
    """
    # make a default outputs with our required shape depending on switches
    if self.return_sequences:
      # need to return all time steps - if going to another LSTM
      retVal = tf.zeros( [ batch_size, timesteps, num_states ], 
                          dtype=keras.backend.floatx() )
    else:
      # this is the standard return case
      retVal = tf.zeros( [ batch_size, num_states ], 
                            dtype=keras.backend.floatx() )
    # end if
    # return
    return retVal



# Other helper methods from keras that needed for LSTM Cell and RNN
#   layers
def _is_multiple_state(state_size):
  """Check whether the state_size contains multiple states."""
  return (hasattr(state_size, '__len__') and
          not isinstance(state_size, tf.TensorShape))


def _generate_zero_filled_state_for_cell(cell, input_list, batch_size, dtype):
  if input_list is not None:
    dyn_inputs = input_list[1]
    batch_size = tf.shape(dyn_inputs)[0]
    dtype = dyn_inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state. '
        f'Received: batch_size={batch_size_tensor}, dtype={dtype}')
  # end if
  # define function
  def create_zeros(unnested_state_size):
    flat_dims = tf.TensorShape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return tf.zeros(init_state_size, dtype=dtype)
  # end function
  if tf.nest.is_nested(state_size):
    return tf.nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)
  # end if


def _generate_dropout_mask(ones, rate, training=None, count=1):
  def dropped_inputs():
    return keras.backend.dropout(ones, rate)
  # end local function
  if count > 1:
    return [
        keras.backend.in_train_phase(dropped_inputs, ones, training=training)
        for _ in range(count)
    ]
  # end if
  return keras.backend.in_train_phase(dropped_inputs, ones, training=training)


#-------------------------------------------------------------------------
# Custom Metrics
#
class NashSutcliffeEfficiency(tf.keras.metrics.Metric):
  """Computes the NashSutcliffeEfficiency between `y_true` and `y_pred`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

  def __init__(self, name="nse", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.nse_denom_sum = self.add_weight( name="nse_denom_sum",
                                          initializer="zeros" )
    self.nse_numer_sum = self.add_weight( name="nse_numer_sum",
                                          initializer="zeros" )
  
  def get_config(self):
    base_config = super(NashSutcliffeEfficiency, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( y_pred,
                                                                y_true )
    obs_mean = tf.math.reduce_mean( y_true )
    sim_mean = tf.math.reduce_mean( y_pred )
    cdenom = tf.math.squared_difference( y_true, obs_mean )
    cnumer = tf.math.squared_difference( y_pred, sim_mean )
    den_sum = tf.math.reduce_sum( cdenom )
    num_sum = tf.math.reduce_sum( cnumer )
    self.nse_denom_sum.assign_add( den_sum )
    self.nse_numer_sum.assign_add( num_sum )

  def result(self):
    cRatio = tf.math.divide_no_nan( self.nse_numer_sum, self.nse_denom_sum )
    ret_NSE = tf.math.subtract( 1.0, cRatio )
    return ret_NSE

  def reset_state(self):
    self.nse_denom_sum.assign(0.)
    self.nse_numer_sum.assign(0.)

class KGBeta(tf.keras.metrics.Metric):
  """Computes Beta between `y_true` and `y_pred`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """
  def __init__(self, name="Beta", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.sim_total = self.add_weight( name="sim_total", 
                                      initializer="zeros" )
    self.obs_total = self.add_weight( name="obs_total",
                                      initializer="zeros" )
    self.count = self.add_weight( name="count", initializer="zeros" )
  
  def get_config(self):
    base_config = super(KGBeta, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( y_pred,
                                                                y_true )
    num_values = math_ops.cast( array_ops.size( y_pred ), self._dtype )
    sim_sum = math_ops.reduce_sum( y_pred )
    obs_sum = math_ops.reduce_sum( y_true )
    self.count.assign_add( num_values )
    self.sim_total.assign_add( sim_sum )
    self.obs_total.assign_add( obs_sum )
  
  def result(self):
    sim_mean = tf.math.divide_no_nan( self.sim_total, self.count )
    obs_mean = tf.math.divide_no_nan( self.obs_total, self.count )
    ret_Beta = tf.math.divide_no_nan( sim_mean, obs_mean )
    return ret_Beta

  def reset_state(self):
    self.sim_total.assign(0.)
    self.obs_total.assign(0.)
    self.count.assign(0.)


class KGAlpha(tf.keras.metrics.Metric):
  """Computes alpha between `y_true` and `y_pred`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """
  def __init__(self, name="alpha", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.sim_smdiff = self.add_weight( name="sim_smdiff", 
                                      initializer="zeros" )
    self.obs_smdiff = self.add_weight( name="obs_smdiff",
                                      initializer="zeros" )
    self.count = self.add_weight( name="count", initializer="zeros" )
  
  def get_config(self):
    base_config = super(KGAlpha, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( y_pred,
                                                                y_true )
    num_values = math_ops.cast( array_ops.size( y_pred ), self._dtype )
    obs_mean = tf.math.reduce_mean( y_true )
    sim_mean = tf.math.reduce_mean( y_pred )
    obs_sqdiff = tf.math.squared_difference( y_true, obs_mean )
    sim_sqdiff = tf.math.squared_difference( y_pred, sim_mean )
    obs_sum = tf.math.reduce_sum( obs_sqdiff )
    sim_sum = tf.math.reduce_sum( sim_sqdiff )
    self.count.assign_add( num_values )
    self.sim_smdiff.assign_add( sim_sum )
    self.obs_smdiff.assign_add( obs_sum )
  
  def result(self):
    sim_std = tf.math.sqrt( tf.math.divide_no_nan( self.sim_smdiff, self.count ) )
    obs_std = tf.math.sqrt( tf.math.divide_no_nan( self.obs_smdiff, self.count ) )
    ret_alpha = tf.math.divide_no_nan( sim_std, obs_std )
    return ret_alpha

  def reset_state(self):
    self.sim_smdiff.assign(0.)
    self.obs_smdiff.assign(0.)
    self.count.assign(0.)


class KGr(tf.keras.metrics.Metric):
  """Computes the linear correlation coefficient between `y_true` and `y_pred`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

  def __init__(self, name="pearson_r", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.sim_smdiff = self.add_weight( name="sim_smdiff", 
                                      initializer="zeros" )
    self.obs_smdiff = self.add_weight( name="obs_smdiff",
                                      initializer="zeros" )
    self.num_diff_simxobs = self.add_weight( name="num_diff_simxobs",
                                             initializer="zeros")
    
  def get_config(self):
    base_config = super(KGr, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( y_pred,
                                                                y_true )
    obs_mean = tf.math.reduce_mean( y_true )
    sim_mean = tf.math.reduce_mean( y_pred )
    diff_obssim = tf.math.reduce_sum( tf.math.multiply( 
                                tf.math.subtract( y_true, obs_mean ), 
                                tf.math.subtract( y_pred, sim_mean ) ) )
    obs_sqdiff = tf.math.squared_difference( y_true, obs_mean )
    sim_sqdiff = tf.math.squared_difference( y_pred, sim_mean )
    obs_sum = tf.math.reduce_sum( obs_sqdiff )
    sim_sum = tf.math.reduce_sum( sim_sqdiff )
    self.sim_smdiff.assign_add( sim_sum )
    self.obs_smdiff.assign_add( obs_sum )
    self.num_diff_simxobs.assign_add( diff_obssim )

  def result(self):
    sqrt_obs_diff = tf.math.sqrt( self.obs_smdiff )
    sqrt_sim_diff = tf.math.sqrt( self.sim_smdiff )
    denom = tf.math.multiply( sqrt_obs_diff, sqrt_sim_diff )
    ret_r = tf.math.divide_no_nan( self.num_diff_simxobs, denom )
    return ret_r

  def reset_state(self):
    self.num_diff_simxobs.assign(0.)
    self.obs_smdiff.assign(0.)
    self.sim_smdiff.assign(0.)


class KGEff(tf.keras.metrics.Metric):
  """Computes the Kling—Gupta Efficiency between `y_true` and `y_pred`.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

  def __init__(self, name="KGE", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.sim_total = self.add_weight( name="sim_total", 
                                      initializer="zeros" )
    self.obs_total = self.add_weight( name="obs_total",
                                      initializer="zeros" )
    self.sim_smdiff = self.add_weight( name="sim_smdiff", 
                                       initializer="zeros" )
    self.obs_smdiff = self.add_weight( name="obs_smdiff",
                                       initializer="zeros" )
    self.count = self.add_weight( name="count", initializer="zeros" )
    self.num_diff_simxobs = self.add_weight( name="num_diff_simxobs",
                                             initializer="zeros" )
  
  def get_config(self):
    base_config = super(KGEff, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( y_pred,
                                                                y_true )
    obs_mean = tf.math.reduce_mean( y_true )
    sim_mean = tf.math.reduce_mean( y_pred )
    obs_sub = tf.math.subtract( y_true, obs_mean )
    sim_sub = tf.math.subtract( y_pred, sim_mean )
    sub_obsXsim = tf.math.multiply( obs_sub, sim_sub )
    tot_obsXsim = tf.math.reduce_sum( sub_obsXsim )
    obs_sqdiff = tf.math.squared_difference( y_true, obs_mean )
    sim_sqdiff = tf.math.squared_difference( y_pred, sim_mean )
    obs_sqsum = tf.math.reduce_sum( obs_sqdiff )
    sim_sqsum = tf.math.reduce_sum( sim_sqdiff )
    self.sim_smdiff.assign_add( sim_sqsum )
    self.obs_smdiff.assign_add( obs_sqsum )
    self.num_diff_simxobs.assign_add( tot_obsXsim )
    num_values = math_ops.cast( array_ops.size( y_pred ), y_pred._dtype )
    self.count.assign_add( num_values )
    sim_sum = math_ops.reduce_sum( y_pred )
    obs_sum = math_ops.reduce_sum( y_true )
    self.sim_total.assign_add( sim_sum )
    self.obs_total.assign_add( obs_sum )

  def result(self):
    # Beta
    sim_mean = tf.math.divide_no_nan( self.sim_total, self.count )
    obs_mean = tf.math.divide_no_nan( self.obs_total, self.count )
    ret_Beta = tf.math.divide_no_nan( sim_mean, obs_mean )
    # alpha
    sim_std = tf.math.sqrt( tf.math.divide_no_nan( self.sim_smdiff, self.count ) )
    obs_std = tf.math.sqrt( tf.math.divide_no_nan( self.obs_smdiff, self.count ) )
    ret_alpha = tf.math.divide_no_nan( sim_std, obs_std )
    # r
    sqrt_obs_diff = tf.math.sqrt( self.obs_smdiff )
    sqrt_sim_diff = tf.math.sqrt( self.sim_smdiff )
    denom = tf.math.multiply( sqrt_obs_diff, sqrt_sim_diff )
    ret_r = tf.math.divide_no_nan( self.num_diff_simxobs, denom )
    # calculate KGE
    r_dist = tf.math.square( tf.math.subtract( ret_r, 1.0 ) )
    alpha_dist = tf.math.square( tf.math.subtract( ret_alpha, 1.0 ) )
    Beta_dist = tf.math.square( tf.math.subtract( ret_Beta, 1.0 ) )
    tot_dist = r_dist + alpha_dist + Beta_dist
    ret_ED = tf.math.sqrt( tot_dist )
    ret_KGE = tf.math.subtract( 1.0, ret_ED )
    return ret_KGE

  def reset_state(self):
    self.count.assign(0.)
    self.obs_total.assign(0.)
    self.sim_total.assign(0.)
    self.sim_smdiff.assign(0.)
    self.obs_smdiff.assign(0.)
    self.num_diff_simxobs.assign(0.)


class NashSutcliffeEfficiency_HSZT(tf.keras.metrics.Metric):
  """Computes the NSE with hydrologic scaling and zero thresholds

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

  def __init__(self, z_threshs, name="nse", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.nse_denom_sum = self.add_weight( name="nse_denom_sum",
                                          initializer="zeros" )
    self.nse_numer_sum = self.add_weight( name="nse_numer_sum",
                                          initializer="zeros" )
    # now set the property for z_threshs
    if isinstance( z_threshs, np.ndarray ):
      self.z_threshs = z_threshs
    elif isinstance( z_threshs, list ):
      self.z_threshs = np.array( z_threshs, dtype=np.float32 )
    else:
      raise TypeError("Only np.ndarray and list are supported z_threshs types" )
    # end if
  
  def get_config(self):
    base_config = super(NashSutcliffeEfficiency, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    z_thr = ops.convert_to_tensor_v2( self.z_threshs, name='z_threshs' )
    # adjust the y_pred values using the thresholds
    adj_pred = tf.where( y_pred < z_thr, z_thr, y_pred )
    # now do our calculations
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( adj_pred,
                                                                y_true )
    obs_mean = tf.math.reduce_mean( y_true )
    sim_mean = tf.math.reduce_mean( y_pred )
    cdenom = tf.math.squared_difference( y_true, obs_mean )
    cnumer = tf.math.squared_difference( y_pred, sim_mean )
    den_sum = tf.math.reduce_sum( cdenom )
    num_sum = tf.math.reduce_sum( cnumer )
    self.nse_denom_sum.assign_add( den_sum )
    self.nse_numer_sum.assign_add( num_sum )

  def result(self):
    cRatio = tf.math.divide_no_nan( self.nse_numer_sum, self.nse_denom_sum )
    ret_NSE = tf.math.subtract( 1.0, cRatio )
    return ret_NSE

  def reset_state(self):
    self.nse_denom_sum.assign(0.)
    self.nse_numer_sum.assign(0.)


class KGBeta_HSZT(tf.keras.metrics.Metric):
  """Computes KGE Beta with hydrologic scaling and zero thresholds

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """
  def __init__(self, z_threshs, name="Beta", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.sim_total = self.add_weight( name="sim_total", 
                                      initializer="zeros" )
    self.obs_total = self.add_weight( name="obs_total",
                                      initializer="zeros" )
    self.count = self.add_weight( name="count", initializer="zeros" )
    # now set the property for z_threshs
    if isinstance( z_threshs, np.ndarray ):
      self.z_threshs = z_threshs
    elif isinstance( z_threshs, list ):
      self.z_threshs = np.array( z_threshs, dtype=np.float32 )
    else:
      raise TypeError("Only np.ndarray and list are supported z_threshs types" )
    # end if
  
  def get_config(self):
    base_config = super(KGBeta, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    z_thr = ops.convert_to_tensor_v2( self.z_threshs, name='z_threshs' )
    # adjust the y_pred values using the thresholds
    adj_pred = tf.where( y_pred < z_thr, z_thr, y_pred )
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( adj_pred,
                                                                y_true )
    num_values = math_ops.cast( array_ops.size( y_pred ), self._dtype )
    sim_sum = math_ops.reduce_sum( y_pred )
    obs_sum = math_ops.reduce_sum( y_true )
    self.count.assign_add( num_values )
    self.sim_total.assign_add( sim_sum )
    self.obs_total.assign_add( obs_sum )
  
  def result(self):
    sim_mean = tf.math.divide_no_nan( self.sim_total, self.count )
    obs_mean = tf.math.divide_no_nan( self.obs_total, self.count )
    ret_Beta = tf.math.divide_no_nan( sim_mean, obs_mean )
    return ret_Beta

  def reset_state(self):
    self.sim_total.assign(0.)
    self.obs_total.assign(0.)
    self.count.assign(0.)


class KGAlpha_HSZT(tf.keras.metrics.Metric):
  """Computes KGE alpha with hydrologic scaling and zero thresholds.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """
  def __init__(self, z_threshs, name="alpha", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.sim_smdiff = self.add_weight( name="sim_smdiff", 
                                      initializer="zeros" )
    self.obs_smdiff = self.add_weight( name="obs_smdiff",
                                      initializer="zeros" )
    self.count = self.add_weight( name="count", initializer="zeros" )
    # now set the property for z_threshs
    if isinstance( z_threshs, np.ndarray ):
      self.z_threshs = z_threshs
    elif isinstance( z_threshs, list ):
      self.z_threshs = np.array( z_threshs, dtype=np.float32 )
    else:
      raise TypeError("Only np.ndarray and list are supported z_threshs types" )
    # end if
  
  def get_config(self):
    base_config = super(KGAlpha, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)
  
  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    z_thr = ops.convert_to_tensor_v2( self.z_threshs, name='z_threshs' )
    # adjust the y_pred values using the thresholds
    adj_pred = tf.where( y_pred < z_thr, z_thr, y_pred )
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( adj_pred,
                                                                y_true )
    num_values = math_ops.cast( array_ops.size( y_pred ), self._dtype )
    obs_mean = tf.math.reduce_mean( y_true )
    sim_mean = tf.math.reduce_mean( y_pred )
    obs_sqdiff = tf.math.squared_difference( y_true, obs_mean )
    sim_sqdiff = tf.math.squared_difference( y_pred, sim_mean )
    obs_sum = tf.math.reduce_sum( obs_sqdiff )
    sim_sum = tf.math.reduce_sum( sim_sqdiff )
    self.count.assign_add( num_values )
    self.sim_smdiff.assign_add( sim_sum )
    self.obs_smdiff.assign_add( obs_sum )
  
  def result(self):
    sim_std = tf.math.sqrt( tf.math.divide_no_nan( self.sim_smdiff, self.count ) )
    obs_std = tf.math.sqrt( tf.math.divide_no_nan( self.obs_smdiff, self.count ) )
    ret_alpha = tf.math.divide_no_nan( sim_std, obs_std )
    return ret_alpha

  def reset_state(self):
    self.sim_smdiff.assign(0.)
    self.obs_smdiff.assign(0.)
    self.count.assign(0.)


class KGr_HSZT(tf.keras.metrics.Metric):
  """Computes the linear correlation coefficient, r, with hydrologic 
  scaling and zero thresholds.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

  def __init__(self, z_threshs, name="pearson_r", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.sim_smdiff = self.add_weight( name="sim_smdiff", 
                                      initializer="zeros" )
    self.obs_smdiff = self.add_weight( name="obs_smdiff",
                                      initializer="zeros" )
    self.num_diff_simxobs = self.add_weight( name="num_diff_simxobs",
                                             initializer="zeros")
    # now set the property for z_threshs
    if isinstance( z_threshs, np.ndarray ):
      self.z_threshs = z_threshs
    elif isinstance( z_threshs, list ):
      self.z_threshs = np.array( z_threshs, dtype=np.float32 )
    else:
      raise TypeError("Only np.ndarray and list are supported z_threshs types" )
    # end if
    
  def get_config(self):
    base_config = super(KGr, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    z_thr = ops.convert_to_tensor_v2( self.z_threshs, name='z_threshs' )
    # adjust the y_pred values using the thresholds
    adj_pred = tf.where( y_pred < z_thr, z_thr, y_pred )
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( adj_pred,
                                                                y_true )
    obs_mean = tf.math.reduce_mean( y_true )
    sim_mean = tf.math.reduce_mean( y_pred )
    diff_obssim = tf.math.reduce_sum( tf.math.multiply( 
                                tf.math.subtract( y_true, obs_mean ), 
                                tf.math.subtract( y_pred, sim_mean ) ) )
    obs_sqdiff = tf.math.squared_difference( y_true, obs_mean )
    sim_sqdiff = tf.math.squared_difference( y_pred, sim_mean )
    obs_sum = tf.math.reduce_sum( obs_sqdiff )
    sim_sum = tf.math.reduce_sum( sim_sqdiff )
    self.sim_smdiff.assign_add( sim_sum )
    self.obs_smdiff.assign_add( obs_sum )
    self.num_diff_simxobs.assign_add( diff_obssim )

  def result(self):
    sqrt_obs_diff = tf.math.sqrt( self.obs_smdiff )
    sqrt_sim_diff = tf.math.sqrt( self.sim_smdiff )
    denom = tf.math.multiply( sqrt_obs_diff, sqrt_sim_diff )
    ret_r = tf.math.divide_no_nan( self.num_diff_simxobs, denom )
    return ret_r

  def reset_state(self):
    self.num_diff_simxobs.assign(0.)
    self.obs_smdiff.assign(0.)
    self.sim_smdiff.assign(0.)


class KGEff_HSZT(tf.keras.metrics.Metric):
  """Computes the Kling—Gupta Efficiency with hydrologic 
  scaling and zero thresholds.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

  def __init__(self, z_threshs, name="KGE", **kwargs):
    super().__init__(name=name, **kwargs)
    # use add weights to essentially initialize tf.Variable instances
    #  so that do not need to specify shape
    self.sim_total = self.add_weight( name="sim_total", 
                                      initializer="zeros" )
    self.obs_total = self.add_weight( name="obs_total",
                                      initializer="zeros" )
    self.sim_smdiff = self.add_weight( name="sim_smdiff", 
                                       initializer="zeros" )
    self.obs_smdiff = self.add_weight( name="obs_smdiff",
                                       initializer="zeros" )
    self.count = self.add_weight( name="count", initializer="zeros" )
    self.num_diff_simxobs = self.add_weight( name="num_diff_simxobs",
                                             initializer="zeros" )
    # now set the property for z_threshs
    if isinstance( z_threshs, np.ndarray ):
      self.z_threshs = z_threshs
    elif isinstance( z_threshs, list ):
      self.z_threshs = np.array( z_threshs, dtype=np.float32 )
    else:
      raise TypeError("Only np.ndarray and list are supported z_threshs types" )
    # end if
  
  def get_config(self):
    base_config = super(KGEff, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    z_thr = ops.convert_to_tensor_v2( self.z_threshs, name='z_threshs' )
    # adjust the y_pred values using the thresholds
    adj_pred = tf.where( y_pred < z_thr, z_thr, y_pred )
    y_pred, y_true = losses_utils.squeeze_or_expand_dimensions( adj_pred,
                                                                y_true )
    obs_mean = tf.math.reduce_mean( y_true )
    sim_mean = tf.math.reduce_mean( y_pred )
    obs_sub = tf.math.subtract( y_true, obs_mean )
    sim_sub = tf.math.subtract( y_pred, sim_mean )
    sub_obsXsim = tf.math.multiply( obs_sub, sim_sub )
    tot_obsXsim = tf.math.reduce_sum( sub_obsXsim )
    obs_sqdiff = tf.math.squared_difference( y_true, obs_mean )
    sim_sqdiff = tf.math.squared_difference( y_pred, sim_mean )
    obs_sqsum = tf.math.reduce_sum( obs_sqdiff )
    sim_sqsum = tf.math.reduce_sum( sim_sqdiff )
    self.sim_smdiff.assign_add( sim_sqsum )
    self.obs_smdiff.assign_add( obs_sqsum )
    self.num_diff_simxobs.assign_add( tot_obsXsim )
    num_values = math_ops.cast( array_ops.size( y_pred ), y_pred._dtype )
    self.count.assign_add( num_values )
    sim_sum = math_ops.reduce_sum( y_pred )
    obs_sum = math_ops.reduce_sum( y_true )
    self.sim_total.assign_add( sim_sum )
    self.obs_total.assign_add( obs_sum )

  def result(self):
    # Beta
    sim_mean = tf.math.divide_no_nan( self.sim_total, self.count )
    obs_mean = tf.math.divide_no_nan( self.obs_total, self.count )
    ret_Beta = tf.math.divide_no_nan( sim_mean, obs_mean )
    # alpha
    sim_std = tf.math.sqrt( tf.math.divide_no_nan( self.sim_smdiff, self.count ) )
    obs_std = tf.math.sqrt( tf.math.divide_no_nan( self.obs_smdiff, self.count ) )
    ret_alpha = tf.math.divide_no_nan( sim_std, obs_std )
    # r
    sqrt_obs_diff = tf.math.sqrt( self.obs_smdiff )
    sqrt_sim_diff = tf.math.sqrt( self.sim_smdiff )
    denom = tf.math.multiply( sqrt_obs_diff, sqrt_sim_diff )
    ret_r = tf.math.divide_no_nan( self.num_diff_simxobs, denom )
    # calculate KGE
    r_dist = tf.math.square( tf.math.subtract( ret_r, 1.0 ) )
    alpha_dist = tf.math.square( tf.math.subtract( ret_alpha, 1.0 ) )
    Beta_dist = tf.math.square( tf.math.subtract( ret_Beta, 1.0 ) )
    tot_dist = r_dist + alpha_dist + Beta_dist
    ret_ED = tf.math.sqrt( tot_dist )
    ret_KGE = tf.math.subtract( 1.0, ret_ED )
    return ret_KGE

  def reset_state(self):
    self.count.assign(0.)
    self.obs_total.assign(0.)
    self.sim_total.assign(0.)
    self.sim_smdiff.assign(0.)
    self.obs_smdiff.assign(0.)
    self.num_diff_simxobs.assign(0.)



#------------------------------------------------------------------------------
# Custom loss functions - in functional form
@tf.function
@dispatch.add_dispatch_support
def nse( y_true, y_pred, ):
  """Computes the Nash Sutcliffe efficiency (NSE) between labels and
  predictions.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    NSE values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # do the initial manipulations
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  # do calculations
  obs_mean = math_ops.reduce_mean( y_true )
  sqDiff_simXobs = math_ops.squared_difference( y_true, y_pred )
  sum_sqDiff = math_ops.reduce_sum( sqDiff_simXobs )
  obs_sqDiff = math_ops.squared_difference( y_true, obs_mean )
  obs_sumSD = math_ops.reduce_sum( obs_sqDiff )
  rSide = math_ops.div_no_nan( sum_sqDiff, obs_sumSD )
  NSE = math_ops.subtract( 1.0, rSide )
  retNSE = math_ops.subtract( 1.0, NSE )
  return retNSE

@tf.function
@dispatch.add_dispatch_support
def nse_star( y_true, y_pred, q_stds,
                        eps_error: float = 0.1, ):
  """Computes the modified Nash Sutcliffe efficiency (NSE) between labels and
  predictions.

  This is the modification used in Kratzert et al.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    q_stds: standard deviation of discharge in the basin from training period.
        shape = `[num_basins, d0, .. dB]
    eps_error: denominator constant, should be around 0.1

  Returns:
    NSE* values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # do the initial manipulations
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  q_stds_e = q_stds + eps_error
  q_stds_e = ops.convert_to_tensor_v2_with_dispatch( q_stds_e, name='q_stds_e' )
  # do calculations
  sq_error = math_ops.squared_difference( y_pred, y_true, name='sq_error' )
  weights = math_ops.inv( math_ops.square( q_stds_e, name='q_weights' ),
                          name='weights' )
  scaled_loss = math_ops.mul( weights, sq_error, name='scaled_loss' )
  return keras.backend.mean(scaled_loss, axis=-1)

@tf.function
@dispatch.add_dispatch_support
def kg_Beta( y_true, y_pred, ):
  """Computes the Beta coefficient between labels and predictions.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Beta values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # make sure that have tensors
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  # calculate the means
  sim_mean = math_ops.reduce_mean( y_pred, axis=-1 )
  obs_mean = math_ops.reduce_mean( y_true, axis=-1 )
  # calculate beta
  rBeta = math_ops.div_no_nan( sim_mean, obs_mean )
  return rBeta

@tf.function
@dispatch.add_dispatch_support
def kg_alpha( y_true, y_pred, ):
  """Computes the alpha coefficient between labels and predictions.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    alpha values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # make sure that have tensors
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  # calculate the means
  sim_mean = math_ops.reduce_mean( y_pred, axis=-1)
  obs_mean = math_ops.reduce_mean( y_true, axis=-1)
  # get N
  num_values = math_ops.cast( array_ops.size( y_pred ), y_pred.dtype )
  # calculate the squared differences from the mean
  sim_sqdiff = math_ops.reduce_sum( math_ops.squared_difference( y_pred, sim_mean ),
                                    axis=-1 )
  obs_sqdiff = math_ops.reduce_sum( math_ops.squared_difference( y_true, obs_mean ),
                                    axis=-1 )
  # calculate standard deviations
  sim_std = math_ops.sqrt( math_ops.div_no_nan( sim_sqdiff, num_values ) )
  obs_std = math_ops.sqrt( math_ops.div_no_nan( obs_sqdiff, num_values ) )
  # calculate beta
  ralpha = math_ops.div_no_nan( sim_std, obs_std )
  return ralpha

@tf.function
@dispatch.add_dispatch_support
def kg_r( y_true, y_pred, ):
  """Computes the linear correlation coefficient, r, between labels and
  predictions.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    r values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # make sure that have tensors
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  # calculate the means
  sim_mean = math_ops.reduce_mean( y_pred, axis=-1)
  obs_mean = math_ops.reduce_mean( y_true, axis=-1)
  # do the initial, partial calculations
  sim_diff = math_ops.subtract( y_pred, sim_mean )
  obs_diff = math_ops.subtract( y_true, obs_mean )
  diff_prod = math_ops.multiply( obs_diff, sim_diff )
  total_diff = math_ops.reduce_sum( diff_prod, axis=-1)
  sim_sqdiff = math_ops.squared_difference( y_pred, sim_mean )
  obs_sqdiff = math_ops.squared_difference( y_true, obs_mean )
  sim_sumsqdiff = math_ops.reduce_sum( sim_sqdiff, axis=-1)
  obs_sumsqdiff = math_ops.reduce_sum( obs_sqdiff, axis=-1)
  sim_sumdiff = math_ops.sqrt( sim_sumsqdiff )
  obs_sumdiff = math_ops.sqrt( obs_sumsqdiff )
  denom = math_ops.multiply( sim_sumdiff, obs_sumdiff )
  rVal = math_ops.div_no_nan( total_diff, denom )
  return rVal

@tf.function
@dispatch.add_dispatch_support
def kg_eff( y_true, y_pred ):
  """Computes the Kling—Gupta Efficiency between labels and
  predictions.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    KGE values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # use our existing functions
  rBeta = kg_Beta( y_true, y_pred )
  ralpha = kg_alpha( y_true, y_pred )
  rVal = kg_r( y_true, y_pred )
  # now calculate ED
  rterm = math_ops.square( math_ops.subtract( rVal, 1.0 ) )
  Bterm =  math_ops.square( math_ops.subtract( rBeta, 1.0 ) )
  aterm = math_ops.square( math_ops.subtract( ralpha, 1.0 ) )
  sum_sq_diffs = rterm + aterm + Bterm
  rED = math_ops.sqrt( sum_sq_diffs )
  # KGE is max of 1 and can be small (i.e., large value negative)
  KGE = math_ops.subtract( 1.0, rED )
  # to put this in terms of losses, subtract from 1.0
  retKGE = math_ops.subtract( 1.0, KGE )
  return retKGE


@tf.function
@dispatch.add_dispatch_support
def mse_hszt( y_true, y_pred, z_threshs ):
  """Custom keras loss function for MSE with hydrologic scaling and zero 
  threshold support.

  Mean Square Error is the MSE. Hydrologic scaling and zero threshold 
  support is implemented by setting the predicted values equal to the
  specified thresholds when predicted values are less than or equal to
  the threshold. This adjustment makes y_true == y_pred when y_pred is
  less than or equal to the zero threshold.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    z_threshs: ZScore value corresponding to the zero threshold in 
               target data set

  Returns:
    MSE* values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # do the initial manipulations to ensure that have the appropriate
  #    tensor formats
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  z_threshs = ops.convert_to_tensor_v2_with_dispatch( z_threshs, name='z_threshs' )
  # adjust the y_pred values using the thresholds
  adj_pred = tf.where( y_pred < z_threshs, z_threshs, y_pred )
  # do calculations
  sq_error = math_ops.squared_difference( adj_pred, y_true, name='sq_error' )
  return keras.backend.mean(sq_error, axis=-1)


@tf.function
@dispatch.add_dispatch_support
def nse_hszt( y_true, y_pred, z_threshs ):
  """Computes NSE with hydrologic scaling and zero threshold support.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    z_threshs: ZScore value corresponding to the zero threshold in 
               target data set

  Returns:
    NSE values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # do the initial manipulations
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  z_threshs = ops.convert_to_tensor_v2_with_dispatch( z_threshs, name='z_threshs' )
  # adjust the y_pred values using the thresholds
  adj_pred = tf.where( y_pred < z_threshs, z_threshs, y_pred )
  # do calculations
  obs_mean = math_ops.reduce_mean( y_true )
  sqDiff_simXobs = math_ops.squared_difference( y_true, adj_pred )
  sum_sqDiff = math_ops.reduce_sum( sqDiff_simXobs )
  obs_sqDiff = math_ops.squared_difference( y_true, obs_mean )
  obs_sumSD = math_ops.reduce_sum( obs_sqDiff )
  obs_denom = np.where( math_ops.abs( obs_sumSD ) < 0.00001, 0.00001, obs_sumSD )
  rSide = math_ops.div_no_nan( sum_sqDiff, obs_denom )
  NSE = math_ops.subtract( 1.0, rSide )
  retNSE = math_ops.subtract( 1.0, NSE )
  return retNSE


@tf.function
@dispatch.add_dispatch_support
def kg_beta_hszt( y_true, y_pred, z_threshs ):
  """Computes KGE Beta component with hydrologic scaling and zero threshold
  support.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    z_threshs: ZScore value corresponding to the zero threshold in 
               target data set

  Returns:
    Beta values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # make sure that have tensors
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  z_threshs = ops.convert_to_tensor_v2_with_dispatch( z_threshs, name='z_threshs' )
  # adjust the y_pred values using the thresholds
  adj_pred = tf.where( y_pred < z_threshs, z_threshs, y_pred )
  # calculate the means
  sim_mean = math_ops.reduce_mean( adj_pred, axis=-1 )
  obs_mean = math_ops.reduce_mean( y_true, axis=-1 )
  # calculate beta
  denom = tf.where( tf.abs( obs_mean ) < 0.00001, 0.00001, obs_mean )
  rBeta = math_ops.div_no_nan( sim_mean, denom )
  return rBeta


@tf.function
@dispatch.add_dispatch_support
def kg_alpha_hszt( y_true, y_pred, z_threshs ):
  """Computes KGE alpha subcomponent with hydrologic scaling and zero threshold
  support.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    z_threshs: ZScore value corresponding to the zero threshold in 
               target data set

  Returns:
    alpha values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # make sure that have tensors
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  z_threshs = ops.convert_to_tensor_v2_with_dispatch( z_threshs, name='z_threshs' )
  # adjust the y_pred values using the thresholds
  adj_pred = tf.where( y_pred < z_threshs, z_threshs, y_pred )
  # calculate the means
  sim_mean = math_ops.reduce_mean( adj_pred, axis=-1)
  obs_mean = math_ops.reduce_mean( y_true, axis=-1)
  # get N
  num_values = math_ops.cast( array_ops.size( adj_pred ), adj_pred.dtype )
  # calculate the squared differences from the mean
  sim_sqdiff = math_ops.reduce_sum( math_ops.squared_difference( adj_pred, sim_mean ),
                                    axis=-1 )
  obs_sqdiff = math_ops.reduce_sum( math_ops.squared_difference( y_true, obs_mean ),
                                    axis=-1 )
  # calculate standard deviations
  sim_std = math_ops.sqrt( math_ops.div_no_nan( sim_sqdiff, num_values ) )
  obs_std = math_ops.sqrt( math_ops.div_no_nan( obs_sqdiff, num_values ) )
  # calculate beta
  denom = tf.where( tf.abs( obs_std ) < 0.00001, 0.00001, obs_std )
  ralpha = math_ops.div_no_nan( sim_std, denom )
  return ralpha


@tf.function
@dispatch.add_dispatch_support
def kg_r_hszt( y_true, y_pred, z_threshs ):
  """Computes the linear correlation coefficient, r, with hydrologic scaling 
  and zero threshold support.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    z_threshs: ZScore value corresponding to the zero threshold in 
               target data set

  Returns:
    r values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # make sure that have tensors
  y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred, name='y_pred')
  y_true = math_ops.cast(y_true, y_pred.dtype, name='y_true')
  z_threshs = ops.convert_to_tensor_v2_with_dispatch( z_threshs, name='z_threshs' )
  # adjust the y_pred values using the thresholds
  adj_pred = tf.where( y_pred < z_threshs, z_threshs, y_pred )
  # calculate the means
  sim_mean = math_ops.reduce_mean( adj_pred, axis=-1)
  obs_mean = math_ops.reduce_mean( y_true, axis=-1)
  # do the initial, partial calculations
  sim_diff = math_ops.subtract( adj_pred, sim_mean )
  obs_diff = math_ops.subtract( y_true, obs_mean )
  diff_prod = math_ops.multiply( obs_diff, sim_diff )
  total_diff = math_ops.reduce_sum( diff_prod, axis=-1)
  sim_sqdiff = math_ops.squared_difference( adj_pred, sim_mean )
  obs_sqdiff = math_ops.squared_difference( y_true, obs_mean )
  sim_sumsqdiff = math_ops.reduce_sum( sim_sqdiff, axis=-1)
  obs_sumsqdiff = math_ops.reduce_sum( obs_sqdiff, axis=-1)
  sim_sumdiff = math_ops.sqrt( sim_sumsqdiff )
  obs_sumdiff = math_ops.sqrt( obs_sumsqdiff )
  denom = math_ops.multiply( sim_sumdiff, obs_sumdiff )
  adjdenom = tf.where( tf.abs( denom ) < 0.00001, 0.00001, denom )
  rVal = math_ops.div_no_nan( total_diff, adjdenom )
  return rVal


@tf.function
@dispatch.add_dispatch_support
def kg_eff_hszt( y_true, y_pred, z_threshs ):
  """Computes the Kling—Gupta Efficiency with hydrologic scaling and zero threshold
  support.

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    z_threshs: ZScore value corresponding to the zero threshold in 
               target data set

  Returns:
    KGE values. shape = `[batch_size, d0, .. dN-1]`.
  """
  # use our existing functions
  rBeta = kg_beta_hszt( y_true, y_pred, z_threshs )
  ralpha = kg_alpha_hszt( y_true, y_pred, z_threshs )
  rVal = kg_r_hszt( y_true, y_pred, z_threshs )
  # now calculate ED
  rterm = math_ops.square( math_ops.subtract( rVal, 1.0 ) )
  Bterm =  math_ops.square( math_ops.subtract( rBeta, 1.0 ) )
  aterm = math_ops.square( math_ops.subtract( ralpha, 1.0 ) )
  sum_sq_diffs = rterm + aterm + Bterm
  rED = math_ops.sqrt( sum_sq_diffs )
  # KGE is max of 1 and can be small (i.e., large value negative)
  KGE = math_ops.subtract( 1.0, rED )
  # to put this in terms of losses, subtract from 1.0
  retKGE = math_ops.subtract( 1.0, KGE )
  return retKGE


# Loss Functions in Class Form ----------------------------------
# Need the class form to pass to compile
class NSEStar(LossFunctionWrapper):
  """Computes the NSE* between labels and predictions.

  """

  def __init__(self, q_stds,
               eps_error = 0.1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='nse*'):
    """Initializes `NashSutcliffeEff` instance.

    Args:
      q_stds: numpy.ndarray of standard deviation of discharge in the basin
          from training period. Has shape of number of basins
      eps_error: denominator constant scalar, should be around 0.1
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'mean_squared_error'.
    """
    super().__init__( nse_star, name=name,
                      reduction=reduction,
                      q_stds = q_stds,
                      eps_error = eps_error )
    self.q_stds = q_stds
    self.eps_error = eps_error
  
  def get_config(self):
    base_config = super(KGEff, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)


class NSE(LossFunctionWrapper):
  """Computes the standard NSE between labels and predictions.

  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO,
               name='nse'):
    """Initializes `NashSutcliffeEff` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'mean_squared_error'.
    """
    super().__init__( nse, name=name, )
  
  def get_config(self):
    base_config = super(KGEff, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)


class KGEffLoss(LossFunctionWrapper):
  """Computes the KGE between labels and predictions.

  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO,
               name='kge'):
    """Initializes `KGEff` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'mean_squared_error'.
    """
    super().__init__( kg_eff, name=name,
                      reduction=reduction, )
  
  def get_config(self):
    base_config = super(KGEff, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)


class MSE_HSZT(LossFunctionWrapper):
  """Computes the MSE with support for hydrologic transform and 
  zero thresholds.

  """

  def __init__(self, z_threshs,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mse_hszt'):
    """Initializes `MSE_HSZT` instance

    Args:
      z_threshs: numpy.ndarray of zero thresholds
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'mean_squared_error'.
    """
    super().__init__( mse_hszt, name=name,
                      reduction=reduction,
                      z_threshs=z_threshs, )
    self.z_threshs = z_threshs
  
  def get_config(self):
    base_config = super(MSE_HSZT, self).get_config()
    return dict(list(base_config.items()))
  
  def from_config(cls, config):
    return cls(**config)


#EOF