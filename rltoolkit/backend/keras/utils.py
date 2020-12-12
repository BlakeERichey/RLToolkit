import os
import types
import GPUtil
import logging
import warnings
import numpy as np
from copy import deepcopy
from rltoolkit.utils import test_network

with warnings.catch_warnings():
  #disable warnings in tensorflow subprocesses
  warnings.simplefilter('ignore')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

  import tensorflow as tf
  tf.compat.logging.set_verbosity(tf.compat.v1.logging.ERROR)

  import keras
  from   keras.models import clone_model
  from   keras.backend.tensorflow_backend import set_session

def backend_test_network(weights, network, env, episodes, seed):
  """
    Wraps environment testing so that network recreation happens on subcore and 
    not main thread. 

    # Arguments
    weights:  list of weights corresponding to Keras NN weights
    network:  a keras Neural network or function to create one
    env:      a Gym environment
    episodes: How many episodes to test an environment.
    seed:     Random seed to ensure consistency across tests
  """

  try:
    env = deepcopy(env)
    keras.backend.clear_session()
    if type(network) == types.FunctionType: #Distributed create_model
      nn = network()
      # nn.summary() #To identify is session is being cleared
    else: #Multicore model
      nn  = clone_model(network)
    nn.set_weights(weights)
    avg = test_network(nn, env, episodes, seed=seed)
  except Exception as e:
    logging.error(f'Exception occured testing network: {e}')
    avg = None
  return avg

def get_model_gpu_allocation(network_function):
  """
    Returns the amount of GPU memory allocated for a singular neural network 
    model as a decimal percantage of 1.  

    network_function: Function that returns a keras model
  """
  
  set_gpu_session()

  nn = network_function()
  input_shape = nn.input_shape[1:]

  fake_inputs = np.random.rand(*nn.input_shape[1:])
  with tf.device('/gpu:0'):
    nn.predict(np.expand_dims(fake_inputs, axis=0))
    gpus = GPUtil.getGPUs()
    return getattr(gpus[0], 'memoryUtil')

def set_gpu_session():
  """
    Sets session to permit gpu growth
  """
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  set_session(sess)