import types
import logging
from copy import deepcopy

with warnings.catch_warnings():
  #disable warnings in tensorflow subprocesses
  warnings.simplefilter("ignore")
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  logging.disable(logging.WARNING)

  #Import predefined context for client spawning
  import keras
  from keras.models import clone_model
  import tensorflow as tf
  import rltoolkit
  from   rltoolkit.utils import test_network

class Packet:
  """
    A basic Packet class for sycnrhonization of data via Proxys.
    Proxies will return a Packet object. Use packet.unpack() to obtain 
    contained data.
  """

  def __init__(self,data):
    self.data = data    
  
  def unpack(self,):
    return self.data

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


  ###CAN THROW ERRORS IN TRY LOOP THAT SHOULD BE REPORTED
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
    logging.warning(f'Exception occured testing network: {e}')
    avg = None
  return avg
  
def clean_noisy_results(results, reference='min'):
  """
    Makes corrections in place to results for tasks that failed or returned 
    None. Ensures a numeric answer for each result.

    #Arguments:
    Results: list of values that are either None or Numeric. 
    reference: One of ['min', 'minimum', 'max', 'maximum'] adjusts non 
      numeric results to the reference value. `numeric_only` must be True, 
      or this parameter is overlooked.
  """
  assert reference in {'min', 'minimum', 'max', 'maximum'}, \
    f"Unidentified keyword for reference: {reference}"

  min_value = reference in {'min', 'minimum'} #use minimum?

  ref_value = None
  for result in results:
    if ref_value is None: #First process could have failed, thus resuults[0] could be None
        ref_value = result
    else:
      if result is not None:
        try:
          if min_value:
            ref_value = min(ref_value, result)
          else:
            ref_value = max(ref_value, result)
        except TypeError:
          pass

  for i, value in enumerate(results):
    if value is None:
      results[i] = ref_value