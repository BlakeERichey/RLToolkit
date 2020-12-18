import zlib
import dill
import pickle
import types
import logging
from copy import deepcopy

class Packet:
  """
    A basic Packet class for synchronization of data via Proxys.
    Proxies will return a Packet object. Use packet.unpack() to obtain 
    contained data.
  """

  def __init__(self,data):
    self.data = data
    self.times_compressed = 0
    self.serialize_method = None
  
  def unpack(self,):
    self.decompress()
    return self.data

  def compress(self, level=1, iterations=None):
    try:
      serialized = pickle.dumps(self.data)
      self.serialize_method = 'pickle'
    except Exception:
      serialized = dill.dumps(self.data) #Can throw dill error, should do so.
      self.serialize_method = 'dill'
    
    data = serialized
    compressed = self._compress(serialized, level)
    if iterations is None:
      while len(compressed) < len(data):
        data = compressed
        compressed = self._compress(compressed, level) #adds 1 and end that must be offset
      self.times_compressed -= 1 #offsetting to omit final compression

    elif iterations > 1:
      for i in range(1, iterations):
        compressed = self._compress(compressed, level)
      data = compressed
    elif iterations == 1:
      data = compressed
    
    self.data = data
  
  def _compress(self, data, level):
    compressed = zlib.compress(data, level=level)
    self.times_compressed += 1 #Should add after compress finished so try/catch can be managed elsewhere
    return compressed
  
  def decompress(self):
    data = self.data
    times_compressed = self.times_compressed
    for i in range(times_compressed):
      data = zlib.decompress(data)
      self.times_compressed -= 1
    
    deserialized = data
    if self.serialize_method:
      if self.serialize_method == 'pickle':
        deserialized = pickle.loads(data)
        self.serialize_method = None
      
      elif self.serialize_method == 'dill':
        deserialized = dill.loads(data)
        self.serialize_method = None
      
      else:
        msg = 'Cant Deserialize Packet. ' + \
          "Expected Serialization method to be one of [\'dill\', \'pickle\'], " + \
          f'but got {self.serialize_method}.'
        raise Exception(msg)
    
    self.data = deserialized

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