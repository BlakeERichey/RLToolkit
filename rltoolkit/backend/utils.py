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
  
  def unpack(self,):
    return self.data

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