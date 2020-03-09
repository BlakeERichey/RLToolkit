import math, random
import numpy as np
from keras.models import clone_model, Sequential
from keras.layers import Dense

from rltoolkit.utils import test_network

class Worker:

  '''
    Worker is a abstract class that implements the testing of an individual
    in an evolutionary strategy
  '''

  def __init__(self, nn, alpha=0.01):

    self.alpha = alpha
    # Not sure what to do with the genes
    # self.genes = np.random.uniform(0.0, 1.0, size=4)
    self.mask = self.gen_mask(nn)



  def fitness(self, nn, env, sharpness=1, validate=False, render=False):
    '''
      should return results from non validation test and validation run

      does not require use of validation run

      returns (results, validation results)

      take nn and apply the mask generated below (add to each layer at the same time)

      If validate is False return 0.0

    '''

    reward = 0.0
    validate_reward = 0.0
    self.apply_mask(nn)

    for _ in range(sharpness):
      reward += test_network(nn=nn, env=env, render=render)

    reward = reward / sharpness
    

    if validate:

      for _ in range(sharpness):
        validate_reward += test_network(nn=nn, env=env, render=render)
      validate_reward = validate_reward/sharpness

      return reward, validate_reward
  
  
    return reward


  def gen_mask(self,nn):
    '''
      Creates a small mask to apply (add) to a a workers
      genes when performing mutations
      Multiply each weight by alpha
    '''

    weights = nn.get_weights()

    self.truncate_weights(weights)

    return weights


  def apply_mask(self, nn):

    weights = nn.get_weights()

    self.truncate_weights(weights)

    for i, layer in enumerate(weights):
      layer += self.mask[i]

    nn.set_weights(weights)

    return None

  def truncate_weights(self, weights):
    """
        Truncates list of weights for a keras network in place
    """
    for i, w in enumerate(weights):
      weights[i] = np.around(w.astype(np.float64), 3)


