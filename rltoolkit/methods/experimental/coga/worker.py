import math, random
import numpy as np
from keras.layers import Dense
from keras.models import clone_model, Sequential
from rltoolkit.utils import truncate_weights, test_network

class Worker:

  """
    A worker is an individual that applies mutations on a Colony and serves
    in determining the overall quality of the Colony
  """

  def __init__(self, nn, alpha=0.01):
    """
      Initializes a worker.

      # Arguments
      nn: A Keras neural network
      alpha: A small number to multiply against newly generated weights to act
        as a mask to apply to mutate Colony networks.
    """
    self.alpha = alpha
    self.mask = self._gen_mask(nn)

  def fitness(self, nn, env, sharpness=1, validate=False, render=False):
    """
      Test a worker by applying its mask, then runs resultant worker through 
      the environment.

      # Arguements
      nn: A keras neural network.
      env: A gym environment.
      sharpness: How many episodes to run of the gym environment.
      render: Pass True to render the environment at each step.
    """
    self._apply_mask(nn)
    reward = test_network(
      nn,
      env,
      verbose=0,
      render=render,
      episodes=sharpness,
    )
    
    #validation reward defaults to 0
    v_reward = 0
    if validate:
      v_reward = test_network(
        nn,
        env,
        verbose=0,
        render=render,
        episodes=sharpness,
      )
  
  
    return reward, v_reward


  def _gen_mask(self, nn,):
    """
      Creates a small mask to apply (add) to a a workers
      genes when performing mutations
    """
    return truncate_weights(nn.get_weights(), alpha=self.alpha, n_decimals=3)


  def _apply_mask(self, nn):
    """
      Applies works weights to keras network. Used as a mutation strategy to 
      update a network prior to testing in an environment.
    """
    
    #Apply mask in place
    weights = nn.get_weights()
    for i, layer in enumerate(weights):
      layer += self.mask[i]

    #set new weights
    nn.set_weights(weights)