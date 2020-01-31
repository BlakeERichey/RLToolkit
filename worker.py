import math, random
import numpy as np

class Worker:

  '''
    Worker is a abstract class that implements the testing of an individual
    in an evolutionary strategy
  '''

  def __init__(self, genes):
    self.genes = genes

  def fitness(self, env):
    '''
      should return results from non validation test and validation run

      does not require use of validation run

      returns (results, validation results)
    '''
    raise NotImplementedError

  def breed(self,worker):
    '''
      takes a worker to breed with and performs the crossover operation
    '''
    raise NotImplementedError
  
  def mutate(self,):
    raise NotImplementedError

  def gen_mask(self,):
    '''
      Creates a small mask to apply (add) to a a workers 
      genes when performing mutations 
    '''
    raise NotImplementedError

  def config(self, episodes=1, validate=False, render=False):
    self.render   = render
    self.episodes = episodes
    self.validate = validate
  
  def clone(self):
    '''
      returns new worker with new genes
    '''
    raise NotImplementedError