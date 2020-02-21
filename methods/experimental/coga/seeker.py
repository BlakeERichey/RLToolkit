import math, random
import numpy as np
from copy import deepcopy

class Seeker:

  '''
    Seeker defines a species for evolutionary strategy. 
    
    Seekers act as a colony of workers and will have each worker 
    run through their respective environments
  '''

  def __init__(self, genes, patience=25,):


    self.history  = []            #history of ranks in population
    self.patience = patience      #number of generations of low performance to accept
    self.genes    = genes  #initial genes

  def fitness(self, env):
    quality = [] #list of results from each worker
    for i, worker in enumerate(self.workers):
      res, valid_res = worker.fitness(env)
      quality.append((i, res, valid_res))
    
    ranked = sorted(quality, key= lambda x: (x[1], x[2]), reverse=True)

    best_worker = ranked[0]
    _id, fitness, validation_fitness = best_worker
    self.genes = [val for val in self.workers[_id].genes]
    return fitness, validation_fitness

  def breed(self,worker):
    self.workers[0].genes = self.genes
    self.workers[0].breed(worker)
    self.genes = deepcopy(self.workers[0].genes)
  
  def mutate(self):
    for worker in self.workers:
      worker.mutate()

  def add_rank(self, rank, pop_size, thresh):
    '''takes rank in a generation and logs it to history '''

    self.history.append(rank)
    if len(self.history)>self.patience:
      del self.history[0]
    
    low_performing = 0
    for _, val in enumerate(self.history):
      if val >= thresh*pop_size:
        low_performing+=1
    
    if low_performing >= self.patience:
      self.reset()
  
  def reset(self):
    self.genes = self.workers[0].clone() #seekers
    self.history = []

  def set_workers(self, workers):
    assert isinstance(workers, list), f"Expected list of Class objects, received {type(workers)}."
    self.workers = workers
    
    for worker in self.workers:
      worker.genes = deepcopy(self.genes)