from seeker import Seeker
import numpy as np

class Evolution:

  def __init__(self,):
    self.seekers = []
    self.configured = False

  def create_species(self, worker):
    assert, self.configured == True, "Evolution params not configured. Run Evolution.config()."

    self.seekers.append(Seeker(worker.genes, self.patience))
    for _ in range(self.num_seekers-1):
      self.seekers.append(Seeker(worker.clone().genes, self.patience))

  def train(self, env, return_worker=False):
    assert len(self.seekers), "No Species Created."
    for gen in range(self.generations):
      ranked = []
      for i, worker in enumerate(self.seekers):
        res, val = worker.fitness(env, self.sharpness, validate, render)
        ranked.append((i, res, val))

      ranked = sorted(ranked, key= lambda x: (x[1], x[2]), reverse=True)
      print("Gen:", gen, "Ranked:", ranked, '\n')
      if self.goal:
        if self.metric == 'reward':
          goal_met = ranked[0][1]>=self.goal
        else:
          goal_met = ranked[0][2]>=self.goal and ranked[0][1]>=self.goal
      
      #next gen
      if gen != self.generations - 1 and not(goal_met):
        #Gen new weights
        mating_pool = self.selection(ranked)
        new_weights = []
        for i, worker in enumerate(mating_pool):
          if len(new_weights) < self.pop_size - self.elites:
            parent1 = self.seekers[worker[0]]
            parent2 = self.seekers[mating_pool[-i][0]]
            weights = parent1.breed(parent2)
            new_weights.append(weights)
        
        #determine if new mask is needed
        for i, tup in enumerate(ranked):
          worker_id = tup[0]
          self.seekers[worker_id].add_rank(i, self.pop_size)

        #apply new weights and mutate
        for i, worker in enumerate(self.seekers):
          if i > self.elites:
            worker.net.weights = new_weights[i-self.elites]
          worker.mutate()
      
      if goal_met:
        break
        
    best_worker = ranked[0][0] #id
    worker = self.seekers[best_worker]
    if return_worker:
      return worker
    return worker.net

  
  def selection(self, ranked):
    mating_pool = []
    for i in range(self.elites):
      mating_pool.append(ranked[i])
    
    remaining = np.random.choice(len(ranked), len(ranked), replace=False)
    for i in remaining:
      worker = ranked[i]
      if worker not in mating_pool:
        mating_pool.append(worker)

    return mating_pool
  
  def config(self, 
    generations, 
    num_seekers, 
    elites,
    num_workers,
    patience=25,
    sharpness=1, 
    goal=None, 
    metric='reward',):

    self.goal        = goal
    self.metric      = metric
    self.elites      = elites
    self.pop_size    = pop_size
    self.patience    = patience
    self.sharpness   = sharpness
    self.generations = generations
    self.num_seekers = num_seekers
    self.num_workers = num_workers

    assert self.elites <= self.pop_size
    assert self.sharpness > 0
    assert self.num_workers >= self.num_seekers
    
    self.configured = True