import random
import numpy as np
from colony import Colony
from worker import Worker

class Evolution:

  def __init__(self, nn, num_colonies=50, num_workers=75, alpha=0.01):
    self.colonies = [Colony(nn) for _ in range(num_colonies)]
    self.workers  = [Worker(nn, alpha) for _ in range(num_workers)]
    self.pyramid =  self._create_pyramid()

  def train(self, env, generations, 
            elites=None, sharpness=1, goal=None, 
            patience=25, return_worker=False):
    assert len(self.colonies), "No Colony Created."
    for gen in range(generations):
      ranked = []
      for i, colony in enumerate(self.colonies):
        res, val = colony.fitness(
          env=env,
          sharpness=sharpness, 
          validate=validate, 
          render=render,
        )
        ranked.append((i, res, val))

      ranked = sorted(ranked, key=lambda x: (x[1], x[2]), reverse=True)
      print("Gen:", gen, "Ranked:", ranked, '\n')
      if goal:
        goal_met = ranked[0][1]>=goal
      
      #WRONG
      #next gen
      if gen != self.generations - 1 and not(goal_met):
        #Gen new weights
        mating_pool = self.selection(ranked)
        new_weights = []
        for i, worker in enumerate(mating_pool):
          if len(new_weights) < self.pop_size - self.elites:
            parent1 = self.colonies[worker[0]]
            parent2 = self.colonies[mating_pool[-i][0]]
            weights = parent1.breed(parent2)
            new_weights.append(weights)
        
        #determine if new mask is needed
        for i, tup in enumerate(ranked):
          worker_id = tup[0]
          self.colonies[worker_id].add_rank(i, self.pop_size)

        #apply new weights and mutate
        for i, worker in enumerate(self.colonies):
          if i > self.elites:
            worker.net.weights = new_weights[i-self.elites]
          worker.mutate()
      
      if goal_met:
        break
        
    best_worker = ranked[0][0] #id
    worker = self.colonies[best_worker]
    if return_worker:
      return worker
    return worker.net
  
  def selection(self, ranked):
    #WRONG
    mating_pool = []
    for i in range(self.elites):
      mating_pool.append(ranked[i])
    
    remaining = np.random.choice(len(ranked), len(ranked), replace=False)
    for i in remaining:
      worker = ranked[i]
      if worker not in mating_pool:
        mating_pool.append(worker)

    return mating_pool

  def _create_pyramid(self,):
    pyramid = [1 for _ in range(len(self.colonies))]

    #generate pyramid [1,1,1,1,2,3,4,5,6,7]
    i = 0
    workers_remaining=len(self.workers)-len(self.colonies)
    while workers_remaining>0:
      if i+1<workers_remaining:
        self.pyramid[i] += i+1
        workers_remaining -= i+1
      else:
        self.pyramid[i]+=workers_remaining
        workers_remaining=0
      i+=1

    pyramid = sorted(pyramid, reverse=True)
    return pyramid
  
  def _assign_workers(self,):
    random.shuffle(self.workers)
    assigned = 0
    for i, colony in enumerate(self.colonies):
      assigning = self.pyramid[i]
      colony.workers = self.workers[assigned:assigning]
      assigned+=assigning
