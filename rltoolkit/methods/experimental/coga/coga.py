import random
import numpy as np
from .colony import Colony
from .worker import Worker
from datetime import datetime
from collections import namedtuple
from rltoolkit.utils import format_time
from rltoolkit.errors import EarlyStopError

class COGA:

  def __init__(self, nn, num_colonies=50, num_workers=75, alpha=0.01):
    #Assert num_workers >= num_colonies and >=0 and an Int
    self.nn       = nn
    self.pop_size = num_colonies
    self.colonies = [Colony(nn) for _ in range(num_colonies)]
    self.workers  = [Worker(nn, alpha) for _ in range(num_workers)]
    self.pyramid  =  self._create_pyramid()
    self._assign_workers()
    
    #load provided network in first colony
    self.colonies[0].set_network(nn)

  def train(self, env, generations=1, elites=None, 
            sharpness=1, goal=None, patience=25, 
            validate=False, verbose=1, return_colony=False, callbacks=[],):
    """
      Trains using COGA RL method.

      # Arguments
      env: A Gym environment
      generations: Int. How many generations to train
      elites: Int. Number of top ranking colonies to ensure remain at each 
        generation
      sharpness: Int. How many episodes to run through environment for each worker
      goal: Float. Object reward for colony to reach. Training ends early if this goal 
        is reached.
      patience: How many generations each colony has to meet or exceed other 
        elites, otherwise the genes are abandoned.
      validate: Runs a second iteration through environment with the same 
        sharpness. Use if the environment is especially stochastic.
      verbose: Int. Reports results of training after this many generations.
      callbacks:  list of functions to call upon completion of a generation.
    """
    assert len(self.colonies), "No Colony Created."
    #Assert elites <= popsize

    if not elites:
      elites = int(0.25*self.pop_size)

    goal_met = False
    Fitness = namedtuple('fitness', 'id reward v_reward')
    start_time = datetime.now()
    print('Starting training:', start_time)
    for gen in range(generations):

      #Evaluate all colonies
      ranked = []
      for i, colony in enumerate(self.colonies):
        res, val = colony.fitness(
          env=env,
          sharpness=sharpness, 
          validate=validate, 
        )
        if verbose == 'testing':
          print(i, res, val)
        ranked.append(Fitness(i, res, val))

      ranked = sorted(
        ranked, 
        key=lambda fitness: (fitness.reward, fitness.v_reward), 
        reverse=True
      )

      #Display Results
      if isinstance(verbose, int) and verbose and gen % verbose == 0 or verbose == 'testing':
        dt = datetime.now() - start_time
        t = format_time(dt.total_seconds())

        rewards = [score.reward   for score in ranked]
        results = f'Gen: {gen+1}/{generations} | ' + \
          f'Max: {max(rewards):.4f} | ' + \
          f'Avg: {sum(rewards)/len(rewards):.4f} | ' + \
          f'Min: {min(rewards):.4f} | ' + \
          f'Time: {t}'
        print(results)
      
      if verbose == 'testing':
        print(f'Gen: {i+1}/{generations}\n {ranked} \n\n')

      if goal:
        if not validate:
          goal_met = ranked[0].reward>=goal
        else:
          goal_met = ranked[0].reward>=goal and ranked[0].v_reward>=goal
      
      #Get next generation of networks
      if gen != generations - 1 and not(goal_met):
        
        elite_ids = set([ranked[i].id for i in range(elites)])
        
        #How many colonies need to be remade?
        remake = 0
        for i, colony in enumerate(self.colonies):
          if not hasattr(colony, 'last_time_elite'):
            colony.last_time_elite = 0
          if colony.last_time_elite > patience and i not in elite_ids:
            remake+=1

        #Gen new replacement colonies
        mating_pool = self._selection(ranked, elites)
        new_colonies  = []
        for i, colony in enumerate(mating_pool):
          if len(new_colonies) < remake:
            parent1 = self.colonies[colony.id] #colony = (index, res, val)
            parent2 = self.colonies[mating_pool[-i].id]
            new_colony = parent1.breed(parent2)
            new_colonies.append(new_colony)
          else:
            break
        
        #Replace bad colonies, update non elites
        remade = 0
        for i, colony in enumerate(self.colonies):

          restart = False #restart patience counter?
          if len(new_colonies):
            if colony.last_time_elite > patience and i not in elite_ids:
              self.colonies[i] = new_colonies[remade]
              remade += 1
              restart = True
          
          if i in elite_ids or restart:
            self.colonies[i].last_time_elite = 0
          else:
            self.colonies[i].mutate() #mutate non elites
            self.colonies[i].last_time_elite += 1
        

        #Logic for generating new workers goes here, currently not necessary
      
      if callbacks:
        #get best network, set to self.nn
        best_colony = self.colonies[ranked[0].id]
        best_worker = best_colony.workers[best_colony.best_worker]
        self.nn.set_weights(best_colony.weights)
        best_worker._apply_mask(self.nn)

        params = {
          'best_total':              ranked[0].reward,
          'best_total_validations':  ranked[0].v_reward,
          'rewards':                 [score.reward   for score in ranked], #res
          'validations':             [score.v_reward for score in ranked], #val
        }
        stop = False
        for callback in callbacks:
          try:
            callback.run(self, params)
          except EarlyStopError:
            stop = True
        
        if stop:
          break

      if goal_met:
        break
        
      self._assign_workers() #done after callbacks to not lose best worker
        
    best_colony = self.colonies[ranked[0].id]
    if return_colony:
      return best_colony

    best_worker = best_colony.workers[best_colony.best_worker]
    self.nn.set_weights(best_colony.weights)
    best_worker._apply_mask(self.nn)
    return self.nn
  
  def _selection(self, ranked, elites):
    """
      Grabs a random pool for breeding next generation with elites at the front

      # Arguments
      ranked: list of (id, rewards, validation rewards) for each colony 
        generated by their fitness method.
      elites: Int. Number of elites in population.

      # Returns
      list of namedtuples with (id, reward, v_reward fields)
    """
    mating_pool = []
    for i in range(elites):
      mating_pool.append(ranked[i])
    
    remaining = random.sample(ranked[elites:], len(ranked)-elites)
    mating_pool = mating_pool+remaining
    return mating_pool

  def _create_pyramid(self,):
    """
      Generates a pyramid quantifying how many workers to assign to each Colony
    """
    pyramid = [1 for _ in range(len(self.colonies))]

    #generate pyramid [1,1,1,1,2,3,4,5,6,7]
    i = 0
    workers_remaining=len(self.workers)-len(self.colonies)
    while workers_remaining>0:
      if i+1<workers_remaining:
        pyramid[i] += i+1
        workers_remaining -= i+1
      else:
        pyramid[i]+=workers_remaining
        workers_remaining=0
      i+=1

    pyramid = sorted(pyramid, reverse=False)
    return pyramid
  
  def _assign_workers(self,):
    """
      Reassigns all workers to new colonies using self.pyramid to assign more 
      workers to the elites
    """
    random.shuffle(self.workers)
    assigned = 0
    for i, colony in enumerate(self.colonies):
      assigning = self.pyramid[i]
      colony.workers = self.workers[assigned:assigned + assigning]
      assigned+=assigning