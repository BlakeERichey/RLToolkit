import random
import numpy as np
from copy import deepcopy
from datetime import datetime
from gym.utils import seeding
from keras.optimizers import Adam
from keras.models import clone_model
from collections import namedtuple
from rltoolkit.errors import EarlyStopError
from rltoolkit.backend import MulticoreBackend, DistributedBackend
from rltoolkit.utils import format_time, test_network, truncate_weights

class Evo:
  """
    NeuroEvolutionary Strategy utilizing a fixed topology.
  """

  def __init__(self, pop_size=None, elites=None):
    """
      Initialized a NeuroEvolution RL method

      #Arguments 
      pop_size: Int. Quantity of neural networks to make. 
        Total population of species.
      elites: Int. A subquantity of the total population that persist between 
        generations. Should be a fraction of the pop_size.
    """

    assert pop_size and elites, "Must declare population size and quantity of elites."
    
    if not isinstance(pop_size, int):
      raise ValueError('Invalid pop_size parameter type. Expected Int, received %s' %type(pop_size))
    if not isinstance(elites, int):
      raise ValueError('Invalid elites parameter type. Expected Int, received %s' %type(elites))

    assert pop_size>0 and elites>=0, "Invalid pop_size or elite quantity."

    self.elites   = elites
    self.pop_size = pop_size
  
  def train(self, nn, env, generations=1, goal=None, episodes=1, verbose=1, 
            callbacks=[], backend=None):
    """
      Trains using NeuroEvolutionary RL method.

      # Arguments
      nn: A keras Neural Network.
      env: A Gym environment
      generations: Int. How many generations to train.
      episodes: Int. How many episodes to run through environment for each individual.
      goal: Float. Object reward for colony to reach. Training ends early if this goal 
        is reached.
      verbose: Int. Reports results of training after this many generations.
      callbacks:  list of functions to call upon completion of a generation.
      backend: If no backend is provided, the training will be done utilizing 
        cores as appropriate for model predictions. Providing a backend allows 
        specific control of how many cores and computers are used. Look at 
        rltoolkit.backend for details.
    """

    print('Creating Population of Size: %s...' %(self.pop_size), end='')
    
    self.nn = nn
    self.env = env
    self.backend = backend
    population = [truncate_weights(nn.get_weights(), n_decimals=3)]

    for i in range(1, self.pop_size):
      nn = clone_model(nn)
      population.append(truncate_weights(nn.get_weights(), n_decimals=3))
    
    print('Done.')
    start_time = datetime.now()
    print('Starting training:', start_time)

    for gen in range(generations):
      ranked = self._evaluate_population(population, env, episodes)

      #Display Results
      if isinstance(verbose, int) and verbose and gen % verbose == 0:
        dt = datetime.now() - start_time
        t = format_time(dt.total_seconds())

        rewards = [score.reward   for score in ranked]
        results = f'Gen: {gen+1}/{generations} | ' + \
          f'Max: {max(rewards):.4f} | ' + \
          f'Avg: {sum(rewards)/len(rewards):.4f} | ' + \
          f'Min: {min(rewards):.4f} | ' + \
          f'Time: {t}'
        print(results)

      goal_met = goal and ranked[0].reward>=goal

      if callbacks:
        #set to self.nn weights for callbacks
        best_weights = population[ranked[0].id]
        self.nn.set_weights(best_weights)

        params = {
          'best_total': ranked[0].reward,
          'rewards':    [score.reward   for score in ranked], #res
        }
        early_stop = False
        for callback in callbacks:
          try:
            callback.run(self, params)
          except EarlyStopError:
            early_stop = True
        
        if early_stop:
          break

      if goal_met:
        break

      #Next generation
      if gen != generations - 1:
        #generate mating pools
        mating_pool = self._selection(ranked, self.elites)

        #perform breeding
        new_individuals  = []
        for i in range(self.pop_size - self.elites):
          parent1 = population[mating_pool[i].id] #mating_pool[i] = (id, reward)
          parent2 = population[mating_pool[-i].id]
          new_weights = self._crossover(parent1, parent2)
          new_individuals.append(new_weights)

        #modify population and mutate, ignoring elites
        for i in range(self.elites, self.pop_size):
          _id = ranked[i].id
          population[_id] = self._mutate(new_individuals[i-self.elites])

    best_weights = population[ranked[0].id]
    self.nn.set_weights(best_weights)
    return self.nn


  #========== Utility Functions ===============================================

  
  def _evaluate_population(self, population, env, episodes):
    """
      Runs each individual of the population through the environment and records
      their results.

      #Arguments
      population: a list of numpy arrays that correspond to a Keras NN's weights.
      env: a gym environment.
      episodes: Int. How many episodes to run through environment for each individual.

      #Returns
      Indivuduals ranked by their avg reward of a number of episodes.
      uses namedtuple (id, reward) where `id` is the individual's 
      index in population and `reward` is the individual's fitness. 
    """
    fitnesses = []
    Fitness = namedtuple('fitness', 'id reward')
    _, seed = seeding.np_random()

    if self.backend is not None:

      #start queueing tasks
      for i, weights in enumerate(population):
        
        nn  = self.nn
        env = self.env
        nn.set_weights(weights)
        self.backend.run(i, test_network, nn, env, episodes, seed=seed)
      
      #Get results
      if isinstance(self.backend, MulticoreBackend):
        res = self.backend.join()
      elif isinstance(self.backend, DistributedBackend):
        res = self.backend.get_results(self.pop_size)

      #put results into order they were called, not order completed  
      res.sort(key= lambda val: val['pid'])
      res = [val['result'] for val in res]
      for i, avg in enumerate(res):
        fitnesses.append(Fitness(i, avg))
    else:
      for i, weights in enumerate(population):
        self.nn.set_weights(weights)
        avg = test_network(self.nn, env, episodes, seed=seed)
        fitnesses.append(Fitness(i, avg))
 
    ranked = sorted(
      fitnesses, 
      key=lambda fitness: (fitness.reward),
      reverse=True
    )

    # print('Ranked:', ranked)

    return ranked

  def _crossover(self, parent1, parent2):
    """
      Performs genetic breeding of parent1 and parent2 to spawn a new individual.

      #Arguments
      parent1: a numpy array defining a keras NN's weights.
      parent2: a numpy array defining a keras NN's weights.

      #Returns
      new_weights to be set as a NN weights via NN.set_weights(new_weights).
      AKA an individual in the population
    """
    #Uncomment print statements to see how this function works
    new_weights = list()
    
    #ensure weight structures are the same
    for layer1, layer2 in zip(parent1, parent2):
        assert layer1.shape == layer2.shape, 'Colonies don\'t have same shape'
        new_weights.append(np.zeros_like(layer1))

    #begin breedings
    for i, layer1, layer2 in zip(range(len(new_weights)), parent1, parent2):
        if new_weights[i].ndim == 1:
            # This method is potentially dangerous since I'm not sure if layer can be other then 2 dimensional
            # and bias can be other than 1 dimensional
            # Bias is always set to 0
            continue
        for j, weight1, weight2 in zip(range(len(new_weights[i])), layer1, layer2):
            seeds = random.sample(range(len(new_weights[i][j])), random.choice(range(len(new_weights[i][j]))))
            #print(i, j, weight1, weight2, seeds)
            '''
            i and j = number of iteration,
            weight1 and weight2 = current row looking at,
            seeds = column location of weight1 that will be in new weights

            example:
            if output = 2 0 [0. 0. 0. 0.] [1. 1. 1. 1.] [3, 2].
            first row of second layer matrix of colony weights will be
            [1. 1. 0. 0.]
            '''
            #print()
            for seed in seeds:
                new_weights[i][j][seed] = weight1[seed]
            for seed in range(len(new_weights[i][j])):
                if seed not in seeds:
                    new_weights[i][j][seed] = weight2[seed]

        new_weights[i] = np.around(new_weights[i].astype(np.float64), 3)

    return new_weights

  def _selection(self, ranked, elites):
    """
      Grabs a random pool for breeding next generation with elites at the front

      # Arguments
      ranked: list of (id, rewards rewards) for each colony 
        generated by their fitness method.
      elites: Int. Number of elites in population.

      # Returns
      list of namedtuples with (id, reward fields)
    """
    mating_pool = []
    for i in range(elites):
      mating_pool.append(ranked[i])
    
    remaining = random.sample(ranked[elites:], len(ranked)-elites)
    mating_pool = mating_pool+remaining
    return mating_pool

  def _mutate(self, weights, alpha=.01):
    alpha = round(np.random.normal(0, .05, 1)[0], 2)
    mask = truncate_weights(weights.copy(), alpha=alpha, n_decimals=3)
    #Apply mask in place
    for i, layer in enumerate(weights):
      layer += mask[i]
    
    return weights