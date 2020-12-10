import types
import keras
import random
import numpy as np
from numba import njit
from copy import deepcopy
from datetime import datetime
from gym.utils import seeding
import tensorflow as tf
from keras.models import clone_model
from collections import namedtuple
from rltoolkit.errors import EarlyStopError
from rltoolkit.backend import MulticoreBackend, DistributedBackend, LocalClusterBackend
from rltoolkit.utils import format_time, test_network, truncate_weights

class Evo:

  def __init__(self, pop_size=None, elites=None):
    """
      NeuroEvolutionary Strategy utilizing a fixed topology.

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

    print('Creating Population of Size: %s...' %(self.pop_size), end='', flush=True)
    
    self.nn = nn
    self.env = env
    self.backend = backend
    population = [truncate_weights(nn.get_weights(), n_decimals=3)]

    if backend is None:
      for i in range(1, self.pop_size):
        nn = clone_model(nn)
        population.append(truncate_weights(nn.get_weights(), n_decimals=3))
    else:
      #Pass population generation to backend
      print('\nPassing population generation to backend...', end='', flush=True)
      task_ids = []
      for i in range(1, self.pop_size):
        if type(backend) == MulticoreBackend:
          backend.run(duplicate_model, nn)
        else:
          task_id = backend.run(duplicate_model, backend.network_generator)
          task_ids.append(task_id)
      
      #Get new weights from backend
      if type(backend) == MulticoreBackend:
        population.extend(backend.join())
      else:
        population.extend(backend.get_results(task_ids))
    
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

    if backend is not None:
      backend.shutdown()
      
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
      
      task_ids = []
      #start queueing tasks
      for i, weights in enumerate(population):
        #dont send network for DistributedBackend, recreate
        task_id = self.backend.test_network(
          weights,
          self.env, episodes, seed,
          (None, self.nn)[isinstance(self.backend, MulticoreBackend)], 
        )
        task_ids.append(task_id)

      #Get results
      if isinstance(self.backend, MulticoreBackend):
        res = self.backend.join(
          values_only=True, 
          numeric_only=True, 
          ref_value='min'
        )
      else:
        res = self.backend.get_results(
          task_ids=task_ids,
          values_only=True, 
          numeric_only=True, 
          ref_value='min'
        )

      #record fitnesses 
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
      parent1: a multi-dimensional list defining a keras NN's weights.
      parent2: a multi-dimensional list defining a keras NN's weights.
    """
    new_weights = []
    for i in range(len(parent1)):    #Layer in network
      strand1 = parent1[i] #Network layers weights as np.array
      strand2 = parent2[i] #Network layers weights as np.array
      new_strand = breed_strand(strand1, strand2)
      new_strand = truncate_weights(new_strand)
      new_weights.append(new_strand)
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

def duplicate_model(network):
  """
    Utility function to employ a backend if present to generate a population.
    DO NOT CALL FROM CORE THREAD.

    reference_model: Keras network if backend is MulticoreBackend. Otherwise 
      a function that returns a Keras network is expected.
  """

  keras.backend.clear_session()
  if type(network) == types.FunctionType: #Distributed create_model
    nn = network()
    # nn.summary() #To identify is session is being cleared
  else: #Multicore model
    nn  = clone_model(network)

  return truncate_weights(nn.get_weights(), n_decimals=3)


@njit
def breed_strand(strand1, strand2):
  """
    Create a new strand by randomly selecting half the 
    new genes from strand1 and the other half from strand2

    This function is compiled at runtime for faster subsequent calls
  """
  original_shape = strand1.shape
  strand1 = strand1.ravel()
  strand2 = strand2.ravel()
  num_genes = strand1.size

  #Randomly select half the new genes from strand1 and the other from strand2
  indices = np.zeros((num_genes,))
  indices[np.random.choice(num_genes, num_genes//2, replace=False)] = 1

  #Randomly select portion of the new genes from strand1 and the rest from strand2
  # indices = np.zeros((num_genes,))
  # indices[np.random.choice(num_genes, np.random.randint(num_genes), replace=False)] = 1
  
  new_strand = np.where(indices==0, strand1, strand2).reshape(original_shape)
  return new_strand