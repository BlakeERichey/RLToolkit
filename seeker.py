import math, random
import numpy as np

class Seeker:

  '''
    Seeker defines a species for evolutionary strategy. 
    
    Seekers act as a colony of workers and will have each worker 
    run through their respective environments
  '''

  def __init__(self, genes, patience=25,):

    self.genes = genes #initial genes
    self.history = [] #history of ranks in population
    self.patience = patience #number of generations of low performance to accept

  def fitness(self, env):
    quality = [] #list of results from each worker
    for worker in self.workers:
      res, valid_res = worker.fitness(env)
      quality.append((res, valid_res))
    
    ranked = sorted(quality, key= lambda x: (x[0], x[1]), reverse=True)

    return ranked[0] #returns (fitness, validation fitness) of best worker

  # def breed(self,worker):
  #   num_layers = len(worker.net.layers)
  #   splice = np.random.randint(2, size=num_layers)
  #   new_weights = []
  #   for i, layer in enumerate(worker.net.layers):
  #     if splice[i] == 0:
  #       new_weights.append(np.copy(self.net.layers[i].weights))
  #     else:
  #       new_weights.append(np.copy(layer.weights))
    
  #   return new_weights
  
  # def mutate(self,):
  #   mutations = self.mutations
  #   if not len(self.history) or (0 in self.history):
  #     mutations /= len(self.net.layers)
  #   for i, layer in enumerate(self.net.layers):
  #     if np.random.uniform() < mutations / len(self.net.layers):
  #       rows, cols = layer.rows, layer.cols
  #       mask = self.mask[i]
  #       layer.weights = np.add(layer.weights, mask[:rows, :cols])

  # def add_rank(self, rank, pop_size, thresh):
  #   '''takes rank in a generation and logs it to history '''

  #   self.history.append(rank)
  #   if len(self.history)>self.patience:
  #     del self.history[0]
    
  #   low_performing = 0
  #   for _, val in enumerate(self.history):
  #     if val >= self.thresh*pop_size:
  #       low_performing+=1
    
  #   if low_performing >= self.patience:
  #     self.gen_mask()
  #     self.history = [self.history[-1]]
  #     self.net = self.net.clone()

  # def gen_mask(self,):
  #   #GENERATE MASKS
  #   self.mask = []
  #   for layer in self.net.layers:
  #     rows, cols = layer.rows, layer.cols
  #     limit = math.sqrt(6/(rows+cols)) #glorot uniform
  #     mask = self.alpha*np.random.uniform(low=-limit, high=limit, size=(rows*cols,)).reshape((rows, cols))
  #     self.mask.append(mask)

  # def predict(self, model, envstate, continuous=False):
  #   ''' decide best action for model. utility function. '''
  #   qvals = model.feed_forward(envstate)
  #   if continuous == True:
  #     action = qvals #continuous action space
  #   else:
  #     action = np.argmax(qvals) #discrete action space
    
  #   return action

  def set_workers(self, workers):
    assert isinstance(workers, list), f"Expected list of Class objects, received {type(workers)}."
    self.workers = workers