import keras
import random
import numpy as np

class ReplayBuffer:

  """
    Class dedicated to managing a replay buffer for RL methods.

    Creates a replay buffer of determined size and enables adding experiences
    to the buffer and taking random samples.
  """

  def __init__(self, max_size=100):
    """
      Initializes an empty ReplayBuffer with a max size of `max_size`.
    """

    self.size = 0
    self.memory = []
    self.max_size = max_size
  
  def remember(self,experience):
    """
      Stores an experience into replay buffer and removes older memories
      if resulting ReplayBuffer excedes max buffer size.

      For DQL an example experiences would be:
      experience = [prev_envstate, action, reward, envstate, done]
    """

    self.memory.append(experience)
    self.size+=1
    
    if len(self.memory) > self.max_size:
      del self.memory[0]
      self.size-=1
  
  def get_batch(self,batch_size=1):
    """
      Grabs `batch_size` random elements from ReplayBuffer and returns
      them as a numpy array.
    """
    batch_size = min(self.size, batch_size)
    return random.sample(self.memory, batch_size)

class Checkpoint:
  """
    Model utility that saves the best model during episode runs
  """

  def __init__(self,filename,save_weights_only=False):
    """
      # Arguments
        filename: target filename to save model file to
        save_weights_only: If set to False, saves model architecture and weights.
          if True, saves only weights.
    """
    self.filename = filename
    self.max_reward = None
    self.best_model = None
    self.save_weights_only = save_weights_only

  def run(self,obj,rewards):
    """
      Determines if the best model stored has been improved.
      If so, saves the new best model. 

      # Arguments
        obj: A RL Method. Used to acquire methods network instance variable.
        rewards: list of rewards obtained during an episode.
    """
    updated = False #save again?
    if self.max_reward is None:
      updated = True
      self.max_reward = sum(rewards)
      self.best_model = obj.nn
    else:
      reward = sum(rewards)
      if reward >= self.max_reward:
        updated = True
        self.max_reward = reward
        self.best_model = obj.nn
    
    if updated:
      if self.save_weights_only:
        self.best_model.save_weights(self.filename)
      else:
        self.best_model.save(self.filename)