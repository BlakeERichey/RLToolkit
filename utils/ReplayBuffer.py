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