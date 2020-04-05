import keras
import random
import numpy as np
from rltoolkit.wrappers import EnvManager

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

def test_network(nn, env, episodes=1, render=False, verbose=0):
  """
    Tests a Keras neural network by running it through an environment

    # Arguments
    nn: Keras nueral network.
    env: Gym environment.
    episodes: How many episodes to run through gym environment.
    render: Pass True to render at each step.
    verbose: Int. Reports results of training after this many episodes. 
  """
  envman = EnvManager(nn, env)
  
  total_rewards = []
  for i in range(episodes):
    rewards = envman.run(render=render) #run 1 episode
    total_rewards.append(sum(rewards))

    if verbose and i % verbose == 0:
      results = f'Episode: {i+1}/{episodes} | ' + \
        f'Reward: {sum(rewards):.4f}'
      print(results)
  
  if render:
    envman.close()

  result = round(sum(total_rewards)/len(total_rewards), 5)
  return result

def truncate_weights(weights, n_decimals=3, alpha=1):
    """
        Truncates list of weights for a keras network in place

        # Arguments:
        weights: list of numpy arrays corresponding to a Keras networks weights.
        n_decimals: number of decimals to round to.
        alpha: small number to multiply each weight by. 
          Set to 1 to keep weights, and simply truncate
        
        Use:
        model.set_weights(truncate_weights(model.get_weights()))
    """
    for i, w in enumerate(weights):
      weights[i]=alpha*np.around(w.astype(np.float64), n_decimals)
    return weights