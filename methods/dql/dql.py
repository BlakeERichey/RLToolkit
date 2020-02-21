import numpy as np
from datetime import datetime
from utils    import ReplayBuffer, format_time

class DQL:

  """
    DQL is a class the encapsulates the Deep Q Learning Method for 
    Reinforcement Learning.
  """

  def __init__(self,gamma=.9,rb_size=100,replay_batch_size=10,callbacks=[]):
    """
      Initializes a DQL Method. 
    
      # Arguments
        gamma: Discount factor. Effectively a ratio of short term rewards 
          significance to long term reward signficance. 
        rb_size: ReplayBuffer size.
        replay_batch_size: size of batches to take from ReplayBuffer at each step
        callbacks:  list of functions to call upon completion of an episode.
    """

    self.gamma = gamma
    self.callbacks = callbacks
    self.replay_buffer = ReplayBuffer(rb_size)
    self.replay_batch_size = replay_batch_size
  
  def train(self, nn, env, episodes, verbose=1, 
            epsilon_start=1, epsilon_decay=.99, min_epsilon=.001, batch_size=1):
    """
      Trains a Neural network to solve a Gym environment

      # Arguments
        nn: Keras neural network
        env: initialized Gym environment
        episodes: How many episodes of an environment to run. 
        verbose: Int. Reports results of training after this many episodes. 
        epsilon_start: Initial exploration factor. This determines percentage 
          of action are randomly determined.
        epsilon_decay: Rate of decay for epsilon after each episode.
        min_epsilon: Minimum value for epsilon. Ensures model continue exploring.
        batch_size: Int. Qty of replay_batch to fit with network at a time. 
          Should be <= `self.replay_batch_size`. 
          Lower number = Longer training, but better results.
    """

    self.nn            = nn
    self.env           = env
    self.min_epsilon   = min_epsilon
    self.epsilon       = epsilon_start
    self.epsilon_decay = epsilon_decay
    self.batch_size    = min([self.replay_batch_size, batch_size, 1])

    self._discrete = hasattr(self.env.action_space, 'n') #environment is discrete

    start_time = datetime.now()
    for i in range(episodes):
      rewards, steps = self._run_episode()
      
      #Display Results
      if verbose and i % verbose == 0:
        dt = datetime.now() - start_time
        t = format_time(dt.total_seconds())

        results = f'Episode: {i+1}/{episodes} | ' + \
          f'Steps: {steps} | ' + \
          f'Epsilon: {self.epsilon} | ' + \
          f'Reward: {sum(rewards)} | ' + \
          f'Time: {t}'
        print(results)
      
      for callback in self.callbacks:
        callback(self)
      self.epsilon = max(self.epsilon*self.epsilon_decay, self.min_epsilon)
    
    return nn

  def _run_episode(self, max_steps=None):
    """
      Utility function that runs through environment using 
      epsilon-greedy strategy.
    """
    steps = 0
    done = False
    rewards = []
    envstate = self.env.reset()
    while (not done,steps<max_steps and not done)[max_steps]:
      prev_envstate = envstate #DQL variable
      
      action = self._predict(envstate)
      envstate, reward, done, info = self.env.step(action)
      rewards.append(reward)
      steps+=1
      
      #Update Replay Buffer
      experience = [prev_envstate, action, reward, envstate, done]
      self.replay_buffer.remember(experience)
      self._learn() #Improve at each step
    
    return rewards, steps

  def _predict(self,envstate):
    """
      Utility function that evaluates what action to take in environment
      based on epsilon-greedy exploration.

      # Arguments
        envstate: envstate from a Gym environment
    """
    if np.random.rand() < self.epsilon:
      action = self.env.action_space.sample()
    else:
      qvals = self.nn.predict(envstate)[0]
      if self._discrete:
        action = np.argmax(qvals)
      else:
        action = qvals
    
    return action
  
  def _learn(self,):
    """
      Needs testing
    """
    batch = self.replay_buffer.get_batch(self.replay_batch_size)

    #Get inputs
    inputs_shape = (len(batch),) + self.env.observation_space.shape
    inputs = np.zeros((inputs_shape))
    for i, episode in enumerate(batch):
      prev_envstate, _, _, _, _ = episode
      inputs[i] = prev_envstate
    
    #Get expected outputs
    targets = self.nn.predict(inputs)

    #Update expectations
    for i, episode in enumerate(batch):
      prev_envstate, action, reward, envstate, done = episode

      if done:
        targets[i, action]
    
    pass
