import numpy as np
from datetime import datetime
from rltoolkit.utils import ReplayBuffer, format_time

class DQL:

  """
    DQL is a class the encapsulates the Deep Q Learning Method for 
    Reinforcement Learning.
  """

  def __init__(self,gamma=.9,rb_size=100,replay_batch_size=10):
    """
      Initializes a DQL Method. 
    
      # Arguments
        gamma: Discount factor. Effectively a ratio of short term rewards 
          significance to long term reward signficance. 
        rb_size: ReplayBuffer size.
        replay_batch_size: size of batches to take from ReplayBuffer at each step
    """

    self.gamma = gamma
    self.replay_buffer = ReplayBuffer(rb_size)
    self.replay_batch_size = replay_batch_size
  
  def train(self, nn, env, episodes, verbose=1, 
            epsilon_start=1, epsilon_decay=.99, 
            min_epsilon=.001, batch_size=1, epochs_per_step=1, callbacks=[]):
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
        epochs_per_step: Int. Number of epochs to fit revised 
          expectations. Higher number = Longer training = Better results.
        callbacks:  list of functions to call upon completion of an episode.
    """

    self.nn            = nn
    self.env           = env
    self.min_epsilon   = min_epsilon
    self.epsilon       = epsilon_start
    self.epsilon_decay = epsilon_decay
    self.epochs        = epochs_per_step
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
          f'Epsilon: {self.epsilon:.4f} | ' + \
          f'Reward: {sum(rewards):.4f} | ' + \
          f'Time: {t}'
        print(results)
      
      for callback in callbacks:
        callback.run(self, rewards)
      self.epsilon = max(self.epsilon*self.epsilon_decay, self.min_epsilon)
    
    return nn

  def test(self, nn, env, episodes=1):
    """
      Test neural network against environment. Renders at each step for
      users to view results.

      # Arguments
        nn: Neural network to be tested.
        env: Gym environment
        episodes: Number of episodes to run nn through environment.
    """
    self.nn = nn
    self.env = env
    self.epsilon = 0
    self._discrete = hasattr(self.env.action_space, 'n') #environment is discrete

    for i in range(episodes):
      steps = 0
      done = False
      rewards = []
      envstate = self.env.reset()
      while not done:
        action = self._predict(envstate)
        envstate, reward, done, _ = self.env.step(action)
        env.render()
        rewards.append(reward)
        steps+=1
      
      results = f'Episode: {i+1}/{episodes} | ' + \
        f'Reward: {sum(rewards):.4f}'
      print(results)
    env.close()

  def _run_episode(self, max_steps=None):
    """
      Utility function that runs through environment using 
      epsilon-greedy strategy.
    """
    steps = 0
    done = False
    rewards = []
    envstate = self.env.reset()
    while not done and (steps < max_steps if max_steps is not None else True):
      prev_envstate = envstate #DQL variable
      
      action = self._predict(envstate)
      envstate, reward, done, _ = self.env.step(action)
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
    num = np.random.rand()
    if num < self.epsilon:
      action = self.env.action_space.sample()
    else:
      qvals = self.nn.predict(np.expand_dims(envstate, axis=0))[0]
      if self._discrete:
        action = np.argmax(qvals)
      else:
        action = qvals
    
    return action
  
  def _learn(self,):
    """
      Updates neural network by peforming batch learning
    """

    inputs, targets = self._get_batch()

    self.nn.fit(
      inputs,
      targets,
      epochs=self.epochs,
      batch_size=self.batch_size,
      verbose=0,
    )

  def _get_batch(self,):
    """
      Gets batch from ReplayBuffer then applies Bellman Equation 
      to expected outputs
    """
    batch = self.replay_buffer.get_batch(self.replay_batch_size)

    #Get inputs
    inputs = []
    next_inputs = [] #used for Q_sa calculation. Single calulcation instead of multiple
    for i, episode in enumerate(batch):
      prev_envstate, _, _, envstate, _ = episode
      inputs.append(prev_envstate)
      next_inputs.append(envstate)
    inputs = np.array(inputs)
    next_inputs = np.array(next_inputs)
    
    #Get expected outputs
    targets          = self.nn.predict(inputs)
    next_state_qvals = self.nn.predict(next_inputs)

    #Update expectations
    for i, episode in enumerate(batch):
      _, action, reward, _, done = episode

      if done:
        if self._discrete:
          targets[i, action] = reward
        else:
          targets[i] = np.array(reward)
      else:
        q_sa = max(next_state_qvals[i])
        if self._discrete:
          #reward = reward + gamma * max_a' Q(s', a')
          targets[i, action] = reward + self.gamma * q_sa
        else:
          targets[i] = np.array(reward + self.gamma * q_sa)
    
    return inputs, targets
