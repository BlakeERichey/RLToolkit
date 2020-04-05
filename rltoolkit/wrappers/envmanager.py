import numpy as np
class EnvManager():
  """
    Manages action predictions for networks interfacing with an environment.
    Implicitly determines if a Time distributed observation is necessary, 
    and evaluates model predictions into interpretable actions.
  """

  def __init__(self, nn, env):
    self.nn = nn
    self.env = env
    self.discrete = hasattr(env.action_space, 'n') #environment is discrete

    self.ob = None
    self.time_dist = (nn.input.shape.ndims == 3) #not correct when dealing with CNNs
    if self.time_dist:
      self.n_timesteps = nn.input.shape[1]
  
  def run(self,render=False):
    """
      Runs an environment through to completion taking models best determined 
      actions.
    """

    self.reset()
    
    done = False
    rewards = []
    while not done:
      action = self.predict()

      #take action
      _, reward, done, _ = self.step(action)
      rewards.append(reward)

      if render:
        self.render()
    
    return rewards
  
  def step(self, action):
    """
      Takes a step into environment and updates self.ob
    """
    envstate, reward, done, _ = self.env.step(action)

    if self.time_dist:
      self.ob = np.concatenate((self.ob, np.expand_dims(envstate, axis=0)), axis=0)[1:]
    else:
      self.ob = envstate

    return self.ob, reward, done, {}

  def reset(self,):
    """
      Resets self.ob
    """
    self.ob = self.env.reset()
    if self.time_dist:
      self.ob = np.array([self.ob for _ in range(self.n_timesteps)])
    
    return self.ob
  
  def render(self,):
    """
      Render self.env
    """
    self.env.render()
  
  def close(self,):
    """
      If self.render, this must be called else kernal will crash upon 
      program completion. Closes GUI for render component.
    """
    self.env.close()
  
  def predict(self):
    """
      Returns nn's determined optimal action given an observation
    """
    qvals = self.nn.predict(np.expand_dims(self.ob, axis=0))[0]
    if self.discrete:
      action = np.argmax(qvals)
    else:
      action = qvals
    
    return action
    