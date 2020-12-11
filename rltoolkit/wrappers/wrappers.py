import numpy as np
class EnvManager():
  def __init__(self, nn, env):
    
    """
      Manages action predictions for networks interfacing with an environment.
      Implicitly determines if a Time distributed observation is necessary, 
      and evaluates model predictions into interpretable actions.
    """

    self.nn = nn
    self.env = env
    self.discrete = hasattr(env.action_space, 'n') #environment is discrete

    self.ob = None

    # Technically correct. (30,256,256,3)[1:] == (256,256,3). 
    # Doesnt test that network is valid. Handled elsewhere
    self.time_dist = (nn.input_shape[1:] != env.observation_space.shape)
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


def subprocess_wrapper(func, res, pid, *args, **kwargs):
  """
    Wraps `func` as a multiprocessing.Process ready 
    subprocess by evaluating func(*args, **kwargs) then 
    enqueues the results in `res`. Uses pid to disinguish the process id.

    # Arguements
    func: a function to be executed in a subprocess.
    res: a multiprocessing.Manager().Queue() to enqueue data into upon completion.
    pid: a process id that is used to distinguish processes from each other 
      when running in parallel.
    args: arguments for `func`.
    kwargs: keyword arguments for `func`.

    #Example Use

    def example_func(a, b=3):
      return a+b
    
    >>>queue = Manager().Queue()
    >>>args = (example_func, queue, 0) #args for subprocess wrapper 
    >>>args += (2,) #args for `example_func` (comma is needed here)
    >>>kwargs = {'b': 5}
    >>>p = Process(target=subprocess_wrapper, args=args, kwargs=kwargs)
    >>>p.start()
    >>>p.join()
    >>>queue.get() #returns (2+5) = 7
  """
  try:
    retval = func(*args, **kwargs)

    #store returned value
    res.put({
      'pid': pid,
      'result': retval,
    }) 
  except ConnectionRefusedError:
    pass
  except ConnectionResetError:
    pass

class CallbackWrapper:
  
  def __init__(self):
    """
      Allows for chaining of functions
    """
    pass
  
  def register(self, name, func):
    setattr(self, name, func)