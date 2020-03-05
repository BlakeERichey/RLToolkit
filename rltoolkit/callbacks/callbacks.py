import matplotlib.pyplot as plt

class Graph:
  """
    Used to graph rewards generated from RL Method after each episode
  """

  def __init__(self,):
    self.episode_rewards = {
      'min':        [],
      'max':        [],
      'epoch':      [],
      'average':    [],
      'cumulative': [],
    }
  
  def run(self,obj,rewards):
    """
      Saves rewards to memory and stores for graphing later.

      # Arguments
        obj: A RL Method. Used to acquire methods network instance variable.
        rewards: list of rewards obtained during an episode.
    """
    min_r = min(rewards)
    max_r = max(rewards)
    avg_r = sum(rewards)/len(rewards)
    
    epoch = len(self.episode_rewards['epoch'])
    self.episode_rewards['epoch'].append(epoch)
    self.episode_rewards['cumulative'].append(sum(rewards))
    self.episode_rewards['min'].append(min_r)
    self.episode_rewards['max'].append(max_r)        
    self.episode_rewards['average'].append(avg_r)
  
  def show(self,version=['cumulative'],loc=4):
    """
      Graphs rewards per epoch on matplotlib pyplot and 
      shows the resulting graph.

      # Arguments
        version: array with options of ['cumulative', 'min', 'max', 'avg']. 
          Determines which stats of recorded rewards are plotted. 
          Pass None to display all metrics on one graph.
        loc: location for legend to appear on graph. Pass None for no legend.
    """
    self._plot(version, loc)
    plt.show()
  
  def _plot(self,version,loc):
    if version is None:
      version = ['cumulative', 'min', 'max', 'avg']
    
    labels = ['Cumulative Rewards','Min Rewards','Max Rewards', 'Avg Rewards']
    for v in version:
      try:
        x = self.episode_rewards.get('epoch')
        y = self.episode_rewards.get(v)
        label = labels[['cumulative', 'min', 'max', 'avg'].index(v)]
        plt.plot(x, y, label=label)
      except:
        #Value error raised by .index
        pass
    
    if loc is not None:
      plt.legend(loc=loc)

  def save(self,filename,version=['cumulative'],loc=4):
    """
      Generates then saves graph of rewards without rendering the graph.
      
      # Arguments
        filename: Name of file to save graph as.
        version: array with options of ['cumulative', 'min', 'max', 'avg']. 
          Determines which stats of recorded rewards are plotted. 
          Pass None to display all metrics on one graph.
        loc: location for legend to appear on graph. Pass None for no legend.
    """
    self._plot(version, loc)
    plt.savefig(filename)
    plt.close()

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