import gym
from rltoolkit.agents import ANN

def create_model():
  env = gym.make('LunarLander-v2')
  model = ANN(env=env, topology=[64,256,128])

  return model