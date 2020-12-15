import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  import gym
  import rltoolkit
  from rltoolkit.utils import test_network
  from config import create_model, ENV_NAME

env = gym.make(ENV_NAME)
model = create_model()
test_network(model, env, episodes=5, verbose=1)