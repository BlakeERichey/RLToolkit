import gym
import tensorflow as tf
import rltoolkit
from rltoolkit.agents import ANN
from rltoolkit.utils import test_network

def test_ann():
  env = gym.make('CartPole-v0')

  model = ANN(env, topology=[32])

  avg = test_network(model, env)
  print('AVG:', avg)

if __name__ == '__main__':
  test_ann()