import gym
import tensorflow as tf
import rltoolkit
from rltoolkit.utils import test_network
from rltoolkit.agents import ANN, CNN, LSTM_ANN, LSTM_CNN

def test_ann():
  env = gym.make('CartPole-v0')

  model = ANN(env, topology=[32])

  avg = test_network(model, env)
  print('AVG:', avg)

def test_agents():
  env = gym.make('CartPole-v0')
  model = ANN(env, topology=[24])
  model = LSTM_ANN(env, topology=[24])

  env = gym.make('BattleZone-v0')
  model = CNN(env, topology=[256, 128, 64])
  model = LSTM_CNN(env, cnn_topology=[64, 128, 256], fcn_topology=[64,32,16])
  print('Agents Created Successfully.')

if __name__ == '__main__':
  test_ann()
  test_agents()