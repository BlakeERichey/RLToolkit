import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from rltoolkit.methods import COGA
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph

import cProfile, pstats, io
def profile(fnc):
  """A decorator that uses cProfile to profile a function"""
  
  def inner(*args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    retval = fnc(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(.05)
    print(s.getvalue())
    return retval

  return inner

env = gym.make('CartPole-v0')
try:
  print(env.unwrapped.get_action_meanings())
except:
  pass

#Build network
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(Adam(0.001), loss='mse')
model.summary()

@profile
def coga(model):
  #Initialize Deep Q Learning Method
  method = COGA(model, 20, 25)

  #Enable graphing of rewards
  graph = Graph()
  #Make a checkpoint to save best model during training
  ckpt = Checkpoint('cartpole.h5')
  #Train neural network for 50 episodes
  nn = method.train(env, 10, 1, patience=2, validate=True, verbose=1, callbacks=[ckpt, graph])

  #Save and show rewards
  graph.show()
  graph.save('cartpole.png')

  #Load best saved model
  model = load_model('cartpole.h5')

  #Test models results for 5 episodes
  avg = test_network(nn, env, episodes=5, render=True, verbose=1)
  # print('Average after 100 episodes:', avg)

coga(model)