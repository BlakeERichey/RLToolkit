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

env = gym.make('MountainCar-v0')
try:
  print(env.unwrapped.get_action_meanings())
except:
  pass

#Build network
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(Adam(0.001), loss='mse')
model.summary()

# @profile
def coga(model):
  #Initialize Deep Q Learning Method
  method = COGA(model, 
                num_colonies=20, 
                num_workers=30,
                alpha=0.1,
              )

  #Enable graphing of rewards
  graph = Graph()
  #Make a checkpoint to save best model during training
  ckpt = Checkpoint('mc.h5')
  #Train neural network for 50 episodes
  nn = method.train(env,
                    goal=200,
                    elites=4, 
                    verbose=1,
                    patience=10,
                    validate=True,
                    generations=100,
                    callbacks=[graph, ckpt], 
                    sharpness=2,
                  )

  #Save and show rewards
  version = ['min', 'max', 'avg']
  graph.show(version=version)
  graph.save('mc.png', version=version)

  #Load best saved model
  model = load_model('mc.h5')
  # model = nn

  #Test models results for 5 episodes
  episodes = 5
  avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
  print(f'Average after {episodes} episodes:', avg)

coga(model)