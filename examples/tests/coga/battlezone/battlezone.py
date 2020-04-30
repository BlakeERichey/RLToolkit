import gym
import keras
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, TimeDistributed, Conv2D, MaxPool2D, \
  BatchNormalization, Flatten
from keras.optimizers import Adam
from rltoolkit.methods import COGA
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop

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

env = gym.make('BattleZone-v0')
try:
  print(env.unwrapped.get_action_meanings())
except:
  pass

print("Obs Space:", env.observation_space.shape)
print("Act Space:", env.action_space.n)

#Build network
n_timesteps = 20
model = Sequential()
#========== Convolutional Base ==============================
#input layer
model.add(TimeDistributed( Conv2D(64, kernel_size=3, activation='relu'), \
  input_shape=(n_timesteps,) + env.observation_space.shape ))

model.add(TimeDistributed( Conv2D(32, kernel_size=3, activation='relu') ))
model.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))
model.add(TimeDistributed( Conv2D(16, kernel_size=3, activation='relu') ))
model.add(TimeDistributed(MaxPool2D(pool_size=(2,2))))
model.add(TimeDistributed(Flatten()))

#========== FCN ==============================
model.add(LSTM(32, return_sequences=True))
model.add(TimeDistributed(BatchNormalization()))
model.add(LSTM(16, return_sequences=True, dropout=.2))
model.add(LSTM(8, return_sequences=True))
model.add(LSTM(env.action_space.n, activation='linear'))
model.compile(loss="mse", optimizer=Adam(lr=0.001))
model.summary()

@profile
def coga(model):
  filename = 'battlezone'

  #Load pretrained model
  try:
    model = load_model(f'{filename}.h5')
  except:
    pass

  # Initialize COGA Learning Method
  method = COGA(model, 
                num_colonies=20, 
                num_workers=30,
                alpha=1E-2,
              )

  #Enable graphing of rewards
  graph = Graph()
  #Enable Early Stopping
  es = EarlyStop(patience=21)
  #Make a checkpoint to save best model during training
  ckpt = Checkpoint(f'{filename}.h5')

  #Train neural network for 25 generations
  nn = method.train(env,
                    goal=30000,
                    elites=4,
                    verbose="testing",
                    patience=3,
                    validate=False,
                    generations=100,
                    callbacks=[graph, ckpt, es], 
                    sharpness=1,
                  )

  #Save and show rewards
  version = ['min', 'max', 'avg']
  graph.show(version=version)
  graph.save(f'{filename}.png', version=version)
  nn.save('nn.h5')

  #Load best saved model
  model = load_model(f'{filename}.h5')
  # model = nn

  # Test models results for 5 episodes
  episodes = 5
  avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
  print(f'Average after {episodes} episodes:', avg)

coga(model)