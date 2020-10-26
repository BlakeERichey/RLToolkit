#====== Mute subprocesses =======
import os
import sys
import warnings
warnings.filterwarnings("ignore")
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
#================================

import gym
import gym_solventx
import tensorflow as tf
from keras.models import load_model
from rltoolkit.methods import Evo
from rltoolkit.agents import LSTM_ANN, ANN
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop
from rltoolkit.backend import MulticoreBackend

filename = 'gym_solventx'
train_from_scratch = True

def create_model():
  env = gym.make(f'{filename}-v0')
  model = ANN(env, topology=[256,256,256])
  return model

if __name__ == '__main__':
  #========== Initialize Environment ============================================
  env = gym.make(f'{filename}-v0')
  try:
    print(env.unwrapped.get_action_meanings())
  except:
    pass

  #========== Build network =====================================================
  model = create_model()

  #Load pretrained model?
  if not train_from_scratch:
    try:
      model = load_model(f'{filename}.h5')
    except:
      pass

  model.summary()

  #========== Configure Callbacks ===============================================
  #Enable graphing of rewards
  graph = Graph()
  #Make a checkpoint to save best model during training
  ckpt = Checkpoint(f'{filename}.h5')
  backend = None#MulticoreBackend(4)

  #========== Train network =====================================================
  method = Evo(pop_size=50, elites=8)
  nn = method.train(model, env, generations=250, episodes=1, callbacks=[graph, ckpt], backend=backend)

  #========== Save and show rewards =============================================
  nn.save('nn.h5')
  version = ['min', 'max', 'avg']
  graph.show(version=version)
  graph.save(f'{filename}.png', version=version)

  #========== Evaluate Results ==================================================
  #Load best saved model
  model = load_model(f'{filename}.h5')

  # Test models results for 5 episodes
  episodes = 1
  avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
  print(f'Average after {episodes} episodes:', avg)

  # episodes = 100
  # avg = test_network(model, env, episodes=episodes, render=False, verbose=0)
  # print(f'Average after {episodes} episodes:', avg)