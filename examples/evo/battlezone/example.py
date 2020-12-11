#====== Mute subprocesses =======
import os
import sys
import gym
import warnings
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
sys.stderr = stderr
#================================

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from rltoolkit.utils import test_network
from rltoolkit.methods import Evo
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop
from rltoolkit.backend import LocalClusterBackend
from rltoolkit.agents import LSTM_CNN

CONTINUE   = False #Continue training from previous BEST_MODEL?
ENV_NAME   = 'BattleZone-v0'
BEST_MODEL = 'BattleZone' #f'{BEST_MODEL}.h5' will be saved as a checkpoint

def create_model():
  env = gym.make(ENV_NAME)

  model = LSTM_CNN(
    env,
    cnn_topology=[64,128,256,512],
    fcn_topology=[256,128,64]
  )
  return model


if __name__ == '__main__':
  #========== Initialize Environment ============================================
  env = gym.make(ENV_NAME)
  try:
    print(env.unwrapped.get_action_meanings())
  except:
    pass

  #========== Build network =====================================================
  model = create_model()

  #========== Demo ==============================================================

  #Load pretrained model
  if CONTINUE:
    try:
      model = load_model(f'{BEST_MODEL}.h5')
    except:
      pass

  model.summary()

  # Test models results for 5 episodes
  import datetime
  print('Starting...',  datetime.datetime.now())
  start_time = datetime.datetime.now()
  for i in range(5):
    avg = test_network(model, env, episodes=1, render=False)
    end_time = datetime.datetime.now()
    dt = end_time - start_time
    print('Total time:', dt.total_seconds())
    start_time = end_time
