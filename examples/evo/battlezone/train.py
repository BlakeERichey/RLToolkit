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

  #========== Configure Callbacks ===============================================
  #Enable graphing of rewards
  graph = Graph()
  #Make a checkpoint to save best model during training
  ckpt = Checkpoint(f'{BEST_MODEL}.h5')

  #========== Train network =====================================================
  ############################# EDITABLE VARIABLES #############################
  backend = LocalClusterBackend(2, network_generator=create_model, timeout=60)
  method = Evo(pop_size=4, elites=3)
  nn = method.train(
    model, 
    env, 
    generations=4, 
    episodes=1, 
    callbacks=[graph, ckpt], 
    backend=backend
  )
  backend.shutdown()
  ##############################################################################

  #========== Save and show rewards =============================================
  version = ['min', 'max', 'avg']
  graph.show(version=version)
  graph.save(f'{BEST_MODEL}.png', version=version)
  nn.save('nn.h5')

  #========== Evaluate Results ==================================================
  #Load best saved model
  model = load_model(f'{BEST_MODEL}.h5')

  # Test models results for 5 episodes
  episodes = 5
  avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
  print(f'Average after {episodes} episodes:', avg)

  episodes = 100
  avg = test_network(model, env, episodes=episodes, render=False, verbose=0)
  print(f'Average after {episodes} episodes:', avg) #~247