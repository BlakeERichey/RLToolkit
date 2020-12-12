import gym
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from rltoolkit.agents import LSTM_CNN

############### BACKEND CONFIG #################################################
PORT           = 50000
AUTHKEY        = b'authkey'
ENV_NAME       = 'BattleZone-v0'
GPUS           = 4
CORES_PER_NODE = 12
TIMEOUT        = 480  #Max time in seconds for a task (run through env) to complete

################################################################################

############### METHOD CONFIG ##################################################

GENERATIONS = 50
POP_SIZE    = 48
ELITES      = 8
GOAL        = None
EPISODES    = 1     #Episodes/Individual/Generation

################################################################################

def create_model():
  env = gym.make(ENV_NAME)

  model = LSTM_CNN(
    env,
    cnn_topology=[64,128,256,512],
    fcn_topology=[256,128,64]
  )
  return model