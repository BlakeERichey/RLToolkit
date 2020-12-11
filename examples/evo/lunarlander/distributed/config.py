import gym
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from rltoolkit.agents import ANN

############### BACKEND CONFIG #################################################
PORT           = 50000
AUTHKEY        = b'authkey'
ENV_NAME       = 'LunarLander-v2'
GPUS           = 0
CORES_PER_NODE = 10
TIMEOUT        = 60  #Max time in seconds for a task (run through env) to complete

################################################################################

############### METHOD CONFIG ##################################################

GENERATIONS = 250
POP_SIZE    = 50
ELITES      = 8
GOAL        = None
EPISODES    = 5     #Episodes/Individual/Generation

################################################################################

def create_model():
  env = gym.make(ENV_NAME)
  model = ANN(env, topology=[64,256,128])
  return model