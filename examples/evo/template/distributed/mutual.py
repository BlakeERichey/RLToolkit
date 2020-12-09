import gym
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam

PORT           = 50000
AUTHKEY        = b'authkey'
ENV_NAME       = 'LunarLander-v2'
CORES_PER_NODE = 1
TIMEOUT        = 60  #Max time in seconds for a task (run through env) to complete

def create_model():
  env = gym.make(ENV_NAME)
  model = Sequential()
  model.add(Dense(64,  activation='relu', input_shape=env.observation_space.shape)) #add input layer
  model.add(Dense(256, activation='relu'))                  #change activation method
  model.add(Dense(128, activation='relu'))                  #another hidden layer
  model.add(Dense(env.action_space.n, activation='softmax')) #add output layer
  model.compile(Adam(0.001), loss='mse')                    #compile network
  return model