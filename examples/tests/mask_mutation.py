import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from rltoolkit.utils import test_network, truncate_weights
from rltoolkit.methods import Evo
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop

if __name__ == '__main__':
  #========== Initialize Environment ============================================
  env = gym.make('CartPole-v0')
  try:
    print(env.unwrapped.get_action_meanings())
  except:
    pass

  #========== Build network =====================================================
  model = Sequential()
  model.add(Dense(64,  activation='relu', input_shape=env.observation_space.shape)) #add input layer
  model.add(Dense(256, activation='relu'))                  #change activation method
  model.add(Dense(128, activation='relu'))                  #another hidden layer
  model.add(Dense(env.action_space.n, activation='linear')) #add output layer
  model.compile(Adam(0.001), loss='mse')                    #compile network

  #========== Demo ==============================================================
  import numpy as np
  state = env.reset()
  pred1 = model.predict(np.expand_dims(state, axis=0))
  weights = truncate_weights(model.get_weights(), n_decimals=3)
  method = Evo(pop_size=50, elites=12)
  new_weights = method._mutate(weights)
  model.set_weights(new_weights)
  pred2 = model.predict(np.expand_dims(state, axis=0))
  print('weights', weights)
  print('new weights', new_weights)
  print(pred1)
  print(pred2)