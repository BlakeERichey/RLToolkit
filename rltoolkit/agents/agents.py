from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Conv2D, MaxPooling2D, \
  BatchNormalization, Flatten
from keras.optimizers import Adam

def get_index(arr, i):
  try:
    rv = arr[i]
  except IndexError:
    rv = None

  return rv

def ANN(env, topology=[10], activations=['relu'], lr=0.001):
  model = Sequential()

  layers = 0
  for i, nodes in enumerate(topology):
    if layers == 0:
      #Input layer
      model.add(Dense(nodes, activation=get_index(activations, i) or 'relu', \
        input_shape=env.observation_space.shape)) 
    else:
      #Hidden layers
      model.add(Dense(nodes, activation=get_index(activations, i) or 'relu'))
    layers+=1
  
  #Output layer
  if hasattr(env.action_space, 'n'):
    nodes = env.action_space.n
  else:
    nodes = env.action_space.shape[0]
  model.add(Dense(nodes, activation='linear')) #add output layer

  model.compile(Adam(lr), loss='mse')

  return model

def CNN(env, topology=[64], activations=['relu'], kernel_size=3, pool_size=2, lr=0.001):
  model = Sequential()

  layers = 0
  for i, filters in enumerate(topology):
    if layers == 0:
      #Input layer
      model.add(Conv2D(filters, activation=get_index(activations, i) or 'relu', \
        kernel_size=kernel_size, input_shape=env.observation_space.shape)) 
    else:
      #Hidden layers
      model.add(Conv2D(filters, activation=get_index(activations, i) or 'relu', \
        kernel_size=kernel_size))
      #Applying pooling
      model.add(MaxPooling2D(pool_size=pool_size))
    layers+=1
  
  #FCN
  model.add(Flatten())
  #Output layer
  if hasattr(env.action_space, 'n'):
    nodes = env.action_space.n
  else:
    nodes = env.action_space.shape[0]
  model.add(Dense(nodes, activation='linear')) #add output layer

  model.compile(Adam(lr), loss='mse')

  return model

def LSTM_ANN(env, n_timesteps=5, topology=[10], activations=['relu'], lr=0.001):
  model = Sequential()

  layers = 0
  for i, nodes in enumerate(topology):
    if layers == 0:
      #Input layer
      model.add(LSTM(nodes, activation=get_index(activations, i) or 'relu', \
        input_shape=((n_timesteps,) + env.observation_space.shape), return_sequences=True))
    else:
      #Hidden layers
      model.add(LSTM(nodes, activation=get_index(activations, i) or 'relu', return_sequences=True))
    layers+=1
  
  #Output layer
  if hasattr(env.action_space, 'n'):
    nodes = env.action_space.n
  else:
    nodes = env.action_space.shape[0]
  model.add(LSTM(nodes, activation='linear')) #add output layer

  model.compile(Adam(lr), loss='mse')

  return model

def LSTM_CNN(env, n_timesteps=5, cnn_topology=[64], cnn_activations=['relu'],
  kernel_size=3, pool_size=2, fcn_topology=[16], fcn_activations=['relu'], lr=0.001):

  model = Sequential()

  #CNN
  layers = 0
  for i, filters in enumerate(cnn_topology):
    if layers == 0:
      #Input layer
      model.add(TimeDistributed( 
        Conv2D(filters, activation=get_index(cnn_activations, i) or 'relu', kernel_size=kernel_size),
        input_shape=(n_timesteps,) + env.observation_space.shape )
      ) 
    else:
      #Hidden layers
      model.add(TimeDistributed(
        Conv2D(filters, activation=get_index(cnn_activations, i) or 'relu', kernel_size=kernel_size),
      ))
      #Applying pooling
      model.add(TimeDistributed(MaxPooling2D(pool_size=pool_size)))
    layers+=1
  
  #FCN
  model.add(TimeDistributed(Flatten()))
  for i, nodes in enumerate(fcn_topology):
    model.add(LSTM(nodes, activation=get_index(fcn_activations, i) or 'relu', return_sequences=True))
    if i != len(fcn_topology) - 1:
      model.add(TimeDistributed(BatchNormalization()))
  
  #Output layer
  if hasattr(env.action_space, 'n'):
    nodes = env.action_space.n
  else:
    nodes = env.action_space.shape[0]
  model.add(LSTM(nodes, activation='linear')) #add output layer

  model.compile(Adam(lr), loss='mse')

  return model