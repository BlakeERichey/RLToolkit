from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Conv2D, MaxPooling2D, \
  BatchNormalization, Flatten
from keras.optimizers import Adam

def get_index(arr, i):
  """
    Helper function that removes the possibility of index error.
  """
  try:
    rv = arr[i]
  except IndexError:
    rv = None

  return rv

def ANN(env, topology=[10], activations=['relu'], lr=0.001):
  """
    Creates an artificial Keras neural network of Densly connected layers.

    #Arguments 
    env: A Gym environment.
    topology: a list of the quantity of nodes used for creating hidden layers 
      of a neural network input and output layers are implicitly determined by 
      the `env`. So `topology` should specifically be for hidden layers.
    activations: the activation function for each hidden layer.
    lr: learning rate for Adam optimizer.

    #Returns 
    A keras neural network with the desired topology tailored to fit the 
    inputs and outputs of the provided gym environment.
  """
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
  """
    Creates an Convultional Keras neural network.

    #Arguments 
    env: A Gym environment.
    topology: a list of the quantity of filters used for creating input and 
      hidden layers of a CNN. Output layer is implicitly determined by 
      the `env`.
    activations: the activation function for each layer excluding output.
    kernel_size: Int. A window size for filters.
    pool_size: When applying MaxPooling, the original image will shrink by this factor.
    lr: learning rate for Adam optimizer.

    #Returns 
    A keras neural network with the desired topology tailored to fit the 
    inputs and outputs of the provided gym environment.
  """
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

def LSTM_ANN(env, n_timesteps=2, topology=[10], activations=['relu'], lr=0.001):
  """
    Creates an LSTM Keras neural network of Densly connected layers.

    #Arguments 
    env: A Gym environment.
    n_timesteps: Int. The number of previous states maintained by the LSTM.
    topology: a list of the quantity of nodes used for creating hidden layers 
      of a neural network input and output layers are implicitly determined by 
      the `env`. So `topology` should specifically be for hidden layers.
    activations: the activation function for each hidden layer.
    lr: learning rate for Adam optimizer.

    #Returns 
    A keras neural network with the desired topology tailored to fit the 
    inputs and outputs of the provided gym environment.
  """
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
  """
    Creates an TimeDistributed Convultional Keras neural network.

    #Arguments 
    env: A Gym environment.
    n_timesteps: Int. The number of previous states maintained by the LSTM.
    cnn_topology: a list of the quantity of filters used for creating input and 
      hidden layers of a CNN. Includes the input layer and generates the 
      convolutional base of the network.
    cnn_activations: the activation function for each layer, excluding output, 
      in the convolutional base.
    kernel_size: Int. A window size for filters.
    pool_size: When applying MaxPooling, the original image will shrink by this factor.
    fcn_topology: a list of the quantity of nodes used for creating hidden layers 
      of a FCN and output layers are implicitly determined by 
      the `env`. So `topology` should specifically be for hidden layers. If an 
      empty list is provided, the FCN will lead directly to an LSTM output layer.
    fcn_activations: the activation function for each hidden layer in the FCN.
    lr: learning rate for Adam optimizer.

    #Returns 
    A keras neural network with the desired topology tailored to fit the 
    inputs and outputs of the provided gym environment.
  """

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