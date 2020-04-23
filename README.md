# RLToolkit
Reinforcement learning library for industry and personal use.  

RLToolkit is a python package intended to solve reinforcement learning 
environments by having users make environments rather than complicated 
reinforcement learning methods.  

If you can provide an environment and a neural network, RLToolkit attempts to 
solve the environment for you.  

# Installation  
Download RLToolkit  
```
git clone https://github.com/BlakeERichey/RLToolkit
```  

Install  
```
cd RLToolkit
pip install -e .
```  

# Usage  

The RLToolkit requires three criteria when training an AI.  
1. Gym environment to solve  
2. Keras neural network  
3. Reinforcement learning method (Provided by RLToolkit)  

## Environments  
Gym is a python package that offers extensive documetation and tools for 
developing robust and highly structured environments that permits assurances when 
RLToolkit trains an AI.  

Useful links:  
* [Gym Documentation](https://gym.openai.com/docs/)  
* [Gym Table of Predefined Environments](https://github.com/openai/gym/wiki/Table-of-environments)  
* [Make a Custom Environment from Scratch](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)  

Once you find a gym environment you would like to solve, initialize it like so
`env = gym.make('YourEnvironmentName')`

## Neural Networks  
Keras is a high level API for easy-to-build neural network development.  
We recommend you you look at [Keras Documentation](https://keras.io/) for a 
comprehensive guide, but some basic neural network examples are shown below (comments for clarity).  

**Artificial Neural Network**  
```python
  model = Sequential()
  model.add(Dense(256, activation='relu', input_shape=env.observation_space.shape)) #add input layer
  model.add(Dense(64, activation='tanh'))                   #add hidden layer
  model.add(Dense(256, activation='relu'))                  #change activation method
  model.add(Dense(64, activation='relu'))                   #another hidden layer
  model.add(Dense(env.action_space.n, activation='linear')) #add output layer
  model.compile(Adam(0.001), loss='mse')                    #compile network
  model.summary()                                           #show network architecture
```  

**Convolutional Neural Network**  
```python
  model = Sequential()
  #Input layer
  model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))

  #Add hidden layer
  model.add(Conv2D(32, kernel_size=3, activation='relu'))
  
  #Applying pooling
  model.add(MaxPooling2D(pool_size=(2,2)))
  
  #Add FCN
  model.add(Flatten())
  model.add(Dense(10, activation='softmax'))

  #compile network
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```  

**Time Distributed (LSTM) Neural Network**  
```python
  n_timesteps = 5
  model = Sequential()
  
  #Input layer
  model.add(LSTM(64, activation='relu', \
    input_shape=((n_timesteps,) + env.observation_space.shape), return_sequences=True))
  
  #add hidden layers
  model.add(LSTM(128, return_sequences=True, activation='tanh'))
  model.add(LSTM(32, return_sequences=True, dropout=.2, activation='relu'))
  
  #output layer
  model.add(LSTM(env.action_space.n))
  
  #compile network
  model.compile(loss="mse", optimizer=Adam(lr=0.001)) 
  
  #show network architecture
  model.summary() 
```  

## Reinforcement Learning Method  
RLToolkit offers multiple methods you can use to solve an environment. 
Each has their own list of requirements and documentation. 
All you need to do is choose which method you want and learn its parameters. 

Included Reinforcement learning methods:  
* Deep Q Learning  
* NeuroEvolution  

Future Implementations:  
* A3C  
* Policy Gradient  

You can initialize a method like so:  
`method = DQL(rb_size=500, replay_batch_size=32)`  

More examples of initializing method can be found in [examples]
(https://github.com/BlakeERichey/RLToolkit/tree/master/examples)  

# Solve the Environment  
Once an environment, neural network, and method have been chose you can create your AI easily:
```python
  import gym
  import keras
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.optimizers import Adam
  from rltoolkit.methods import DQL

  #Initialize environment
  env = gym.make('CartPole-v0')

  #Build network
  model = Sequential()
  model.add(Dense(16, activation='relu', input_shape=env.observation_space.shape))
  model.add(Dense(64, activation='relu'))
  model.add(Dense(2, activation='linear'))
  model.compile(Adam(0.001), loss='mse')
  model.summary()

  #Initialize Deep Q Learning Method
  method = DQL(rb_size=500, replay_batch_size=32)

  #Train neural network for 50 episodes
  ai = method.train(model, env, 50, epsilon_decay=.9)
```  

Each method has its own parameters and lets you fine tune them in the `method.train` function call.  

# Test your AI  
We want the make testing of your AI easy, so we provided a for doing so. 
With an AI, or Keras neural network, and a gym environment, simply call the 
`test_network` function like so:  
```python
  from rltoolkit.utils import test_network
  #Test models results for 5 episodes
  avg = test_network(model, env, episodes=5, render=True)
  print('Average after 5 episodes:', avg)
```