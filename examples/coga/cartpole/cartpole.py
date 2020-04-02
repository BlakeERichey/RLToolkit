import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from rltoolkit.methods import COGA
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph

env = gym.make('CartPole-v0')
try:
  print(env.unwrapped.get_action_meanings())
except:
  pass

#Build network
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=env.observation_space.shape))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(Adam(0.001), loss='mse')
model.summary()

#Initialize COGA Learning Method
method = COGA(model, 
              num_colonies=20, 
              num_workers=30,
              alpha=0.1,
            )

#Enable graphing of rewards
graph = Graph()
#Make a checkpoint to save best model during training
ckpt = Checkpoint('cartpole.h5')

#Train neural network for 25 generations
nn = method.train(env,
                  goal=200,
                  elites=4, 
                  verbose=1,
                  patience=3,
                  validate=True,
                  generations=25,
                  callbacks=[graph, ckpt], 
                  sharpness=20,
                )

#Save and show rewards
version = ['min', 'max', 'avg']
graph.show(version=version)
graph.save('cartpole.png', version=version)

#Load best saved model
model = load_model('cartpole.h5')

#Test models results for 5 episodes
episodes = 5
avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
print(f'Average after {episodes} episodes:', avg)