import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from rltoolkit.methods import COGA
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph

env = gym.make('CartPole-v1')
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

filename = 'cartpole'

#Load pretrained model
try:
  model = load_model(f'{filename}.h5')
except:
  pass

#Initialize COGA Learning Method
method = COGA(model, 
              num_colonies=20, 
              num_workers=50,
              alpha=0.2,
            )

#Enable graphing of rewards
graph = Graph()
#Make a checkpoint to save best model during training
ckpt = Checkpoint(f'{filename}.h5')

#Train neural network for 25 generations
nn = method.train(env,
                  goal=500,
                  elites=5,
                  verbose=1,
                  patience=10,
                  validate=True,
                  generations=100,
                  callbacks=[graph, ckpt], 
                  sharpness=25,
                )

#Save and show rewards
version = ['min', 'max', 'avg']
graph.show(version=version)
graph.save(f'{filename}.png', version=version)
nn.save('nn.h5')

#Load best saved model
model = load_model(f'{filename}.h5')

#Test models results for 5 episodes
episodes = 5
avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
print(f'Average after {episodes} episodes:', avg)