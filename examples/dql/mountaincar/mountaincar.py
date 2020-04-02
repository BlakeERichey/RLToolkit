import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from rltoolkit.methods import DQL
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint

env = gym.make('MountainCar-v0')
try:
  print(env.unwrapped.get_action_meanings())
except:
  pass

#Build network
model = Sequential()
model.add(Dense(2, activation='relu', input_shape=env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='linear'))
model.compile(Adam(0.001), loss='mse')
model.summary()

#Initialize DQL method
method = DQL(rb_size=500, replay_batch_size=128)

#Checkpoint to save best model
ckpt = Checkpoint('mountaincar.h5')

#Train neural network for 10000 episodes
nn = method.train(model, 
                  env,
                  10000,
                  epsilon_start=.99,
                  epsilon_decay=.9992,
                  min_epsilon=.01,
                  batch_size=64,
                  callbacks=[ckpt],
                )

#load best model
model.load_weights('mountain_best.h5')

#Test model results for 10 episodes
episodes = 10
avg = test_network(model, env, episodes, render=True, verbose=1)
print(f'Average after {episodes} episodes:', avg)
