import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from rltoolkit.methods import DQL
from rltoolkit.utils import Checkpoint

env = gym.make('FrozenLake-v0')
env.unwrapped.get_action_meanings()

#Build network
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(Adam(0.001), loss='mse')
model.summary()

#Initialize Deep Q Learning Method
method = DQL(rb_size=500, replay_batch_size=32)

#Make a checkpoint to save best model during training
ckpt = Checkpoint('frozenlake.h5')
#Train neural network for 50 episodes
method.train(model, env, 200, callbacks=[ckpt])

#Load best saved model
model = load_model('frozenlake.h5')

#Test models results for 5 episodes
method.test(model, env, 5)

