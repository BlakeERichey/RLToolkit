import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from rltoolkit.methods import DQL
from rltoolkit.utils import Checkpoint

env = gym.make('CartPole-v0')
# env = gym.make('MountainCarContinuous-v0')

#Build network
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=env.observation_space.shape))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(Adam(0.001), loss='mse')
model.summary()


#Initialize Deep Q Learning Method
method = DQL(rb_size=500, replay_batch_size=32)

#Make a checkpoint to save best model during training
ckpt = Checkpoint('cartpole.h5')
#Train neural network for 50 episodes
method.train(model, env, 50, epsilon_decay=.9, callbacks=[ckpt])

#Load best saved model
model = load_model('cartpole.h5')

#Test models results for 5 episodes
method.test(model, env, 5)

