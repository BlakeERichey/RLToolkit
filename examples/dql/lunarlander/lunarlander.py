import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from rltoolkit.methods import DQL
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph

env = gym.make('LunarLander-v2')
try:
  print(env.unwrapped.get_action_meanings())
except:
  pass

#Build network
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=env.observation_space.shape))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(Adam(0.001), loss='mse')
model.summary()

filename = 'lunarlander'

#Load pretrained model
try:
  model = load_model(f'{filename}.h5')
except:
  pass

#Initialize DQL method
method = DQL(rb_size=2500, replay_batch_size=256)

#Enable graphing of rewards
graph = Graph()
#Checkpoint to save best model
ckpt = Checkpoint(f'{filename}.h5')

#Train neural network for 2000 episodes
nn = method.train(model, 
                  env,
                  2000,
                  epsilon_start=.99,
                  epsilon_decay=.995,
                  min_epsilon=.01,
                  batch_size=64,
                  callbacks=[ckpt, graph],
                )

#Save and show rewards
graph.show()
graph.save(f'{filename}.png')
nn.save('nn.h5')

#load best model
model.load_weights(f'{filename}.h5')

#Test model results for 10 episodes
episodes = 10
avg = test_network(model, env, episodes, render=True, verbose=1)
print(f'Average after {episodes} episodes:', avg)
