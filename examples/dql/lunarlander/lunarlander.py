import gym
import keras
from keras.models import load_model
from rltoolkit.agents import ANN
from rltoolkit.methods import DQL
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph

env = gym.make('LunarLander-v2')
try:
  print(env.unwrapped.get_action_meanings())
except:
  pass

#Build network
model = ANN(env, topology=[64,256,128])

filename = 'lunarlander'

#Load pretrained model
try:
  model = load_model(f'{filename}.h5')
except:
  pass

#Initialize DQL method
method = DQL(rb_size=200000, replay_batch_size=1024)

#Enable graphing of rewards
graph = Graph()
#Checkpoint to save best model
ckpt = Checkpoint(f'{filename}.h5')

#Train neural network for 2000 episodes
nn = method.train(model, 
                  env,
                  1000,
                  epsilon_start=1,
                  epsilon_decay=.99,
                  min_epsilon=.05,
                  batch_size=64,
                  callbacks=[ckpt, graph],
                )

#Save and show rewards
graph.show()
graph.save(f'{filename}.png')
nn.save('nn.h5')

#load best model
model.load_weights(f'{filename}.h5')

# Test models results for 5 episodes
episodes = 5
avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
print(f'Average after {episodes} episodes:', avg)

episodes = 100
avg = test_network(model, env, episodes=episodes, render=False, verbose=0)
print(f'Average after {episodes} episodes:', avg)
