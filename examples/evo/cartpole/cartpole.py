import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from rltoolkit.methods import Evo
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop

#========== Initialize Environment ============================================
env = gym.make('CartPole-v0')
try:
  print(env.unwrapped.get_action_meanings())
except:
  pass

#========== Build network =====================================================
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(Adam(0.001), loss='mse')
model.summary()

#========== Demo ==============================================================
filename = 'cartpole'
load_saved = True

#Load pretrained model
if load_saved:
  try:
    model = load_model(f'{filename}.h5')
  except:
    pass

#========== Configure Callbacks ===============================================
#Enable graphing of rewards
graph = Graph()
#Enable Early Stopping
es = EarlyStop(patience=21)
#Make a checkpoint to save best model during training
ckpt = Checkpoint(f'{filename}.h5')

#========== Train network =====================================================
method = Evo(pop_size=30, elites=10)
nn = method.train(model, env, generations=50, episodes=10, callbacks=[graph, ckpt, es])

#========== Save and show rewards =============================================
version = ['min', 'max', 'avg']
graph.show(version=version)
graph.save(f'{filename}.png', version=version)
nn.save('nn.h5')

#========== Evaluate Results ==================================================
#Load best saved model
model = load_model(f'{filename}.h5')

# Test models results for 5 episodes
episodes = 5
avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
print(f'Average after {episodes} episodes:', avg)