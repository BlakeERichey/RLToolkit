import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from rltoolkit.methods import Evo
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop

#========== Initialize Environment ============================================
env = gym.make('MountainCar-v0')
try:
  print(env.unwrapped.get_action_meanings())
except:
  pass

#========== Build network =====================================================
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

#========== Demo ==============================================================
filename = 'mountaincar'
load_saved = False

#Load pretrained model
if load_saved:
  try:
    model = load_model(f'{filename}.h5')
  except:
    pass

model.summary()

#========== Configure Callbacks ===============================================
#Enable graphing of rewards
graph = Graph()
#Make a checkpoint to save best model during training
ckpt = Checkpoint(f'{filename}.h5')

#========== Train network =====================================================
method = Evo(pop_size=50, elites=12)
nn = method.train(model, env, generations=100, episodes=10, callbacks=[graph, ckpt], goal=-105)

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

episodes = 100
avg = test_network(model, env, episodes=episodes, render=False, verbose=0)
print(f'Average after {episodes} episodes:', avg)