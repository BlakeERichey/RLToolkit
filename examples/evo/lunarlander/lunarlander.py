import gym
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from rltoolkit.utils import test_network
from rltoolkit.methods import Evo
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop

if __name__ == '__main__':
  #========== Initialize Environment ============================================
  env = gym.make('LunarLander-v2')
  try:
    print(env.unwrapped.get_action_meanings())
  except:
    pass

  #========== Build network =====================================================
  model = Sequential()
  model.add(Dense(64,  activation='relu', input_shape=env.observation_space.shape)) #add input layer
  model.add(Dense(256, activation='relu'))                  #change activation method
  model.add(Dense(128, activation='relu'))                  #another hidden layer
  model.add(Dense(env.action_space.n, activation='softmax')) #add output layer
  model.compile(Adam(0.001), loss='mse')                    #compile network

  #========== Demo ==============================================================
  filename = 'lunarlander'
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
  method = Evo(pop_size=50, elites=8)
  nn = method.train(model, env, generations=250, episodes=5, callbacks=[graph, ckpt])

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
  print(f'Average after {episodes} episodes:', avg) #~220