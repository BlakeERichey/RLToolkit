###NOT SOLVED YET###

import gym
from keras.models import load_model
from rltoolkit.methods import Evo
from rltoolkit.agents import LSTM_ANN, ANN
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop

if __name__ == '__main__':
  #========== Initialize Environment ============================================
  filename = 'BipedalWalker'
  train_from_scratch = True

  env = gym.make(f'{filename}-v2')
  try:
    print(env.unwrapped.get_action_meanings())
  except:
    pass

  #========== Build network =====================================================
  model = ANN(env, topology=[2,64,256,256,64])

  #Load pretrained model?
  if not train_from_scratch:
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
  method = Evo(pop_size=150, elites=30)
  nn = method.train(model, env, generations=2000, episodes=1, callbacks=[graph, ckpt], cores=1)

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