import gym
from keras.models import load_model
from rltoolkit.methods import Evo
from rltoolkit.agents import ANN
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop
from rltoolkit.backend import LocalClusterBackend

def create_model():
  env = gym.make('CartPole-v0')
  model = ANN(env, topology=[32])
  return model

if __name__ == '__main__':
  #========== Initialize Environment ============================================
  filename = 'CartPole'
  train_from_scratch = True

  env = gym.make(f'{filename}-v0')
  try:
    print(env.unwrapped.get_action_meanings())
  except:
    pass

  #========== Build network =====================================================
  model = ANN(env, topology=[32])

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
  backend = LocalClusterBackend(4, network_generator=create_model)

  #========== Train network =====================================================
  method = Evo(pop_size=20, elites=4)
  nn = method.train(model, env, generations=25, episodes=10, callbacks=[graph, ckpt], goal=200, backend=None)

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