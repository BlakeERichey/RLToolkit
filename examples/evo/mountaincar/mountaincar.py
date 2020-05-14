import sys
import gym
import keras
from keras.models import load_model
from rltoolkit.methods import Evo
from rltoolkit.agents import LSTM_ANN, ANN
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop
from rltoolkit.backend import MulticoreBackend, DistributedBackend

if __name__ == '__main__':
  server_info = ('127.0.0.1', 50000, b'password') #(ip, port, authkey)

  args = sys.argv[1:]
  for i, arg in enumerate(args):
    if arg == '--server':
      print('Launching Server.')
      DistributedBackend(*server_info).spawn_server()

    elif arg == '--client':
      cores = int(args[i+1])
      print('Spawning client using', cores, 'cores.')
      DistributedBackend(*server_info).spawn_client(cores)

    elif arg == '--driver':
      #========== Initialize Environment ============================================
      filename = 'MountainCar'
      train_from_scratch = True

      env = gym.make(f'{filename}-v0')
      try:
        print(env.unwrapped.get_action_meanings())
      except:
        pass

      #========== Build network =====================================================
      # model = LSTM_ANN(env, n_timesteps=10, topology=[2,64,64,16])
      model = ANN(env, topology=[4,64,64,4])

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
      # backend = DistributedBackend(*server_info)
      backend = MulticoreBackend(4)

      #========== Train network =====================================================
      method = Evo(pop_size=50, elites=8)
      nn = method.train(model, env, generations=250, episodes=1, callbacks=[graph, ckpt], backend=backend)

      #========== Save and show rewards =============================================
      version = ['min', 'max', 'avg']
      graph.show(version=version)
      graph.save(f'{filename}.png', version=version)
      nn.save('nn.h5')

      #========== Evaluate Results ==================================================
      #Load best saved model
      model = load_model('nn.h5')

      # Test models results for 5 episodes
      episodes = 5
      avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
      print(f'Average after {episodes} episodes:', avg)

      episodes = 100
      avg = test_network(model, env, episodes=episodes, render=False, verbose=0)
      print(f'Average after {episodes} episodes:', avg)