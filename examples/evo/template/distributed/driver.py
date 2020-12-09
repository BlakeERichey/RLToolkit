import gym
import socket
from keras.models import load_model
from mutual import create_model, ENV_NAME, PORT, AUTHKEY, CORES_PER_NODE, TIMEOUT
from rltoolkit.methods import Evo
from rltoolkit.utils import test_network
from rltoolkit.backend import DistributedBackend
from rltoolkit.callbacks import Graph, Checkpoint

if __name__ == '__main__':
  #========== Initialize Backend ===============================================
  ip = socket.gethostbyname(socket.gethostname())
  with open('server_ip.log', 'r') as f:
    ip = f.readline()
  backend = DistributedBackend(
    port=PORT, 
    timeout=TIMEOUT,
    server_ip=ip, 
    authkey=AUTHKEY,
    network_generator=create_model
  )

  #========== Initialize Environment ============================================
  env = gym.make(ENV_NAME)
  try:
    print(env.unwrapped.get_action_meanings())
  except:
    pass

  #========== Build network =====================================================
  model = create_model()                #compile network

  #========== Demo ==============================================================
  filename = 'best_model'
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
  ############################# EDITABLE VARIABLES #############################
  method = Evo(pop_size=5, elites=2)
  nn = method.train(
    model,
    env, 
    generations=250, 
    episodes=10, 
    callbacks=[graph, ckpt],
    backend=backend
  )
  ##############################################################################

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

  print('Testing 100 times!')
  episodes = 100
  avg = test_network(model, env, episodes=episodes, render=False, verbose=0)
  print(f'Average after {episodes} episodes:', avg)