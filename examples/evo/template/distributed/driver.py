import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  import gym
  import socket
  from keras.models import load_model
  from config import create_model, ENV_NAME, PORT, AUTHKEY, TIMEOUT, GPUS,\
    CORES_PER_NODE, GENERATIONS, POP_SIZE, ELITES, GOAL, EPISODES, FILENAME, \
    CONTINUE
  from rltoolkit.methods import Evo
  from rltoolkit.utils import test_network
  from rltoolkit.callbacks import Graph, Checkpoint
  from rltoolkit.backend.keras import DistributedBackend, set_gpu_session

if GPUS:
  set_gpu_session()

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

  #Load pretrained model
  if CONTINUE:
    try:
      model = load_model(f'{FILENAME}.h5')
    except:
      model = create_model() #compile new network
  else:
    model = create_model() #compile new network
      
  model.summary()

  #========== Configure Callbacks ===============================================
  #Enable graphing of rewards
  graph = Graph()
  #Make a checkpoint to save best model during training
  ckpt = Checkpoint(f'{FILENAME}.h5')

  #========== Train network =====================================================
  method = Evo(pop_size=POP_SIZE, elites=ELITES)
  nn = method.train(
    model,
    env, 
    generations=GENERATIONS, 
    episodes=EPISODES,
    goal=GOAL, 
    callbacks=[graph, ckpt],
    backend=backend
  )

  #========== Save and show rewards =============================================
  nn.save('nn.h5')
  version = ['min', 'max', 'avg']
  try:
    graph.show(version=version)
  except: #Node is headless
    pass
  graph.save(f'{FILENAME}.png', version=version)