import time
import gym
import datetime
import rltoolkit
from rltoolkit.agents import ANN
from rltoolkit.backend import LocalClusterBackend

########### Helper Functions ###################################################

def calc_big_number(number):
  total=0
  for i in range(1, number+1):
    time.sleep(i)
    total+=i
  
  return total

def create_model():
  filename = 'MountainCar'
  env = gym.make(f'{filename}-v0')
  # model = LSTM_ANN(env, n_timesteps=10, topology=[2,64,64,16])
  # model = ANN(env, topology=[4,64,64,4])
  model = ANN(env, topology=[256,256,256])
  return model


########### TESTS ##############################################################

def test_localhost_cluster():
  backend = LocalClusterBackend(10)

  hashes = []
  for i in range(10):
    res = backend.run(calc_big_number, i, timeout=10)
    hashes.append(res)
  assert hashes == [str(i) for i in range(1, 11)], f'Task IDs not returning properly {hashes}'

  results = backend.get_results(hashes, values_only=True) 
  assert results == [0,1,3,6,None,None,None,None,None,None], \
    f'Invalid Localhost Cluster Results {results}'
  backend.shutdown()

def test_localhost_cluster_env():
  backend = LocalClusterBackend(3, network_generator=create_model)

  env = gym.make(f'MountainCar-v0')
  model = create_model()
  weights = model.get_weights()
  hashes = []
  for i in range(10):
    task_id = backend.test_network(weights, env, 1, 1)
    hashes.append(task_id)
  
  assert hashes == [str(i) for i in range(1, 11)], f'Task IDs not returning properly {hashes}'
  
  res = backend.get_results(hashes, values_only=False)
  answer = dict(zip([str(i) for i in range(1,11)], [-200 for _ in range(10)]))
  assert res == answer, \
    f'Invalid Localhost Cluster Results {res}'
  backend.shutdown()

if __name__ == '__main__':
  test_localhost_cluster()
  test_localhost_cluster_env()