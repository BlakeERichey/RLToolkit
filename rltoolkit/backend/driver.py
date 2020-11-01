import time
import gym
import datetime
import hashlib
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager
from test_manager import ParallelManager, calc_big_number, create_model
from backend import DistributedBackend, MulticoreBackend

filename = 'MountainCar'
train_from_scratch = True

if __name__ == '__main__':
  backend = DistributedBackend('127.0.0.1', 50000, authkey=b'123', 
    network_generator=create_model)
  # # backend.manager.connect()
  # # print(backend.manager.get_active_tasks().unpack())
  # # print(backend.manager.kill_tasks(['1','2','3','4','5','6']))
  
  # env = gym.make('MountainCar-v0')
  # model = create_model()
  # hashes = []
  # for i in range(10):
  #   # res = backend.test_network(model.get_weights(), env, 1, 1, model, timeout=14)
  #   res = backend.run(calc_big_number, i, timeout=14)
  #   print('Task Hash:', res)
  #   hashes.append(res)
  # print('requesting:', hashes)
  # backend.get_results(hashes, values_only=False, numeric_only=True, min_value=True) 


  #Multicore backend
  # env = gym.make(f'{filename}-v0')
  # try:
  #   print(env.unwrapped.get_action_meanings())
  # except:
  #   pass

  # #========== Build network =====================================================
  # model = create_model()
  backend = MulticoreBackend(4)
  for i in range(10):
    # task_id = backend.test_network(model.get_weights(), env, 1, 1, model)
    task_id = backend.run(calc_big_number, i)
    print('task_id:', task_id)
  
  res = backend.join(values_only=False, numeric_only=True, min_value=False)
  print('Resuults:', res)
  
  