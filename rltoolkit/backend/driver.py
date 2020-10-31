import time
import datetime
import hashlib
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager
from test_manager import ParallelManager, calc_big_number
from backend import DistributedBackend

if __name__ == '__main__':
  backend = DistributedBackend('127.0.0.1', 50000, authkey=b'123')
  # backend.manager.connect()
  # print(backend.manager.get_active_tasks().unpack())
  # print(backend.manager.kill_tasks(['1','2','3','4','5','6']))
  hashes = []
  for i in range(6):
    res = backend.run(calc_big_number, i, timeout=8)
    print('Task Hash:', res)
    hashes.append(res)
  print('requesting:', hashes)
  backend.get_results(hashes) 
  