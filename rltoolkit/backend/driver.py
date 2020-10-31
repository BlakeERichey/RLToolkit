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
  hashes = []
  for i in range(6):
    res = backend.run(0, calc_big_number, i)
    print('Task Hash:', res)
    hashes.append(res)
  
  backend.get_results(hashes, hash_keys=False) 
  