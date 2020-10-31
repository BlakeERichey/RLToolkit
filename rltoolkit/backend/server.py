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
  backend.spawn_server()
  print('Serving...')
  while True:
    pass
  