import time
import gym
import datetime
import hashlib
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager
from test_manager import ParallelManager, calc_big_number, create_model
from backend import DistributedBackend, MulticoreBackend

if __name__ == '__main__':
  backend = DistributedBackend('127.0.0.1', 50000, authkey=b'123')
  print('Serving...')
  backend.spawn_server()
  