import time
import datetime
import hashlib
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager
from test_manager import ParallelManager, calc_big_number

if __name__ == '__main__':
  manager = ParallelManager(address=('127.0.0.1', 50000), authkey=b'123')
  manager.connect()
  print('Connected.')
  task_hash = manager.schedule(32, calc_big_number, 3)
  print('Scheduling task.', task_hash, type(task_hash))
  manager.shutdown()