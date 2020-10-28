import time
import datetime
import hashlib
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager
from test_manager import ParallelManager, calc_big_number

if __name__ == '__main__':
  manager = ParallelManager(address=('127.0.0.1', 50000), authkey=b'123')
  manager.start()
  print('Serving...')
  time.sleep(100)