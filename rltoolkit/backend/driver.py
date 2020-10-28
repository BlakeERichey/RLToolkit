import time
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager

from test_manager import ParallelManager, calc_big_number


if __name__ == '__main__':
  manager = ParallelManager(address=('127.0.0.1', 50000), authkey=b'123')
  print('Serving')
  manager.connect()
  q = manager.get_results()
  print('Got Queue')
  answer = manager.schedule(32, calc_big_number, 3)
  print(answer)

  # manager.get_server().serve_forever()
  # manager.start()
  