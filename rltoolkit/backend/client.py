import time
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager

from test_manager import ParallelManager, calc_big_number


if __name__ == '__main__':
  manager = ParallelManager(address=('127.0.0.1', 50000), authkey=b'123')
  print('Operating as Client')
  manager.connect()
  q = manager.scheduler.get_answer()
  print(q)
  print(repr(q))
  print(type(q))
  manager.scheduler.change_question()
  manager.respond('a', 'b', 'c')
  # active = False
  # while str(manager.monitor()) == 'False':
  #   active = manager.monitor()
  
  # print('Server has tasks queued')
  # task = manager.request()
  # print(task['start_time'])

  