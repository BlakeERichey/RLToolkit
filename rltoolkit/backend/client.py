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
  active='False'
  while True:
    if str(active) == 'False':
      active = manager.monitor()
      time.sleep(1)
    else:
      print('Task queued:', active)
      packet = manager.request()
      print(packet, type(packet))
      data = packet.get_data()
      print('Packet received...', data, '\n\n')
      tash_hash = data['hash']
      func      = data['func']
      args      = data['args']
      kwargs    = data['kwargs']
      retval    = func(*args, **kwargs)
      
      manager.respond(tash_hash, retval)
      active='False'
      print('Complete.', retval)
      print('Requesting new task.')
