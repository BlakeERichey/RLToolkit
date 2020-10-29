import time
import datetime
import hashlib
import asyncio
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager

def calc_big_number(number):
  total=0
  for i in range(1, number+1):
    time.sleep(i)
    total+=i
  
  return total


class ParallelManager(SyncManager):  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.results_queue = Manager().Queue()
    self.processes_queue = Manager().Queue()

    self.timeout = 500 #max_time for task to run, in seconds

    self.current_hash    = 0
    self.hash_table      = {} #{hash: task_id}
    self.queued_tasks    = set() #hashes
    self.active_tasks    = set() #hashes
    self.completed_tasks = set() #hashes
    self.tasks = {} #{task_id hash: Task}... 
    # Task SCHEMA:
    # task = {
    #   'hash':       hash
    #   'func':       func,
    #   'args':       args,
    #   'kwargs':     kwargs,
    #   'start_time': None, 
    #   'running':    False, 
    #   'result:':    None
    # }
    
    self.register('schedule',    callable=self.schedule)
    self.register('monitor',     callable=self.monitor)
    self.register('request',     callable=self.request)
    self.register('respond',     callable=self.respond)
    self.register('get_results', callable=self.get_results)

  def _get_new_hash(self):
    """
      Returns a new hash.
      Used for identifying tasks with the same task_id
    """
    # byte_string = bytes(str(self.current_hash), encoding='utf-8')
    # new_hash = hashlib.md5(byte_string).hexdigest()
    new_hash = self.current_hash
    self.current_hash += 1
    return new_hash

  def schedule(self, task_id, func, *args, **kwargs):
    task_hash = str(self._get_new_hash())
    self.queued_tasks.add(task_hash)
    self.hash_table[task_hash] = task_id

    task = {
      'hash':       task_hash,
      'func':       func,
      'args':       args,
      'kwargs':     kwargs,
      'start_time': None, 
      'running':    False, 
      'result:':    None
    }
    self.tasks[task_hash] = task

    print('Task', task_id, 'queued under', task_hash)
    print(self.queued_tasks)
    return task_hash

  def monitor(self,):
    """
      Interface for client to monitor the server for active tasks
    """
    return len(self.queued_tasks) > 0
  
  def request(self,):
    """
      Inteface for clients to request a task
    """
    packet = None
    try:
      task_hash = self.queued_tasks.pop()
      self.active_tasks.add(task_hash)
      task = self.tasks[task_hash]
      modInfo = {
        'start_time': datetime.datetime.now(), 
        'running':    True, 
      }
      task.update(modInfo)
      packet = Packet(task)
    except:
      pass

    return packet

  def respond(self, task_hash, retval, info=None):
    """
      Interface for clients to submit answers to tasks
    """
    end_time = datetime.datetime.now()
    task = self.tasks[task_hash]
    if (end_time - task['start_time']).total_seconds() <= self.timeout:
      mod_info = {
        'result':  retval,
        'running': False
      }
    print('Result', task_hash, retval)

    self.active_tasks.remove(task_hash)
    self.completed_tasks.add(task_hash)

  def get_results(self, task_hashes=[], hash_keys=False):
    """
      Interface for driver to request completed tasks' results
      hash_keys: returned dictionary should use hashes as keys. 
      Defaults to using original task_id
    """
    print('In results:', self.queued_tasks)
    pass
    # while True:
    #   pass
    # results = {}
    # for task_hash in task_hashes:
    #   task = self.tasks[task_hash]
    #   if hash_keys:
    #     _id = task_hash
    #   else:
    #     _id = task.get('task_id')
      
    #   results[_id] = task.get('result')
      


class Packet:

  def __init__(self,data):
    self.data = data    
  
  def get_data(self,):
    return self.data