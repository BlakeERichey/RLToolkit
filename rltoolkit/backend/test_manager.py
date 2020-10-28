import time
import datetime
import hashlib
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager

def calc_big_number(number):
  total=1
  for i in range(1, number+1):
    time.sleep(i)
    total+=i
  
  return total


class ParallelManager(BaseManager):  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.results_queue = Manager().Queue()
    self.processes_queue = Manager().Queue()

    self.current_hash    = 0
    self.hash_table      = {} #{hash: task_id}
    self.queued_tasks    = set() #hashes
    self.active_tasks    = set() #hashes
    self.completed_tasks = set() #hashes
    self.tasks = {} #{task_id hash: Task}... 
    # Task SCHEMA:
    # task = {
    #   'func':       func,
    #   'args':       args,
    #   'kwargs':     kwargs,
    #   'start_time': None, 
    #   'running':    False, 
    #   'result:':    None
    # }
    
    self.scheduler = TaskScheduler()
    self.register('schedule',    callable=self.schedule)
    self.register('monitor',     callable=self.monitor)
    self.register('request',     callable=self.request)
    self.register('respond',     callable=self.respond)
    self.register('get_results', callable=self.get_results)
    self.register('scheduler', callable=self.scheduler)

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
    task = {
      'func':       func,
      'args':       args,
      'kwargs':     kwargs,
      'start_time': None, 
      'running':    False, 
      'result:':    None
    }
    task_hash = str(self._get_new_hash())
    self.queued_tasks.add(task_hash)
    self.hash_table[task_hash] = task_id
    self.tasks[task_hash] = task
    print('Task', task_id, 'queued under', task_hash)
    return True

  def monitor(self,):
    """
      Interface for client to monitor the server for active tasks
    """
    return len(self.queued_tasks) > 0
  
  def request(self,):
    """
      Inteface for clients to request a task
    """
    task_hash = self.queued_tasks.pop()
    self.active_tasks.add(task_hash)
    task = self.tasks[task_hash]
    modInfo = {
      'start_time': datetime.datetime.now(), 
      'running':    True, 
    }
    task.update(modInfo)
    print(self.tasks[task_hash])
    return task

  def respond(self, task_id, retval, info):
    """
      Interface for clients to submit answers to tasks
    """
    print(self.scheduler.asking)
    pass

  def get_results(self,):
    """
      Interface for driver to request completed tasks' results
    """
    pass

class TaskScheduler:

  def __init__(self,):
    self.asking = 'what is your question'
  
  def get_question(self,):
    return self.asking

  def get_answer(self,):
    return True

  def change_question(self,):
    self.asking = 'This is the new question'
