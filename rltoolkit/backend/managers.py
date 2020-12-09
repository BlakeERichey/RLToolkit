import datetime
import multiprocessing
from multiprocessing          import Queue, Process, Manager
from multiprocessing.managers import SyncManager
import rltoolkit
from rltoolkit.backend.utils import Packet, clean_noisy_results, backend_test_network
from rltoolkit.wrappers import subprocess_wrapper

if os.name != 'nt':
  try:
    multiprocessing.set_start_method('forkserver')
  except Exception as e:
    if str(e) != 'context has already been set':
      raise e

class ParallelManager(SyncManager):  
  def __init__(self, *args, timeout=None, task_limit=None, **kwargs):
    """
      Initlizes a parallel manager for distributed task management.

      # Arguments
      address: (string server_ip, int port). 
        Port to open on the server and its IP address for remote connections.
      authkey: authorization key to connect to remote manager.
      timeout: Default time in seconds to permit for a task. 
        If a task takes longer, the server ceases to await a response.
      task_limit: Int. Max number of tasks for server to remember. 
        This monitors the total number of active, completed, and queued tasks.
    """
    super().__init__(*args, **kwargs)

    self.timeout    = timeout
    self.task_limit = task_limit

    self.current_task_id    = 1  
    self.queued_tasks    = set() #task_ids
    self.active_tasks    = set() #task_ids
    self.completed_tasks = set() #task_ids
    self.tasks = {} #{task_id: Task}... 
    # Task SCHEMA:
    # task = {
    #   'task_id':    task_ids
    #   'func':       func,
    #   'args':       args,
    #   'kwargs':     kwargs,
    #   'start_time': None, 
    #   'running':    False, 
    #   'result:':    None
    #   'timeout:':   None
    # }
    
    #Exposed methods to remote clients and drivers (via authkey)
    self.register('schedule',            callable=self.schedule)
    self.register('monitor',             callable=self.monitor)
    self.register('request',             callable=self.request)
    self.register('respond',             callable=self.respond)
    self.register('kill_tasks',          callable=self.kill_tasks)
    self.register('get_results',         callable=self.get_results)
    self.register('get_active_tasks',    callable=self.get_active_tasks)
    self.register('get_completed_tasks', callable=self.get_completed_tasks)

  def _get_new_task_id(self):
    """
      Returns a new task_id.
      Used for internal monitoring of tasks.
    """
    task_id = self.current_task_id
    self.current_task_id += 1
    return task_id

  def schedule(self, func, *args, timeout=None, **kwargs):
    """
      Interface for 'clients' to submit a problem and its dependencies to the 
      problem hoster, the 'server'. The server hosts this problem and loads it 
      into a self regulated task queue. 

      #Returns server's identifying task id. 

      # Arguments:
      func:    an exectuable function.
      args:    all arguments to be passed into func.
      kwargs:  all keyword arguments to be passed into func
      timeout: the max time in seconds to permit the function to be in 
        operation. If None, the default for the manager, as created
        by __init__(), will be used.
    """
    packet = Packet(None)
    if self.task_limit is None or len(self.tasks) < self.task_limit:
      task_id = str(self._get_new_task_id())
      self.queued_tasks.add(task_id)

      task = {
        'task_id':    task_id,
        'func':       func,
        'args':       args,
        'kwargs':     kwargs,
        'start_time': None, 
        'running':    False, 
        'result':    None,
        'timeout':   timeout or self.timeout
      }
      self.tasks[task_id] = task

      packet = Packet(task_id)
    
    return packet

  def monitor(self,):
    """
      Interface for client to monitor the server for active tasks

      Returns True if tasks are waiting in the servers task queue.
    """
    return Packet(len(self.queued_tasks) > 0)
  
  def request(self,):
    """
      Inteface for clients to request a task. 
      When a task is available in the task queue, the server respond 
      to clients with all the information needed to compute the hosted task.
    """
    packet = Packet(None)
    try:
      task_id = self.queued_tasks.pop() #if no tasks available, throws err
      self.active_tasks.add(task_id)
      task = self.tasks[task_id]
      modInfo = {
        'start_time': datetime.datetime.now(), 
        'running':    True, 
      }
      task.update(modInfo)
      packet = Packet(task)

    except:
      pass

    return packet

  def respond(self, task_id, retval, info=None):
    """
      Interface for clients to submit answers to tasks.
      If task is still marked as active and has not exceeded its maximum 
      duration, as determined by self.schedule(), then the server marks the 
      task as completed.

      # Arguments
      task_id: the task_id of the problem
      retval:  the computed answer to the task.
    """
    if task_id in self.active_tasks:
      if retval is None:
        logging.warning(f'Task {task_id} returned None.')
      task = self.tasks.get(task_id)
      end_time = datetime.datetime.now()
      duration = (end_time - task['start_time']).total_seconds()
      max_duration = task.get('timeout') or self.timeout
      if max_duration is None or duration <= max_duration:
        mod_info = {
          'result':  retval,
          'running': False
        }
        task.update(mod_info)
        self.active_tasks.remove(task_id)
        self.completed_tasks.add(task_id)
  
  def kill_tasks(self, task_ids):
    """
      Terminates active tasks and closes manager to listening for a 
      response for those specific tasks.
      This is called to ensure tasks do not exceed their maximum duration.
    """
    for task_id in task_ids:
      task = self.tasks.get(task_id)
      if task and task['running']:
        task['running'] = False
        try:
          self.active_tasks.remove(task_id)
          self.completed_tasks.add(task_id)
        except Exception as e:
          logging.warning(f'Error occured killing tasks: {e}')

  def get_active_tasks(self,):
    """
      Returns all active tasks' task_ids, starting times, and max duration.
      To be used by the backend for monitoring if active tasks have 
      exceeded their maximum duration.
    """
    active_tasks = self.active_tasks
    partial_active_tasks_dict = {} #active task dict with partial keys, value pairs
    for task_id in active_tasks:
      task = self.tasks[task_id]
      sub_dict = { #exposed values (others would constitute significant data)
        'timeout':    task.get('timeout'),
        'start_time': task.get('start_time')
      }
      partial_active_tasks_dict[task_id] = sub_dict

    return Packet(partial_active_tasks_dict)
  
  def get_completed_tasks(self,):
    """
      Returns task_ids of completed tasks
    """
    return Packet(self.completed_tasks)
  
  def clear_task(self, task_id):
    """
      Removes all traces of a task being present on the server.
      Removes the task_id from all task queues and opens memory for additional 
      tasks.
    """

    for collection in [self.queued_tasks, self.active_tasks, self.completed_tasks]:
      if task_id in collection:
        collection.remove(task_id)
    
    self.tasks.pop(task_id, None)

  def get_results(self, task_ids=[], values_only=True, clear=True):
    """
      Interface for driver to request completed tasks' results

      values_only: Remove task_id before returning values. Returned answers are in 
        the order of the 'task_ids' parameter.
      clear: If True, removes task from server memory after returning results.
    """
    results = {}
    as_list = []
    for task_id in task_ids:
      _id = task_id
      task = self.tasks[task_id]
      
      res = task.get('result')
      results[_id] = res
      if values_only:
        as_list.append(res)
      if clear:
        self.clear_task(task_id)
    
    return Packet((results, as_list)[values_only])