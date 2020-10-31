import os
import time
import types
import queue
import socket
import logging
import datetime
import multiprocessing
from   copy                     import deepcopy
from   keras.models             import clone_model
from   rltoolkit.utils          import test_network
from   rltoolkit.wrappers       import subprocess_wrapper
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager

#disable warnings in tensorflow subprocesses
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)

#Import predefined context for client spawning
import keras
import rltoolkit
import tensorflow as tf

if os.name != 'nt':
  try:
    multiprocessing.set_start_method('forkserver')
  except Exception as e:
    if str(e) != 'context has already been set':
      raise e


#========== MANAGERS ===========================================================

import time
import datetime
import hashlib
import asyncio
import multiprocessing
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import SyncManager, BaseManager

class ParallelManager(SyncManager):  
  def __init__(self, *args, timeout=None, task_limit=None, **kwargs):
    """
      task_limit: max number of tasks to remember and be able to queue
    """
    super().__init__(*args, **kwargs)

    self.timeout = timeout #max_time for task to run, in seconds
    self.task_limit = task_limit

    self.current_hash    = 1
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
    #   'timeout:':   None
    # }
    
    self.register('schedule',            callable=self.schedule)
    self.register('monitor',             callable=self.monitor)
    self.register('request',             callable=self.request)
    self.register('respond',             callable=self.respond)
    self.register('get_results',         callable=self.get_results)
    self.register('hash_mapping',        callable=self.hash_mapping)
    self.register('get_active_tasks',    callable=self.get_active_tasks)
    self.register('get_completed_tasks', callable=self.get_completed_tasks)

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

  def schedule(self, task_id, func, *args, timeout=None, **kwargs):
    """
      Place a task into the task queue and return a task hash
    """
    packet = Packet(None)
    if len(self.tasks) < self.task_limit:
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
        'result:':    None,
        'timeout:':   timeout or self.timeout
      }
      self.tasks[task_hash] = task

      print('Task', task_id, 'queued under', task_hash)
      packet = Packet(task_hash)
    
    return packet

  def monitor(self,):
    """
      Interface for client to monitor the server for active tasks
    """
    return Packet(len(self.queued_tasks) > 0)
  
  def request(self,):
    """
      Inteface for clients to request a task
    """
    packet = Packet(None)
    try:
      task_hash = self.queued_tasks.pop() #if no tasks available, throws err
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
    if task_hash in self.active_tasks:
      task = self.tasks.get(task_hash)
      end_time = datetime.datetime.now()
      duration = (end_time - task['start_time']).total_seconds()
      max_duration = task.get('timeout') or self.timeout
      if max_duration is None or duration <= max_duration:
        mod_info = {
          'result':  retval,
          'running': False
        }
        task.update(mod_info)
        print('Result', task_hash, retval)
        self.active_tasks.remove(task_hash)
        self.completed_tasks.add(task_hash)
  
  def kill_tasks(self, task_hashes):
    """
      Terminates an active task and closes manager to listening for a response.
    """
    for task_hash in task_hashes:
      task = self.active_tasks.get(task_hash)
      if task:
        task['running'] = False
        self.active_tasks.remove(task_hash)
        self.completed_tasks.add(task_hash)

  def get_active_tasks(self,):
    """
      Returns all active tasks' hashes, starting times, and max duration
    """
    active_tasks = self.active_tasks
    partial_active_tasks_dict = {} #active task dict with partial keys, value pairs
    for task_hask in active_tasks:
      task = self.active_tasks[task_hask]
      sub_dict = { #exposed values (others would constitute significant data)
        'timeout':    task.get('timeout')
        'start_time': task.get('start_time')
      }
      partial_active_tasks_dict[task_hask] = sub_dict

    return Packet(partial_active_tasks_dict)
  
  def get_completed_tasks(self,):
    """
      Returns hashes of completed tasks
    """
    return Packet(self.completed_tasks)
  
  def clear_task(self, task_hash):
    """
      Removes all traces of a task being present on the server.
    """

    for collection in [self.queued_tasks, self.active_tasks, self.completed_tasks]:
      if task_hash in collection:
        collection.remove(task_hash)
    
    self.tasks.pop(task_hash, None)
    self.hash_table.pop(task_hash, None)

  def get_results(self, task_hashes=[], hash_keys=True, clear=True):
    """
      Interface for driver to request completed tasks' results
      hash_keys: returned dictionary should use hashes as keys. 
      Defaults to using original task_id

      clear: Remove hash after returning values
    """
    results = {}
    for task_hash in task_hashes:
      task = self.tasks[task_hash]
      print('task:', task)
      if hash_keys:
        _id = task_hash
      else:
        _id = self.hash_table[task_hash]
      
      results[_id] = task.get('result')
      if clear:
        self.clear_task(task_hash)

    print('Returning:', results)
    return Packet(results)

  def hash_mapping(self, task_hashes=[]):
    """
      Returns the task_ids that correspond to the task_hashes
    """
    mapping = [self.hash_table[task_hash] for task_hash in task_hashes]

    return Packet(mapping)


class Packet:

  def __init__(self,data):
    self.data = data    
  
  def unpack(self,):
    return self.data

#========== BACKENDS ===========================================================

class DistributedBackend:

  def __init__(self, server_ip='127.0.0.1', port=50000, authkey=b'rltoolkit', 
              network_generator=None, use_gpu=False, require_gpu=False):
    """
      Initializes a Distributed & Multicore Backend Remote Manager.

      # Arguments
      network_generator: function. Function that returns a Keras model. 
        Used for client nodes to interpret network architecture and graph context.
      server_ip: String. IP address for Remote Server. Client machines must use 
        and be able to see this machine.
      port: Int. Port number to open for clients to interface with the manager.
      authkey: Byte string. Used to authenticate access to the manager.

      ## NOT IMPLEMENTED
      use_gpu: Specifies if the Backend should look for GPUs.
      require_gpu: Specifies if the Backend should spawn a process in absence 
        of an available GPU. Set to True if the Backend should wait for a GPU 
        to become available. Overrides passed in value of `use_gpu`. Sets to True.
    """
    
    if network_generator is not None:
      assert type(network_generator) == types.FunctionType, \
        'Expected function for network generator.'

    self.tasks     = {}      #task_id: Task, intended for tracking crashed processes
    self.port      = port
    self.authkey   = authkey
    self.server_ip = server_ip
    self.network_generator = network_generator
    
    # Start a shared manager server and access its queues
    self.manager = ParallelManager(address=(server_ip, port), authkey=authkey)
  
  def spawn_server(self):
    """
      Initializes a server on the active thread. 
      Hangs the process until it is terminated.
      Should be called in a subprocess.
    """
    manager = self.manager
    manager.start()
    

    # server = manager.get_server()
    ip = socket.gethostbyname(socket.gethostname())
    print(f'Server started. Port {manager.address[1]}. Local IP: {ip}')
  
  def _monitor_active_tasks(self,):
    """
      Run on server. Monitors active tasks to ensure they complete within 
      timeout limit. Hangs the active thread.
    """

    manager=self.manager
    manager.connect()

    while True:
      time.sleep(1)
      tasks_to_kill = set()
      active_tasks = manager.get_active_tasks().unpack()
      for task_hash, task in active_tasks.items():
        start_time   = task.get('start_time')
        max_duration = task.get('timeout')
        duration = (datetime.datetime.now() - start_time).total_seconds()

        if duration > max_duration:
          tasks_to_kill.add(task_hash)
      
      if len(tasks_to_kill):
        manager.kill_tasks(tasks_to_kill)

  def spawn_client(self, cores=1):
    """
      Uses the active thread to connect to the remote server.
      Performs processes out of managers process Queue in parallel.

      # Arguments
      cores: Int. How many cores to utilize in addition to the active thread.
    """
    manager = self.manager
    manager.connect()
    
    print('Connected.', manager.address)
    active=False
    while True:
      time.sleep(1)
      if active is False:
        active = manager.monitor().unpack()
      else:
        packet = manager.request()
        data = packet.unpack()
        if data is not None:
          print('Task received...', end='', flush=True)
          tash_hash = data['hash']
          func      = data['func']
          args      = data['args']
          kwargs    = data['kwargs']
          retval    = func(*args, **kwargs)
        
          manager.respond(tash_hash, retval)
          active='False'
          print('Complete.\nResult:', retval)
  
  def test_network(self, task_id, weights, env, episodes, seed, network=None):
    """
      Neural Network DistributedBackend.run() utility function. 

      Wraps and enqueues environment testing so that network recreation 
      happens on subcore and not main thread.

      # Arguments
      weights:  list of weights corresponding to Keras NN weights
      env:      a Gym environment
      episodes: How many episodes to test an environment.
      seed:     Random seed to ensure consistency across tests
      network:  No functionality, kept for identical argument structure to 
        MulticoreBackend.test_network().
    """
    self.run(
      task_id, backend_test_network, weights,      #model creation params
      self.network_generator, env, episodes, seed  #test network params
    )

  def run(self, task_id, func, *args, **kwargs):
    """
      Places a process onto the server process Queue.

      # Arguments
      task_id: id for the task when placed into the server process queue. 
        Used because the queue may complete out of order and will allow for 
        reordering upon completion.
      func: function to be run via clients. Needs to be a function visible to 
        the requesting machine, the server, and the clients.
      args: arguments for `func`.
      kwargs: keyword arguments for `func`.
    """
    manager = self.manager
    manager.connect()
    packet = manager.schedule(task_id, func, *args, **kwargs)
    task_hash = packet.unpack()
    return task_hash
  
  def get_results(self, task_hashes=[], clean=True, hash_keys=True):
    """
      Gets a number of results from the results queue. Defaults to 1 result
      Hangs current thread until this quantity has been retreived.
    """
    manager = self.manager
    manager.connect()

    #When all the requested tasks are completed
    task_hashes = set(task_hashes)
    request_results = False #request results from server?
    while not request_results:
      time.sleep(1)
      completed_tasks = manager.get_completed_tasks().unpack()
      request_results = task_hashes.difference(completed_tasks) == set()
      print(task_hashes)
      print(completed_tasks)
      print(request_results)
    
    print('tasks complete')
    results = manager.get_results(task_hashes, hash_keys=hash_keys).unpack()
    print('Received Results', results)
    
    if num_results == 1:
      result = manager.get_results().get()
      if clean:
        result = result['result']
    else:
      result = []
      for _ in range(num_results):
        res = manager.get_results().get()
        result.append(res)

      if clean:
        result.sort(key= lambda val: val['pid'])
        result = [val['result'] for val in result]


    return result

class MulticoreBackend():

  def __init__(self, cores=1, timeout=None):
    """
      Initializes a MulticoreBackend with a configurable number of cores
      
      #Arguments
      cores: Int. Max number of cores to utilize.
      timeout: Max time in seconds to permit a Process to run.
    """
    
    self.active   = 0       #number of active processes
    self.tasks    = {}      #task_id: Task
    self.cores    = cores   #max number of processes to spawn at one time
    self.results  = Queue() #results queue
    self.timeout  = timeout #max time for process
  
  def test_network(self, task_id, weights, env, episodes, seed, network):
    """
      Neural Network MulticoreBackend.run() utility function. 

      Wraps and enqueues environment testing so that network recreation 
      happens on subcore and not main thread.

      # Arguments
      weights:  list of weights corresponding to Keras NN weights
      network:  a keras Neural network
      env:      a Gym environment
      episodes: How many episodes to test an environment.
      seed:     Random seed to ensure consistency across tests
    """
    self.run(
      task_id, backend_test_network, weights, #model creation params
      network, env, episodes, seed            #test network params
    )

  def run(self, task_id, func, *args, **kwargs):
    """
      Spawns a subprocess passing args into func(). 
      Queues a subprocess in the event of no available cores
      # Arguements
      task_id: id for the task when placed into the server process queue. 
        Used because the queue may complete out of order and will allow for 
        reordering upon completion.
      func: a function
      args: value based arugements for function `func`
      kwargs: keyword based arguements for `func`
      # Example
      def example_func(i, name=None):
        print(i, name)
      >>>MulticoreBackend.run(example_func, 1, name='test')
      >>>'1 test'
    """

    #add callbacks features and function wrapper
    args = (func,self.results,task_id) + (args, (args,))[isinstance(args, int)]

    p = Process(target=subprocess_wrapper, args=args, kwargs=kwargs)
    if self.active < self.cores:
      #Start processes
      p.start()
      self.active+=1
      task = {
        'process':    p, 
        'start_time': datetime.datetime.now(), 
        'running':    True, 
        'result:':    None
      }
    else:
      #Queue process
      task = {
        'process':    p, 
        'start_time': None, 
        'running':    False, 
        'result':     None
      }
    
    self.tasks[task_id] = task

  def join(self, clean=True): #####IMPROVE DOCUMENTATION FOR DOCSTRING#####
    """
      Syncronously awaits all subprocesses competion and returns 
      when this condition is met

      clean: if True, function sorts results by pid then 
        strips process ids from returned list
    """

    done = {} #completed tasks
    keys = list(self.tasks.keys()) #keys for queued tasks
    while len(self.tasks):

      #check status of current processes
      i = 0
      while i < len(keys):
        task_id = keys[i]
        task    = self.tasks[task_id] #{process start_time running result}
        process = task['process']
        
        if task['running']:
          #terminate process, if done or timeout reached
          if not process.is_alive(): #Task completed normally
            process.join()

            #Remove Task from queued/active
            keys.pop(i)
            self.active -= 1
            task['running'] = False
            done[task_id] = task
            self.tasks.pop(task_id)
            i-=1 #back up an index, due to keys.pop()
          
          elif self._time_limit_reached(task): #Task has taken too long, kill it
            process.kill()
            self.results.put({
              'pid':    task_id,
              'result': None,
            })

            #Remove Task from queued/active
            keys.pop(i)
            self.active -= 1
            task['running'] = False
            done[task_id] = task
            self.tasks.pop(task_id)
            i-=1 #back up an index, due to keys.pop()

        elif self.active < self.cores: #Open core, spawn new process
          process.start()
          task['running'] = True
          task['start_time'] = datetime.datetime.now()
          self.active += 1

          
        i+=1

    return results_from_queue(self.results, done, clean)
  
  def _time_limit_reached(self, task):
    dt = (datetime.datetime.now() - task['start_time']).total_seconds()
    return dt > self.timeout if self.timeout is not None else False

#========== UTILITIES ==========================================================
def backend_test_network(weights, network, env, episodes, seed):
  """
    Wraps environment testing so that network recreation happens on subcore and 
    not main thread. 

    # Arguments
    weights:  list of weights corresponding to Keras NN weights
    network:  a keras Neural network
    env:      a Gym environment
    episodes: How many episodes to test an environment.
    seed:     Random seed to ensure consistency across tests
  """

  try:
    # time.sleep(15)
    env = deepcopy(env)
    # print('Network is func:', type(network) == types.FunctionType, network)
    if type(network) == types.FunctionType: #Distributed create_model
      nn = network()
    else:
      nn  = clone_model(network)
    nn.set_weights(weights)
    # foo = bar
    avg = test_network(nn, env, episodes, seed=seed)
    # print('Avg:', avg)
  except:
    avg = None
    # print('Avg:', avg)
  return avg

def results_from_queue(q, done, clean): #####IMPROVE DOCUMENTATION FOR DOCSTRING#####
    """
      Helper function for getting results from self.results Queue().
      Offers cleaning utility to control returned format.

      #Arguments
      q: results queue
      done: completed Tasks dictionary
      clean: If True returns list of rewards sorted by task_id
    """
    #Stage all results
    i = 0
    results = []
    min_score = None #Determine value to replace for timed out tasks or errors
    n_tasks = len(done.keys()) #how many results to pull from queue
    while i < n_tasks:
      res = q.get()
      
      #Find minimum score of all tasks, for processes that failed
      result = res['result']
      if min_score is None: #First process could have failed, thus i==0 is wrong
        min_score = result
      else:
        if result is not None:
          try:
            min_score = min(min_score, result)
          except TypeError:
            pass

      results.append(res)
      task_id = int(res['pid'])
      done[task_id]['result'] = result

      i+=1
    
    #Resolve processes that threw errors
    if clean:
      results = []
      sorted_tasks = sorted(done.items(), key= lambda item: item[0])
      for _, task in sorted_tasks:
        result = task['result']
        if result is None:
          result = min_score
        
        results.append(result)
    
    return results