import os
import time
import types
import queue
import socket
import logging
import warnings
import datetime
import multiprocessing
from   copy                     import deepcopy
from   rltoolkit.utils          import test_network, silence_function
from   rltoolkit.wrappers       import subprocess_wrapper
from   multiprocessing          import Queue, Process, Manager
from   multiprocessing.managers import BaseManager, SyncManager

with warnings.catch_warnings():
  #disable warnings in tensorflow subprocesses
  warnings.simplefilter("ignore")
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  logging.disable(logging.WARNING)

  #Import predefined context for client spawning
  import tensorflow as tf
  import keras
  from keras.models import clone_model
  import rltoolkit

# if os.name != 'nt':
#   try:
#     multiprocessing.set_start_method('forkserver')
#   except Exception as e:
#     if str(e) != 'context has already been set':
#       raise e


#========== MANAGERS ===========================================================

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


class Packet:

  def __init__(self,data):
    self.data = data    
  
  def unpack(self,):
    return self.data

#========== BACKENDS ===========================================================

class DistributedBackend:

  def __init__(self, server_ip='127.0.0.1', port=50000, authkey=b'rltoolkit', 
              timeout=None, network_generator=None, 
              use_gpu=False, require_gpu=False):
    """
      Initializes a Distributed & Multicore Backend Remote Manager.

      # Arguments
      server_ip: String. IP address for Remote Server. Client machines must use 
        and be able to see this machine.
      port: Int. Port number to open for clients to interface with the manager.
      authkey: Byte string. Used to authenticate access to the manager.
      timeout: Default time in seconds to permit on ther server for a task. 
        If a task takes longer, the server ceases to await a response.
      network_generator: function. Function that returns a Keras model. 
        Used for client nodes to interpret network architecture and graph context.

      ## NOT IMPLEMENTED
      use_gpu: Specifies if the Backend should look for GPUs.
      require_gpu: Specifies if the Backend should spawn a process in absence 
        of an available GPU. Set to True if the Backend should wait for a GPU 
        to become available. Overrides passed in value of `use_gpu`. Sets to True.
    """
    
    if network_generator is not None:
      assert type(network_generator) == types.FunctionType, \
        'Expected function for network generator.'

    self.port      = port
    self.authkey   = authkey
    self.server_ip = server_ip
    self.manager_creds = (server_ip, port, authkey)
    self.network_generator = network_generator
    
    # Start a shared manager server and access its queues
    self.manager = ParallelManager(
      address=(server_ip, port), 
      authkey=authkey, 
      timeout=timeout
    )
  
  def shutdown(self,):
    """
      Does nothing. Kept for consistency between backends.
    """
    pass
  
  def spawn_server(self):
    """
      Initializes a server on the active thread and monitors the server's 
      active task queue. Hangs the process until it is terminated.
    """
    manager = self.manager
    manager.start()
    

    # server = manager.get_server()
    ip = socket.gethostbyname(socket.gethostname())
    print(f'Server started. Port {manager.address[1]}. Local IP: {ip}')
    self._monitor_active_tasks()
  
  def _monitor_active_tasks(self,):
    """
      Run on server. Monitors active tasks to ensure they complete within 
      timeout limit. Hangs the active thread.
    """

    manager=self.manager
    manager.connect()

    while True:
      time.sleep(1) #delay as to not overwhelm servers incoming packets
      tasks_to_kill = set()
      active_tasks = manager.get_active_tasks().unpack()

      #If an active task duration exceeds its timeout limit, set to kill.
      for task_id, task in active_tasks.items():
        start_time   = task.get('start_time')
        max_duration = task.get('timeout')
        duration = (datetime.datetime.now() - start_time).total_seconds()

        if max_duration and duration > max_duration:
          tasks_to_kill.add(task_id)
      
      if len(tasks_to_kill):
        manager.kill_tasks(tasks_to_kill)

  def spawn_client(self, cores=1):
    """
      Uses the active thread to connect to the remote server.
      Sets the client to monitor the connected server for tasks. When tasks are 
      available, client will request the necessary functions and data to 
      complete, and then submit the computed result to the server.

      # Arguments
      cores: Int. How many cores to utilize in addition to the active thread.
    """

    port      = self.port
    authkey   = self.authkey
    server_ip = self.server_ip
    if cores > 1:
      multi_backend = MulticoreBackend(cores=cores)
      for i in range(cores):
        multi_backend.run(
          DistributedBackend._spawn_client_wrapper, 
          server_ip, port, authkey
        )
      multi_backend.join()
    else:
      DistributedBackend._spawn_client_wrapper(server_ip, port, authkey)

  @staticmethod
  def _spawn_client_wrapper(server_ip, port, authkey):
    """
      Wrapper for multiprocessing backend to spawn clients in subprocesses.
    """
    manager = ParallelManager(address=(server_ip, port), authkey=authkey)
    manager.connect()
    
    print('Connected.', manager.address)
    tasks_queued = False
    while True:
      time.sleep(1) #Time delay to not overload the servers incoming packets
      if tasks_queued is False:
        tasks_queued = manager.monitor().unpack()
      else:
        #Request info to complete task
        packet = manager.request()
        
        #Unpack info and compute result
        data = packet.unpack()
        if data is not None:
          task_id   = data['task_id']
          func      = data['func']
          args      = data['args']
          kwargs    = data['kwargs']
          retval    = func(*args, **kwargs)
        
          manager.respond(task_id, retval)
          tasks_queued=False
           
  def test_network(self, weights, env, episodes, seed, network=None, timeout=None):
    """
      Neural Network DistributedBackend.run() utility function. 

      Wraps and enqueues environment testing so that network recreation 
      happens on subcore and not main thread.

      Returns associated task_id for reordering results.

      # Arguments
      weights:  list of weights corresponding to Keras NN weights
      env:      a Gym environment
      episodes: How many episodes to test an environment.
      seed:     Random seed to ensure consistency across tests
      network:  No functionality, kept for identical argument structure to 
        MulticoreBackend.test_network().
      timeout: the max time in seconds to permit the function to be in 
        operation. If None, the default for the manager, as created
        by __init__(), will be used.
    """
    assert self.network_generator is not None, "Unable to test network without generator function"
    return self.run(
      backend_test_network, weights,      #model creation params
      self.network_generator, env, episodes, seed,  #test network params
      timeout=timeout
    )

  def run(self, func, *args, timeout=None, **kwargs):
    """
      Places a task into the servers queued tasks for client completion.

      # Returns the associated task id to recover the results.

      # Arguments
      func: function to be run via clients. Needs to be a function visible to 
        the requesting machine, the server, and the clients.
      args: arguments for `func`.
      kwargs: keyword arguments for `func`.
      timeout: the max time in seconds to permit the function to be in 
        operation. If None, the default for the manager, as created
        by __init__(), will be used.
    """
    manager = self.manager
    manager.connect()
    packet = manager.schedule(func, *args, timeout=timeout, **kwargs)
    task_id = packet.unpack()
    return task_id
  
  def get_results(self, task_ids=[], 
                  values_only=True, numeric_only=False, ref_value='min'):
    """
      Gets a number of results from the results queue. Defaults to 1 result
      Hangs current thread until this quantity has been retreived.


      task_ids: task_ids as generated by self.run(). These are used by the 
        server to identify which task to return the results for.
      values_only: if False returns dictionary that includes the task ids with its 
        results. Otherwise, returns the values computed in order of the 
        requested task ids.
      numeric_only: if True, adjusts non numeric results to a `ref_value` 
        numeric value of the results, as determine by the `ref_valuue` parameter.
      ref_value: One of ['min', 'minimum', 'max', 'maximum'] adjusts non 
        numeric results to the reference value. `numeric_only` must be True, 
        or this parameter is overlooked.
    """
    manager = self.manager
    manager.connect()

    #When all the requested tasks are completed, get results
    task_set = set(task_ids)
    request_results = False #request results from server?
    while not request_results:
      time.sleep(1) #delay to limit servers incoming packets
      completed_tasks = manager.get_completed_tasks().unpack()

      #test if all observed tasks are among the completed
      request_results = task_set.difference(completed_tasks) == set()
    
    results = manager.get_results(task_ids, values_only=values_only).unpack()
    if numeric_only:
      if values_only:
        clean_noisy_results(results, reference=ref_value)
      else:
        temp_results = list(results.values())
        clean_noisy_results(temp_results, reference=ref_value)
        for i, key in enumerate(results.keys()):
          results[key] = temp_results[i]
    
    return results

class MulticoreBackend():

  def __init__(self, cores=1, timeout=None):
    """
      Initializes a MulticoreBackend with a configurable number of cores
      
      #Arguments
      cores: Int. Max number of cores to utilize.
      timeout: Max time in seconds to permit a Process to run.
    """
    
    self.active          = 0       #number of active processes
    self.tasks           = {}      #task_id: Task
    self.cores           = cores   #max number of processes to spawn at one time
    self.results         = Queue() #results queue
    self.timeout         = timeout #max time for process
    self.current_task_id = 1       #Manages task ids

  def shutdown(self,):
    """
      Terminates all tasks processes.
    """
    #Terminate all processes (These loops are done asynchronously)
    for task_id in self.tasks:
      task = self.tasks[task_id]
      task['running'] = False
      p = task['process']
      if p._popen is not None: #task has started
        p.terminate()
    
    #Free Memeory
    for task_id in self.tasks:
      task = self.tasks[task_id]
      p = task['process']
      #Process never started         #process was terminated
      while p._popen is not None and p.exitcode is None:
        pass #Wait until process has ended
      p.close()
    
    self.active = 0
  
  def test_network(self, weights, env, episodes, seed, network, timeout=None):
    """
      Neural Network MulticoreBackend.run() utility function. 

      Wraps and enqueues environment testing so that network recreation 
      happens on subcore and not main thread.

      Returns associated task id.

      # Arguments
      weights:  list of weights corresponding to Keras NN weights
      network:  a keras Neural network
      env:      a Gym environment
      episodes: How many episodes to test an environment.
      seed:     Random seed to ensure consistency across tests
      timeout: the max time in seconds to permit the function to be in 
        operation. If None, the default for the manager, as created
        by __init__(), will be used.
    """
    return self.run(
      backend_test_network, weights, #model creation params
      network, env, episodes, seed,            #test network params
      timeout=timeout
    )

  def run(self, func, *args, timeout=None, **kwargs):
    """
      Spawns a subprocess passing args into func(). 
      Queues a subprocess in the event of no available cores

      Returns the task_id associated with the task.

      # Arguements
      func: a function
      args: value based arugements for function `func`
      kwargs: keyword based arguements for `func`
      timeout: the max time in seconds to permit the function to be in 
        operation. If None, the default for the manager, as created
        by __init__(), will be used.

      # Example
      def example_func(i, name=None):
        print(i, name)
      >>>MulticoreBackend.run(example_func, 1, name='test')
      >>>'1 test'
    """

    #add callbacks features and function wrapper
    task_id = self.current_task_id
    self.current_task_id += 1
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
        'result:':    None,
        'timeout':    timeout or self.timeout
      }
    else:
      #Queue process
      task = {
        'process':    p, 
        'start_time': None, 
        'running':    False, 
        'result':     None,
        'timeout':    timeout or self.timeout
      }
    
    self.tasks[task_id] = task
    return task_id

  def join(self, values_only=True, numeric_only=False, ref_value='min'): #####IMPROVE DOCUMENTATION FOR DOCSTRING#####
    """
      Syncronously awaits all subprocesses competion and returns 
      when this condition is met

      values_only: if True, function sorts results by pid then 
        strips process ids from returned list
      numeric_only: if True, adjusts non numeric results to a `ref_value` 
        numeric value of the results, as determine by the `ref_valuue` parameter.
      ref_value: One of ['min', 'minimum', 'max', 'maximum'] adjusts non 
        numeric results to the reference value. `numeric_only` must be True, 
        or this parameter is overlooked.
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

    results = self._results_from_queue(done, values_only)
    if numeric_only:
      if values_only:
        clean_noisy_results(results, reference=ref_value)
      else:
        temp_results = list(results.values())
        clean_noisy_results(temp_results, reference=ref_value)
        for i, key in enumerate(results.keys()):
          results[key] = temp_results[i]
    return results
  
  def _results_from_queue(self, done, values_only=True):
    """
      Helper function for getting results from self.results Queue().

      #Arguments
      q: results queue
      done: completed Tasks dictionary
      values_only: If True returns list of retruned values. If False, returns dict
        with attached task_ids
    """
    q = self.results
    #Stage all results
    i = 0
    results = [] if values_only else {}
    n_tasks = len(done.keys()) #how many results to pull from queue
    while i < n_tasks:
      res = q.get()
  
      task_id = res['pid']
      result  = res['result']
      done[task_id]['result'] = result
      i+=1
    
    #Add returned values to results in order of task_id
    sorted_tasks = sorted(done.items(), key= lambda item: item[0]) #sort by task_id
    for task_id, task in sorted_tasks:
      result = task['result']
      if values_only:
        results.append(result)
      else:
        results[task_id] = result

    return results
  
  def _time_limit_reached(self, task):
    dt = (datetime.datetime.now() - task['start_time']).total_seconds()
    timeout = task['timeout']
    return dt > timeout if timeout is not None else False

class LocalClusterBackend(DistributedBackend):

  def __init__(self, cores=1, *args, **kwargs):
    """
      Initializes a DistributedBackend run off localhost to run tasks 
      concurrently without reloading contexts.
      
      #Arguments
      cores: Int. Max number of cores to utilize.
      timeout: Max time in seconds to permit a Process to run.
      network_generator: function. Function that returns a Keras model. 
        Used for client nodes to interpret network architecture and graph context.
    """
    super().__init__(*args, **kwargs)
    print('Initializing LocalClusterBackend Backend.')
    self.manager.start()

    port      = self.port
    authkey   = self.authkey
    server_ip = self.server_ip
    self.manager_creds = (server_ip, port, authkey)

    self.multi_backend = MulticoreBackend(cores+1) #Negligible monitoring core
    for _ in range(cores):
      self.multi_backend.run(
        silence_function, 1, 
        LocalClusterBackend._spawn_client_wrapper,
        *self.manager_creds
      )
    self.multi_backend.run(LocalClusterBackend.monitor_active_tasks, *self.manager_creds)

  @staticmethod
  def monitor_active_tasks(server_ip, port, authkey):
    """
      Run on server. Monitors active tasks to ensure they complete within 
      timeout limit. Hangs the active thread.
    """
    manager = ParallelManager(address=(server_ip, port), authkey=authkey)
    manager.connect()

    while True:
      time.sleep(1) #delay as to not overwhelm servers incoming packets
      tasks_to_kill = set()
      active_tasks = manager.get_active_tasks().unpack()

      #If an active task duration exceeds its timeout limit, set to kill.
      for task_id, task in active_tasks.items():
        start_time   = task.get('start_time')
        max_duration = task.get('timeout')
        duration = (datetime.datetime.now() - start_time).total_seconds()

        if max_duration and duration > max_duration:
          tasks_to_kill.add(task_id)
      
      if len(tasks_to_kill):
        manager.kill_tasks(tasks_to_kill)


  def shutdown(self,):
    """
      Terminates open tasks.
    """
    logging.debug('Shutting down cluster.')
    self.manager.shutdown()
    # task_schedulers subprocesses will be terminated due to manager.shutdown(), 
    self.multi_backend.shutdown()
    print('Cluster shutdown.')

#========== UTILITIES ==========================================================
def backend_test_network(weights, network, env, episodes, seed):
  """
    Wraps environment testing so that network recreation happens on subcore and 
    not main thread. 

    # Arguments
    weights:  list of weights corresponding to Keras NN weights
    network:  a keras Neural network or function to create one
    env:      a Gym environment
    episodes: How many episodes to test an environment.
    seed:     Random seed to ensure consistency across tests
  """


  ###CAN THROW ERRORS IN TRY LOOP THAT SHOULD BE REPORTED
  try:
    env = deepcopy(env)
    keras.backend.clear_session()
    if type(network) == types.FunctionType: #Distributed create_model
      nn = network()
      # nn.summary() #To identify is session is being cleared
    else: #Multicore model
      nn  = clone_model(network)
    nn.set_weights(weights)
    avg = test_network(nn, env, episodes, seed=seed)
  except Exception as e:
    logging.error(f'Exception occured testing network: {e}')
    avg = None
  return avg
  
def clean_noisy_results(results, reference='min'):
  """
    Makes corrections in place to results for tasks that failed or returned 
    None. Ensures a numeric answer for each result.

    #Arguments:
    Results: list of values that are either None or Numeric. 
    reference: One of ['min', 'minimum', 'max', 'maximum'] adjusts non 
      numeric results to the reference value. `numeric_only` must be True, 
      or this parameter is overlooked.
  """
  assert reference in {'min', 'minimum', 'max', 'maximum'}, \
    f"Unidentified keyword for reference: {reference}"

  min_value = reference in {'min', 'minimum'} #use minimum?

  ref_value = None
  for result in results:
    if ref_value is None: #First process could have failed, thus resuults[0] could be None
        ref_value = result
    else:
      if result is not None:
        try:
          if min_value:
            ref_value = min(ref_value, result)
          else:
            ref_value = max(ref_value, result)
        except TypeError:
          pass

  for i, value in enumerate(results):
    if value is None:
      results[i] = ref_value