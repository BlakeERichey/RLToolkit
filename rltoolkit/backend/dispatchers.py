import time
import math
import socket
import logging
import datetime
import multiprocessing
from   copy                     import deepcopy
from   rltoolkit.utils          import silence_function
from   rltoolkit.wrappers       import subprocess_wrapper
from   rltoolkit.backend        import BaseDispatcher, ParallelManager
from   multiprocessing          import Queue, Process

#========== DISPATCHERS ========================================================

class DistributedDispatcher(BaseDispatcher):

  def __init__(self, server_ip='127.0.0.1', port=50000, authkey=b'rltoolkit', 
              timeout=None):
    """
      Initializes a Distributed & Multicore Backend Remote Manager.

      # Arguments
      server_ip: String. IP address for Remote Server. Client machines must use 
        and be able to see this machine.
      port: Int. Port number to open for clients to interface with the manager.
      authkey: Byte string. Used to authenticate access to the manager.
      timeout: Default time in seconds to permit on ther server for a task. 
        If a task takes longer, the server ceases to await a response.
    """

    self.port      = port
    self.authkey   = authkey
    self.server_ip = server_ip
    self.manager_creds = (server_ip, port, authkey)
  
    # Start a shared manager server and access its queues
    self.manager = ParallelManager(
      address=(server_ip, port), 
      authkey=authkey, 
      timeout=timeout
    )
  
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
    DistributedDispatcher._monitor_active_tasks(*self.manager_creds)
  
  @staticmethod
  def _monitor_active_tasks(server_ip, port, authkey):
    """
      Run on server. Monitors active tasks to ensure they complete within 
      timeout limit. Hangs the active thread.
    """
    logging.debug('Monitoring tasks')
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
        logging.debug('Killing tasks', tasks_to_kill)
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

    if cores > 1:
      dispatcher = MulticoreDispatcher(cores=cores)
      for i in range(cores):
        dispatcher.run(
          DistributedDispatcher._spawn_client_wrapper, 
          *self.manager_creds
        )
      dispatcher.join()
    else:
      DistributedDispatcher._spawn_client_wrapper(*self.manager_creds)

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
      logging.debug('Checking for tasks')
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
  
  def shutdown(self,):
    """
      No memory management possible do to expected operation on a cluster.
    """
    pass

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
      logging.debug('Completed Tasks:', completed_tasks)

      #test if all observed tasks are among the completed
      request_results = task_set.difference(completed_tasks) == set()
    
    results = []
    for task_id in task_ids: #Doing each task sequentially (instead of in bulk) results in 30x speedup
      result = manager.get_results(task_ids=[task_id], values_only=values_only).unpack()
      results.extend(result)
    results = self._clean_results(results, values_only, numeric_only, ref_value)
    return results

class LocalClusterDispatcher(DistributedDispatcher):

  def __init__(self, cores=1, *args, **kwargs):
    """
      Initializes a DistributedDispatcher run off localhost to run tasks 
      concurrently without reloading contexts.
      
      #Arguments
      cores: Int. Max number of cores to utilize.
      timeout: Max time in seconds to permit a Process to run.
      network_generator: function. Function that returns a Keras model. 
        Used for client nodes to interpret network architecture and graph context.
    """
    silence_function(1, super().__init__, *args, **kwargs)
    self.manager.start()

    self.dispatcher = MulticoreDispatcher(cores+1) #Negligible monitoring core
    self.dispatcher.run(
      LocalClusterDispatcher._monitor_active_tasks, *self.manager_creds
    )
    for _ in range(cores):
      self.dispatcher.run(
        silence_function, 1, 
        LocalClusterDispatcher._spawn_client_wrapper,
        *self.manager_creds
      )

  def shutdown(self,):
    """
      Terminates open tasks.
    """
    self.manager.shutdown()
    silence_function(1, self.dispatcher.shutdown)

class MulticoreDispatcher(BaseDispatcher):

  def __init__(self, cores=1, timeout=None):
    """
      Initializes a MulticoreDispatcher with a configurable number of cores
      
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
      
      if hasattr(p, 'close'): #python 3.7+
        p.close()
    self.active = 0
    print('MulticoreDispatcher shutdown.')

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
      >>>MulticoreDispatcher.run(example_func, 1, name='test')
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
  
  def get_results(self,):
    pass

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
            process.terminate()
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
    results = self._clean_results(results, values_only, numeric_only, ref_value)
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
