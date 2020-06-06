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

class ParallelManager(SyncManager):  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    results_queue = Manager().Queue()
    processes_queue = Manager().Queue()

    self.register('get_results',   callable=lambda:results_queue)
    self.register('get_processes', callable=lambda:processes_queue)

#========== BACKENDS ===========================================================

class DistributedBackend:

  def __init__(self, network_generator, server_ip='127.0.0.1', port=50000, 
              authkey=b'rltoolkit', use_gpu=False, require_gpu=False):
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

    server = manager.get_server()
    ip = socket.gethostbyname(socket.gethostname())
    print(f'Server started. Port {manager.address[1]}. Local IP: {ip}')
    server.serve_forever()

  def spawn_client(self, cores=1):
    """
      Uses the active thread to connect to the remote server.
      Performs processes out of managers process Queue in parallel.

      # Arguments
      cores: Int. How many cores to utilize in addition to the active thread.
    """
    manager = self.manager
    manager.connect()
    
    results   = manager.get_results()
    processes = manager.get_processes()
    print('Client Running.')
    print('Utilizing', cores, 'cores.')
    print('Current Process:', processes.qsize())
    print('Current Results:', results.qsize())

    active = []
    retvals = Queue()
    while True:
      #Initiate additional processes up to core limit
      if len(active) < cores:
        #If more processes are available
        try:
          params  = processes.get_nowait()
        except queue.Empty as e:
          params = None
        
        if params:
          func    = params.get('func')
          task_id = params.get('task_id')
          args    = params.get('args', ())
          kwargs  = params.get('kwargs', {})

          args = (func,retvals,task_id) + args
          print('Spawning new Process:', task_id)

          p = Process(target=subprocess_wrapper, args=args, kwargs=kwargs)
          active.append(p)
          p.start()
      
      #pop() completed processes, continue checking remaining processes
      i = 0
      while i < len(active):
        process = active[i]
        if not process.is_alive():
          #terminate process, once done
          process.join()
          active.pop(i)
          i-=1 #back up an index, due to active.pop()

          val = retvals.get()
          results.put(val)
        
        i+=1
  
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
    
    processes = manager.get_processes()
    
    params = {
      'func': func,
      'args': args,
      'kwargs': kwargs,
      'task_id': task_id,
    }

    processes.put(params)
  
  def get_results(self, num_results=1, clean=True):
    """
      Gets a number of results from the results queue. Defaults to 1 result
      Hangs current thread until this quantity has been retreived.
    """
    manager = self.manager
    manager.connect()
    
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
          min_score = min(min_score, result)

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
    else:
      for item in results:
        if item['result'] is None:
          item['result'] = min_score
    
    return results