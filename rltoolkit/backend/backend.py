import os
import queue
import socket
import logging
import multiprocessing
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

  def __init__(self, server_ip='127.0.0.1', port=50000, authkey=b'rltoolkit',
              use_gpu=False, require_gpu=False):
    """
      Initializes a Distributed & Multicore Backend Remote Manager.

      # Arguments
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
    self.port      = port
    self.authkey   = authkey
    self.server_ip = server_ip
    
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
  
  def get_results(self, num_results=1):
    """
      Gets a number of results from the results queue. Defaults to 1 result
      Hangs current thread until this quantity has been retreived.
    """
    manager = self.manager
    manager.connect()
    
    if num_results == 1:
      result = manager.get_results().get()
    else:
      result = []
      for _ in range(num_results):
        res = manager.get_results().get()
        result.append(res)

    return result

class MulticoreBackend():

  def __init__(self, cores=1):
    """
      Initializes a MulticoreBackend with a configurable number of cores
      
      #Arguments
      cores: Int. Max number of cores to utilize.
    """
    
    self.queued    = []    #process that have yet to run
    self.processes = []    #processes actively running
    self.results   = Queue() #results queue
    self.cores = cores

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
    if len(self.processes) < self.cores:
      self.processes.append(p)
      p.start()
    else:
      self.queued.append(p)

  def join(self,):
    """
      Syncronously awaits all subprocesses competion and returns 
      when this condition is met
    """

    results = []

    while len(self.processes) or len(self.queued):
      
      #check status of current processes
      i = 0
      while i < len(self.processes):
        process = self.processes[i]
        
        #terminate process, once done
        if not process.is_alive():
          process.join()
          self.processes.pop(i)
          i-=1 #back up an index, due to active.pop()

          #stage returned values into results
          res = self.results.get()
          results.append(res)

          #Start a queued process
          if len(self.queued) and len(self.processes) < self.cores:
            new_process = self.queued.pop(0)
            self.processes.append(new_process)
            new_process.start()
        
        i+=1
    
    return results