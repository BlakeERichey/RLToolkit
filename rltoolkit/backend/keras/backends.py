import os
import time
import math
import types
import socket
import GPUtil
import logging
import warnings
import datetime
import numpy as np
from   copy               import deepcopy
from   .utils             import set_gpu_session, backend_test_network, \
  get_model_gpu_allocation, kill_proc_tree
from   rltoolkit.utils    import test_network, silence_function
from   rltoolkit.wrappers import subprocess_wrapper
from   rltoolkit.backend  import ParallelManager, MulticoreDispatcher, \
  LocalClusterDispatcher, DistributedDispatcher
from multiprocessing import Array, Process

#========== BACKENDS ===========================================================

class DistributedBackend(DistributedDispatcher):

  def __init__(self, server_ip='127.0.0.1', port=50000, authkey=b'rltoolkit', 
              timeout=None, network_generator=None, 
              gpus=None, processes_per_gpu=None):
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
      gpus: Specifies how many GPUs the Backend should look for.
      processes_per_gpu: Specifies how many processes should be run in parallel
        on each available GPU. If None, the Backend will auto infer the maximum 
        based on network size from `network_generator`.
    """
    print('Initializing DistributedBackend Backend.')
    assert type(network_generator) == types.FunctionType, \
      'Expected function for network generator.'
    self.network_generator = network_generator

    if gpus:
      num_visibile = len(GPUtil.getGPUs()) #Number of visible GPUs

      if num_visibile<gpus:
        msg = f'Backend requested {gpus} GPUs, but can only see {num_visibile}.\n' + \
          f'Setting requested GPU count to {num_visibile}.'
        logging.warning(msg)
        gpus = num_visibile

      if gpus:
        if processes_per_gpu is None:
          processes_per_gpu = self._get_max_gpu_processes()
    
    self.dispatchers = []
    self.gpus = gpus
    self.processes_per_gpu = processes_per_gpu #Max allowable processes if gpus>0
  
    super().__init__(server_ip, port, authkey, timeout)
  
  def shutdown(self,):
    """
      Terminates all initialized dispatchers and closes all orphaned 
      subprocesses caused by gpu_wrapper.
    """
    if hasattr(self, 'gpu_process_ids'):
      for ppid in self.gpu_process_ids:
        kill_proc_tree(ppid)
    for dispatcher in self.dispatchers:
      silence_function(1, dispatcher.shutdown)
    print('DistributedBackend shutdown.')

  def _get_max_gpu_processes(self):
    """
      returns the maximum number of GPU processes permissible given a 
      `network_generator` by measuring that networks memory utilization
    """
    mem_usage = self._get_gpu_mem_usage()
    print('Mem Usage:', mem_usage)

    num_processes = int(1 / mem_usage)
    return num_processes
  
  def _get_gpu_mem_usage(self):
    """
      returns the GPU memory allocation of `network_generator`'s session
    """
    assert self.network_generator is not None, \
      "Unable to measure network memory utilization without generator function"

    dispatcher = MulticoreDispatcher(1)
    dispatcher.run(get_model_gpu_allocation, self.network_generator)
    mem_usage = dispatcher.join()[0]
    mem_usage = math.ceil(mem_usage / .1) * .1 #Round up to nearest 10%
    dispatcher.shutdown()
    return mem_usage

  def spawn_client(self, cores=1, hang=True):
    """
      Uses the active thread to connect to the remote server.
      Sets the client to monitor the connected server for tasks. When tasks are 
      available, client will request the necessary functions and data to 
      complete, and then submit the computed result to the server.

      # Arguments
      cores: Int. How many cores to utilize in addition to the active thread.
      hang: If `True`, the current process hangs until the spawned client 
        subprocesses are terminated. Otherwise, the dispatcher managing the 
        subprocesses is returned.
    """

    cores = self._get_client_cores(cores)

    #Spawn with GPU wrapper
    if self.gpus is not None and self.gpus>0:
      print('Spawning with GPU wrapper')
      gpu_process_ids = Array('i', self.gpus) #Shared data type between subprocesses
      dispatcher = MulticoreDispatcher(cores=self.gpus)
      
      remaining = cores #Remaining subprocesses to spawn
      for i in range(self.gpus):
        processes_to_spawn = min(remaining, self.processes_per_gpu)
        remaining = max(remaining - self.processes_per_gpu, 0)
        dispatcher.run(
          DistributedBackend._spawn_gpu_client_wrapper, 
          *self.manager_creds, processes_to_spawn, i,
          gpu_process_ids
        )

      self.gpu_process_ids = gpu_process_ids
      print('Added GPU PPIDs', self.gpu_process_ids)
    
    #Spawn with CPU wrapper
    else:
      print('Spawning with CPU wrapper')
      dispatcher = MulticoreDispatcher(cores=cores)
      for i in range(cores):
        dispatcher.run(
          DistributedBackend._spawn_client_wrapper, 
          *self.manager_creds
        )
    
    self.dispatchers.append(dispatcher)
    if hang:
      dispatcher.join() #wont terminate until subprocesses and orphaned subprocess terminate
    return dispatcher
  
  @staticmethod  
  def _spawn_gpu_client_wrapper(server_ip, port, authkey, 
      processes_per_gpu, gpu_id, shared_arr):
    """
      If GPUs are utilized, spawns a subprocess for each GPU up to the 
      `process_per_gpu` limit.

      # Arguments 
      processes_per_gpu: Int. Number of subprocesses to spawn
      gpu_id: Int. Which GPU to spawn all subprocess on
      shared_arr: multiprocessing.Array for reporting process_ids to terminate 
        child processes via `self.shutdown()`
    """
    print('Spawning GPU client')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) #Only this GPU visible

    manager_creds = (server_ip, port, authkey)
    for i in range(processes_per_gpu):
      p = Process(
        target=DistributedBackend._spawn_client_wrapper, args=(manager_creds)
      )
      p.start()
    
    pid = os.getpid()
    print('Process ID:', pid)
    shared_arr[gpu_id] = pid

  def _get_client_cores(self, cores):
    """
      Takes into account number of GPUs and requested number of cores to 
      identify if cores exceeds maximum gpu allowance. If not using GPUs, then 
      simply returns `cores`.

      cores: Int. Requested number of cores to be utilized.
    """

    if self.gpus:
      limit = self.processes_per_gpu * self.gpus
      if limit < cores:
        msg = 'Requested core count exceeds maximum gpu allowance.\n' + \
          'Setting to core limit: ' + str(limit)
        logging.warning(msg)
        cores = limit
    
    return cores
           
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
    return self.run(
      backend_test_network, weights,      #model creation params
      self.network_generator, env, episodes, seed,  #test network params
      timeout=timeout
    )

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
      gpus: Specifies how many GPUs the Backend should look for.
      processes_per_gpu: Specifies how many processes should be run in parallel
        on each available GPU. If None, the Backend will auto infer the maximum 
        based on network size from `network_generator`.
    """
    print('Initializing LocalClusterBackend Backend.')
    silence_function(1, super().__init__, *args, **kwargs)
    self.manager.start()

    silence_function(0, self.spawn_client, cores, hang=False)
    
    dispatcher = MulticoreDispatcher(1) #Negligible monitoring core
    dispatcher.run(
      silence_function, 0,
      LocalClusterBackend._monitor_active_tasks, *self.manager_creds
    )
    self.dispatchers.append(dispatcher)


  def shutdown(self,):
    """
      Terminates open tasks.
    """
    logging.debug('Shutting down cluster.')
    silence_function(1, super().shutdown)
    self.manager.shutdown()
    print('Cluster shutdown.')