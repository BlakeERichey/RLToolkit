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
from   .utils             import get_model_gpu_allocation, backend_test_network
from   rltoolkit.utils    import test_network, silence_function
from   rltoolkit.wrappers import subprocess_wrapper
from   rltoolkit.backend  import BaseDispatcher, MulticoreDispatcher, \
  LocalClusterDispatcher, DistributedDispatcher

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
    
    assert type(network_generator) == types.FunctionType, \
      'Expected function for network generator.'
    self.network_generator = network_generator

    if gpus and processes_per_gpu is None:
      processes_per_gpu = self._get_max_gpu_processes()
    
    self.gpus = gpus
    self.processes_per_gpu = processes_per_gpu #Max allowable processes if gpus>0
  
    super().__init__(server_ip, port, authkey, timeout)

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
    mem_usage = math.ceil(mem_usage / .05) * .05 #Round up to nearest 5%
    dispatcher.shutdown()
    return mem_usage

  def spawn_client(self, cores=1):
    """
      Uses the active thread to connect to the remote server.
      Sets the client to monitor the connected server for tasks. When tasks are 
      available, client will request the necessary functions and data to 
      complete, and then submit the computed result to the server.

      # Arguments
      cores: Int. How many cores to utilize in addition to the active thread.
    """

    if self.gpus:
      if self.processes_per_gpu < cores:
        msg = 'Requested core count exceeds maximum gpu allowance.\n' + \
          'Setting to core limit: ' + str(self.processes_per_gpu)
        logging.warning(msg)
        cores = self.processes_per_gpu

    super().spawn_client(cores)
           
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
    """
    print('Initializing LocalClusterBackend Backend.')
    silence_function(1, super().__init__, *args, **kwargs)
    self.manager.start()

    if self.gpus:
      #To enter this loop, processes_per_core must be an int
      if self.processes_per_gpu < cores:
        msg = 'Requested core count exceeds maximum gpu allowance.\n' + \
          'Setting to core limit: ' + str(self.processes_per_gpu)
        logging.warning(msg)
        cores = self.processes_per_gpu

    self.dispatcher = MulticoreDispatcher(cores+1) #Negligible monitoring core
    self.dispatcher.run(
      LocalClusterBackend._monitor_active_tasks, *self.manager_creds
    )

    for _ in range(cores):
      self.dispatcher.run(
        silence_function, 1, 
        LocalClusterBackend._spawn_client_wrapper,
        *self.manager_creds
      )

  def shutdown(self,):
    """
      Terminates open tasks.
    """
    logging.debug('Shutting down cluster.')
    self.manager.shutdown()
    silence_function(1, self.dispatcher.shutdown)
    print('Cluster shutdown.')