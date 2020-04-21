import os
import logging
import multiprocessing
from   multiprocessing import Process, Manager

class TaskScheduler:

  """Manages tasks and assigns them to available cores"""

  def __init__(self, num_cores=1):
    """
      Initializes TaskScheduler with a configurable number of cores
    """
    #disable warnings in subprocess
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.disable(logging.WARNING)
    
    self.pid       = 0     #process id, links queues results with order of process execution
    self.queued    = []    #process that have yet to run
    self.processes = []    #processes actively running
    self.res       = Manager().Queue() #results queue
    self.num_cores = num_cores

  @staticmethod
  def function_wrapper(func, res, pid, *args, **kwargs):
    """
      wraps func as a subprocess by evaluating func(*args, **kwargs) then 
      enqueus the results in `res` that is injected from 
      TashScheduler.run()
    """
    retval = func(*args, **kwargs)

    #store returned value
    res.put({
      'pid': pid,
      'retval': retval,
    }) 

  def run(self, func, *args, **kwargs):
    """
      Spawns a subprocess passing args into func(). 
      Queues a subprocess in the event of no available cores

      # Arguements
      func: a function
      args: value based arugements for function `func`
      kwargs: keyword based arguements for `func`

      # Example
      def example_func(i, name=None):
        print(i, name)

      >>>TaskScheduler.run(example_func, 1, name='test')
      >>>'1 test'
    """

    #add callbacks features and function wrapper
    args = (func,self.res,self.pid) + args
    self.pid += 1

    p = Process(target=TaskScheduler.function_wrapper, args=args, kwargs=kwargs)
    if len(self.processes) < self.num_cores:
      self.processes.append(p)
      p.start()
    else:
      self.queued.append(p)

  def join(self,):
    """
      Syncronously awaits all subprocesses competion and returns 
      when this condition is met
    """
    
    #Join completed processes, initiate queued processes
    while len(self.processes) or len(self.queued):
      for i, process in enumerate(self.processes):
        if not process.is_alive():
          #terminate process, once done
          process.join()
          self.processes.pop(i)

          #if queued processes exist, begin one
          if len(self.queued) and len(self.processes) < self.num_cores:
            new_process = self.queued.pop(0)
            self.processes.append(new_process)
            new_process.start()

    results = []
    while not self.res.empty():
      item = self.res.get()
      results.append(item)

    #sort in event processes finished asychronously
    results = sorted(results, key= lambda val: val['pid'])
    
    return [item['retval'] for item in results] #remove pids
  
