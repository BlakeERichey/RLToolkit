#Define global context, imports, and functions here
import time
import random
import datetime
from rltoolkit.backend import MulticoreBackend

def get_factors(number):
  factors = []

  for i in range(1, int((number+1)//2)):
    if(number % i == 0):
      factors.append(i)
  factors.append(number)
  
  return factors


if __name__ == '__main__':
  #These are only run on the central process, not subprocesses.
  import cProfile, pstats, io, sys
  def profile(fnc):
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      retval = fnc(*args, **kwargs)
      pr.disable()
      s = io.StringIO()
      sortby = 'cumulative'
      ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
      ps.print_stats(0.05)
      print(s.getvalue())
      return retval

    return inner
  
  @profile
  def main():
    backend = MulticoreBackend(cores=10)

    N = 50
    numbers_to_factor = []
    for i in range(N):
      numbers_to_factor.append(99999999+random.randint(5_000_000, 100_000_000))
    print('Numbers to Factor:', numbers_to_factor)

    start_time = datetime.datetime.now()
    for task_id in range(N):
      backend.run(task_id, get_factors, numbers_to_factor[task_id])

    results = backend.join()

    end_time = datetime.datetime.now()
    total = (end_time - start_time).total_seconds()
    print(results)
    print('Time taken (Multicore):', total, 'sec') #~84 secs
  
  main()