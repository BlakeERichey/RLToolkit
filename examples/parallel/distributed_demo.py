#Define global context, imports, and functions here
import time
import random
import datetime
from rltoolkit.backend import DistributedBackend

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
    server_info = ('127.0.0.1', 50000, b'password') #(ip, port, authkey)

    args = sys.argv[1:]
    for i, arg in enumerate(args):
      if arg == '--server':
        print('Launching Server.')
        DistributedBackend(*server_info).spawn_server()

      elif arg == '--client':
        cores = int(args[i+1])
        print('Spawning client using', cores, 'cores.')
        DistributedBackend(*server_info).spawn_client(cores)

      elif arg == '--driver':
        print('Running driver code')
        backend = DistributedBackend(*server_info)

        #Make N large numbers to factor
        N = 50
        numbers_to_factor = []
        for i in range(N):
          numbers_to_factor.append(99999999+random.randint(5_000_000, 100_000_000))
        print('Numbers to Factor:', numbers_to_factor)

        start_time = datetime.datetime.now()
        #Queue up tasks
        for task_id in range(N):
          backend.run(task_id, get_factors, numbers_to_factor[task_id])

        #Wait for N results to come in
        results = backend.get_results(N)
        #Sort them since they likely came out of order
        results.sort(key= lambda x: x['pid'])

        end_time = datetime.datetime.now()
        total = (end_time - start_time).total_seconds()
        print(results)
        print('Time taken (Distributed):', total, 'sec') #~84 secs
  
  main()