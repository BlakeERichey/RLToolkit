import time
import random
from rltoolkit.backend import TaskScheduler

def example(i, qty, name=None):
  for idx in range(qty):
    time.sleep(random.randint(1, 5))
    print(f'Process {i}: {idx}')
  return name

# =========== DEMO ============================================
if __name__ == '__main__':
  backend = TaskScheduler(10) #start task manager with 3 cores available
  
  #Start 3 tasks, queue 7
  for i in range(10):
    backend.run(example, i, 10, name=f'{i}process')
  
  #wait for all 10 tasks to finish
  res = backend.join()

  #print results
  print('Results', res)