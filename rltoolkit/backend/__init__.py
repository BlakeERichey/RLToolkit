from .utils import Packet
from .base import BaseDispatcher
from .managers import ParallelManager
from .dispatchers import DistributedDispatcher, MulticoreDispatcher, \
  LocalClusterDispatcher

from multiprocessing import set_start_method

try:
  set_start_method('spawn')
except:
  pass