import os
import sys
from io import StringIO

# This is a small utility for printing readable time strings:
def format_time(seconds):
  h = int(seconds / 3600)
  seconds = seconds - 3600*h 
  m = int(seconds / 60)
  seconds = seconds - 60*m
  s = int(seconds)

  time = ''
  if h:
    time+=f'{h}h'
  if h or m:
    time+=f'{m}m'
  if h or m or s:
    time+=f'{s}s'
  return time

def silence_function(silence_level, func, *args, **kwargs):
  '''
    Replaces stdout temporarily to silence print statements inside a function
    silence_level: one of [1,2,3]. 
      1: Mute STDOUT
      2: Mute STDOUT && STDIN
      3: Mute STDOUT && STDIN && STDERR
  '''
  #mask standard output
  actualstdin  = sys.stdin
  actualstdout = sys.stdout
  actualstderr = sys.stderr

  sys.stdout   = StringIO()
  if silence_level > 1:
    sys.stdin    = StringIO()
  if silence_level > 2:
    sys.stderr   = StringIO()

  try:
    retval = func(*args, **kwargs)
  finally: #set stdout but dont catch error
    sys.stdin  = actualstdin
    sys.stdout = actualstdout
    sys.stderr = actualstderr

  return retval