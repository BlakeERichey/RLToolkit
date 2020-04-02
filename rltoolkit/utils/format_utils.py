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