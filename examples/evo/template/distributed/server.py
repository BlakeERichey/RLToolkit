import gym
import socket
from mutual import create_model, ENV_NAME, PORT, AUTHKEY
from rltoolkit.backend import DistributedBackend

if __name__ == '__main__':
  ip = socket.gethostbyname(socket.gethostname())
  backend = DistributedBackend(ip, PORT, authkey=AUTHKEY)
  with open('server_ip.log', 'w') as f:
    f.write(ip)
  backend.spawn_server()
  