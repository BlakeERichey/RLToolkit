import gym
import socket
from mutual import create_model, ENV_NAME, PORT, AUTHKEY, CORES_PER_NODE
from rltoolkit.backend import DistributedBackend

if __name__ == '__main__':
  ip = socket.gethostbyname(socket.gethostname())
  with open('server_ip.log', 'r') as f:
    ip = f.readline()
  backend = DistributedBackend(ip, PORT, authkey=AUTHKEY)
  backend.spawn_client(CORES_PER_NODE)