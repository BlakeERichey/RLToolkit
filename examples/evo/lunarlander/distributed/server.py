import gym
import socket
from config import create_model, ENV_NAME, PORT, AUTHKEY, TIMEOUT, GPUS,\
  CORES_PER_NODE, GENERATIONS, POP_SIZE, ELITES, GOAL, EPISODES
from rltoolkit.backend.keras import DistributedBackend

if __name__ == '__main__':
  ip = socket.gethostbyname(socket.gethostname())
  
  backend = DistributedBackend(
    port=PORT, 
    timeout=TIMEOUT,
    server_ip=ip,
    authkey=AUTHKEY,
    network_generator=create_model
  )

  with open('server_ip.log', 'w') as f:
    f.write(ip)
  backend.spawn_server()
  