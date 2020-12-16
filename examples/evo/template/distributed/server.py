import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  import gym
  import socket
  from config import create_model, ENV_NAME, PORT, AUTHKEY, TIMEOUT, GPUS,\
    CORES_PER_NODE, GENERATIONS, POP_SIZE, ELITES, GOAL, EPISODES, FILENAME, \
    CONTINUE
  from rltoolkit.backend.keras import DistributedBackend, set_gpu_session

if GPUS:
  set_gpu_session()

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
  