import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  import socket
  from config import create_model, ENV_NAME, PORT, AUTHKEY, TIMEOUT, GPUS,\
    CORES_PER_NODE, GENERATIONS, POP_SIZE, ELITES, GOAL, EPISODES
  from rltoolkit.backend.keras import DistributedBackend, set_gpu_session

if GPUS:
  set_gpu_session()

if __name__ == '__main__':
  ip = socket.gethostbyname(socket.gethostname())
  with open('server_ip.log', 'r') as f:
    ip = f.readline()
  backend = DistributedBackend(
    port=PORT, 
    timeout=TIMEOUT,
    server_ip=ip,
    gpus=GPUS,
    authkey=AUTHKEY,
    network_generator=create_model
  )
  backend.spawn_client(CORES_PER_NODE)