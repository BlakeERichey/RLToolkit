import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  import time
  import gym
  import datetime
  import rltoolkit
  from rltoolkit.agents import LSTM_CNN
  from rltoolkit.backend import LocalClusterDispatcher
  from rltoolkit.backend.keras import LocalClusterBackend, set_gpu_session

set_gpu_session()
def create_model():
  env = gym.make('BattleZone-v0')

  model = LSTM_CNN(
    env,
    cnn_topology=[64,128,256,512],
    fcn_topology=[256,128,64]
  )
  return model

def test_multi_gpu():
  backend = LocalClusterBackend(3, gpus=2, processes_per_gpu=None, network_generator=create_model)

  env = gym.make(f'BattleZone-v0')
  model = create_model()
  weights = model.get_weights()
  hashes = []
  print('Testing MultiGPU...')
  for i in range(3):
    task_id = backend.test_network(weights, env, 1, 1, timeout=2)
    hashes.append(task_id)
  
  res = backend.get_results(hashes, values_only=False)
  backend.shutdown()

if __name__ == '__main__':
  test_multi_gpu()