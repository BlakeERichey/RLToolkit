import gym
from keras.models import load_model
from rltoolkit.agents import ANN, CNN, LSTM_ANN, LSTM_CNN
from rltoolkit.methods import Evo
from rltoolkit.utils import test_network
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop

if __name__ == '__main__':
  #========== Demo  =============================================================
  filename = 'BattleZone'

  #========== Initialize Environment ============================================
  env = gym.make('BattleZone-v0')
  try:
    print(env.unwrapped.get_action_meanings())
  except:
    pass

  #========== Build network =====================================================
  model = LSTM_CNN(env, n_timesteps=4, cnn_topology=[128, 64, 64, 32], fcn_topology=[32, 64], pool_size=4)
  model.summary()
  model.save(f'{filename}.h5')

  #========== Evaluate Results ==================================================
  #Load best saved model
  model = load_model(f'{filename}.h5')

  # Test models results for 5 episodes
  print('Testing...')
  episodes = 5
  avg = test_network(model, env, episodes=episodes, render=True, verbose=1)
  print(f'Average after {episodes} episodes:', avg)

  episodes = 100
  avg = test_network(model, env, episodes=episodes, render=False, verbose=0)
  print(f'Average after {episodes} episodes:', avg)