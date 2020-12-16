import warnings
with warnings.catch_warnings():
  warnings.simplefilter('ignore')
  import gym
  import rltoolkit
  from keras.models import load_model
  from rltoolkit.utils import test_network
  from config import create_model, ENV_NAME, FILENAME

#========== Evaluate Results ==================================================
#Load best saved model
env = gym.make(ENV_NAME)
model = load_model(f'{FILENAME}.h5')
model.summary()

# Test models results for 5 episodes
episodes = 5
avg = test_network(model, env, episodes=episodes, render=False, verbose=1)
print(f'Average after {episodes} episodes:', avg)

print('Testing 100 times!')
episodes = 100
avg = test_network(model, env, episodes=episodes, render=False, verbose=0)
print(f'Average after {episodes} episodes:', avg)