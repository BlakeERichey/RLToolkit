import gym
import keras
import rltoolkit

from keras.models import load_model
from rltoolkit.utils import test_network

fn = 'KungFuMaster'
env = gym.make(fn+'-v0')
model = load_model(fn+'.h5')
print(test_network(model, env, episodes=100, render=False, verbose=True))