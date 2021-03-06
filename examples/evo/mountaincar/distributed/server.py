import gym
import rltoolkit
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from rltoolkit.agents import ANN
from rltoolkit.utils import test_network
from rltoolkit.methods import Evo
from rltoolkit.callbacks import Checkpoint, Graph, EarlyStop
from rltoolkit.backend import DistributedBackend
from driver import create_model

if __name__ == '__main__':
  backend = DistributedBackend(
    server_ip='127.0.0.1',
    port=50000, 
    authkey=b'rltoolkit',
    timeout=60,
    network_generator=create_model
  )
  backend.spawn_server()