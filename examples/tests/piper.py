import gym
import pickle
from rltoolkit.backend import Packet
from rltoolkit.agents import LSTM_CNN, LSTM_ANN
from rltoolkit.utils import truncate_weights
from multiprocessing import Process, Pipe
from keras.models import clone_model

class Example:
  def __init__(self, name):
    self.example = {name: set()}
    self.starter = 0
  
  def get_data(self,):
    yield self.starter+1

def create_pop(model, size=100):
  pop = []
  weights = model.get_weights()
  pop.append(truncate_weights(weights))
  # pop.append(model)
  for i in range(1, size):
    model = clone_model(model)
    weights = model.get_weights()
    pop.append(truncate_weights(weights))
    # pop.append(model)
  
  return pop
    

def send_big_data(conn):
    # data = lambda x: x
    # data = b"a" * (1 << 30) # 1GB
    env = gym.make('BattleZone-v0')
    model = LSTM_CNN(env, 5, [512,512,512], fcn_topology=[32,32,32])
    # env = gym.make('MountainCar-v0')
    # model = LSTM_ANN(env, n_timesteps=10, topology=[2,64,64,16])
    model.summary()
    print('Creating Population.')
    pop = create_pop(model, size=24)
    print('Sending...')
    for ind in pop:
      data = pickle.dumps(ind)
      print('Original Size:', len(data))
      packet = Packet(ind)
      # print('Compressing...')
      # packet.compress()
      # print('Compressed:', len(packet.data))
      saved = len(data) - len(packet.data)
      print('Bytes saved:', saved, 'Ratio:', saved/len(data))
      print('Times compressed', packet.times_compressed, 'Method:', packet.serialize_method)
      conn.send(packet)
      print('Data buffered into Pipe')
    conn.close() # eof

if __name__ == '__main__':
  parent_conn, child_conn = Pipe()
  child = Process(target=send_big_data, args=(child_conn,))
  child.start()
  child_conn.close() # child must be the only with it opened
  for i in range(24):
    packet = parent_conn.recv()
    print('Receiver: Compressed:', len(packet.data))
    print('Receiver: Times compressed', packet.times_compressed, 'Method:', packet.serialize_method)
    packet.unpack()
    print('Receiver: Times compressed', packet.times_compressed, 'Method:', packet.serialize_method)
    print('Receiver: Len:', len(packet.data))
  print("Finished Receiving.")
