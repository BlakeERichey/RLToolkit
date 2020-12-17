import gym
import pickle
from rltoolkit.backend import Packet
from rltoolkit.agents import LSTM_CNN
from multiprocessing import Process, Pipe

class Example:
  def __init__(self, name):
    self.example = {name: set()}
    self.starter = 0
  
  def get_data(self,):
    yield self.starter+1

def send_big_data(conn):
    # data = lambda x: x
    # data = b"a" * (1 << 30) # 1GB
    env = gym.make('BattleZone-v0')
    model = LSTM_CNN(env, 5, [128,256, 512], fcn_topology=[32,32,32]).get_weights()
    data = pickle.dumps(model)
    print('Original Size:', len(data))
    packet = Packet(model)
    print('Compressing...')
    packet.compress()
    print('Compressed:', len(packet.data))
    print('Times compressed', packet.times_compressed, 'Method:', packet.serialize_method)
    conn.send(packet)
    conn.close() # eof

if __name__ == '__main__':
  parent_conn, child_conn = Pipe()
  child = Process(target=send_big_data, args=(child_conn,))
  child.start()
  child_conn.close() # child must be the only with it opened
  packet = parent_conn.recv()
  print('Receiver: Compressed:', len(packet.data))
  print('Receiver: Times compressed', packet.times_compressed, 'Method:', packet.serialize_method)
  packet.decompress()
  print('Receiver: Times compressed', packet.times_compressed, 'Method:', packet.serialize_method)
  print('Receiver: Len:', len(packet.data))
  # lossless = data==packet.unpack()
  # print('Lossless decompression:', lossless)
  # if not lossless:
  #   print(data)
  #   print(packet.data)

