import keras
import random
import numpy as np
from keras.layers import Dense
from collections import namedtuple
from keras.models import clone_model


class Colony:

    def __init__(self, nn):
        self.workers = list()
        self.best_worker = None
        self.nn = clone_model(nn)
        self.weights = self.nn.get_weights()

    def fitness(self, env=None, sharpness=1, validate=False):
        assert len(self.workers), 'Colony must have at least 1 worker'
        
        model = self.nn
        model.set_weights(self.weights)

        Result = namedtuple('result', 'id reward v_reward')
        reward, v_reward = self.workers[0].fitness(model, env, sharpness, validate)
        results = [Result(0, reward, v_reward)]

        if len(self.workers) > 1:
          i = 0
          for worker in self.workers[1:]:
            model.set_weights(self.weights)
            reward, v_reward = worker.fitness(model, env, sharpness, validate)
            results.append(Result(i, reward, v_reward))   
            i+=1
        
        results.sort(key= lambda res: (res.reward, res.v_reward), reverse=True)
        self.best_worker = results[0].id

        return results[0].reward, results[0].v_reward

    def breed(self, colony2):
        #Uncomment print statements to see how this function works
        new_weights = list()
        
        for layer1, layer2 in zip(self.weights, colony2.weights):
            assert layer1.shape == layer2.shape, 'Colonies don\'t have same shape'
            new_weights.append(np.zeros_like(layer1))

        for i, layer1, layer2 in zip(range(len(new_weights)), self.weights, colony2.weights):
            if new_weights[i].ndim == 1:
                # This method is potentially dangerous since I'm not sure if layer can be other then 2 dimensional
                # and bias can be other than 1 dimensional
                # Bias is always set to 0
                continue
            for j, weight1, weight2 in zip(range(len(new_weights[i])), layer1, layer2):
                seeds = random.sample(range(len(new_weights[i][j])), random.choice(range(len(new_weights[i][j]))))
                #print(i, j, weight1, weight2, seeds)
                '''
                i and j = number of iteration,
                weight1 and weight2 = current row looking at,
                seeds = column location of weight1 that will be in new weights

                example:
                if output = 2 0 [0. 0. 0. 0.] [1. 1. 1. 1.] [3, 2].
                first row of second layer matrix of colony weights will be
                [1. 1. 0. 0.]
                '''
                #print()
                for seed in seeds:
                    new_weights[i][j][seed] = weight1[seed]
                for seed in range(len(new_weights[i][j])):
                    if seed not in seeds:
                        new_weights[i][j][seed] = weight2[seed]

            new_weights[i] = np.around(new_weights[i].astype(np.float64), 3)
            #print(new_weights[i])
            #print()

        colony = Colony(self.nn)
        colony.nn.set_weights(new_weights)
        
        return colony

    def mutate(self):
      # self.workers[self.best_worker]._apply_mask(self.nn)
      pass
      # [worker.mutate() for worker in range(self.workers)]

def test_breed():
    '''
    Run this code to see how the breeding function works
    '''

    nn1 = keras.models.Sequential()
    nn1.add(Dense(12, input_dim=3, activation='relu'))
    nn1.add(Dense(4, input_dim=3, kernel_initializer='zero', activation='relu'))

    nn2 = keras.models.Sequential()
    nn2.add(Dense(12, input_dim=3, activation='relu'))
    nn2.add(Dense(4, input_dim=3, kernel_initializer='one', activation='relu'))

    colony1 = Colony(nn1)
    colony2 = Colony(nn2)

    '''
    #Raw weights output
    print(colony1.weights)
    print()
    print(colony2.weights)
    print()
    '''

    print('Colony1: ')
    for i in range(0, len(colony1.weights), 2):
        print('layer:')
        print(colony1.weights[i])
        print('bias:')
        print(colony1.weights[i + 1])
    print()
    print('Colony2: ')
    for i in range(0, len(colony2.weights), 2):
        print('weights:')
        print(colony2.weights[i])
        print('bias:')
        print(colony2.weights[i + 1])
    print()

    new_colony = colony1.breed(colony2)

    '''
    #Raw weights output
    print(new_colony.weights)
    '''

    print('New colony: ')
    for i in range(0, len(new_colony.weights), 2):
        print('layer:')
        print(new_colony.weights[i])
        print('bias:')
        print(new_colony.weights[i + 1])
    print()

def test_time():
    import time as t

    nn1 = keras.models.Sequential()
    nn1.add(Dense(12, input_dim=3, activation='relu'))
    nn1.add(Dense(4, input_dim=3, kernel_initializer='zero', activation='relu'))

    colony1 = Colony(nn1)

    start_time = t.time()
    print(nn1.get_weights())
    print("%s seconds" % (t.time() - start_time))
    print()

    start_time = t.time()
    print(colony1.weights)
    print("%s seconds" % (t.time() - start_time))
    print()

# if __name__ == '__main__':
    # test_breed()
    # test_time()
