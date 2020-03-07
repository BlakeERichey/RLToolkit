import numpy as np
import random
import keras
from keras.layers import Dense


class Colony:

    def __init__(self, nn):
        self.nn = nn
        self.weights = nn.get_weights()
        self.workers = list()

    def fitness(self, env, sharpness=1, validate=False):
        assert len(self.workers) > 1, 'Colony has to have at least 2 workers'
        
        model = keras.models.clone_model(self.nn)
        model = model.set_weights(self.weights)

        result, v_result = self.workers[0].fitness(env, model, sharpness, validate)
        best_worker_result = result
        best_worker_v_result = v_result

        for worker in self.workers[1:]:
            result, v_result = worker.fitness(env, model, sharpness, validate)

            if result > best_worker_result:
                best_worker_result = result
                best_worker_v_result = v_result       
        
        return best_worker_result, best_worker_v_result

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

        model = keras.models.clone_model(self.nn)
        model.set_weights(new_weights)
        
        return Colony(model)

    def mutate(self):
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

if __name__ == '__main__':
    #test_breed()
    test_time()