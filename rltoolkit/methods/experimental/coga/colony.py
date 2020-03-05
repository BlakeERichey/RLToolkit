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
        if self.worker == []:
            print('Colony has no worker.')
            return None
        
        model = keras.models.clone_model(self.nn)
        results = list()

        best_worker = self.workers[0]
        result, v_result = worker.fitness(env, self.model, sharpness, validate)
        best_worker_result = result
        best_worker_v_result = v_result

        if validate:
            for worker in self.workers[1:]:
                result, v_result = worker.fitness(env, self.model, sharpness, validate)

                if result > best_worker_result and v_result > best_worker_v_result:
                    best_worker = worker
                    best_worker_result = result
                    best_worker_v_result = v_result       
        else:
            for worker in self.workers[1:]:
                result, v_result = worker.fitness(env, self.model, sharpness, validate)

                if result > best_worker_result:
                    best_worker = worker
                    best_worker_result = result
        
        return best_worker.fitness()

    def breed(self, colony2):
        weights1 = self.weights[0]
        weights2 = colony2.weights[0]

        if weights1.shape != weight2.shape:
            print('Not same shape.')
            return None

        new_weights = np.zeros_like(weights1)
        
        for i, weight1, weight2 in zip(range(len(new_weights)), weights1, weights2):
            seeds = random.sample(range(len(new_weights[i])), random.choice(range(len(new_weights[i]))))
            # print(i, weight1, weight2, seeds)
        for seed in seeds:
            new_weights[i][seed] = weight1[seed]
        for seed in range(len(new_weights[i])):
            if seed not in seeds:
                new_weights[i][seed] = weight2[seed]   

        new_colony = Colony(self.nn)
        new_colony.weights = self.weights
        new_colony.workers = self.workers
        new_colony.weights[0] = new_weights

        return new_colony

    def mutate(self):
        pass
        # [worker.mutate() for worker in range(self.workers)]
        

if __name__ == '__main__':
    nn = keras.models.Sequential()
    nn.get_weights()
    
    model1 = keras.models.Sequential()
    model1.add(Dense(12, input_dim=3, activation='relu'))

    print(model1.get_weights())
    print()
    print(len(model1.get_weights()))
    print()
