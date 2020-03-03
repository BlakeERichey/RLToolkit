import keras


class Colony:

    def __init__(self, nn):
        self.nn = nn
        self.weights = nn.get_weights()
        self.workers = list()

    def fitness(self):
        model = keras.models.clone_model(self.nn)
        best_worker = None
        for worker in self.workers:
            model.set_weights(model.get_weights())
            # set original weights
            # collect results
            # validation results
            best_worker = worker

        #return best_worker.fitness()
        raise Exception('StillWorkingOnThis')

    def breed(self, colony2):
        # Take in two colonies and splice them together
        
        new_colony = None
        
        #return new_colony
        raise Exception('StillWorkingOnThis')

    def set_workers(self, worker):
        self.workers.append(worker)

    def mutate(self):
        pass

if __name__ == '__main__':
    nn = keras.models.Sequential()
    colony = Colony(nn)
    colony.fitness()

