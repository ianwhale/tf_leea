#
# Main evolution class.
#
# Adapted from original LEEA C# library.
#

#TODO: Consider parallelization in evaluation, initialization, and reproduction.

import copy
import numpy as np
import random as r
from sys import maxsize as MAXSIZE
from utils.roulette import RouletteWheel
from .params import Params
from .genome import Genome

INT_MIN = -MAXSIZE - 1

class Evolver:
    """
    """

    def __init__(self, trainable_variables, evaluator):
        """
        :param trainable_variables: TensorFlow's trainable variables collection.
        """

        #
        # Evolutionary Parameters.
        #
        self.rate_decay = (1 - Params.MUTATION_RATE_DECAY) ** (1.0 / Params.MAX_GENERATIONS)
        self.power_decay = (1 - Params.MUTATION_POWER_DECAY) ** (1.0 / Params.MAX_GENERATIONS)
        self.population_size = Params.POPULATION_SIZE
        self.current_generation = 0
        self.vars = trainable_variables ## Need to remember to rebuild the tensors.
        self.flattened = self.flatten_tensors()
        self.population = []

        self.evaluator = evaluator
        self.createPopulation()
        self.evaluatePopulation()

    def flatten_tensors(self):
        """
        Turn all trainable variables into a real valued vector.
        :return np.array:
        """
        to_flatten = tuple(var.eval().flatten() for var in self.vars)
        return np.concatenate(to_flatten)

    def unflatten(self, vector):
        """
        Unflatten a vector into a list of np.array's with the appropriate shapes.
        :param vector:
        :return:
        """
        unflattened = []
        initial = 0
        for var in self.vars:
            slice = var.get_shape()[0] * var.get_shape[1] ## How many entries we need to slice from genome.

            unflattened.append(vector[initial : initial + slice].reshape(var.get_shape()[0], var.get_shape()[1]))
            initial += slice

        return unflattened


    def createPopulation(self):
        """
        Create the initial random population of genomes.
        """
        ## TODO: Parallelize me?
        for _ in range(self.population_size):
            self.population.append(Genome(copy.deepcopy(self.flattened), Params.INITIAL_WEIGHTS_DELTA))

    def evaluatePopulation(self):
        pass

    def run(self):
        while self.current_generation <= Params.MAX_GENERATIONS:
            self.current_generation += 1

            if (self.current_generation % Params.TRACKING_STRIDE) == 0:
                self.updateStats()

            self.produceOffspring()
            self.evaluatePopulation()

    def produceOffspring(self):
        sorted(self.population, key=lambda genome: genome.fitness)

        num_selected = int(Params.POPULATION_SIZE * Params.SELECTION_PROPORTION)

        probabilities = []
        for i in range(num_selected):
            probabilities.append(self.population[i])

        wheel = RouletteWheel(probabilities)

        ## Pairs of genomes to mate.
        matings = [[INT_MIN, INT_MIN] for _ in range(Params.POPULATION_SIZE)]
        for mating in matings:
            index = wheel.throw()

            if r.random < Params.SEX_PROPORTION:
                mating[0] = index

                parent2 = index
                while parent2 == index:
                    parent2 = wheel.throw()

                mating[1] = parent2
            else:
                mating[0] = index

        new_generation = []
        ## TODO: Paralellize me?
        for mating in matings:
            if mating[1] > INT_MIN:
                ## Sexual reproduction
                child = self.population[mating[0]].reproduce(self.population[mating[1]])
                child.fitness = (self.population[mating[0]].fitness +
                                 self.population[mating[1]].fitness) / 2

            else:
                ## Asexual reproduction
                child = self.population[mating[0]].repdroduce()
                child.fitness = self.population[mating[0]].fitness

            new_generation.append(child)

        ## Prompt some garbage collection from the previous population.
        for genome in self.population:
            del genome.weights

        del self.population

        self.population = new_generation

    def updateStats(self):
        pass

    @staticmethod
    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])

class Evaluator:
    """
    """
    def __init__(self, ):
        """
        Needs outside support from TensorFlow to get evaluation metrics.
        """
        pass