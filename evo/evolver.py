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
        :param evaluator: Evaluator instance with the accuracy metric from TensorFlow.
        """
        #
        # Evolutionary Parameters.
        #
        self.mutation_power = Params.MUTATION_POWER
        self.mutation_rate = Params.MUTATION_RATE
        self.rate_decay = (1 - Params.MUTATION_RATE_DECAY) ** (1.0 / Params.MAX_GENERATIONS)
        self.power_decay = (1 - Params.MUTATION_POWER_DECAY) ** (1.0 / Params.MAX_GENERATIONS)
        self.population_size = Params.POPULATION_SIZE
        self.current_generation = 0
        self.variables = trainable_variables ## Need to remember to rebuild the tensors.
        self.flattened = Evolver.flatten_tensors(trainable_variables)
        self.population = []

        self.evaluator = evaluator
        evaluator.evolver = self
        self.createPopulation()
        self.evaluatePopulation()

    @staticmethod
    def flatten_tensors(variables):
        """
        Turn all trainable variables into a real valued vector.
        :param variables: flatten me
        :return np.array:
        """
        to_flatten = tuple(var.eval().flatten() for var in variables)
        return np.concatenate(to_flatten)

    @staticmethod
    def unflatten_tensors(vector, variables):
        """
        Unflatten a vector into a list of np.array's with the appropriate shapes.
        :param vector:
        :param variables: unflatten me
        :return:
        """
        unflattened = []
        initial = 0
        for var in variables:
            shape_list = var.get_shape().as_list()

            slice = 1
            for i in range(len(shape_list)):
                slice *= shape_list[i] ## How many entries to slice from the genome.

            new_shape = tuple(shape_list[i] for i in range(len(shape_list)))

            unflattened.append(vector[initial : initial + slice].reshape(new_shape))
            initial += slice

        return unflattened

    @staticmethod
    def restore_variables(variables, session, *matrices):
        """
        Put the provided matrices into the variables.
        :param variables: list of tf.Variables.
        :param session: the tf.session
        :param matrices: list of np.arrays (can be vectors or matrices), likely unflattened in Evolver.unflatten_tensors
        :return list: variables with whatever values were in the matrices.
        """
        for i in range(len(variables)):
            op = variables[i].assign(matrices[i])
            session.run(op)

        return variables



    def createPopulation(self):
        """
        Create the initial random population of genomes.
        """
        ## TODO: Parallelize me?
        for _ in range(self.population_size):
            self.population.append(Genome(copy.deepcopy(self.flattened), Params.INITIAL_WEIGHTS_DELTA))

    def evaluatePopulation(self):
        ## TODO: Parellelize me!!

        pass

    def run(self):
        """
        Execute the evolution loop.
        :return:
        """
        while self.current_generation <= Params.MAX_GENERATIONS:
            self.current_generation += 1

            if (self.current_generation % Params.TRACKING_STRIDE) == 0:
                self.updateStats()

            self.produceOffspring()
            self.evaluatePopulation()

            self.mutation_power *= self.power_decay
            self.mutation_rate *= self.rate_decay

    def produceOffspring(self):
        """
        Standard genetic algorithm things. Pick the top performers and generate offspring with roulette selection.
        """
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

            ## Determine asexual or sexual reproduction.
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
                child = self.population[mating[0]].reproduce(self.mutation_power,
                                                             self.mutation_rate,
                                                             self.population[mating[1]])

                ## Fitness inheritance scheme.
                child.fitness = (self.population[mating[0]].fitness +
                                 self.population[mating[1]].fitness) / 2

            else:
                ## Asexual reproduction
                child = self.population[mating[0]].repdroduce(self.mutation_power,
                                                              self.mutation_rate)
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
    def __init__(self, loss):
        """
        Needs outside support from TensorFlow to get evaluation metrics.
        :param loss: measure of network accuracy
        """
        self.evolver = None ## Needs to be set!
        self.loss = loss ## The loss function from TensorFlow.
        pass

    def checkIfSetup(self):
        if not self.evolver:
            raise AttributeError("Need to set evolver before calling this function!")

    def evaluate(self, batch):
        self.checkIfSetup()

        for individual in self.evolver.population:
            pass
