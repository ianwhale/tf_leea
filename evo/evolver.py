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

    def getBest(self):
        """
        Get the best individual.
        :return Genome:
        """
        self.population.sort(key=lambda genome: genome.fitness, reverse=Params.LOW_FITNESS_BETTER)
        return self.population[0]

    def createPopulation(self):
        """
        Create the initial random population of genomes.
        """
        ## TODO: Parallelize me?
        for _ in range(self.population_size):
            self.population.append(Genome(copy.deepcopy(self.flattened), Params.INITIAL_WEIGHTS_DELTA))

    def doGeneration(self, feed_dict):
        """
        Execute the evolution loop.
        :return:
        """
        self.current_generation += 1

        if (self.current_generation % Params.TRACKING_STRIDE) == 0:
            self.updateStats()

        self.evaluator.evaluate(feed_dict)
        self.produceOffspring()

        self.mutation_power *= self.power_decay
        self.mutation_rate *= self.rate_decay

    def produceOffspring(self):
        """
        Standard genetic algorithm things. Pick the top performers and generate offspring with roulette selection.
        """
        self.population.sort(key=lambda genome: genome.fitness, reverse=Params.LOW_FITNESS_BETTER)

        num_selected = int(Params.POPULATION_SIZE * Params.SELECTION_PROPORTION)

        probabilities = []
        for i in range(num_selected):
            probabilities.append(self.population[i].fitness)

        wheel = RouletteWheel(probabilities)

        ## Pairs of genomes to mate.
        matings = [[INT_MIN, INT_MIN] for _ in range(Params.POPULATION_SIZE)]
        for mating in matings:
            index = wheel.throw()

            ## Determine asexual or sexual reproduction.
            if r.random() < Params.SEX_PROPORTION:
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
                child = self.population[mating[0]].sexual(self.population[mating[1]])

                ## Fitness inheritance scheme.
                child.fitness = (self.population[mating[0]].fitness +
                                 self.population[mating[1]].fitness) / 2

            else:
                ## Asexual reproduction
                child = self.population[mating[0]].asexual(self.mutation_power,
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

class Evaluator:
    """
    """
    def __init__(self, session, loss, final_op):
        """
        Needs outside support from TensorFlow to get evaluation metrics.
        :param session: the tf.session
        :param loss: measure of network accuracy
        :param final_op: the final operation, serving as the output of the neural network
        """
        self.evolver = None ## Needs to be set!
        self.session = session
        self.loss = loss
        self.final_op = final_op
        pass

    def checkIfSetup(self):
        if not self.evolver:
            raise AttributeError("Need to set evolver before calling this function!")

    def evaluate(self, feed_dict):
        self.checkIfSetup()

        min_loss = -INT_MIN

        for individual in self.evolver.population:
            variables = self.evolver.restore_variables(self.evolver.variables,
                                                       self.session,
                                                       *self.evolver.unflatten_tensors(individual.weights, self.evolver.variables))

            calc_loss = -1 * self.session.run(self.loss, feed_dict=feed_dict)

            min_loss = calc_loss if calc_loss < min_loss else min_loss
            individual.fitness = calc_loss

        print("Minimum loss this generation: ", -1 * min_loss)