#
# Utility class for representing a genome.
#
# Adapted from original LEEA C# library.
#

import copy
import random as r

class Genome:
    """
    Genomes in this system are a vector of real values.
    """

    def __init__(self, weights, delta=1):
        """
        """
        self.fitness = 0        ## Fitness, maximum of 1.
        self.alt_fitness = 0    ## Alternate fitness.
        self.weights = weights  ## The flattened weights.

        self.initialize(delta)

    def initialize(self, delta):
        """
        Initialized the genome to have a random "weight" vector.
        :param delta:
        :return:
        """
        for i in range(len(self.weights)):
            self.weights[i] = r.random() * delta * 2 - delta

    def reproduce(self, mutation_power, mutation_rate, parent2=None):
        """
        Create a new genome.
        :param mutation_power: how much to perturb a value by
        :param mutation_rate: how often to mutate
        :param parent2:
        :return: Genome
        """
        return self.sexual(parent2) if parent2 else self.asexual(mutation_power, mutation_rate)

    def asexual(self, mutation_power, mutation_rate):
        """
        Asexual reproduction with mutation only.

        :return:
        """
        child = copy.deepcopy(self)

        for i in range(len(child.weights)):
            if r.random() < mutation_rate:
                child.weights[i] += float(r.random() * mutation_power * 2 - mutation_power)

        return child

    def sexual(self, parent2):
        """
        Sexual reproduction with n-point crossover.

        :param parent2: another Genome
        :return: Genome
        """
        child = copy.deepcopy(self)

        ## TODO: consider this function and its inability to preserve building blocks.

        for i in range(len(child.weights)):
            if r.random() < 0.5:
                child.weights[i] = parent2.weights[i]

        return child