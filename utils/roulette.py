#
# Simple roulette wheel functionality.
#

import random as r

def abs_sum(items):
    val = 0
    for x in items:
        val += abs(x)
    return val

class RouletteWheel:
    """
    Roulette wheel class used for selection.
    """

    def __init__(self, probabilities):
        self.probabilities = [abs(n) for n in probabilities]
        self.total = abs_sum(probabilities)
        self.labels = [i for i in range(len(probabilities))]

    def throw(self):
        """
        Returns an index in the list that was provided in the constructor.
        :return:
        """
        value = self.total * r.random()
        accumulator = 0.0

        for i in range(len(self.probabilities)):
            accumulator += self.probabilities[i]

            if value < accumulator:
                return self.labels[i]

        ## Rounding error occurred if execution makes it here.
        for i in range(len(self.probabilities)):
            if self.probabilities[i] != 0.0:
                return self.labels[i]

        raise AttributeError("Invalid operation, faulty distribution provided.")