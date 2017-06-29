#
# Tests for the roulette class.
#

from unittest import TestCase
from test import support
from utils.roulette import RouletteWheel
from collections import defaultdict

class TestRoulette(TestCase):
    distribution = [60, 30, 10]
    neg_distribution = [-60, -30, -10]

    def test_throw(self):
        """
        Test the throw functionality.
        :return:
        """
        wheel = RouletteWheel(TestRoulette.distribution)
        counts = defaultdict(int)

        for _ in range(1000):
            counts[wheel.throw()] += 1

        ## Higher values in the distribution should get more throws.
        for i in range(len(TestRoulette.distribution) - 1):
            self.assertTrue(counts[i] > counts[i + 1])

        wheel = RouletteWheel(TestRoulette.neg_distribution)
        counts = defaultdict(int)

        for _ in range(1000):
            counts[wheel.throw()] += 1

        print(counts)

        ## Same as before...
        for i in range(len(TestRoulette.distribution) - 1):
            self.assertTrue(counts[i] > counts[i + 1])

if __name__ == '__main__':
    support.run_unittest(TestRoulette)

