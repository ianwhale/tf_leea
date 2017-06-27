#
# Tests for the evolver class.
#

import tensorflow as tf
import numpy as np
from test import support
from unittest import TestCase
from evo.evolver import Evolver


class EvolverTest(TestCase):
    matrix = [[i * i for i in range(j * 10, (j * 10) + 10)] for j in range(10)]
    vector = [i / float(i) for i in range(2, 12)]

    def test_flatten_and_unflatten(self):
        """
        Test the unflatten and flatten methods to make sure order is retained.
        :return:
        """
        graph = tf.Graph()
        with graph.as_default():
            mat = tf.Variable(EvolverTest.matrix)
            vec = tf.Variable(EvolverTest.vector)

        with tf.Session(graph=graph) as session:
            op = tf.variables_initializer(tf.trainable_variables())
            session.run(op)

            flat = Evolver.flatten_tensors(tf.trainable_variables())
            self.complex_equality(EvolverTest.matrix, EvolverTest.vector, flat)

            unflattened = Evolver.unflatten_tensors(flat, tf.trainable_variables())
            self.complex_equality(unflattened[0], unflattened[1], flat)

    def complex_equality(self, matrix, vector, flat):
        """
        Comparing big lists against big np.arrays is rough, so we just loop through them to test for equality.
        :param matrix:
        :param vector:
        :param flat: flattened matrix or vector.
        :return:
        """
        i = 0
        j = 0
        index = 0
        while i < len(matrix):
            while j < len(matrix[0]):
                self.assertEquals(matrix[i][j], flat[index])
                index += 1
                j += 1
            j = 0
            i += 1

        for k in range(len(vector)):
            self.assertEquals(vector[k], flat[index])
            index += 1

if __name__ == '__main__':
    support.run_unittest(EvolverTest)