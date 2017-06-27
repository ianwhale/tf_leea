#
# Tests for the evolver class.
#

import tensorflow as tf
import numpy as np
from test import support
from unittest import TestCase
from evo.evolver import Evolver

tf.set_random_seed(42)

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
            self.complex_comparison(EvolverTest.matrix, EvolverTest.vector, flat)

            unflattened = Evolver.unflatten_tensors(flat, tf.trainable_variables())
            self.complex_comparison(unflattened[0], unflattened[1], flat)

    def test_restore_variables(self):
        graph = tf.Graph()
        with graph.as_default():
            mat = tf.Variable(
            tf.truncated_normal([len(EvolverTest.matrix), len(EvolverTest.matrix[0])])
            )
            vec = tf.Variable(tf.zeros(len(EvolverTest.vector)))

        with tf.Session(graph=graph) as session:
            op = tf.variables_initializer(tf.trainable_variables())
            session.run(op)

            flattened = [EvolverTest.matrix[i][j] for i in range(len(EvolverTest.matrix))
                         for j in range(len(EvolverTest.matrix))] + EvolverTest.vector

            # Initialized to random values, should not be equal to the class tester values.
            self.complex_comparison(mat.eval(), vec.eval(), flattened, inequality=True)

            mat, vec = Evolver.restore_variables([mat, vec], session, EvolverTest.matrix, EvolverTest.vector)

            self.complex_comparison(mat.eval(), vec.eval(), flattened)

    def complex_comparison(self, matrix, vector, flat, inequality=False):
        """
        Comparing big lists against big np.arrays/matrices is rough, so we just loop through them to test for equality.
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
                if inequality:
                    self.assertNotEquals(matrix[i][j], flat[index])
                else:
                    self.assertEquals(matrix[i][j], flat[index])

                index += 1
                j += 1
            j = 0
            i += 1

        for k in range(len(vector)):
            if inequality:
                self.assertNotEquals(vector[k], flat[index])
            else:
                self.assertEquals(vector[k], flat[index])
            index += 1

if __name__ == '__main__':
    support.run_unittest(EvolverTest)