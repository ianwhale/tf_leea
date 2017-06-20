#
# Main entry point.
#
# Code modified from the TensorFlow MOOC.
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Disable annoying warnings.
import random as r
import numpy as np
import tensorflow as tf
from evo.evolver import Evolver, Evaluator
from utils.loaders import load_and_pickle_mnist

SEED = 42 # Should be taken from command line
r.seed(SEED)

image_size = 28
num_labels = 10

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def sgd_experiment():
    """
    Optimize a neural network that classifies MNIST data.
    :return:
    """
    batch_size = 128
    layer_1_hidden_nodes = 80 ## Starting small so my computer can keep up with the ram requirements of LEEA :)

    (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels) = get_mnist()

    graph = tf.Graph()
    with graph.as_default():
        ## Data variables.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        ## Weights describing single layer.
        weights1 = tf.Variable(
            tf.truncated_normal([image_size * image_size, layer_1_hidden_nodes])
        )
        biases1 = tf.Variable(tf.zeros([layer_1_hidden_nodes]))
        weights2 = tf.Variable(
            tf.truncated_normal([layer_1_hidden_nodes, num_labels])
        )
        biases2 = tf.Variable(tf.zeros([num_labels]))

        ## Training variables.
        lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
        logits = tf.matmul(lay1_train, weights2) + biases2
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)
        )

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
        valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
        lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
        test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)

    num_steps = 3001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)

            if (step % 250) == 0:
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels)
                )

        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


def leea_experiment():
    """
    Optimize a neural network with LEEA and the MNIST dataset.
    :return:
    """
    batch_size = 128
    layer_1_hidden_nodes = 80  ## Starting small so my computer can keep up with the ram requirements of LEEA :)

    (train_dataset, train_labels), (valid_dataset, valid_labels), (test_dataset, test_labels) = get_mnist()

    ## Copy pasted, I'm so sorry.
    graph = tf.Graph()
    with graph.as_default():
        ## Data variables.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        ## Weights describing single layer.
        weights1 = tf.Variable(
            tf.truncated_normal([image_size * image_size, layer_1_hidden_nodes])
        )
        biases1 = tf.Variable(tf.zeros([layer_1_hidden_nodes]))
        weights2 = tf.Variable(
            tf.truncated_normal([layer_1_hidden_nodes, num_labels])
        )
        biases2 = tf.Variable(tf.zeros([num_labels]))

        ## Training variables.
        lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
        logits = tf.matmul(lay1_train, weights2) + biases2
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels)
        )

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        train_prediction = tf.nn.softmax(logits)
        lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
        valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
        lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
        test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)

    # op = tf.variables_initializer(tf.trainable_variables())
    # session.run(op)
    # for var in tf.trainable_variables():
    # print(var.eval())
    #
    # evaluator = Evaluator(tf.contrib.losses.softmax_cross_entropy)
    # evolver = Evolver(tf.trainable_variables(), evaluator)

def get_mnist():
    mnist = load_and_pickle_mnist()

    def reformat(data, labels):
        return data.reshape((-1, image_size * image_size)).astype(np.float32), \
               labels.astype(np.float32)

    return reformat(mnist.train.images, mnist.train.labels),  \
           reformat(mnist.validation.images, mnist.validation.labels), \
           reformat(mnist.test.images, mnist.test.labels)

if __name__ == '__main__':
    # Optimize with Stochastic Gradient Descent.
    # sgd_experiment()

    # Optimize with Limited Evaluation Evolutionary Algorithm.
    leea_experiment()