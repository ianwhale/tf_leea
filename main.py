#
# Main entry point.
#
# Code modified from the TensorFlow MOOC.
#

import random as r
import numpy as np
import tensorflow as tf
from evo.evolver import Evolver, Evaluator
from utils.loaders import load_and_pickle_mnist

SEED = 42 # Should be taken from command line

r.seed(SEED)

mnist = load_and_pickle_mnist()

train_dataset = mnist.train.images
train_labels = mnist.train.labels

valid_dataset = mnist.validation.images
valid_labels = mnist.validation.labels

test_dataset = mnist.test.images
test_labels = mnist.test.labels

batch_size = 128
image_size = 28
num_labels = 10
leea_batch = 2

graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(SEED)

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 3001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
    op = tf.variables_initializer(tf.trainable_variables())
    session.run(op)
    for var in tf.trainable_variables():
        print(var.eval())

    evaluator = Evaluator()
    evolver = Evolver(tf.trainable_variables())

    # for step in range(num_steps):



    #   offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    #   # Generate a minibatch.
    #   batch_data = train_dataset[offset:(offset + batch_size), :]
    #   batch_labels = train_labels[offset:(offset + batch_size), :]
    #
    #   feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    #   _, l, predictions = session.run(
    #     [optimizer, loss, train_prediction], feed_dict=feed_dict)
    #   if (step % 500 == 0):
    #     print("Minibatch loss at step %d: %f" % (step, l))
    #     print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
    #     print("Validation accuracy: %.1f%%" % accuracy(
    #       valid_prediction.eval(), valid_labels))
    # print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))