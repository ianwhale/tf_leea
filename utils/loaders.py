#
# Loader utilities for saving data.
#

import os
import glob
from six.moves import cPickle as pickle
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

me = os.path.dirname(__file__)
data_dir = os.path.join(me, '..', 'data')
pickle_file = os.path.join(data_dir, 'mnist.pkl')

def load_and_pickle_mnist():
    """
    If needed, download mnist dataset.
    If not, simply load it from the pickle file.
    :return mnist: DataSet object
    """

    if os.path.exists(pickle_file):
        print("Pickle file found! Unpickling...")
        with open(pickle_file, "rb") as pf:
            mnist = pickle.load(pf)
    else:
        mnist = read_data_sets(data_dir, one_hot=True)

        with open(pickle_file, "wb") as pf:
            pickle.dump(mnist, pf, pickle.HIGHEST_PROTOCOL)

        # Remove .gz files from the mnist download.
        for ptr in glob.glob(os.path.join(data_dir, "*.gz")):
            os.remove(ptr)

    return mnist