import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math



mnist = input_data.read_data_sets("MNIST_data/")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

print(type(train_data), type(train_data[0]), type(train_data[0][0]))
print(type(train_labels), type(train_labels[0]))
print(train_labels[0])
