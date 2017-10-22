""" Auto Encoder for image reconstruction

Build a 1 hidden layer auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from local_lr import data_loader

# Import MNIST data

mnist = data_loader.load_mnist(if_demean=True, if_one_hot=True)
X_train = mnist['X_train']

# Training parameters

learning_rate = 0.01
epochs = 30000
batch_size = 256

# Network parameters
n_input = 784
n_hidden1 = 40
n_hidden2 = 40
n_output = 784

# tf Graph input



