# %load BCM.py
import numpy as np
import tensorflow as tf

# Write QBCM objective function in tensorflow
def QBCM_gradient(xi, w, thres, act=None, decay = 2, p=2):
    """Update weights.
        Args:
            x: input, shape: m * n_output
            y: output, shape: m * n_output
            param:
                w: old weights: n_input * n_output
                thres: old threshold: 1 * n_output
        Returns:
            param:
                w: new weights: n_input * n_output
                thres: new threshold, 1 * n_output
        """
    y = tf.matmul(xi,w)
    
    if act is 'relu':
        y = tf.nn.relu(y)

    y_thres_diff = y - thres
    y_thres_mult = tf.multiply(y, y_thres_diff)
    gradient = tf.matmul(tf.transpose(xi),y_thres_mult) - decay * w
    thres = ema(x=thres, y=y, power=p)
    
    return gradient, thres

def ema(x, y, tau=100, power=2):
    # x is the iterative variable, y is the function being averaged
    h = tf.cast(tf.exp(-1 / tau), tf.float32)
    return tf.cast(x * h + tf.pow(y, power) * (1 - h), tf.float32)

def QBCM_obj(X, w):
    # Calculate objective function using all data
    c = tf.matmul(X, w)
    c = tf.nn.relu(c)

    obj1 = tf.reduce_mean(tf.pow(c, 3), axis=0)
    obj2 = tf.reduce_mean(tf.pow(c, 2), axis=0)
    obj = tf.divide(tf.pow(obj2, 2), 4) - tf.divide(obj1, 3) 

    return obj

