# %load BCM.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
from scipy.special import expit
import pdb


# Define Foldiak's local learning rule
# Relu/Sigmoid/tanh activation function
# Update weight per sample
# Enable objective function
# Enable batch learning
# Hasn't applied decay time constant
def get_act(act):
    """Get an activation function and its derivative.
    Args:
        act: name of the activation function (sigmoid, relu, linear).
    Returns:
        fn, dfn: the activation function and its derivative.
    """
    relu = lambda x: np.maximum(0, x)
    relu_deriv = lambda y: np.sign(y) * 0.5 + 0.5
    sigmoid = lambda x: expit(x)
    sigmoid_deriv = lambda y: y * (1 - y)
    linear = lambda x: x
    linear_deriv = lambda y: 1
    tanh = lambda x: np.tanh(x)
    tanh_deriv = lambda y: 1.0 - np.tanh(y) ** 2
    if act == 'relu':
        return relu, relu_deriv
    elif act == 'sigmoid':
        return sigmoid, sigmoid_deriv
    elif act == 'tanh':
        return tanh, tanh_deriv
    return linear, linear_deriv


def foldiak(eta=0.01, act=None, inhi_strength=0.3):
    """A fully connected linear layer with Foldiak's rule."""

    def forward(x, **kwargs):
        """Run forward.
        Args:
            x: input, shape: m * n
            param: 
            w: feedforward weight, shape: n * p
            q: lateral weights, shape: n_output * n_output, lower trangular matrix
        Returns:
            y: output, m * p
        """
        w = kwargs['w']
        q = kwargs['q']
        y1 = x.dot(w)
        y2 = y1.dot(q)
        y = y1 + y2
        # w2 = w.dot(np.linalg.inv(np.eye(q.shape[0]) - q))
        # y = x.dot(w2)
        if act is not None:
            fwd, bak = get_act(act)
            y = fwd(y)
        return y

    def update(x, y, **kwargs):
        """Update weights.
        Args:
            x: input, shape: m * n
            y: output, m * p
            param:
                w: old weights, shape: n * p
        Returns:
            param:
                w: new weights, shape: n * p
        """
        w = kwargs['w']
        q = kwargs['q']
        dw = x.T.dot(y) - w * np.sum(y.T.dot(y) * np.eye(w.shape[1]), keepdims=1, axis=0).reshape(1, -1)
        dq = -1 * np.tril(y.T.dot(y), -1)
        # _y = np.expand_dims(y, 1)
        # y_ = np.expand_dims(y, 2)
        # dq = -1 * eta * (_y * y_).mean(axis=0)

        w = w + eta * dw
        q = q + inhi_strength * eta * dq

        return {'w': w, 'q': q}

    return forward, update
