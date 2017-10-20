# %load BCM.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
from scipy.special import expit
from sklearn.preprocessing import normalize

# Define QBCM's local learning rule with foldiak lateral inhibition layer.
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
    tanh_deriv = lambda y: 1.0 - np.tanh(y)**2
    if act == 'relu':
        return relu, relu_deriv
    elif act == 'sigmoid':
        return sigmoid, sigmoid_deriv
    elif act == 'tanh':
        return tanh, tanh_deriv
    return linear, linear_deriv



def qbcm_foldiak(eta=0.001, decay=0.0, p=2, tau=100, act=None, batch_size=1, inhi_strength=0):
    """One layer BCM network with foldiak lateral inhibition
    Args:
        eta: learning rate on weights.
        decay: weight decay constant.
        p: order of the threshold function.
        tau: threshold EMA constant.
        act: activation function.
    Returns:
        forward: run forward in the network.
        update: run update step.
    """

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
        y1 = x.dot(w)
        if inhi_strength is not 0:
            q = kwargs['q']
            y2 = y1.dot(q)
            y = y1+y2
        else:
            y = y1
            
        if act is not None:
            fwd, bak = get_act(act)
            y = fwd(y)
        return y



    def update(x, **kwarg):
        """Update weights.
        Args:
            x: input, shape: m * n_output
            y: output, shape: m * n_output
            param:
                w: old weights: n_input * n_output
                thres: old threshold: 1 * n_output
                q: old lateral inhibition weights: low triangular matrix, n_output * n_output
        Returns:
            param:
                w: new weights: n_input * n_output
                thres: new threshold, 1 * n_output
                q:new lateral inhibition weights: low triangular matrix, n_output * n_output
        """
        w = kwarg['w']
        thres = kwarg['thres']
        q = kwarg['q']
        
        for i in range(x.shape[0]//batch_size):
            xi = x[i*batch_size:(i+1)*batch_size, :]
            y1 = xi.dot(w)
            if inhi_strength is not 0:
                q = kwarg['q']
                y2 = y1.dot(q)
                y = y1+y2
            else:
                y = y1
            if act is not None:
                fwd, bak = get_act(act)
                y = fwd(y)
                dy = bak(y)

            y_thres_diff = y - thres
            if (act == None)|(act == 'relu'):
                y_thres_mult = np.multiply(y, y_thres_diff)
                w = w + eta * xi.T.dot(y_thres_mult) - eta * decay * w
            else:
                y_thres_mult = np.multiply(dy, np.multiply(y, y_thres_diff))
                w = w + eta * xi.T.dot(y_thres_mult) - eta * decay * w

            dq = -1 * np.tril(y.T.dot(y), -1)
            q = q + inhi_strength * eta * dq
            thres = ema(x=thres, y=y, power=p, tau=tau)
                            
        return {'w': w, 'thres': thres, 'q':q}

    return forward, update

def ema(x, y, tau=100, power=2):
    # x is the iterative variable, y is the function being averaged
    h = np.float32(np.exp(-1 / tau))
    return np.float32(x * h + (y ** power) * (1 - h))