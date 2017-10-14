# %load BCM.py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
import pdb

# Define BCM class by quadratic local learning rule
# Relu/Sigmoid activation function
# Update weight per sample
# Enable 'QBCM' and 'kurtosis' objective function
# Hasn't applied batch learning
# Enable decay time constant
# Enable matrix calculation on output neurons

class bcm_mat:
    """BCM learning
    Parameter:
    eta: float, learning rate (between 0.0 - 1.0)
    n_epoch: int, passes over the training dataset
    ny: int, number of output neurons
    batchsize: int, percentage of data that are used to update the weight once
    tau: float, time constant for BCM leanring rule
    thres: float, initial BCM threshold
    p: int, power of BCM threshold function
    random_state: int, seed for random number generator
    shuffle: Boolean, whether shuffle whole datasize for each epoch
    nonlinear: String, type of activation function, can be choosen from 'Relu', 'Sigmoid' amd None
    obj_type: type of local learning rule, can be choosen from 'QBCM' and 'kurtosis'
    decay: float, decay time constant, 0.0 means no decay

    Attributes:
    w_: array input dimension * num of output neurons
    w_track: list, trakcing the trajectory of weights, list length is number of weight updats, and each list contains array with input dimension * num of output neurons
    y_thres: list, trakcing the trajectory of output, list length is number of weight updats, and each list contains array with 1 * num of output neurons
    obj: list, trakcing the trajectory of values of certain objective function, list length is number of weight updats, and each list contains array with 1 * num of output neurons
    """

    def __init__(self, eta=0.1, n_epoch=10, ny=1, batch=1, tau=100.0, thres=0, p=2, seed=None, random_state=0,
                 nonlinear='ReLU', obj_type='QBCM', decay=0.0):
        self.eta = eta
        self.n_epoch = n_epoch
        self.ny = ny
        self.batch = batch
        self.tau = tau
        self.thres = np.float32(thres * np.ones(ny))
        self.p = p
        self.nonlinear = nonlinear
        self.obj_type = obj_type
        self.y_thres = []  # Storaged y for studying effect of threshold
        self.decay = decay
        self.random_state = random_state
        self.seed = seed

    def fit(self, X):

        # Weights initialized as normal distribution
        # if self.seed:

        r_gen = np.random.RandomState(self.seed)
        self.w_ = r_gen.randn(X.shape[1], self.ny)  # 2*1
        self.w_track = []
        self.obj = []

        # Use elementwise training 
        threshold = self.thres  # Auxillary veriables for calculating thresholds
        self.thres = []
        obj_x1 = np.zeros(self.ny)  # Two iterative terms in objective function
        obj_x2 = np.zeros(self.ny)  # Two iterative terms in objective function
        bcm_obj = np.zeros(self.ny)
        for _ in range(self.n_epoch):
        # for _ in range(1):
            # X = self._shuffle(X)
            for i, xi in enumerate(X):  # elementwise training for all samples
                y = np.float32(np.zeros(self.ny))
                y = self._activation(np.dot(xi, self.w_))
                y_thres_diff = y - threshold
                y_thres_mult = np.multiply(y, y_thres_diff)
                y_thres_covex = np.multiply(y_thres_mult, 2 * y)
                if self.nonlinear == 'ReLU':
                    self.w_ = self.w_ + self.eta * np.dot(xi[:, None], y_thres_mult[None, :]) - self.eta * self.decay * self.w_
                    threshold = self._ema(x=threshold, y=y, power=self.p)
                elif self.nonlinear == 'Convex':
                    self.w_ = self.w_ + self.eta * np.dot(xi[:, None], y_thres_covex[None, :]) - self.eta * self.decay * self.w_
                    threshold = self._ema(x=threshold, y=y, power=self.p)
                else:
                    print('Wrong nonlinearity')
                # bcm_obj = np.float32(obj(X, w=self.w_, obj_type=self.obj_type, nonlinear=self.nonlinear))
                w_tmp = np.concatenate(self.w_.T, axis=0)  # make 2*2 matrix into 1*4, preparing for weight tracking
                self.y_thres.append(y)
                self.thres.append(threshold.tolist())
                self.w_track.append(w_tmp.tolist())
                # self.obj.append(bcm_obj.tolist())
        return self

    def _shuffle(self, X):
        r_gen = np.random.RandomState(self.random_state)
        r = r_gen.permutation(len(X))
        return X[r]

    # Implementing exponential moving average
    def _ema(self, x, y, power=2):
        # x is the iterative variable, y is the function being averaged
        h = np.float32(np.exp(-1 / self.tau))
        return np.float32(x * h + (y ** power) * (1 - h))

    def _activation(self, y):
        if self.nonlinear == 'Sigmoid':
            y = sigmoid(y)
        elif self.nonlinear == 'ReLU':
            y = (y >= 0) * y
        elif self.nonlinear == 'Convex':
            y0 = (y >= 0)
            y2 = y ** 2
            y = np.multiply(y0, y2)
        else:
            print('Wrong activation function')

        return y


# Implement differentiation of sigmoid function
def dsigmoid(z):
    return sigmoid(z) * (1 - (sigmoid(z)))


# Implement sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# objective function using all data
def obj(X, w, obj_type='QBCM', nonlinear='Sigmoid'):
    c = np.dot(X, w)
    if nonlinear == 'Sigmoid':
        c = sigmoid(c)
    elif nonlinear == 'ReLU':
        c = (c >= 0) * c

    obj = 0

    if obj_type == 'QBCM':
        obj1 = (c ** 3).mean(axis=0)
        obj2 = (c ** 2).mean(axis=0)
        obj = obj1 / 3 - obj2 ** 2 / 4
    # obj = - obj2/2
    elif obj_type == 'kurtosis':
        obj1 = (c ** 4).mean(axis=0)
        obj2 = (c ** 2).mean(axis=0)
        obj = obj1 - obj2 ** 2 * 3
    elif obj_type == 'skewness':
        obj1 = (c ** 3).mean(axis=0)
        obj2 = (c ** 2).mean(axis=0)
        obj = np.divide(obj1, obj2 ** 1.5)
    else:
        print('Wrong objective function')

    return obj
