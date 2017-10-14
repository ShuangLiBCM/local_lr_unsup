__author__ = 'fabee'
import numpy as np
#from theano import tensor as T
#import theano as th
from scipy.optimize import fmin_l_bfgs_b, fmin_bfgs, fmin_cg
from matplotlib.pyplot import fill, cm, Rectangle
import matplotlib.pyplot as plt
from scipy.special import gammaln, betaln
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
from sklearn.cluster import KMeans
import warnings
from sklearn import cross_validation
import pandas as pd
from itertools import product
from scipy import stats

def chooseln(k, N):
    return gammaln(N + 1) - gammaln(N - k + 1) - gammaln(k + 1)


def layer(name):
    if 'L1' in name:
        return 1
    elif 'L23' in name:
        if 'Pyr' in name:
            return 3
        else:
            return 2
    else:
        if 'Pyr' in name:
            return 5
        else:
            return 4


def load_data(transpose=False, collapse=None, remove=None):
    with open('files/matrix09112015.csv', 'r') as fid:
    # with open('files/matrix.csv', 'r') as fid:
        labels = [e.strip() for e in fid.readline().split(',')[1:]]
        K, N = [], []
        for l in fid.readlines():
            K.append([list(map(float, e.strip().split('/')))[0] for e in l.split(',')[1:]])
            N.append([list(map(float, e.strip().split('/')))[1] for e in l.split(',')[1:]])
    layers = [layer(name) for name in labels]
    K = np.asarray(K)
    N = np.asarray(N)

    if collapse is not None:
        for k, v in collapse.items():
            idx = list(sorted([labels.index(e) for e in v]))
            i = idx[0]
            labels[i] = k
            K[i, :] = K[idx, :].sum(axis=0)
            K[:, i] = K[:, idx].sum(axis=1)
            N[i, :] = N[idx, :].sum(axis=0)
            N[:, i] = N[:, idx].sum(axis=1)
            K = np.delete(K, idx[1:], axis=0)
            K = np.delete(K, idx[1:], axis=1)
            N = np.delete(N, idx[1:], axis=0)
            N = np.delete(N, idx[1:], axis=1)
            labels = [labels[i] for i in range(len(labels)) if i not in idx[1:]]
            layers = [layers[i] for i in range(len(layers)) if i not in idx[1:]]
    if remove is not None:
        idx = list(sorted([labels.index(e) for e in remove]))
        K = np.delete(K, idx, axis=0)
        K = np.delete(K, idx, axis=1)
        N = np.delete(N, idx, axis=0)
        N = np.delete(N, idx, axis=1)
        labels = [labels[i] for i in range(len(labels)) if i not in idx]
        layers = [layers[i] for i in range(len(layers)) if i not in idx]

    if transpose:
        return labels, layers, K.T, N.T
    else:
        return labels, layers, K, N


def extended_hinton(ax, V, C, vmax=None, cmin=None, cmax=None, cmap=None, matrix_style=False, alpha=1,
                    enforce_box=False):
    if cmap is None:
        cmap = cm.jet

    if vmax is None:  vmax = np.amax(np.abs(V))
    if cmax is None:  cmax = np.amax(C)
    if cmin is None:  cmin = np.amin(C)

    cnorm = Normalize(vmin=cmin, vmax=cmax, clip=True)
    cmapable = ScalarMappable(norm=cnorm, cmap=cmap)

    if matrix_style:
        V, C = V.T, C.T

    ax.patch.set_facecolor([0, 0, 0, 0])

    for (x, y), w in np.ndenumerate(V):
        s = C[x, y]
        color = cmap(cnorm(s))  # cmap(s / cmax)
        size = np.abs(w / vmax)
        rect = Rectangle([x - size / 2, y - size / 2], size, size,
                         facecolor=color, edgecolor=color, alpha=alpha)
        ret = ax.add_patch(rect)

    if enforce_box:
        ax.axis('tight')
        try:
            ax.set_aspect('equal', 'box')
        except:
            pass
    #ax.autoscale_view()
    #ax.invert_yaxis()
    return cnorm




class MatrixModel:
    def __init__(self, alpha, beta, coupled=None):
        self.alpha = alpha
        self.beta = beta
        self.coupled = coupled if coupled is not None else []
        self.dof = 0

    def fit(self, K, N):
        self.dof = 0
        # check whether constraints are consistent
        check = np.zeros_like(K)
        for mr in self.coupled:
            check += 1 * mr
        assert np.all(check <= 1), "Constraints are inconsistent"

        alpha, beta = self.alpha, self.beta

        P = np.zeros_like(K)
        if len(self.coupled) > 0:
            for mr in self.coupled:
                P[mr] = (K[mr].sum() + alpha - 1) / (N[mr].sum() + alpha + beta - 2)
                self.dof += 1
        else:
            P = (K + alpha - 1) / (N + alpha + beta - 2)
            self.dof = P.size
        self.P = P

    def loglik(self, K, N):
        P = self.P
        L = K * np.log(P) + (N - K) * np.log(1 - P) + chooseln(K, N)
        return np.nansum(L)

    def bootstrap_cross_entropy(self, K, N, labels, cv=10):
        Pmle = K/N # bootstrapping bernoulli trials is the same as sampling from the corresponding binomial
        N = np.array(N, dtype=int)
        ret = []
        for _ in range(cv):
            Ktrain = stats.binom.rvs(N,Pmle)
            Ktest = stats.binom.rvs(N,Pmle)
            self.fit(Ktrain.astype(float), N.astype(float))
            ret.append(self.cross_entropy(Ktest, N))
        return np.array(ret)

    def cross_entropy(self, K, N):
        return -self.loglik(K, N) / np.log(2) / K.size


