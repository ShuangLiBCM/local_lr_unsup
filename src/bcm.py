import numpy as np

class BCM:
    """BCM learning
    Parameter:
    eta: float, learning rate (between 0.0 - 1.0)
    n_iter: int, passes over the training dataset
    ny: number of output neurons
    batchsize: float, percentage of data that are used to update the weight once

    Attributes:
    w_: 1d-array, weights after fitting
    error_: list, number of misclassification in every epoch
    """
    def __init__(self, eta=0.1, n_iter=10, ny=1, tau=10, batchsize=10, thres=0, p=2):
        self.eta = eta
        self.n_iter = n_iter
        self.ny = ny
        self.batchsize = batchsize
        self.tau = tau,  # Time constant for calculating thresholds
        self.thres = [thres*np.ones(ny)]
        self.p = p  # int, power for threshold computation
        self.y_thres = []  # Storaged y for studying effect of threshold

    def fit(self, X):
        """fitting training data
        Parameter:
        X: {array-like}, shape = [n_samples,n_features]
        Returns: self:object
        ny: value, number of output neurons
        """
        # Weights initialized as normal distribution
        self.w_ = np.random.randn(X.shape[1], self.ny)  # 2*1

        # Use elementwise training
        for _ in range(self.n_iter):
            for i, xi in enumerate(X):  # 150, elementwise training for all samples
                threshold = self.thres[-1]
                y = np.dot(xi, self.w_)  # 1
                y[y < 0] = 0
                self.y_thres.append(y)
                y = np.atleast_2d(y)
                self.w_ += self.eta * xi[:, None] * y * (y - threshold)

                thres_sum = 0
                mv_ave = np.zeros(len(self.y_thres) + 1)
                y_power = np.vstack(self.y_thres).T ** self.p  # Element wise computation?
                n = y_power.shape[1]
                t = np.arange(n)
                h = np.exp(-t / self.tau)
                theta = np.vstack([np.convolve(yp, h, mode='full')[:n] for yp in y_power]) / self.tau
                self.thres.append(theta[:, -1])

        return self


