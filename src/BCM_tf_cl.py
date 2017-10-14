# Impletment BCM class and related functions in tensorflow
# inference() - Builds the model as far as is required for running the network forward
# training() - Use the local learning rule to update the weights and thresholdss

import numpy as np
import tensorflow as tf
import pdb

class BCM_tf:
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
    def __init__(self, eta=0.1, n_output=1, batch=1, tau=100.0, p=2, nonlinear='ReLU', obj_type='QBCM', decay=0.0, seed = None):
        self.eta = eta
        self.n_output = n_output
        self.batch = batch
        self.tau = tau  
        self.p = p
        self.nonlinear = nonlinear
        self.obj_type = obj_type
        self.decay = decay
        self.seed = seed

    def inference(self, input_single, inputs, n_output=2, obj_type='QBCM', nonlinear='ReLU'):

        """
        Generate output estimate based on current inputs, weights and thresholds
        Also generate value for objective function
        Args:
        inputs: 1 * dim_input array, one sample data input for forward inference
        dim_input: int, input dimension, for variable initialization

        Returns:
        outputs: 1 * dim_output array, estimated output
        """
        dim_input = int(input_single.get_shape()[1])
        # self.weights = tf.Variable(tf.random_normal([dim_input, n_output]))
        r_gen = np.random.RandomState(self.seed)
        self.weights = tf.Variable(r_gen.randn(dim_input, n_output), dtype = tf.float64)
        self.thresholds = tf.Variable(tf.zeros([1, n_output], dtype=tf.float64), dtype=tf.float64)

        self.net_inputs = tf.matmul(input_single, self.weights)
        if self.nonlinear == 'Sigmoid':
            self.outputs = tf.Sigmoid(self.net_inputs)
        elif (self.nonlinear == 'ReLU')| (self.nonlinear == None):
            self.outputs = tf.nn.relu(self.net_inputs)
        elif (self.nonlinear == 'Convex'):
            self.outputs = tf.nn.relu(tf.square(self.net_inputs))
        else:
            print('Wrong nolinearity')

        self.objs = obj(inputs, self.weights, self.obj_type, self.nonlinear)

    def training(self, input_single, n_output = 2, obj_type = 'QBCM', nonlinear = 'ReLU', eta = 0.00001, decay = 0.01, tau = 200, p = 2):
        """Update the weights and thresholds with BCM learning rule
        """

        out_thre_diff = self.outputs - self.thresholds
        out_thre_diff_p2 = tf.pow(self.outputs, 2) - self.thresholds

        if self.obj_type == 'QBCM':
            if self.nonlinear == 'Sigmoid':
                delta_w = self.eta * tf.matmul(input_single, tf.multiply(tf.multiply(self.outputs, out_thre_diff), dsigmoid(self.outputs)),
                                          transpose_a=True) - self.eta * self.decay * self.weights
            elif (self.nonlinear == 'ReLU') | (self.nonlinear == 'None'):
                delta_w = self.eta * tf.matmul(input_single, tf.multiply(self.outputs, out_thre_diff),
                                      transpose_a=True) - self.eta * self.decay * self.weights
            elif (self.nonlinear == 'Convex'):
                delta_w = self.eta * tf.matmul(input_single, tf.multiply(tf.multiply(self.outputs, out_thre_diff), 2 * self.outputs),
                                          transpose_a=True) - self.eta * self.decay * self.weights
            else:
                print('Wrong nonlinearty')
        elif self.obj_type == 'kurtosis':
            if self.nonlinear == 'Sigmoid':
                delta_w = 4 * self.eta * tf.matmul(input_single, tf.multiply(tf.multiply(self.outputs, out_thre_diff_p2), dsigmoid(self.outputs)),
                                          transpose_a = True) - self.eta * self.decay * self.weights
            elif (self.nonlinear == 'ReLU') | (self.nonlinear == 'None'):
                delta_w = 4 * self.eta * tf.matmul(input_single, tf.multiply(self.outputs, out_thre_diff_p2),
                                          transpose_a = True) - self.eta * self.decay * self.weights
            else:
                print('Wrong nonlinearty')
        else:
            print('Wrong objective function')

        new_w = self.weights + delta_w
        update_w = tf.assign(self.weights, new_w)

        # Update threshold
        h = tf.cast(tf.exp(-1 / tau), tf.float64)
        new_thres = self.thresholds * h + tf.pow(self.outputs, self.p) * (1 - h)
        update_thres = tf.assign(self.thresholds, new_thres)

        return update_w, update_thres

# objective function using all data
def obj(X, w, obj_type='QBCM', nonlinear='Sigmoid'):
    c = tf.matmul(X, w)
    if nonlinear == 'Sigmoid':
        c = tf.Sigmoid(c)
    elif nonlinear == 'ReLU':
        c = tf.nn.relu(c)

    obj = 0

    if obj_type == 'QBCM':
        obj1 = tf.reduce_mean(tf.pow(c, 3))
        obj2 = tf.reduce_mean(tf.pow(c, 2))
        obj = obj1 / 3 - obj2 ** 2 / 4
    # obj = - obj2/2
    elif obj_type == 'kurtosis':
        obj1 = tf.reduce_mean(tf.pow(c, 4))
        obj2 = tf.reduce_mean(tf.pow(c, 2))
        obj = obj1 - obj2 ** 2 * 3
    elif obj_type == 'skewness':
        obj1 = tf.reduce_mean(tf.pow(c, 3))
        obj2 = tf.reduce_mean(tf.pow(c, 2))
        obj = obj1/obj2 ** 1.5
    else:
        print('Wrong objective function')

    return obj

def dsigmoid(z):
    return tf.nn.sigmoid(z) * (1 - (tf.nn.sigmoid(z)))

# Plot the weights trajectory on top of objective function landscape
def bcm_obj(s_rt_wt, w_min, w_max, reso, para, obj_select=None, nonlinear_select=None, ori_w=0):
    """
    Parameter: 
    s_rt_wt: input data, num of samples * dimension
    w_min: mininum range of objective function landscape
    w_max: maximun range of objective function landscape
    reso: resolution of weights grid
    para: parameters for training local learnin rule 
    ori_w: for laplace data, plot the original weights
    obj_select: str, type of objective function if specified, sweep across all combination if none
    Nolinear_select: str, type of nonlinearity if specified, sweep across all combination if none
    """

    w = tf.linspace(w_min, w_max, reso)
    wx, wy = np.meshgrid(w, w)
    w = np.vstack((wx.ravel(), wy.ravel()))
    obj_choice = ['QBCM', 'kurtosis']
    nonlinear_choice = ['Relu', 'Sigmoid', 'None']

    # parameter passed through para
    p = para[0]
    n_output = para[1]
    tau = para[2]
    batch = para[3]
    n_epoch = para[4]
    decay = para[5]
    eta = para[6]

    # Plot a gallery of images
    # if nolinearty and objective defined, train for a specific case, otherwise, train all the combinations
    if obj_select == None:
        n_row = len(obj_choice)
    else:
        n_row = 1
        obj_index = obj_choice.index(obj_select)

    if nonlinear_select == None:
        n_col = len(nonlinear_choice)
    else:
        n_col = 1
        nonlinear_index = nonlinear_choice.index(nonlinear_select)

    fig, ax = plt.subplots(n_row, n_col, figsize=(12, 6), sharex=True, sharey=True)
    ori_w = ori_w * (w_max ** 0.5)

    for i in range(n_row):
        for j in range(n_col):
            if (n_row + n_col) > 2:
                obj_type = obj_choice[i]
                nonlinear = nonlinear_choice[j]
                para_index = i * n_col + j
            else:
                obj_type = obj_select
                nonlinear = nonlinear_select
                para_index = obj_index * len(nonlinear_choice) + nonlinear_index

            obj_landscape = obj(s_rt_wt, w, obj_type=obj_type, nonlinear=nonlinear)
            title_set = [obj_type, nonlinear]

            nbins = 20
            levels = np.percentile(obj_landscape, np.linspace(0, 100, nbins))

            with sns.axes_style('white'):
                if (n_row + n_col) > 2:
                    g = ax[i, j]
                else:
                    g = ax

                c = g.contour(wx, wy, obj_landscape.reshape(wx.shape), levels=levels, zorder=-10,
                              cmap=plt.cm.get_cmap('viridis'))
                g.plot(s_rt_wt[:, 0], s_rt_wt[:, 1], '.k', ms=4)
                g.set_aspect(1)

            plt.grid('on')
            plt.colorbar(c, ax=g)

            # Training with BCM local learning rule
            bcm_data = bcm_tf(eta = eta[para_index], n_epoch = n_epoch[para_index], batch = batch, n_output = n_output, tau = tau, p = p,
                        nonlinear = nonlinear, obj_type = bj_type,
                           decay = decay[para_index], seed = seed)

            bcm_data.fit(s_rt_wt)
            bcm_data_w = np.vstack(bcm_data.w_track)

            if len(ori_w) > 1:
                g.plot([0, ori_w[0][0]], [0, ori_w[0][1]])
                g.plot([0, ori_w[1][0]], [0, ori_w[1][1]])

            g.plot(bcm_data_w[:, 0], bcm_data_w[:, 1], 'g')
            g.plot(bcm_data_w[-1, 0], bcm_data_w[-1, 1], 'y*', ms=15)
            g.plot(bcm_data_w[:, 2], bcm_data_w[:, 3], 'r')
            g.set_title(title_set)
            g.plot(bcm_data_w[-1, 2], bcm_data_w[-1, 3], 'y*', ms=15)
