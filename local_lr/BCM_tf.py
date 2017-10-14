# Impletment BCM class and related functions in tensorflow
# inference() - Builds the model as far as is required for running the network forward
# training() - Use the local learning rule to update the weights and thresholdss
import numpy as np
import tensorflow as tf

def inference(input_single, inputs, n_output = 2, obj_type = 'QBCM', nonlinear = 'ReLU'):

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
    weights = tf.Variable(tf.random_normal([dim_input, n_output]))
    thresholds = tf.Variable(tf.zeros([1, n_output]))

    net_inputs = tf.matmul(input_single, weights)
    if nonlinear == 'Sigmoid':
        outputs = tf.Sigmoid(net_inputs)
    elif (nonlinear == 'ReLU')| (nonlinear == None):
        outputs = tf.nn.relu(net_inputs)
    else:
        print('Wrong nolinearity')

    objs = obj(inputs, weights, obj_type, nonlinear)

    return outputs, objs, weights, thresholds

def training(input_single, weights, thresholds, outputs, n_output = 2, obj_type = 'QBCM', nonlinear = 'ReLU', eta = 0.00001, decay = 0.01, tau = 200, p = 2):
    """Update the weights and thresholds with BCM learning rule
    """

    out_thre_diff = outputs - thresholds
    out_thre_diff_p2 = tf.pow(outputs, 2) - thresholds

    if obj_type == 'QBCM':
        if nonlinear == 'Sigmoid':
            delta_w = eta * tf.matmul(input_single, tf.multiply(tf.multiply(outputs, out_thre_diff), dsigmoid(outputs)),
                                      transpose_a=True) - eta * decay * weights
        elif (nonlinear == 'ReLU') | (nonlinear == 'None'):
            delta_w = eta * tf.matmul(input_single, tf.multiply(outputs, out_thre_diff),
                                      transpose_a=True) - eta * decay * weights
        else:
            print('Wrong nonlinearty')
    elif obj_type == 'kurtosis':
        if nonlinear == 'Sigmoid':
            delta_w = 4 * eta * tf.matmul(input_single, tf.multiply(tf.multiply(outputs, out_thre_diff_p2), dsigmoid(outputs)),
                                      transpose_a = True) - eta * decay * weights
        elif (nonlinear == 'ReLU') | (nonlinear == 'None'):
            delta_w = 4 * eta * tf.matmul(input_single, tf.multiply(outputs, out_thre_diff_p2),
                                      transpose_a = True) - eta * decay * weights
        else:
            print('Wrong nonlinearty')
    else:
        print('Wrong objective function')

    new_w = weights + delta_w
    update_w = tf.assign(weights, new_w)

    # Update threshold
    h = tf.exp(-1 / tau)
    new_thres = thresholds * h + tf.pow(outputs, p) * (1 - h)
    update_thres = tf.assign(thresholds, new_thres)

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
        obj = np.divide(obj1, obj2 ** 1.5)
    else:
        print('Wrong objective function')

    return obj

def dsigmoid(z):
    return tf.Sigmoid(z) * (1 - (tf.Sigmoid(z)))

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

    w = np.linspace(w_min, w_max, reso)
    wx, wy = np.meshgrid(w, w)
    w = np.vstack((wx.ravel(), wy.ravel()))
    obj_choice = ['QBCM', 'kurtosis']
    nonlinear_choice = ['Relu', 'Sigmoid', 'None']

    # parameter passed through para
    p = para[0]
    ny = para[1]
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

            bcm_data = bcm(eta=eta[para_index], n_epoch=n_epoch[para_index], batch=batch, ny=ny, tau=tau, thres=0, p=p,
                           random_state=None, nonlinear=nonlinear, obj_type=obj_type,
                           decay=decay[para_index])

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





