# Build graph for training fully connected network work BCM rule
# Implement as a class

# %load BCM.py
import numpy as np
import tensorflow as tf
from . import BCM_tf
import time
from . import data_loader
import matplotlib.pyplot as plt
from local_lr import BCM_tf_cl


class BCM_fully_connect:
    """Build fully connected network for BCM training
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
	sample_size: int, number of training samples
	decay: float, decay time constant, 0.0 means no decay

	Attributes:
	w_: array input dimension * num of output neurons
	w_track: list, trakcing the trajectory of weights, list length is number of weight updats, and each list contains array with input dimension * num of output neurons
	y_thres: list, trakcing the trajectory of output, list length is number of weight updats, and each list contains array with 1 * num of output neurons
	obj: list, trakcing the trajectory of values of certain objective function, list length is number of weight updats, and each list contains array with 1 * num of output neurons

	"""
    def __init__(self, eta=0.1, n_epoch=10, n_output=1, batch=1, tau=100.0, p=2, seed= None, random_state= None,
                 nonlinear='ReLU', obj_type='QBCM', decay=0.0):
        self.eta = eta
        self.n_epoch = n_epoch
        self.n_output = n_output
        self.batch = batch
        self.tau = tau
        self.p = p
        self.nonlinear = nonlinear
        self.obj_type = obj_type
        self.decay = decay
        self.random_state = random_state
        self.seed = seed
        self.w_track = []
        self.thres_track = []
        

    def placeholder_inputs(self, X):
        """
		Generate placeholder variables to represent the input tensors.

		Args:
			sample_size: int, size of the total sample inputs for calculating objective function, eg.: dim_x = X.shape[0]
			dim_x: int, input dimension, eg.: sample_size = X.shape[1]

		Returns:
			single sample placeholder for stochastic training
			Whole sample placeholder for objective function calculation

		"""
        self.sample_size = X.shape[0]
        self.dim_input = X.shape[1]
        self.input_placeholder = tf.placeholder(tf.float64, [1, self.dim_input])
        self.obj_placeholder = tf.placeholder(tf.float64, [self.sample_size, self.dim_input])

    def fill_feed_dict(self, data_single, data_all, images_pl, obj_pl):
        """
		Fills the feed_dict for training the given step
		Args:
			data_set: The set of images and labels, from input_data
			images_pl: The images placeholder, from place_holder_inputs()
			lables_pl: The labels placeholder, from place_holder_inputs()

		Return:
			feed_dict: The feed dictionary mapping from placeholders to values.
		"""
        images_feed = data_single
        obj_feed = data_all

        self.feed_dict = {
            images_pl: images_feed,
            obj_pl: obj_feed,
        }

    def run_training(self, data_sets):
        """
		Train data for a number of steps
		# Get the sets of images for unsupervised training
		Args:
		Returns:
			data_sets: input data for training
		"""
        # data_sets, data_w= data_loader.load_laplace(loc=0, scale=1, sample_size=1000, dimension=2, skew=False, whiten=True, rotation=True)

        # Tell tensorflow that the model will be built into the default gragh
        with tf.Graph().as_default():
            # Generate placeholders for the single image and all training images for objective function
            self.placeholder_inputs(data_sets)

            # Initialize BCM class
            BCM_model = BCM_tf_cl.BCM_tf(eta=self.eta, n_output=self.n_output, batch=self.batch, tau=self.tau, p=self.p,
                                         nonlinear=self.nonlinear, obj_type=self.obj_type, decay=self.decay,
                                         seed=self.seed)

            # Build a graph that computes predictions from the inference model
            BCM_model.inference(input_single=self.input_placeholder, inputs=self.obj_placeholder,
                                n_output=self.n_output, obj_type=self.obj_type,
                                nonlinear=self.nonlinear)

            # Add to the Graph that Ops that train the model
            update_w, update_thres = BCM_model.training(input_single=self.input_placeholder, n_output=self.n_output,
                                                        obj_type=self.obj_type,
                                                        nonlinear=self.nonlinear, eta=self.eta, decay=self.decay,
                                                        tau=self.tau, p=self.p)

            # Build the summary Tensor based on the TF collections fo Summaries
            # summary = tf.merge_all_summaries()

            # Add the variable initializer Op.
            init = tf.global_variables_initializer()

            # Create a saver for writing training checkpoints.
            # saver = tf.train.Saver()

            # Create a session for running Ops on the Graph
            sess = tf.Session()

            # Instantiate a Summary Writer to output summaries and the Graph
            # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

            # After everything is built:

            # Run the Op to initialize the variables.
            sess.run(init)

            # Start the training loop

            for step in range(self.n_epoch):
                # for step in range(1):
                tf.random_shuffle(data_sets)
                #r_gen = np.random.RandomState(self.random_state)
                #r = r_gen.permutation(len(data_sets0))
                #data_sets = data_sets0[r]
                for sample in range(data_sets.shape[0]):
                    # for sample in range(1):
                    # Fill a feed dictionary with the actual set of images and labels
                    # For this particular training step
                    self.fill_feed_dict(data_sets[sample, :].reshape([1, 2]), data_sets, self.input_placeholder,
                                        self.obj_placeholder)
                    # Run one step of the mode. The return values are the outputs
                    sess.run([update_w, update_thres], feed_dict=self.feed_dict)
                    # sess.run(update_thres, feed_dict=self.feed_dict)
                    w_val, t_val = sess.run([BCM_model.weights, BCM_model.thresholds])
                    self.w_track.append(w_val.reshape(1, self.dim_input * self.n_output))
                    self.thres_track.append(t_val.reshape(1, self.n_output))
                    # w_val, t_val = sess.run([BCM_model.weights, BCM_model.thresholds])
                    # print("%.16f" % t_val[0][0])
                    # Write summaries and print overview fairly often
                    # if (step+1) * sample % 100 == 0:
                    # Print status to stdout
                    #   print('Iteration %d:' % (weights[0, 0]))
                    # Update the event file
                    # summary_str = sess.run(summary, feed_dict=feed_dict)
                    # summary_writer.add_summary(summary_str, step)
                    # summary_writer.flush()

            self.final_w = sess.run(BCM_model.weights).reshape(1, 4)


