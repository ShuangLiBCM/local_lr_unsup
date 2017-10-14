# Build graph for training fully connected network work BCM rule
# Implement as a module
# For class, see BCM_fully_connect_feed_class.py

# %load BCM.py
import numpy as np
import tensorflow as tf
from . import BCM_tf
import time
from . import data_loader
import matplotlib.pyplot as plt


# Define BCM class by quadratic local learning rule
# Relu/Sigmoid activation function
# Update weight per sample
# Enable 'QBCM' and 'kurtosis' objective function
# Hasn't applied batch learning
# Enable decay time constant

# Basic model parameters as external flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('eta', 0.0005, 'Initial learning rate')
flags.DEFINE_integer('epochs', 40, 'Number of epcohs')
flags.DEFINE_integer('batch_size', 1, 'Batch_size, must divide evenly into the dataset sizes')
# flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data')
flags.DEFINE_integer('p', 2, 'Power for BCM dictionary')
flags.DEFINE_float('tau', 200, 'Time constant for exponential moveing average')
flags.DEFINE_string('obj_type', 'QBCM', 'Type of objective function')
flags.DEFINE_string('nonlinear', 'ReLU', 'Type of nonlinearity')
flags.DEFINE_integer('n_output', 2, 'Number of output neurons')
flags.DEFINE_float('decay', 0.01, 'decay time constant for the weights')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data')

def placeholder_inputs(sample_size, dim_input):
	"""
	Generate placeholder variables to represent the input tensors.

	Args:
		sample_size: int, size of the total sample inputs for calculating objective function, eg.: dim_x = X.shape[0]
		dim_x: int, input dimension, eg.: sample_size = X.shape[1]

	Returns:
		single sample placeholder for stochastic training
		Whole sample placeholder for objective function calculation

	"""
	input_placeholder = tf.placeholder(tf.float32, [1, dim_input])
	obj_placeholder = tf.placeholder(tf.float32, [sample_size, dim_input])

	return input_placeholder, obj_placeholder

def fill_feed_dict(data_single, data_all, images_pl, obj_pl):
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

	feed_dict = {
		images_pl: images_feed,
		obj_pl: obj_feed,
	}

	return feed_dict

def run_training():
	"""
	Train data for a number of steps
	# Get the sets of images for unsupervised training
	Args:
	Returns:
		data_sets: input data for training
	"""
	data_sets, data_w= data_loader.load_laplace(loc=0, scale=1, sample_size=1000, dimension=2, skew=False, whiten=True, rotation=True)

	# Tell tensorflow that the model will be built into the default gragh
	with tf.Graph().as_default():
		# Generate placeholders for the single image and all training images for objective function
		input_placeholder, obj_placeholder = placeholder_inputs(data_sets.shape[0], data_sets.shape[1])

		# Build a graph that computes predictions from the inference model
		outputs, objs, weights, thresholds = BCM_tf.inference(input_placeholder, obj_placeholder, FLAGS.n_output, FLAGS.obj_type, FLAGS.nonlinear)

		# Add to the Graph that Ops that train the model
		update_w, update_thres = BCM_tf.training(input_placeholder, weights, thresholds, outputs, FLAGS.n_output, FLAGS.obj_type, FLAGS.nonlinear, FLAGS.eta, FLAGS.decay, FLAGS.tau, FLAGS.p)

		# Build the summary Tensor based on the TF collections fo Summaries
		summary = tf.summary.merge_all()

		# Add the variable initializer Op.
		init = tf.global_variables_initializer()

		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver()

		# Create a session for running Ops on the Graph
		sess = tf.Session()

		# Instantiate a Summary Writer to output summaries and the Graph
		summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

		# After everything is built:

		# Run the Op to initialize the variables.
		sess.run(init)

		# Start the training loop
		for step in range(FLAGS.epochs):
			tf.random_shuffle(data_sets)
			for sample in range(data_sets.shape[0]):
				# Fill a feed dictionary with the actual set of images and labels
				# For this particular training step
				feed_dict = fill_feed_dict(data_sets[sample, :].reshape([1, 2]), data_sets, input_placeholder, obj_placeholder)

				# Run one step of the mode. The return values are the outputs
				sess.run(update_w, feed_dict=feed_dict)
				sess.run(update_thres, feed_dict=feed_dict)

				# Write summaries and print overview fairly often
				# if (step+1) * sample % 100 == 0:
					# Print status to stdout
				 #   print('Iteration %d:' % (weights[0, 0]))
					# Update the event file
					#summary_str = sess.run(summary, feed_dict=feed_dict)
					#summary_writer.add_summary(summary_str, step)
					#summary_writer.flush()

		final_w = sess.run(weights).reshape(1,4)

	return final_w, data_w

def main():
	start_time = time.time()
	final_w, data_w = run_training()
	print('Done in %0.3fs' % (time.time() - start_time))

	# Plot the final weight
	plt.figure()
	plt.plot([0, data_w[0][0]], [0, data_w[0][1]], 'k')
	plt.plot([0, data_w[1][0]], [0, data_w[1][1]], 'k')
	g = plt.gca()
	g.set_aspect(1)
	plt.plot([0, final_w[0][0]], [0, final_w[0][1]], 'r')
	plt.plot([0, final_w[0][2]], [0, final_w[0][3]], 'r')
	plt.show()

if __name__ == '__main__':
	tf.app.run()


