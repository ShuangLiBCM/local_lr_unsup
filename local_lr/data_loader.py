# Load data and prepare the data for BCM training
# Implemented Iris data and laplace data

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
from sklearn import datasets
from scipy import stats
import pandas as pd

# Import Iris data from sklean
# This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray
# http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
def load_Iris(whiten = True):
	
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
	#df = datasets.load_iris()
	df.tail()   # Visualize in the table last rows of the dataframe

	y = df.iloc[:, 4].values  # pandas dataframe, select data by location index
	y = np.where(y == 'Iris-setosa',1,-1)
	s_rt = df.iloc[:,[0,2]].values
	
	## Remove the mean
	s_rt = s_rt - s_rt.mean(axis = 0)

	## Perform zca whitenning
	if whiten:
		ZCAMatrix = zca_whitening_matrix(s_rt.T)
		s_rt_wt = np.dot(s_rt,ZCAMatrix)
	else: 
		s_rt_wt = s_rt

	# Generate 2d satter plot of the whitened data
	df = pd.DataFrame({'x':s_rt_wt[:,0],'y':s_rt_wt[:,1]})
	g = sns.jointplot(x="x", y="y", data=df)
	g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
	g.ax_joint.collections[0].set_alpha(0)
	g.set_axis_labels("Dimension 1", "Dimension 2")

	plt.scatter(s_rt_wt[:50,0],s_rt_wt[:50,1],color = 'red',marker = 'o',label = 'setosa')
	plt.scatter(s_rt_wt[50:150,0],s_rt_wt[50:150,1],color = 'blue', marker = '*', label = 'versicolor')
	plt.xlabel('petal length')
	plt.ylabel('sepal length')
	plt.legend(loc = 'upper left')

	return s_rt_wt, g


# Generate laplace distributed data

def load_laplace(loc = 0, scale = 1, sample_size = 1000 , dimension = 2,skew = False, whiten = True, rotation = False, Affine = False, iffigure = True):
	"""Generate laplacian distributed data
		Args:
			loc: float, the position of the distribution peak. Default is 0.
			scale: float, the exponential decay. Default is 1
			sample_size: int, number of samples
			dimension: int, number of dimensions
			skew: boolean, whether to introduce skewness to the data
			whiten: boolean, whether to perform whitening on the data
			rotation: boolean, whether to introduce rotation
			Affine: boolean, whether to introduce affine transformation
			iffigure: whether to plot the sampled data
		Returns:
			s_rt_wt: sample_size * dimension, generated data
			w_rt_wt: 1 * dimension, ground truth independent direction to track transformation
	"""

	# Sample from the laplace distribution
	s = np.random.laplace(loc,scale,[sample_size,dimension])  
	w = np.eye(2)

	# make data skewed with a half-squaring

	if skew:
		s[:,1] = stats.skewnorm.rvs(14., size=len(s))
		title_ori = 'Skewed original distribution'
	else:
		title_ori = 'Original distribution'

	# # plot original distribution
	# if iffigure:
	# 	df = pd.DataFrame({'x':s[:,0],'y':s[:,1]})
	# 	g = sns.jointplot(x="x", y="y", data=df)
	# 	g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
	# 	# sns.plt.title(title_ori)
	# 	g.ax_joint.collections[0].set_alpha(0)
	# 	g.set_axis_labels("Dimension 1", "Dimension 2")

	# Conduct rotation and mixture
	# Generate rotation matrix
	A = np.eye(2)
	if rotation:
		theta = np.pi/4      # 45 degree rotation
		A = np.array(((np.cos(theta),- np.sin(theta)),(np.sin(theta), np.cos(theta))))
		title_trans = 'Rotated distribution'
	elif Affine:
		A = np.random.randn(dimension,dimension)
		title_trans = 'Affine tranformed distribution'
	#A = np.dot(A, A.T)

	s_rt = np.dot(s,A)
	w_rt = np.dot(w,A)

	# # plot mixed distribution
	# if iffigure:
	# 	df = pd.DataFrame({'x':s_rt[:,0],'y':s_rt[:,1]})
	# 	# sns.plt.title(title_trans)
	# 	g = sns.jointplot(x="x", y="y", data=df)
	# 	g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
	# 	g.ax_joint.collections[0].set_alpha(0)
	# 	g.set_axis_labels("Dimension 1", "Dimension 2")

	if whiten:
		# Demean is critical, especially for skewed data
		s_rt = s_rt - s_rt.mean(axis = 0)
		ZCAMatrix = zca_whitening_matrix(s_rt.T)
		s_rt_wt = np.dot(s_rt,ZCAMatrix)
		w_rt_wt = np.dot(w_rt,ZCAMatrix)
		# plot mixed distribution
	else:
		s_rt_wt = s_rt
		w_rt_wt = w_rt

	if iffigure:
		data_visu_2d(s_rt_wt)

	return s_rt_wt, w_rt_wt
		
def load_uniform(loc = 0, scale = 1, sample_size = 1000 , dimension = 2,skew = False, whiten = True, rotation = False, Affine = False, iffigure = True):
	# Sample from the laplace distribution
	s = np.random.uniform(loc,scale,[sample_size,dimension])  
	w = np.eye(2)
	# make data skewed with a half-squaring

	# make data skewed with a half-squaring

	if skew:
		s[:,1] = stats.skewnorm.rvs(14., size=len(s))
		title_ori = 'Skewed original distribution'
	else:
		title_ori = 'Original distribution'

	# plot original distribution
	if iffigure:
		df = pd.DataFrame({'x':s[:,0],'y':s[:,1]})
		g = sns.jointplot(x="x", y="y", data=df)
		g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
		# sns.plt.title(title_ori)
		g.ax_joint.collections[0].set_alpha(0)
		g.set_axis_labels("Dimension 1", "Dimension 2")

	# Conduct rotation and mixture
	# Generate rotation matrix
	A = np.eye(2)
	if rotation:
		theta = np.pi/4      # 45 degree rotation
		A = np.array(((np.cos(theta),- np.sin(theta)),(np.sin(theta), np.cos(theta))))
		title_trans = 'Rotated distribution'
	elif Affine:
		A = np.random.randn(dimension,dimension)
		title_trans = 'Affine tranformed distribution'
	#A = np.dot(A, A.T)

	s_rt = np.dot(s,A)
	w_rt = np.dot(w,A)

	# plot mixed distribution
	if iffigure:
		df = pd.DataFrame({'x':s_rt[:,0],'y':s_rt[:,1]})
		# sns.plt.title(title_trans)
		g = sns.jointplot(x="x", y="y", data=df)
		g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
		g.ax_joint.collections[0].set_alpha(0)
		g.set_axis_labels("Dimension 1", "Dimension 2")

	if whiten:
		# Demean is critical, especially for skewed data
		s_rt = s_rt - s_rt.mean(axis = 0)
		ZCAMatrix = zca_whitening_matrix(s_rt.T)
		s_rt_wt = np.dot(s_rt,ZCAMatrix)
		w_rt_wt = np.dot(w_rt,ZCAMatrix)
		# plot mixed distribution
		if iffigure:
			df = pd.DataFrame({'x':s_rt_wt[:,0],'y':s_rt_wt[:,1]})
			g = sns.jointplot(x="x", y="y", data=df)
			g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
			g.ax_joint.collections[0].set_alpha(0)
			g.set_axis_labels("Dimension 1", "Dimension 2")
	else: 
		s_rt_wt = s_rt
		w_rt_wt = w_rt

	return s_rt_wt, w_rt_wt 

# Perform zca whitening
def zca_whitening_matrix(X):
	"""
	Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
	INPUT:  X: [M x N] matrix.
		Rows: Variables
		Columns: Observations
	OUTPUT: ZCAMatrix: [M x M] matrix
	"""
	# Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
	sigma = np.cov(X, rowvar=True) # [M x M]
	# Singular Value Decomposition. X = U * np.diag(S) * V
	U,S,V = np.linalg.svd(sigma)
		# U: [M x M] eigenvectors of sigma.
		# S: [M x 1] eigenvalues of sigma.
		# V: [M x M] transpose of U
	# Whitening constant: prevents division by zero
	epsilon = 1e-5
	# ZCA Whitening matrix: U * Lambda * U'
	ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
	#ZCAMatrix = np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))) # [M x M]
	return ZCAMatrix

def data_visu_2d(data):
	df = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]})
	g = sns.jointplot(x="x", y="y", data=df)
	g.plot_joint(plt.scatter, c="gray", s=10, linewidth=.1, marker=".")
	g.ax_joint.collections[0].set_alpha(0)
	g.set_axis_labels("Dimension 1", "Dimension 2")

