from time import time

from sklearn.cross_validation import train_test_split

from bcm import BCM
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()  # Visualize in the table last rows of the dataframe

y = df.iloc[:, 4].values  # pandas dataframe, select data by location index
y = np.where(y == 'Iris-setosa', 1, -1)
X = df.iloc[:, [0, 2]].values

# Shuffle the iris data preparing for training
shuffle_idx = np.random.permutation(len(X[:, 1]))
X = X[shuffle_idx]  # 150*2
# X -= X.mean(axis = 0)
y = y[shuffle_idx]

data_train, _, _, _ = train_test_split(X, y, test_size=0, random_state=8723)

ny = 1
n_iter = 10
BCM_iris = BCM(eta=0.0001, n_iter=n_iter, ny=2, tau=10, batchsize=0, thres=0, p=2)

# print("Training with Oja's rule from %d digits" % (n_components,data_train.shape[0]))
t0 = time()
BCM_iris.fit(data_train)
print("done in %0.3fs" % (time() - t0))

x_plot = range(n_iter * len(X))
plot_lim = 100
thres = np.vstack(BCM_iris.thres)

sns.set_style('ticks')
fig, ax = plt.subplots()
ax.plot(x_plot[:plot_lim], thres[:plot_lim], 'g^', label='Output')
# ax.plot(x_plot[:plot_lim], BCM_mnist.y_thres[:plot_lim], 'bs', label=r'$\theta$')

ax.set_xlabel('updates')
sns.despine(fig)
fig.savefig('figures/results.png')
# h = 14
# w = 14
# BCMdigits = []
# for i in range(ny):
#     BCMdigits_tmp = BCM_mnist.w_[:,i];
#     BCMdigits.append(BCMdigits_tmp.reshape(h,w))

# BCMdigits_titles = ["ojadigits %d" % i for i in range (ny)]
# plot_gallery(BCMdigits,BCMdigits_titles,h,w,n_row= 3, n_col=3)
