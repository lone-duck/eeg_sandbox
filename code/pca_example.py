"""
This script loads a processed data set (in X,y form) and uses pca to visualize how well
a given subset of the features separates the data.
"""

# import required libraries
import numpy as numpy
import matplotlib.pyplot as plt
from eeg_sandbox import *
import pickle as pickle
from sklearn import decomposition
from matplotlib.ticker import NullFormatter

# load and standardize X
X_save_path = './X_filtered_non_standardized.pkl'
X, y = load_X_and_y(X_save_path)
standardize_X(X)

# the column numbers of the feature set we wish to consider
#reduced1 = [50, 59, 72, 106, 108, 134, 147, 226, 236, 244, 252, 323]
reduced2 = [50, 59, 72, 106, 108, 134, 147, 154, 192, 226, 236, 244, 252, 315, 323]

# remove all features aside from those chosen
X = X[:, reduced2]

# get shape of X
n_samples, d = X.shape

# number of principal components
n_components = 2

# vectors used to index concussed and healthy patients in X
concussed = y == 1
healthy = y == 0

# create plot
fig = plt.figure()
# create PCA object and transform X
pca = decomposition.PCA(n_components=2)
Y = pca.fit_transform(X)
# plot concussed and healthy subjects
plt.scatter(Y[concussed,0], Y[concussed, 1], c="r")
plt.scatter(Y[healthy, 0], Y[healthy, 1], c="g")
plt.axis("tight")




plt.show()