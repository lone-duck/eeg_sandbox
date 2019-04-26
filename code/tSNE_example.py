"""
This script loads a preprocessed data set (in X,y form) and uses t-SNE to visualize 
how well a particular subset of the features separates the data.
"""
# import necessary libraries
import numpy as numpy
import matplotlib.pyplot as plt
from eeg_sandbox import *
import pickle as pickle
from sklearn import manifold
from matplotlib.ticker import NullFormatter

# load and standardize X
X_save_path = './X_filtered_non_standardized.pkl'
X, y = load_X_and_y(X_save_path)
standardize_X(X)

# Column numbers of features for feature set we wish to visualize
#reduced1 = [50, 59, 72, 106, 108, 134, 147, 226, 236, 244, 252, 323]
reduced2 = [50, 59, 72, 106, 108, 134, 147, 154, 192, 226, 236, 244, 252, 315, 323]

# remove features outside the set we wish to consider
X = X[:, reduced2]

# get size of X
n_samples, d = X.shape

# t-SNE parameters, more can be read about these in scikit-learn's t-SNE docuementation
n_components = 2
perplexities = [30,50,100]

# vectors used to index concussed/healthy rows from X
concussed = y == 1
healthy = y == 0

# set up figure
(fig, subplots) = plt.subplots(1, 3)

# for each perplexity value
for i, perplexity in enumerate(perplexities):
	ax = subplots[i]
	tsne = manifold.TSNE(n_components=n_components, init='random',
						 random_state=0, perplexity=perplexity)
	# transform data using t-SNE
	Y = tsne.fit_transform(X)
	ax.set_title("Perplexity=%d" % perplexity)
	# plot concussed, healthy subjects
	ax.scatter(Y[concussed,0], Y[concussed, 1], c="r")
	ax.scatter(Y[healthy, 0], Y[healthy, 1], c="g")
	ax.xaxis.set_major_formatter(NullFormatter())
	ax.yaxis.set_major_formatter(NullFormatter())
	ax.axis("tight")


plt.show()
