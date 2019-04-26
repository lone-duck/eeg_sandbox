"""
This script uses scikit-learn's RFECV class (recursive feature elimination
with cross validation) to find important features. On each iteration, 
RFECV eliminates the feature with smallest weight. K-fold cross validation is 
used to determine the optimal number of features. It was found that the picked features depend on the ordering of the training data
(which affects the cross validation splits), but that many features appear frequently
regardless of ordering. This script runs RFECV 100 times, counts the number of times
each feature appeared, and prints out the 30 most commonly selected features alongside
the number of times they appeared. Both SVMs and logistic regression (or any other 
parametric model) can be used by changing the model. 
"""

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as numpy
import pickle as pickle
from eeg_sandbox import *
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV 
from collections import Counter

# import X and y, standardize X
X_save_path = './X_filtered_non_standardized.pkl'
X, y = load_X_and_y(X_save_path)
standardize_X(X)

# list to hold features selected by the algorithm
selections = []

# 100 times do...
for _ in range(100):
	# shuffle X and y
	X,y = shuffle_X_and_y(X,y)
	# choose the underlying model. to use log. reg. change which line is commented out
	model = SVC(kernel="linear", C = 0.5)
	#model = LogisticRegression(C = 0.5)
	# Create the RFECV object and compute a cross-validated score.
	# The "accuracy" scoring is proportional to the number of correct
	# classifications
	rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(4),
	              scoring='accuracy')
	rfecv.fit(X, y)

	# create list of selected features for this iteration
	mask = rfecv.support_
	chosen = [i for i in range(len(mask)) if mask[i]]
	# add this to selections
	selections += chosen

# once complete, create a dict to hold occurence counts for each feature and populate
fcount = {}
for f in selections:
	if f not in fcount:
		fcount[f] = 1
	else:
		fcount[f] += 1

# n_choose selects the threshold; we wish to look at features chosen n_chooose or more times
n_choose = 30
# this list will store the features that meet the above criteria
final_selected = []

# object used to find most commonly occuring features
f_counter = Counter(fcount)

# populate final_selected
for f, count in f_counter.most_common(n_choose):
	print(f, ": ", count)
	final_selected.append(f)

# sort final_selected
final_selected.sort()
# display
print(final_selected)




