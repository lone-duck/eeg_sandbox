"""
This script uses the forward_selection function defined in eeg_sanbox.py
to do feature selection on a given data set.
"""

# import necessary libraries
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as numpy
import pickle as pickle
from eeg_sandbox import *
import matplotlib.pyplot as plt


# BaggedSVM parameters chosen from gridsearch (deprecated)
C = 16	
gamma = 0.0027472527472527475
# import X and y
X_save_path = './X_filtered_non_standardized.pkl'

X, y = load_X_and_y(X_save_path)
X, y = shuffle_X_and_y(X,y)
standardize_X(X)

# give a list of starting features, perhaps found using some other feature selection method.
# note this is optional - to use forward_selection alone simply omit the starting_features parameter
# when calling forward_selection.
starting_features = [30, 43, 49, 64, 108, 134, 159, 167, 200, 281, 299, 330]

# create model
model = BaggingClassifier(SVC(C=C, gamma=gamma), n_estimators=500, max_features=1.0)
#model = SVC(C=C, gamma=gamma)

# run algorithm and display selected features
picked_features, errors = forward_selection(X, y, model, 10, max_error=True, k=4, starting_features=starting_features)

print(picked_features)



