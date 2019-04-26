"""
This script performs a grid search for optimal hyperparameters for a bagged SVM.
"""

# important necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as numpy
import pickle as pickle
from eeg_sandbox import *

# feature sets to consider
reduced1 = [50, 59, 72, 106, 108, 134, 147, 226, 236, 244, 252, 323]
reduced2 = [50, 59, 72, 106, 108, 134, 147, 154, 192, 226, 236, 244, 252, 315, 323]


# import X and y, here they have already been standardized and shuffled
X_save_path = './X_filtered_non_standardized.pkl'

# load, shuffle data
X, y = load_X_and_y(X_save_path)
# to use a different feature set, change which set indexes X below
X = X[:, reduced2]
n,d = X.shape
standardize_X(X)
X,y = shuffle_X_and_y(X,y)

# split the data up for quick validation of hyperparameter choice
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# the base gamma value to use, to be multiplied by powers of 10 in the grid
# 1/d is a common choice for gamma when you don't want to optimize at all.
gamma_base = 1/d

# a dict of (string, list) pairs : strings are parameter names, lists are candidate values
parameters = {'base_estimator__gamma' : [gamma_base*10.0**k for k in np.arange(-2, 5)], 
			  'base_estimator__C' : [4.0**k for k in np.arange(-1,5)]}


# create model and fit using training data
model = GridSearchCV(BaggingClassifier(SVC(), n_estimators=100, max_features=1.0), parameters, cv=4, scoring='accuracy')
model.fit(X_train, y_train)

print("Best parameters: ")
print(model.best_params_)


# print the error rates resulting from each set of parameters in the grid
print()
print("Grid scores:")
print()
means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, model.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    print()
print()
print("Best score achieved: ")
print(model.best_score_)
print()

# test the best parameters using the validation set
y_true, y_pred = y_test, model.predict(X_test)
print("Test error on non-training data:")
print(np.mean(y_pred != y_true))






