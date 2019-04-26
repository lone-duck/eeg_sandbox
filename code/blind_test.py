"""
This script is used for testing a classifier (or multiple classifiers) on blind test data.
"""

# import necessary libraries
import numpy as numpy
from eeg_sandbox import *
import pickle as pickle
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# path for training data
Xtrain_save_path = './X_filtered_non_standardized.pkl'
# path for blind data
Xtest_save_path = './X_blind_first_third.pkl'
#Xtest_save_path = './X_blind_second_third.pkl'

# load X and y
X, y = load_X_and_y(Xtrain_save_path)
n,d = X.shape

# Feature sets chosen using RFECV
full = range(d)
reduced1 = [50, 59, 72, 106, 108, 134, 147, 226, 236, 244, 252, 323]
reduced2 = [50, 59, 72, 106, 108, 134, 147, 154, 192, 226, 236, 244, 252, 315, 323]


# SVM parameters chosen using gridsearch
C_full = 64
gamma_full = 0.00027472527472527475
C_r1_d = 4
gamma_r1_d = 1/len(reduced1)
C_r1_10d = 16
gamma_r1_10d = 1/len(reduced1)/10
C_r2_d = 1
gamma_r2_d = 1/len(reduced2)
C_r2_10d = 16
gamma_r2_10d = 1/len(reduced2)/10


# load blind data, filenames (the blind_data_load_example script should be run beforehand)
Xtest, filenames = load_Xtest_and_filenames(Xtest_save_path)

# standardize both X and Xtest
means, stds = standardize_X(X)
standardize_X(Xtest, means=means, stds=stds)


# create the model to test
model = BaggingClassifier(SVC(C=C_full, gamma=gamma_full), n_estimators=1000, max_features=1.0)

print("Using full feature set:")
print()
# fit the model on the training set using the specified features
model.fit(X[:, full], y)
# make predictions using the same features on the blind test set
y_pred = model.predict(Xtest[:, full])
# display the filenames in the blind set along with the prediction made by the model
for i in range(len(y_pred)):
	print(filenames[i] + ": %d" % y_pred[i])

model = BaggingClassifier(SVC(C=C_r1_d, gamma=gamma_r1_d), n_estimators=1000, max_features=1.0)
print()
print("Using reduced1 feature set, gamma = 1/d:")
print()
model.fit(X[:, reduced1], y)
y_pred = model.predict(Xtest[:, reduced1])
for i in range(len(y_pred)):
	print(filenames[i] + ": %d" % y_pred[i])

model = BaggingClassifier(SVC(C=C_r2_d, gamma=gamma_r2_d), n_estimators=1000, max_features=1.0)
print()
print("Using reduced2 feature set, gamma = 1/d:")
print()
model.fit(X[:, reduced2], y)
y_pred = model.predict(Xtest[:, reduced2])
for i in range(len(y_pred)):
	print(filenames[i] + ": %d" % y_pred[i])

model = BaggingClassifier(SVC(C=C_r1_10d, gamma=gamma_r1_10d), n_estimators=1000, max_features=1.0)
print()
print("Using reduced1 feature set, gamma = 1/10d:")
print()
model.fit(X[:, reduced1], y)
y_pred = model.predict(Xtest[:, reduced1])
for i in range(len(y_pred)):
	print(filenames[i] + ": %d" % y_pred[i])

model = BaggingClassifier(SVC(C=C_r2_10d, gamma=gamma_r2_10d), n_estimators=1000, max_features=1.0)
print()
print("Using reduced2 feature set, gamma = 1/10d:")
print()
model.fit(X[:, reduced2], y)
y_pred = model.predict(Xtest[:, reduced2])
for i in range(len(y_pred)):
	print(filenames[i] + ": %d" % y_pred[i])

