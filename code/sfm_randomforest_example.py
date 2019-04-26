"""
This script does feature selection using the SelectFromModel class from
scikit-learn. This class fits a random forest classifier and then removes
features with importances less than a specified threshold. Tends to give 
very different results each time using the original full feature set.
"""
import scipy.io as sio
import numpy as np 
import os
import time
from eeg_sandbox import *
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# paths to load mutex training.validation data
featurized_training_path = './featurized_subjectwise_train1.pkl'
featurized_val_path = './featurized_subjectwise_val1.pkl'

# load data
concussed_val_data, control_val_data, _ = load_data(featurized_val_path)
concussed_train_data, control_train_data, _ = load_data(featurized_training_path)

# create X's and y's, standardize
Xtrain, ytrain = create_X_and_y(concussed_train_data, control_train_data)
Xval, yval = create_X_and_y(concussed_val_data, control_val_data)
means, stds = standardize_X(Xtrain)
standardize_X(Xval, means=means, stds=stds)

# model is Random Forest with 5000 trees
model = RandomForestClassifier(n_estimators=5000)
# run vanilla Random Forest (all features used)
model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtrain)
print("Training error (random forest, full feature set): %.3f" % np.mean(y_pred != ytrain))
y_pred = model.predict(Xval)
print("Validation error (random forest, full feature set): %.7f" % np.mean(y_pred != yval))
# use SelectFromModel to find "important" features
sfm = SelectFromModel(model, threshold=1e-2, prefit=True)
# remove unimportant features from training and validation sets
X_important_train = sfm.transform(Xtrain)
X_important_test = sfm.transform(Xval)
# create a separate model
model_important = RandomForestClassifier(n_estimators=5000)
# run this model, training only on "important features"
model_important.fit(X_important_train, ytrain)
y_pred = model_important.predict(X_important_train)
print("Training error (random forest, reduced feature set using sfm): %.3f" % np.mean(y_pred != ytrain))
y_pred = model_important.predict(X_important_test)
print("Validation error (random forest, reduced feature set using sfm): %.7f" % np.mean(y_pred != yval))

# show features used
mask = sfm.get_support()
chosen = [i for i in range(len(mask)) if mask[i]]
print(chosen)