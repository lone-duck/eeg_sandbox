"""
This script is used to compare a set of models using validation error
from mutex and non-mutex validation sets.
"""

# import necessary libraries 
import numpy as numpy
from eeg_sandbox import *
import pickle as pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

# paths for data used in the script
mutex_train_path_1 = './featurized_subjectwise_train1.pkl'
mutex_val_path_1 = './featurized_subjectwise_val1.pkl'
mutex_train_path_2 = './featurized_subjectwise_train2.pkl'
mutex_val_path_2 = './featurized_subjectwise_val2.pkl'
mutex_train_path_3 = './featurized_subjectwise_train3.pkl'
mutex_val_path_3 = './featurized_subjectwise_val3.pkl'
mutex_train_path_4 = './featurized_subjectwise_train4.pkl'
mutex_val_path_4 = './featurized_subjectwise_val4.pkl'
mutex_train_path_5 = './featurized_subjectwise_train5.pkl'
mutex_val_path_5 = './featurized_subjectwise_val5.pkl'
X_save_path = './X_filtered_non_standardized.pkl'


# load non-mutex data, shuffle, standardize
X, y = load_X_and_y(X_save_path)
n,d = X.shape
shuffle_X_and_y(X,y)
standardize_X(X)

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

# load mutex data, 3 sets for cross validation
concussed_val_data_1, control_val_data_1, _ = load_data(mutex_val_path_1)
concussed_train_data_1, control_train_data_1, _ = load_data(mutex_train_path_1)

concussed_val_data_2, control_val_data_2, _ = load_data(mutex_val_path_2)
concussed_train_data_2, control_train_data_2, _ = load_data(mutex_train_path_2)

concussed_val_data_3, control_val_data_3, _ = load_data(mutex_val_path_3)
concussed_train_data_3, control_train_data_3, _ = load_data(mutex_train_path_3)

concussed_val_data_4, control_val_data_4, _ = load_data(mutex_val_path_4)
concussed_train_data_4, control_train_data_4, _ = load_data(mutex_train_path_4)

concussed_val_data_5, control_val_data_5, _ = load_data(mutex_val_path_5)
concussed_train_data_5, control_train_data_5, _ = load_data(mutex_train_path_5)

# create X's and y's, consolidate into "data" variables
Xtrain_1, ytrain_1 = create_X_and_y(concussed_train_data_1, control_train_data_1)
Xval_1, yval_1 = create_X_and_y(concussed_val_data_1, control_val_data_1)
data1 = [Xtrain_1, ytrain_1, Xval_1, yval_1]

Xtrain_2, ytrain_2 = create_X_and_y(concussed_train_data_2, control_train_data_2)
Xval_2, yval_2 = create_X_and_y(concussed_val_data_2, control_val_data_2)
data2 = [Xtrain_2, ytrain_2, Xval_2, yval_2]

Xtrain_3, ytrain_3 = create_X_and_y(concussed_train_data_3, control_train_data_3)
Xval_3, yval_3 = create_X_and_y(concussed_val_data_3, control_val_data_3)
data3 = [Xtrain_3, ytrain_3, Xval_3, yval_3]

Xtrain_4, ytrain_4 = create_X_and_y(concussed_train_data_4, control_train_data_4)
Xval_4, yval_4 = create_X_and_y(concussed_val_data_4, control_val_data_4)
data4 = [Xtrain_4, ytrain_4, Xval_4, yval_4]

Xtrain_5, ytrain_5 = create_X_and_y(concussed_train_data_5, control_train_data_5)
Xval_5, yval_5 = create_X_and_y(concussed_val_data_5, control_val_data_5)
data5 = [Xtrain_5, ytrain_5, Xval_5, yval_5]

# put data variables into a list, this is the format accepted by the mutex_errors function
mutex_data = [data1, data2, data3, data4, data5]

# For each model to test, create the model and compute/display errors

#Random Forest Full Feature Set
model = RandomForestClassifier(n_estimators=5000)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Random Forest:")
print()
print("Full:")
print()
# compute errors and display. 
non_mutex_max_val_error, non_mutex_max_train_error = k_fold_errors(X[:, full], y, model, ret_max=True)
non_mutex_mean_val_error, non_mutex_mean_train_error = k_fold_errors(X[:, full], y, model, ret_max=False)
mutex_max_val_error, mutex_max_train_error = mutex_errors(mutex_data, full, model, ret_max=True)
mutex_mean_val_error, mutex_mean_train_error = mutex_errors(mutex_data, full, model, ret_max=False)
print("Non-mutex:")
print("Mean training error: %f" % non_mutex_mean_train_error)
print("Max training error: %f" % non_mutex_max_train_error)
print("Mean validation error: %f" % non_mutex_mean_val_error)
print("Max validation error: %f" % non_mutex_max_val_error)
print()
print("Mutex:")
print("Mean training error: %f" % mutex_mean_train_error)
print("Max training error: %f" % mutex_max_train_error)
print("Mean validation error: %f" % mutex_mean_val_error)
print("Max validation error: %f" % mutex_max_val_error)
print()
print()


#Random Forest w/ reduced1
print("reduced1:")
print()
# note a specific feature set can be used by indexing X using X[:, desired_features]
non_mutex_max_val_error, non_mutex_max_train_error = k_fold_errors(X[:, reduced1], y, model, ret_max=True)
non_mutex_mean_val_error, non_mutex_mean_train_error = k_fold_errors(X[:, reduced1], y, model, ret_max=False)
mutex_max_val_error, mutex_max_train_error = mutex_errors(mutex_data, reduced1, model, ret_max=True)
mutex_mean_val_error, mutex_mean_train_error = mutex_errors(mutex_data, reduced1, model, ret_max=False)
print("Non-mutex:")
print("Mean training error: %f" % non_mutex_mean_train_error)
print("Max training error: %f" % non_mutex_max_train_error)
print("Mean validation error: %f" % non_mutex_mean_val_error)
print("Max validation error: %f" % non_mutex_max_val_error)
print()
print("Mutex:")
print("Mean training error: %f" % mutex_mean_train_error)
print("Max training error: %f" % mutex_max_train_error)
print("Mean validation error: %f" % mutex_mean_val_error)
print("Max validation error: %f" % mutex_max_val_error)
print()
print()

#Random Forest w/ reduced2
print("reduced2:")
print()
non_mutex_max_val_error, non_mutex_max_train_error = k_fold_errors(X[:, reduced2], y, model, ret_max=True)
non_mutex_mean_val_error, non_mutex_mean_train_error = k_fold_errors(X[:, reduced2], y, model, ret_max=False)
mutex_max_val_error, mutex_max_train_error = mutex_errors(mutex_data, reduced2, model, ret_max=True)
mutex_mean_val_error, mutex_mean_train_error = mutex_errors(mutex_data, reduced2, model, ret_max=False)
print("Non-mutex:")
print("Mean training error: %f" % non_mutex_mean_train_error)
print("Max training error: %f" % non_mutex_max_train_error)
print("Mean validation error: %f" % non_mutex_mean_val_error)
print("Max validation error: %f" % non_mutex_max_val_error)
print()
print("Mutex:")
print("Mean training error: %f" % mutex_mean_train_error)
print("Max training error: %f" % mutex_max_train_error)
print("Mean validation error: %f" % mutex_mean_val_error)
print("Max validation error: %f" % mutex_max_val_error)
print()
print()

#BaggedSVM w/ Full Feature Set
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Bagged SVM:")
print()
print("Full:")
print()
model = BaggingClassifier(SVC(C=C_full, gamma=gamma_full), n_estimators=500, max_features=1.0)
non_mutex_max_val_error, non_mutex_max_train_error = k_fold_errors(X[:, full], y, model, ret_max=True)
non_mutex_mean_val_error, non_mutex_mean_train_error = k_fold_errors(X[:, full], y, model, ret_max=False)
mutex_max_val_error, mutex_max_train_error = mutex_errors(mutex_data, full, model, ret_max=True)
mutex_mean_val_error, mutex_mean_train_error = mutex_errors(mutex_data, full, model, ret_max=False)
print("Non-mutex:")
print("Mean training error: %f" % non_mutex_mean_train_error)
print("Max training error: %f" % non_mutex_max_train_error)
print("Mean validation error: %f" % non_mutex_mean_val_error)
print("Max validation error: %f" % non_mutex_max_val_error)
print()
print("Mutex:")
print("Mean training error: %f" % mutex_mean_train_error)
print("Max training error: %f" % mutex_max_train_error)
print("Mean validation error: %f" % mutex_mean_val_error)
print("Max validation error: %f" % mutex_max_val_error)
print()
print()

#BaggedSVM w/ reduced1
model = BaggingClassifier(SVC(C=C_r1_d, gamma=gamma_r1_d), n_estimators=1000, max_features=1.0)
print("reduced1, gamma = 1/d:")
print()
non_mutex_max_val_error, non_mutex_max_train_error = k_fold_errors(X[:, reduced1], y, model, ret_max=True)
non_mutex_mean_val_error, non_mutex_mean_train_error = k_fold_errors(X[:, reduced1], y, model, ret_max=False)
mutex_max_val_error, mutex_max_train_error = mutex_errors(mutex_data, reduced1, model, ret_max=True)
mutex_mean_val_error, mutex_mean_train_error = mutex_errors(mutex_data, reduced1, model, ret_max=False)
print("Non-mutex:")
print("Mean training error: %f" % non_mutex_mean_train_error)
print("Max training error: %f" % non_mutex_max_train_error)
print("Mean validation error: %f" % non_mutex_mean_val_error)
print("Max validation error: %f" % non_mutex_max_val_error)
print()
print("Mutex:")
print("Mean training error: %f" % mutex_mean_train_error)
print("Max training error: %f" % mutex_max_train_error)
print("Mean validation error: %f" % mutex_mean_val_error)
print("Max validation error: %f" % mutex_max_val_error)
print()
print()

#BaggedSVM w/ reduced2
model = BaggingClassifier(SVC(C=C_r2_d, gamma=gamma_r2_d), n_estimators=1000, max_features=1.0)

print("reduced2, gamma = 1/d:")
print()
non_mutex_max_val_error, non_mutex_max_train_error = k_fold_errors(X[:, reduced2], y, model, ret_max=True)
non_mutex_mean_val_error, non_mutex_mean_train_error = k_fold_errors(X[:, reduced2], y, model, ret_max=False)
mutex_max_val_error, mutex_max_train_error = mutex_errors(mutex_data, reduced2, model, ret_max=True)
mutex_mean_val_error, mutex_mean_train_error = mutex_errors(mutex_data, reduced2, model, ret_max=False)
print("Non-mutex:")
print("Mean training error: %f" % non_mutex_mean_train_error)
print("Max training error: %f" % non_mutex_max_train_error)
print("Mean validation error: %f" % non_mutex_mean_val_error)
print("Max validation error: %f" % non_mutex_max_val_error)
print()
print("Mutex:")
print("Mean training error: %f" % mutex_mean_train_error)
print("Max training error: %f" % mutex_max_train_error)
print("Mean validation error: %f" % mutex_mean_val_error)
print("Max validation error: %f" % mutex_max_val_error)
print()
print()

#BaggedSVM w/ reduced1
model = BaggingClassifier(SVC(C=C_r1_10d, gamma=gamma_r1_10d), n_estimators=1000, max_features=1.0)
print("reduced1, gamma = 1/10d:")
print()
non_mutex_max_val_error, non_mutex_max_train_error = k_fold_errors(X[:, reduced1], y, model, ret_max=True)
non_mutex_mean_val_error, non_mutex_mean_train_error = k_fold_errors(X[:, reduced1], y, model, ret_max=False)
mutex_max_val_error, mutex_max_train_error = mutex_errors(mutex_data, reduced1, model, ret_max=True)
mutex_mean_val_error, mutex_mean_train_error = mutex_errors(mutex_data, reduced1, model, ret_max=False)
print("Non-mutex:")
print("Mean training error: %f" % non_mutex_mean_train_error)
print("Max training error: %f" % non_mutex_max_train_error)
print("Mean validation error: %f" % non_mutex_mean_val_error)
print("Max validation error: %f" % non_mutex_max_val_error)
print()
print("Mutex:")
print("Mean training error: %f" % mutex_mean_train_error)
print("Max training error: %f" % mutex_max_train_error)
print("Mean validation error: %f" % mutex_mean_val_error)
print("Max validation error: %f" % mutex_max_val_error)
print()
print()

#BaggedSVM w/ reduced2
model = BaggingClassifier(SVC(C=C_r2_10d, gamma=gamma_r2_10d), n_estimators=1000, max_features=1.0)

print("reduced2, gamma = 1/10d:")
print()
non_mutex_max_val_error, non_mutex_max_train_error = k_fold_errors(X[:, reduced2], y, model, ret_max=True)
non_mutex_mean_val_error, non_mutex_mean_train_error = k_fold_errors(X[:, reduced2], y, model, ret_max=False)
mutex_max_val_error, mutex_max_train_error = mutex_errors(mutex_data, reduced2, model, ret_max=True)
mutex_mean_val_error, mutex_mean_train_error = mutex_errors(mutex_data, reduced2, model, ret_max=False)
print("Non-mutex:")
print("Mean training error: %f" % non_mutex_mean_train_error)
print("Max training error: %f" % non_mutex_max_train_error)
print("Mean validation error: %f" % non_mutex_mean_val_error)
print("Max validation error: %f" % non_mutex_max_val_error)
print()
print("Mutex:")
print("Mean training error: %f" % mutex_mean_train_error)
print("Max training error: %f" % mutex_max_train_error)
print("Mean validation error: %f" % mutex_mean_val_error)
print("Max validation error: %f" % mutex_max_val_error)
print()
print()





