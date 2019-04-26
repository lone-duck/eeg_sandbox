"""
This script creates sets of featurized training and validation data 
which are mutually exclusive at the subject level. That is, there is no 
case where data from a subseries of a particular subject is contained 
in both the training and validation set. This can be used to better 
approximates a model's performance on blind data.
"""

#import used libraries
import scipy.io as sio
import numpy as np 
import os
import time
from eeg_sandbox import *
import pickle as pkl
import random

# names of directories that contain or will store the data
# NOTE: these are relative paths... assumes the script is in the same directory as the data (for now)
concussed_path = '../10-10 Montage 27 channel data filtered ICA corrected and artifacts removed/Concusseddata_filteredICAcorrectedartifactsremoved'
control_path = '../10-10 Montage 27 channel data filtered ICA corrected and artifacts removed/Control data_training set_filteredICAcorrectedandartifactsremoved'
featurized_training_path = './featurized_subjectwise_train5.pkl'
featurized_val_path = './featurized_subjectwise_val5.pkl'

# subseries_length is the length (in seconds) of the subseries' we want to consider. 
subseries_length = 90

# this loads data from original 27 channel matlab struct files
concussed_data, control_data, sample_rate = load_raw_training_data(concussed_path, control_path)
# cuts off the start
concussed_data, control_data = truncate_data_end(concussed_data, control_data, 10, sample_rate)
# cuts off the end
concussed_data, control_data = truncate_data_start(concussed_data, control_data, 5, sample_rate)

# shuffle the data to create randomized mutex sets
r = random.SystemRandom()
r.shuffle(concussed_data)
r.shuffle(control_data)

# split into train and validation sets
concussed_val_data = concussed_data[0:4]
concussed_train_data = concussed_data[4:]
control_val_data = control_data[0:7]
control_train_data = control_data[7:]

#remove unimportant channels
get_important_channels(concussed_val_data)
get_important_channels(control_val_data)
get_important_channels(concussed_train_data)
get_important_channels(control_train_data)

# chops data into subseries of length subseries_length 
concussed_val_data = chop_data(concussed_val_data, subseries_length, sample_rate)
concussed_train_data = chop_data(concussed_train_data, subseries_length, sample_rate)
control_val_data = chop_data(control_val_data, subseries_length, sample_rate)
control_train_data = chop_data(control_train_data, subseries_length, sample_rate)

# bandpass filter chopped data
bandpass_filter_data(concussed_val_data, sample_rate)
bandpass_filter_data(control_val_data, sample_rate)
bandpass_filter_data(concussed_train_data, sample_rate)
bandpass_filter_data(control_train_data, sample_rate)

# featurize the data
featurize(concussed_val_data, sample_rate)
featurize(concussed_train_data, sample_rate)
featurize(control_val_data, sample_rate)
featurize(control_train_data, sample_rate)

# save featurized data
save_data(concussed_val_data, control_val_data, sample_rate, featurized_val_path)
save_data(concussed_train_data, control_train_data, sample_rate, featurized_training_path)


