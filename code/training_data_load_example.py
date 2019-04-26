"""
This script loads and pre-processes raw data from .mat files, featurizes
the data, then saves both the featurized data and an (X,y) pair. This 
(X,y) pair can be used for experimentation and for training in the 
submitted classifier.
"""
#import used libraries
import scipy.io as sio
import numpy as np 
import os
import time
from eeg_sandbox import *
import pickle as pkl

# names of directories that contain the data
# NOTE: these are relative paths... assumes the script is in the same directory as the data (for now)
concussed_path = '../10-10 Montage 27 channel data filtered ICA corrected and artifacts removed/Concusseddata_filteredICAcorrectedartifactsremoved'
control_path = '../10-10 Montage 27 channel data filtered ICA corrected and artifacts removed/Control data_training set_filteredICAcorrectedandartifactsremoved'
X_save_path = './X_filtered_non_standardized.pkl'
featurized_path = './featurized.pkl'
# subseries_length is the length (in seconds) of the subseries' we want to consider. 
subseries_length = 90

# this loads data from original 27 channel matlab struct files
concussed_data, control_data, sample_rate = load_raw_training_data(concussed_path, control_path)
# cuts off the start
concussed_data, control_data = truncate_data_end(concussed_data, control_data, 10, sample_rate)
# cuts off the end
concussed_data, control_data = truncate_data_start(concussed_data, control_data, 5, sample_rate)
# chops data into subseries of length subseries_length 
concussed_data = chop_data(concussed_data, subseries_length, sample_rate)
control_data = chop_data(control_data, subseries_length, sample_rate)
# unimportant channels
get_important_channels(concussed_data)
get_important_channels(control_data)
# filter the chopped data
bandpass_filter_data(concussed_data, sample_rate)
bandpass_filter_data(control_data, sample_rate)
# featurize the data
featurize(concussed_data, sample_rate)
featurize(control_data, sample_rate)
# save featurized data
save_data(concussed_data, control_data, sample_rate, featurized_path)
# create X and y, shuffle, and save:
X, y = create_X_and_y(concussed_data, control_data)
X, y = shuffle_X_and_y(X,y)
save_X_and_y(X, y, X_save_path)