"""
This script loads and pre-processes blind data from .mat files and then saves it in a 
pickle file along with the associated filenames.
"""


#import used libraries
import scipy.io as sio
import numpy as np 
import os
from eeg_sandbox import *
import pickle as pkl

# path for blind data
test_data_save_path =  '../First Third Blind Data_90s for 10-10 Montage 27'
#test_data_save_path = '../Second third Blind Data_90s for 10-10 Montage 27'

# where to save
X_save_path = './X_blind_first_third.pkl'
#X_save_path = './X_blind_second_third.pkl'

# load testing data
test_data, filenames, sample_rate = load_blind_data(test_data_save_path)
# remove unimportant channels
get_important_channels(test_data)
# filter testing data
bandpass_filter_data(test_data, sample_rate)
# featurize testing data
featurize(test_data, sample_rate)
# create Xtest
Xtest = create_Xtest(test_data)
# save this Xtest
save_Xtest_and_filenames(Xtest, filenames, X_save_path)