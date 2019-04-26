import scipy.io as sio
import numpy as np 
import os
from functools import reduce
import catch22
import pickle as pkl
from scipy.signal import butter, sosfiltfilt, sosfreqz
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# suggested channels of particular importance
IMPORTANT_CHANNELS = [0, 2, 3, 5, 7, 8, 11, 15, 16, 19, 21, 23, 24, 26]

def load_raw_training_data(concussed_path, control_path):
	"""
    Loads data (i.e. amplitude time series) from all .mat files in the given paths.

    Parameters
    ----------
    concussed_path : string
        Relative path to directory of concussed data
    control_path : string
        Relative path to directory of control data

    Returns
    -------
    concussed_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length
    control_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length 
    sample_rate : int
    	Sample rate in Hz
    """

	# variable to hold raw data from matlab structs
	concussed_data = []
	control_data = []

	# load concussed data
	for filename in os.listdir(concussed_path):
		# load matlab struct
		mstruct = sio.loadmat(concussed_path + '/' + filename)
		# transpose matrix, change datatype
		concussed_data.append(np.transpose(mstruct['besa_channels_artRej']['amplitudes'][0][0].astype(float)))

	# load control data
	for filename in os.listdir(control_path):
		# load matlab struct
		mstruct = sio.loadmat(control_path + '/' + filename)
		# transpose matrix, change datatype
		control_data.append(np.transpose(mstruct['besa_channels_artRej']['amplitudes'][0][0].astype(float)))


	sample_rate = mstruct['besa_channels_artRej']['samplingrate'][0][0][0][0]

	return concussed_data, control_data, sample_rate

def load_blind_data(test_path):
	"""
	Loads data (i.e. amplitude time series) from all .mat files in the given paths.
    Specifically for use with the blind data directories.

    Parameters
    ----------
    test_path : string
    	Relative path to where the blind data is stored.

    Returns
    -------
    test_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length
    filenames : list(string)
    	The filenames of the .mat files the matrices were extracted from.
    sample_rate : int
    	The sample rate in Hz (assumed to be 250).
    """

	filenames = []
	test_data = []

	for filename in os.listdir(test_path):
		# keep name in list
		filenames.append(filename)
		# load matlab struct
		mstruct = sio.loadmat(test_path + '/' + filename)
		# transpose matrix, change datatype
		test_data.append(list(mstruct.values())[3][:, :-1].astype(float))

	sample_rate = 250 # assume this will always stay the same

	return test_data, filenames, sample_rate


def truncate_data_end(concussed_data, control_data, truncate_time, sample_rate):
	"""
	Given two lists of numpy arrays (containing multivariate time series), a truncate time, and a sample rate,
	removes truncate_time seconds from the end of each time series.

    Parameters
    ----------
    concussed_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length
    control_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length 
	truncate_time : int
    	Time (in seconds) to remove from the end of each time series
    sample_rate : int
    	Sample rate in Hz

    Returns
    -------
    concussed_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length (truncated)
    control_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length (truncated)
    """
	# compute number of samples to remove
	samples_to_remove = truncate_time * sample_rate
	# create lambda for truncation
	truncate = lambda m : m[:, 0:-samples_to_remove]
	# truncate samples
	concussed_data = list(map(truncate,concussed_data)) 
	control_data = list(map(truncate,control_data)) 
	
	return concussed_data, control_data

def truncate_data_start(concussed_data, control_data, truncate_time, sample_rate):
	"""
	Given two lists of numpy arrays (containing multivariate time series), a truncate time, and a sample rate,
	removes truncate_time seconds from the beginning of each time series.

    Parameters
    ----------
    concussed_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length
    control_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length 
	truncate_time : int
    	Time (in seconds) to remove from the end of each time series
    sample_rate : int
    	Sample rate in Hz

    Returns
    -------
    concussed_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length (truncated)
    control_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length (truncated)
    """

	# compute number of samples to remove
	samples_to_remove = truncate_time * sample_rate
	# create lambda for truncation
	truncate = lambda m : m[:, 0:-samples_to_remove]
	# truncate samples
	concussed_data = list(map(truncate,concussed_data)) 
	control_data = list(map(truncate,control_data)) 
	
	return concussed_data, control_data

def get_important_channels(data):
	"""
	Given a list of numpy arrays containing multivariate time series, removes channels deemed 
	"unimportant" from each matrix (in place). 

    Parameters
    ----------
    param_name : type
    	what is it?

    Returns
    -------
    param_name : type
    	what is it?
    """

    # for each matrix in the list, keep only the important channels
	for i in range(len(data)):
		data[i] = data[i][IMPORTANT_CHANNELS, :]


def bandpass_filter_data(data, sample_rate):
	"""
	Given a list of numpy arrays containing multivariate time series and a sample rate, applies a 4-40Hz bandpass 
	filter to each univariate time series (row) in the each data matrix (in place).

    Parameters
    ----------
    data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length
    sample_rate : int
    	Sample rate in Hz
    """

	lowcut = 4
	highcut = 40

	for matrix in data:
		for row in range(matrix.shape[0]):
			matrix[row, :] = butter_bandpass_filter(matrix[row, :], lowcut, highcut, sample_rate)


def butter_bandpass(lowcut, highcut, fs, order=5):
	"""
	For discussion of this function please see:
	https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
	%
	"""
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	sos = butter(order, [low, high], analog=False, btype='band', output='sos')
	return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	"""
	For discussion of this function please see:
	https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
	%
	"""
	sos = butter_bandpass(lowcut, highcut, fs, order=order)
	y = sosfiltfilt(sos, data)
	return y
    
def featurize(data, sample_rate):
	"""
	Given a list of numpy arrays containing multivariate time series and a sample rate, computes a number of "features"
	from each univariate time series (row) in the matrix and replaces the row with this list of features (in place). The 
	end result is that the list of multivariate time series matrices is replaced with a list of "feature matrices", where 
	position (i,j) contains the j-th scalar valued feature computed on the i-th channel in the EEG signal.

    Parameters
    ----------
    data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length
    sample_rate : int
    	Sample rate in Hz
    """

	# for each matrix in the list...
	tau = 5
	num_channels = data[0].shape[0]

	for i in range(len(data)):
		# create a list to temporarily hold features
		features = []
		j = 0

		# for each row in the matrix...
		for channel in data[i]:
			# this print statement can be removed, but lets you know that things are moving
			print("working on: " + str((i,j)))
			# compute a list of features using catch22 and append to the features variable
			features += catch22.catch22_all(channel.tolist())['values']
			features += mean_var_features(channel, tau, sample_rate)
			j += 1

		# convert the 2d list to a numpy array and assign to data[i]
		data[i] = np.array(features).reshape(num_channels, -1)

def mean_var_features(data, tau, sample_rate):
	"""
	Given a univariate time series, a length (in seconds) tau, and a sample rate, divides the original
	time series into subseries' of length tau, computes the means and variances of these subseries, and
	returns the means and variances of these means and variances.

    Parameters
    ----------
    data : numpy.array(float)
    	A univariate time series
    tau : int
    	The time divisions (in seconds) over which to compute the means and variances
    sample_rate : int
    	Sample rate in Hz

    Returns
    -------
    ret_val : list(float) 
    	Contains the mean of means, mean of variances, variance of means and variance of variances
    	for the subseries resulting from dividing the time series into sub-series of length tau.
    """

	length_in_samples = tau*sample_rate
	# split the matrices in submatrices of width length_in_samples
	subseries = np.split(data, len(data)/length_in_samples)
	# define mean, var lambdas
	fm = lambda v : np.mean(v) 
	fv = lambda v : np.var(v)
	# compute means and vars of all subseries'
	means = list(map(fm, subseries))
	variances = list(map(fv, subseries))
	# return means and vars of means and vars
	return [fm(means), fm(variances), fv(means), fv(variances)]


def chop_data(data, subseries_length, sample_rate):
	"""
	Given a list of matrices containing multivariate time series, a subseries length (in seconds), and a sample rate,
	splits each time series temporally into the maximum possible number of mutually exclusive multivariate time series'
	of length subseries_length.

    Parameters
    ----------
    data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length
    subseries_length : int
    	Length (in seconds) to chop time series' into
    sample_rate : int
    	Sample rate in Hz

    Returns
    -------
    chopped_data : list(numpy.ndarray(float))
    	The list of submatrices resulting from chopping the original matrices in time series of length subseries_length
    """

	# compute the number of samples needed for subseries_length in seconds
	length_in_samples = subseries_length*sample_rate
	# create a variable to hold the chopped matrices
	chopped_data = []
	num_channels = data[0].shape[0]

	# for each matrix/subject in the list...
	for i in range(len(data)):
		# compute the number of subseries available from series in this matrix
		num_subseries = data[i].shape[1]//length_in_samples	
		# split matrix, ignoring end samples which do not fit into a group of length_in_samples samples
		submatrices = np.hsplit(data[i][:, 0:num_subseries*length_in_samples], num_subseries)
		# place split matrices into chopped_data
		chopped_data += submatrices

	return chopped_data

def save_data(concussed_data, control_data, sample_rate, save_path):
	"""
	Saves the given data sets and their sample rate in a pickle file.

    Parameters
    ----------
    concussed_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length
    control_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length 
    sample_rate : int
    	Sample rate in Hz
    save_path : string
    	The relative path where the pickle file should be saved.

    """

	session_data = {
		"concussed_data" : concussed_data,
		"control_data" : control_data,
		"sample_rate" : sample_rate
	}

	with open(save_path, 'wb') as f:
		pkl.dump(session_data, f, pkl.HIGHEST_PROTOCOL)

def load_data(load_path):
	"""
	Loads concussed_data, control_data, and sample rate from a pickle file. The inverse operation of save_data.

    Parameters
    ----------
    load_path : string
    	The relative path where the data is stored. 

    Returns
    -------
    [concussed_data, control_data, sample_rate] : [list(numpy.ndarray(float)), list(numpy.ndarray(float)), int]
    	what is it?
    """

	with open(load_path, 'rb') as f:
		session_data = pkl.load(f)

	return [*session_data.values()]




def create_X_and_y(concussed_data, control_data):
	"""
	Given lists of featurized data matrices (one control and one concussed), flattens these matrices into rows
	and puts them together in a matrix. Corresponding entries in y are 1 if the matrix comes from a concussed subject,
	0 otherwise. 

    Parameters
    ----------
    concussed_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length
    control_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length 

    Returns
    -------
    X : numpy.ndarray(float)
    	A matrix where each row corresponds to a subject (or a submatrix from a particular subject) and 
    	each column to a particular feature computed on a particular channel.
    y : numpy.array(float)
    	A vector where the i-th entry contains a 1 if the i-th row in X corresponds to a concussed subject, 
    	and a 0 otherwise. 
    """

    # flatten every matrix in both lists
	f = lambda m : m.flatten()
	concussed_data = list(map(f, concussed_data))
	control_data = list(map(f, control_data))
	# concatenate lists of lists, turn these into a numpy matrix
	X = np.array(concussed_data + control_data)
	# create a corresponding vector with 1's for concussed, 0's for control
	y = np.array([1]*len(concussed_data) + [0]*len(control_data))
	return X,y


def create_Xtest(test_data):
	"""
	Given a list of featurized data matrices, flattens these matrices into rows and puts them together in a matrix. 
    Parameters
    ----------
    concussed_data : list(numpy.ndarray(float))
    	List of matrices of data for individual subjects, size channels by time series length

    Returns
    -------
	X : numpy.ndarray(float)
    	A matrix where each row corresponds to a subject (or a submatrix from a particular subject) and 
    	each column to a particular feature computed on a particular channel.
    """

    # flatten matrices in test_data list
	f = lambda m : m.flatten()
	test_data = list(map(f, test_data))
	# convert to numpy vector
	Xtest = np.array(test_data)
	return Xtest
	
def normalize_X(X, means=None):
	"""
	Normalizes X column-wise (i.e. each column has mean 0). If the optional means parameter is included, subtracts
	the column means from another matrix of the same width from X.

    Parameters
    ----------
    X : numpy.ndarray(float)
    	the matrix to be normalized
    means: numpy.array(float), default None
    	OPTIONAL: the column-wise means from another matrix of the same width. If included, these means
    	are subtracted from the columns of X. 

    Returns
    -------
    means : numpy.array(float)
    	The means subtracted from X (either its original column means or the means passed in).
    """

	if means is None:
		means = np.mean(X, axis=0)
	X -= means
	return means


def standardize_X(X, means=None, stds=None):
	"""
	Standardizes X column-wise (i.e. each column has mean 0 and standard deviation 1). If the optional means and stds 
	parameters are included, uses the column means/stddevs from another matrix of the same width from X.

    Parameters
    ----------
    X : numpy.ndarray(float)
    	the matrix to be normalized
    means: numpy.array(float), default None
    	OPTIONAL: the column-wise means from another matrix of the same width. If included, these means
    	are subtracted from the columns of X. 
    stds: numpy.array(float), default None
    	OPTIONAL: the column-wise std devs from another matrix of the same width. If included, these std devs
    	are used in place of X's std devs.

    Returns
    -------
    means : numpy.array(float)
    	The means subtracted from X (either its original column means or the means passed in).
    means : numpy.array(float)
    	The std devs used in the calculations (either X's original stds or the stds passed in).
    """

	if means is None:
		means = np.mean(X, axis=0)
		stds = np.std(X, axis=0)
	X -= means
	X /= stds
	return means, stds

def shuffle_X_and_y(X,y):
	"""
	Shuffle the rows of X and y simultaneously.
    Parameters
    ----------
    X : numpy.ndarray(float)
    	A data matrix
    y : numpy.array(float)
    	A target vector

    Returns
    -------
    X_shuffled : numpy.ndarray(float)
    	A row-wise shuffled data matrix
    y_shuffled : numpy.array(float)
    	A row-wise shuffled target vector
    """

	inds = np.random.permutation(len(X))
	return X[inds], y[inds]

def save_X_and_y(X,y, save_path):
	"""
	Saves the given X and y in a pickle file. 

    Parameters
    ----------
    X : numpy.ndarray(float)
    	A data matrix
    y : numpy.array(float)
    	A target vector  
    save_path : string
    	Relative path to where the data should be saved.  
    """

	session_data = {
		"X" : X,
		"y" : y
	}

	with open(save_path, 'wb') as f:
		pkl.dump(session_data, f, pkl.HIGHEST_PROTOCOL)

def load_X_and_y(load_path):
	"""
	Given a relative path, loads an X and y stored in a pickle files (inverse of save_X_and_y).
    Parameters
    ----------
    save_path : string
    	Relative path to where the data is saved.

    Returns
    -------
   [X, y] : [numpy.ndarray(float), numpy.array(float)]
    	A data matrix and target vector pair.
    """
	
	with open(load_path, 'rb') as f:
		session_data = pkl.load(f)
	
	return [*session_data.values()]

def save_Xtest_and_filenames(Xtest, filenames, save_path):
	"""
	Saves the given Xtest and associated filenames in a pickle file. 

    Parameters
    ----------
    X : numpy.ndarray(float)
    	A data matrix
    y : numpy.array(float)
    	A target vector  
    save_path : string
    	Relative path to where the data should be saved.  
    """

	session_data = {
		"Xtest" : Xtest,
		"filenames" : filenames
	}

	with open(save_path, 'wb') as f:
		pkl.dump(session_data, f, pkl.HIGHEST_PROTOCOL)

def load_Xtest_and_filenames(load_path):
	"""
	Given a relative path, loads an Xtest and associated filenames
	stored in a pickle file (inverse of save_X_and_y).
    
    Parameters
    ----------
    save_path : string
    	Relative path to where the data is saved.

    Returns
    -------
   [X, y] : [numpy.ndarray(float), numpy.array(float)]
    	A data matrix and target vector pair.
    """
	
	with open(load_path, 'rb') as f:
		session_data = pkl.load(f)
	
	return [*session_data.values()]


def k_fold_errors(X, y, model, k=4, ret_max=True):
	"""
	Given a data matrix X, target vector y, and a model, evaluates the model using 
	k-fold cross validation.  
    Parameters
    ----------
    X : numpy.ndarray(float)
    	A data matrix
    y : numpy.array(float)
    	A target vector  
    model : object
    	The model (classifier) to evaluate. Must have fit and predict functions implemented.
    k : int, default 4
    	The number of folds to use in cross validation.
    ret_max : bool, default True
    	If true, the maximum errors across folds is returned, otherwise the mean errors are returned.


    Returns
    -------
    (validation_error, training_error) : (float, float)
    	If ret_max = True, returns the maximum validation and training error across the four folds. 
    	Otherwise returns the mean validation and training error. 
    """

    # variables to hold validation and training errors
	val_errors = []
	train_errors = []
	# create StratifiedKFold object. Note this class attempts to have the same distribution of 
	# concussed/healthy entries in each fold as is found in the overall data set.
	kf = StratifiedKFold(n_splits=k, shuffle=False)
	# for each fold
	for train_index, test_index in kf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		model.fit(X_train, y_train)
		# compute training and validation errors
		y_pred = model.predict(X_train)
		train_errors.append(np.mean(y_pred != y_train))
		y_pred = model.predict(X_test)
		val_errors.append(np.mean(y_pred != y_test))

	# return the max or the mean
	if ret_max == True:
		return np.max(val_errors), np.max(train_errors)
	else:
		return np.mean(val_errors), np.mean(train_errors)  


def forward_selection(X, y, model, n_features_to_add, max_error=True, k=4, starting_features = None):
	"""
	Given a data matrix X, target vector y, and a model, uses the forward selection algorithm to 
	pick up to n_features_to_add features to add to the orginal (possibly empty) feature set. 

    Parameters
    ----------
    X : numpy.ndarray(float)
    	A data matrix
    y : numpy.array(float)
    	A target vector  
    model : object
    	The model (classifier) to evaluate features. Must have fit and predict functions implemented.
    n_features_to_add: int
    	The maximum number of features to add to the original (possibly empty) feature set.
    max_error : bool, default True
    	If true, the maximum errors across folds is used to evaluate feature performance (otherwise the mean).
    k : int, default 4
    	The number of folds to use in cross validation.
    starting_features : list(int), default None
    	A set of starting_features (derived from some other feature selection method) can be included. In this
    	case, the algorithm attempts to add features to this set which improve performance of the model in terms
    	of k-fold cross validation.


    Returns
    -------
    picked_features : list(int)
    	The features selected by the algorithm. If starting_features was included, picked_features will be a 
    	superset of these features.
    """

	n,d = X.shape
	errors = []
	# in the case where no features are passed...
	if starting_features is None:
		# the current best error is 100% (any feature does better than none)
		current_best_error = 1.0
		# no features in the picked set
		picked_features = set()
		# all features are candidates
		candidate_features = set(range(d))
	#in the case where features are passed...
	else:
		# the starting features are part of the feature set
		picked_features = set(starting_features)
		# and thus should not be included in the set of candidate features
		candidate_features = set(range(d)) - set(starting_features)
		# the current best error is the error attained with the starting features
		max_val_error, _ = k_fold_errors(X[:, starting_features], y, model, ret_max=max_error, k=k)
		current_best_error = max_val_error
		print("Starting error: %.3f" % current_best_error)
		print()

	for i in range(n_features_to_add):

		current_best_feature = None

		print("Iteration %d" % (i + 1))
		print()

		# for each candidate feature...
		for feature in candidate_features:
			
			# we want to evaluate the performance of picked_features + candidate
			temp_features = list(picked_features.union({feature}))
			# compute the k-fold error
			max_val_error, _ = k_fold_errors(X[:, temp_features], y, model, ret_max=max_error, k=k)
			# if we do better, this is the current best
			if max_val_error < current_best_error:
				current_best_error = max_val_error
				current_best_feature = feature
				print("Current best error: %.3f" % current_best_error)
				print("Current best feature: %d" % feature)

		# if no feature was picked, no feature improves performance. exit.
		if current_best_feature is None:
			print("TERMINATED EARLY.")
			picked_features = list(picked_features)
			picked_features.sort()
			return picked_features, errors
		# otherwise, add the current best to picked and remove from candidates
		else:
			picked_features.add(current_best_feature)
			candidate_features = candidate_features - {current_best_feature}
			errors.append(current_best_error)
			print()
			print("Added feature %d" % current_best_feature)
			print()
			print(picked_features)

	picked_features = list(picked_features)
	picked_features.sort()
	return picked_features, errors

def mutex_errors(mutex_data, features, model, ret_max):
	"""
	Given a list of mutually exclusive data sets, a feature set and a model, evaluates the mean or maximum
	training/validation error across the data sets. In this context, mutually exclusive refers to data which 
	was split into training/validation sets at the subject level, meaning that the classifier will not have 
	seen any data coming from a subject included in the validation set during training. 
    Parameters
    ----------
    mutex_data : list([numpy.ndarray(float), numpy.array(float), numpy.ndarray(float), numpy.array(float)])
		A list of mutually exclusive data sets, where each set is a list formatted 
		as [X_train, y_train, X_validation, y_validation].
	features : list(int)
		Column indices of the features to use.
	model : object
		The model (classifier) to evaluate features. Must have fit and predict functions implemented.
	ret_max : bool
		Determines whether the max or mean training/validation errors across sets is returned.
    Returns
    -------
    (validation_error, training_error) : (float, float)
    	The maximum or mean validation/training errors across the data sets.
    """

	val_errors = []
	train_errors = []
	# for each mutex set
	for data in mutex_data:
		# extract the training/test sets (using passed features only)
		Xtrain, Xval = data[0][:, features], data[2][:, features]
		ytrain, yval = data[1], data[3]
		# standardize these sets
		means, stds = standardize_X(Xtrain)
		standardize_X(Xval, means=means, stds=stds)
		# compute training and validation scores
		model.fit(Xtrain, ytrain)
		y_pred = model.predict(Xtrain)
		train_errors.append(np.mean(y_pred != ytrain))
		y_pred = model.predict(Xval)
		# the print statements below are useful if you want to inspect the nature of the errors
		#print("y_pred")
		#print(y_pred)
		#print("yval")
		#print(yval)
		val_errors.append(np.mean(y_pred != yval))

	# return max or mean or training and validation error
	if ret_max == True:
		return np.max(val_errors), np.max(train_errors)
	else:
		return np.mean(val_errors), np.mean(train_errors)










