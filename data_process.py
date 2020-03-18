import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_multiFrames

from scipy.optimize import curve_fit
from scipy.special import i0
from numpy import exp, sin, cos

def toLinear(data):
	new_data = data.copy(deep=True)
	for i in range(len(new_data['stimulusID'])):
		if abs(new_data.loc[i,'morphID'] - new_data.loc[i, 'stimulusID']) >= 80:
			if new_data.loc[i, 'stimulusID'] < 75:
				new_data.loc[i, 'stimulusID'] += 147
			else:
				new_data.loc[i, 'stimulusID'] -= 147
	return new_data

def fromLinear(data):
	new_data = data.copy(deep=True)
	for i in range(len(new_data['stimulusID'])):
		if new_data.loc[i, 'stimulusID'] <= 0:
			new_data.loc[i, 'stimulusID'] += 147
		elif new_data.loc[i, 'stimulusID'] > 147: 
			new_data.loc[i, 'stimulusID'] -= 147
		else:
			continue
	return new_data

def recenter(x):
	for i in range(len(x)):
		if x[i] > 74 - 1:
			x[i] = x[i] - 2 * 74
		elif x[i] < -74:
			x[i] = x[i] + 2 * 74
	return x

def polyFunc(x, coeffs):

	# Compute the value of a polynomial function given its coefficients.

    # Args:
    #     x (float)
    #     coefs (List[float]): coefficients of a polynomial function. Order from high to low.
	
	# Returns:
	# 	y (float): the polynomial function value.

	y = 0
	for i in range(len(coeffs)):
		y += coeffs[i]*(x**(8-i))
	return y

def vonmise_derivative(xdata, a = 25, kai = 4):
    xdata = xdata / 75 * np.pi
    return - a / (i0(kai) * 2 * np.pi) * exp(kai * cos(xdata)) * kai * sin(xdata) # Derivative of vonmise formula

def polyCorrection(stimulusID, morphID, order=8):

	# Apply polynomial fitting on the response to remove idiosyncratic bias

    # Args:
    #     stimulusID (List[int]): stimulus IDs.
    #     morphID (List[int]): morph responses.
	
	# Returns:
	# 	responseError (List[int]): list of corrected response errors.

	coefs = np.polyfit(stimulusID, morphID, order) # polynomial coefs
	responseError = [y - polyFunc(x, coefs) for x,y in zip(stimulusID,morphID)]

	return responseError

def getnBack_diff(data, nBack):

	# Compute the stimulus difference between different trials according the nBack param.

    # Args:
    #     trial_number (List[int]): trail numbers of the experiment.
	# 	stimulusID (List[int]): stimulus IDs.
    #     responseError (List[int]): list of corrected response errors.
	# 	nBack: number of trials which need to trace back.
	
	# Returns:
	# 	diff_Stimulus (List[int]): the difference between given two trial stimuli.
	# 	filtered_y (List[int]): the errors correspond to the differences

	differencePrevious = []
	filtered_y = []
	for i in range(len(data['stimulusID'])):
		if data.loc[i, 'trialNumber'] <= nBack:
			continue
		else:
			differencePrevious.append(data.loc[i-nBack, 'stimulusID'] - data.loc[i, 'stimulusID'])
			filtered_y.append(data.loc[i, 'responseError'])

	return differencePrevious, filtered_y

def outlier_removal_RT(data, threshold=20):

	# Remove outliers from dataframe according to reaction time.

    # Args:
    #     data (DataFrame): the original data.
	# 	threshold (float): the exclusion criteria. (seconds)
	
	# Returns:
	# 	new_data (DataFrame): the new dataFrame after outlier removal.

	new_data = data[data['RT'] <= threshold]

	return new_data

def outlier_removal_error(data, std_factors=3): ## check here

	# Remove outliers from dataframe according to error.

    # Args:
    #     data (DataFrame): the original data.
	# 	std_factors (float): # of std.
	
	# Returns:
	# 	new_data (DataFrame): the new dataFrame after outlier removal.

	new_data = data.copy(deep=True)
	error_mean = np.mean(new_data['responseError'])
	error_std = np.std(new_data['responseError'])
	new_data = new_data[new_data['responseError'] <= error_mean + std_factors * error_std]
	new_data = new_data[new_data['responseError'] >= error_mean - std_factors * error_std]

	return new_data

def save_RTfigure(morphID, RTdata, filename):

	# Save the figure for morph response and stumulus ID

    # Args:
    #     morphID (List[int]): morph responses.
	# 	RTdata (List[float]): corresponding reaction time.
    #     filename (str): the saved figure filename.

	plt.figure()
	plt.rcParams["figure.figsize"] = (10,6)
	plt.rcParams.update({'font.size': 22})
	plt.title('Morph ID & Reaction Time')
	plt.xlabel('Morph ID')
	plt.ylabel('Reaction Time')
	plt.plot(morphID, RTdata, 'o', color ='orange', alpha=0.5, markersize=10)
	plt.savefig(filename, dpi=150)
	

def save_SRfigure(stimulusID, morphID, filename):

	# Save the figure for morph response and stumulus ID

    # Args:
    #     stimulusID (List[int]): stimulus IDs.
    #     morphID (List[int]): morph responses.
    #     filename (str): the saved figure filename.

	plt.figure()
	plt.rcParams["figure.figsize"] = (10,6)
	plt.rcParams.update({'font.size': 22})
	plt.title('Stimulus & Response')
	plt.xlabel('Stimulus ID')
	plt.ylabel('Morph Response')
	plt.axhline(y=75, linewidth=4, linestyle = "--", color='b', label = 'y = 75' )
	plt.plot(stimulusID, stimulusID + 0, linewidth=4, linestyle = "-", color='g', label = 'x = y')
	plt.plot(stimulusID, morphID, 'mo', alpha=0.5, markersize=10)
	plt.savefig(filename, dpi=150)

if __name__ == '__main__':
	### Read data ###
	path = './' ## the folder path containing all experiment csv files
	data = get_multiFrames(path)

	### Reaction Time ###
	save_RTfigure(data['stimulusID'], data['RT'], 'ReactionTime.pdf')
	data = outlier_removal_RT(data)
	data = data.reset_index()

	### Stimulus and Response ###

	## Raw Data ## 
	save_SRfigure(data['stimulusID'], data['morphID'], 'RawData.pdf')	

	## Polynomial Correction ##
	Linear_data = toLinear(data) ## Make data linear
	save_SRfigure(Linear_data['stimulusID'], Linear_data['morphID'], 'CorrectedData.pdf')
	responseError = polyCorrection(Linear_data['stimulusID'], Linear_data['morphID'], 8)
	data = fromLinear(Linear_data) ## Restore data structure
	data['responseError'] = responseError
	# save_SRfigure(data['stimulusID'], responseError, 'Error.pdf')
	data = outlier_removal_error(data)
	data = data.reset_index()

	## Compute the stimulus difference ##
	stimuli_diff, filtered_responseError = getnBack_diff(data, 1)
	stimuli_diff = recenter(stimuli_diff)
	filtered_responseError = recenter(filtered_responseError)

	## Von Mise fitting ##
	init_vals = [25, 4]
	best_vals, covar = curve_fit(vonmise_derivative, stimuli_diff, filtered_responseError, p0=init_vals)
	print('Von Mise Parameters: amplitude {0:.4f}, Kai {1:.4f}.'.format(best_vals[0],best_vals[1]))

	plt.figure()
	plt.plot(stimuli_diff, filtered_responseError, 'co', alpha=0.5, markersize=10)
	x = np.linspace(-75, 75, 300)
	y = [vonmise_derivative(xi,best_vals[0],best_vals[1]) for xi in x]
	plt.plot(x, y, '-', linewidth = 1)
	plt.savefig('FittingCurve.pdf', dpi=150)

	print('Half Amplitude: {0:.4f}'.format(np.max(y)))
	print('Half Width: {0:.4f}'.format(x[np.argmax(y)]))