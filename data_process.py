import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_multiFrames

from scipy.optimize import curve_fit
from scipy.special import i0
from numpy import exp, sin, cos

def toLinear(x, y):
    for i in range(len(x)):
        if abs(y[i] - x[i]) >= 80:
            if x[i] < 75:
                x[i] += 147
            else:
                x[i] -= 147
    return x, y

def fromLinear(x, y):
    for i in range(len(x)):
        if x[i] <= 0:
            x[i] += 147
        elif x[i] > 147: 
            x[i] -= 147
        else:
            continue
    return x, y

def recenter(x, y):
	for i in range(0,len(x)):
		if x[i] > 74 - 1:
			x[i] = x[i] - 2 * 74
		elif x[i] < -74:
			x[i] = x[i] + 2 * 74
		
        if y[i] > 74 - 1:
            y[i] = y[i] - 2 * 74
        elif y[i] < -74:
            y[i] = y[i] + 2 * 74
	return x, y

def polyFunc(x, coeffs):
	"""
	Compute the value of a polynomial function given its coefficients.

    Args:
        x (float)
        coefs (List[float]): coefficients of a polynomial function. Order from high to low.
	
	Returns:
		y (float): the polynomial function value.
    """
	y = 0
	for i in range(len(coeffs)):
		y += coeffs[i]*(x**(8-i))
	return y

def vonmise_derivative(xdata, a = 25, kai = 4, mu = 0):
    xdata = xdata / 75 * np.pi
    mu = mu / 75 * np.pi
    return - a / (i0(kai) * 2 * np.pi) * exp(kai * cos(xdata - mu)) * kai * sin(xdata - mu) # Derivative of vonmise formula

def polyCorrection(stimulusID, morphID, order=8):
	"""
	Apply polynomial fitting on the response to remove idiosyncratic bias

    Args:
        stimulusID (List[int]): stimulus IDs.
        morphID (List[int]): morph responses.
	
	Returns:
		responseError (List[int]): list of corrected response errors.
    """
	coefs = np.polyfit(stimulusID, morphID, order) # polynomial coefs
	responseError = [morphID - polyFunc(x, coefs) for x in stimulusID]

	return responseError

def getnBack_diff(trial_number, stimulusID, responseError, nBack):
	"""
	Compute the stimulus difference between different trials according the nBack param.

    Args:
        trial_number (List[int]): trail numbers of the experiment.
		stimulusID (List[int]): stimulus IDs.
        responseError (List[int]): list of corrected response errors.
		nBack: number of trials which need to trace back.
	
	Returns:
		diff_Stimulus (List[int]): the difference between given two trial stimuli.
		filtered_y (List[int]): the errors correspond to the differences
    """
	differencePrevious = []
	filtered_y = []
	for i in range(len(stimulusID)):
		if trial_number[i] <= nBack:
			continue
		else:
			differencePrevious.append(stimulusID[i-nBack] - stimulusID[i])
			filtered_y.append(responseError[i])

	return differencePrevious, filtered_y

def save_RTfigure(morphID, RTdata, filename):
	"""
	Save the figure for morph response and stumulus ID

    Args:
        morphID (List[int]): morph responses.
		RTdata (List[float]): corresponding reaction time.
        filename (str): the saved figure filename.
    """
	plt.figure()
	plt.rcParams["figure.figsize"] = (10,6)
	plt.rcParams.update({'font.size': 22})
	plt.title('Morph ID & Reaction Time')
	plt.xlabel('Morph ID')
	plt.ylabel('Reaction Time')
	plt.plot(morphID, RTdata, 'o', color ='orange', alpha=0.5, markersize=10)
	plt.savefig(filename, dpi=150)
	

def save_SRfigure(stimulusID, morphID, filename):
	"""
	Save the figure for morph response and stumulus ID

    Args:
        stimulusID (List[int]): stimulus IDs.
        morphID (List[int]): morph responses.
        filename (str): the saved figure filename.
    """
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

	### Reaction Time ### (outlier removal according to RT)
	save_RTfigure(data['morphID'], data['RT'], 'ReactionTime.pdf')

	### Stimulus and Response ###

	## Raw Data ## 
	save_SRfigure(data['stimulusID'], data['morphID'], 'RawData.pdf')	

	## Polynomial Correction ## (outlier removal according to 3 std)
	stimulusID, morphID = toLinear(data['stimulusID'], data['morphID']) ## Make data linear
	save_SRfigure(stimulusID, morphID, 'CorrectedData.pdf')
	responseError = polyCorrection(stimulusID, morphID, 8)
	responseError, stimulusID = fromLinear(stimulusID, responseError) ## Restore data structure

	## Compute the stimulus difference ##
	stimuli_diff, filtered_responseError = getnBack_diff(data['trialNumber'], stimulusID, responseError, 1)
	stimuli_diff, filtered_responseError = recenter(stimuli_diff, filtered_responseError)

	## Von Mise fitting ##
	init_vals = [10, 30, 0]
	best_vals, covar = curve_fit(vonmise_derivative, stimuli_diff, filtered_responseError, p0=init_vals)
	print('Von Mise Parameters: amplitude {0:.4f}, Kai {1:.4f}, mu {2:.4f}.'.format(best_vals[0],best_vals[1],best_vals[2]))

	plt.figure()
	plt.plot(stimuli_diff, filtered_responseError, 'co', alpha=0.5, markersize=10)
	x = np.linspace(-75, 75, 300)
	y = [vonmise_derivative(xi,best_vals[0],best_vals[1],best_vals[2]) for xi in x]
	plt.plot(x, y, '-', linewidth = 1)
	plt.savefig('FittingCurve.pdf', dpi=150)

	print('Half Amplitude: {0:.4f}'.format(np.max(y)))
	print('Half Width: {0:.4f}'.format(x[np.argmax(y)]))