import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_multiFrames

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

	### Reaction Time ###
	plt.figure()
	plt.rcParams["figure.figsize"] = (10,6)
	plt.rcParams.update({'font.size': 22})
	plt.title('Morph ID & Reaction Time')
	plt.xlabel('Morph ID')
	plt.ylabel('Reaction Time')
	plt.plot(data['morphID'], data['RT'], 'o', color ='orange', alpha=0.5, markersize=10)
	plt.savefig('ReactionTime.pdf', dpi=150)

	### Stimulus and Response ###

	## Raw Data ##
	save_SRfigure(data['stimulusID'], data['morphID'], 'RawData.pdf')

	## From circular to linear ##
	stimulusID, morphID = toLinear(plot_data['stimulusID'], plot_data['morphID'])
	save_SRfigure(stimulusID, morphID, 'CorrectedData.pdf')

	## Polynomial Fit ##
	