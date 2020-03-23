import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util import get_multiFrames

from scipy.optimize import curve_fit
from scipy.special import i0
from numpy import exp, sin, cos

class Subject:
    def __init__(self, dataFrame, RT_threshold=20, std_factors=3, polyfit_order=8, stimulus_maxID=147):
        self.data = dataFrame
        self.std_factors = std_factors
        self.RT_threshold = RT_threshold
        self.polyfit_order = polyfit_order
        self.stimulus_maxID = stimulus_maxID
        self.data['Error'] = [x - y for x,y in zip(self.data['stimulusID'],self.data['morphID'])]

        self.current_stimuliDiff = []

    def toLinear(self):
        for i in range(len(self.data['stimulusID'])):
            if abs(self.data.loc[i,'morphID'] - self.data.loc[i, 'stimulusID']) >= 80: ## threshold need to change accroding to different patterns
                if self.data.loc[i, 'stimulusID'] < self.stimulus_maxID / 2.0:
                    self.data.loc[i, 'stimulusID'] += self.stimulus_maxID
                else:
                    self.data.loc[i, 'stimulusID'] -= self.stimulus_maxID

    def fromLinear(self):
        for i in range(len(self.data['stimulusID'])):
            if self.data.loc[i, 'stimulusID'] <= 0:
                self.data.loc[i, 'stimulusID'] += self.stimulus_maxID
            elif self.data.loc[i, 'stimulusID'] > self.stimulus_maxID:
                self.data.loc[i, 'stimulusID'] -= self.stimulus_maxID
            else:
                continue

    #### CHANGED BY CG - removed 3std deviation removal since we do it earier now
    def polyCorrection(self):
        coefs = np.polyfit(self.data['stimulusID'], self.data['morphID'], self.polyfit_order) # polynomial coefs
        self.data['responseError'] = [y - polyFunc(x, coefs) for x,y in zip(self.data['stimulusID'],self.data['morphID'])]
        self.data['responseError'] = recenter(self.data['responseError'])

    def getnBack_diff(self, nBack):
        differencePrevious_stimulusID = []
        differencePrevious_stimulusLoc = []
        filtered_y = []
        filter_RT = []
        for i in range(len(self.data['stimulusID'])):
            if self.data.loc[i, 'trialNumber'] <= nBack:
                continue
            else:
                differencePrevious_stimulusID.append(self.data.loc[i-nBack, 'stimulusID'] - self.data.loc[i, 'stimulusID'])
                differencePrevious_stimulusLoc.append(self.data.loc[i-nBack, 'stimLocationDeg'] - self.data.loc[i, 'stimLocationDeg'])
                filtered_y.append(self.data.loc[i, 'responseError'])
                filter_RT.append(self.data.loc[i, 'RT'])

        differencePrevious_stimulusID = recenter(differencePrevious_stimulusID)
        differencePrevious_stimulusLoc = recenter(differencePrevious_stimulusLoc, threshold=180)
        self.current_stimuliDiff = differencePrevious_stimulusID

        return differencePrevious_stimulusID, differencePrevious_stimulusLoc, filtered_y, filter_RT

    def outlier_removal_RT(self):
        self.data = self.data[self.data['RT'] <= self.RT_threshold]
        self.data = self.data.reset_index()

    #### CHANGED BY CG- made this based on raw error before polynomial correction
    def outlier_removal_SD(self):
        error_mean = np.mean(self.data['Error'])
        error_std = np.std(self.data['Error'])
        self.data = self.data[self.data['Error'] <= error_mean + self.std_factors * error_std]
        self.data = self.data[self.data['Error'] >= error_mean - self.std_factors * error_std]
        self.data = self.data.reset_index()

    def save_RTfigure(self, filename):
        plt.figure()
        plt.rcParams["figure.figsize"] = (10,6)
        plt.rcParams.update({'font.size': 22})
        plt.title('stimulus ID & Reaction Time')
        plt.xlabel('stimulus ID')
        plt.ylabel('Reaction Time')
        plt.plot(self.data['stimulusID'], self.data['RT'], 'o', color ='orange', alpha=0.5, markersize=10)
        plt.savefig(filename, dpi=150)


    def save_SRfigure(self, filename):
        plt.figure()
        plt.rcParams["figure.figsize"] = (10,6)
        plt.rcParams.update({'font.size': 22})
        plt.title('Stimulus & Response')
        plt.xlabel('Stimulus ID')
        plt.ylabel('Morph Response')
        plt.axhline(y=75, linewidth=4, linestyle = "--", color='b', label = 'y = 75' )
        plt.plot(self.data['stimulusID'], self.data['stimulusID'], linewidth=4, linestyle = "-", color='g', label = 'x = y')
        plt.plot(self.data['stimulusID'], self.data['morphID'], 'mo', alpha=0.5, markersize=10)
        plt.savefig(filename, dpi=150)

    #### CHANGED BY CG - made a new function to graph and display the polyfit (based on save_SRfigure)
    def save_Polyfigure(self, filename):
        plt.figure()
        plt.rcParams["figure.figsize"] = (10,6)
        plt.rcParams.update({'font.size': 22})
        plt.title('Stimulus & Response')
        plt.xlabel('Stimulus ID')
        plt.ylabel('Morph Response')
        plt.axhline(y=75, linewidth=4, linestyle = "--", color='b', label = 'y = 75' )
        plt.plot(self.data['stimulusID'], self.data['stimulusID'], linewidth=4, linestyle = "-", color='g', label = 'x = y')
        plt.plot(self.data['stimulusID'], self.data['morphID'], 'mo', alpha=0.5, markersize=10)
        coefs = np.polyfit(self.data['stimulusID'], self.data['morphID'], self.polyfit_order) #### CHANGED BY CG- added
        xarray = np.array(range(-30, 170 + 1)) #### CHANGED BY CG -added
        PolyLine = np.polyval(coefs, xarray) #### CHANGED BY CG -added
        plt.plot(xarray, PolyLine, label = 'poly', color = 'c', linewidth = 3) #### CHANGED BY CG -added
        plt.savefig(filename, dpi=150)

    def save_Errorfigure(self, filename):
        plt.figure()
        plt.rcParams["figure.figsize"] = (10,6)
        plt.rcParams.update({'font.size': 22})
        plt.title('Stimulus &Error')
        plt.xlabel('Stimulus ID')
        plt.ylabel('Error')
        plt.xlim(-20, 170)
        plt.ylim(-60, 60)
        plt.axhline(y=0, linewidth=4, linestyle = "--", color='b', label = 'y = 0' )
        plt.plot(self.data['stimulusID'], self.data['Error'], 'mo', alpha=0.5, markersize=10)
        plt.savefig(filename, dpi=150)

    #### CHANGED BY CG - made 2 save_Errorfigures for before and after polyfit (I know this is the long way of doing it lol).
    def save_Errorfigure2(self, filename):
        plt.figure()
        plt.rcParams["figure.figsize"] = (10,6)
        plt.rcParams.update({'font.size': 22})
        plt.title('Stimulus & Response Error after Bias Removal')
        plt.xlabel('Stimulus ID')
        plt.ylabel('Response Error')
        plt.xlim(-20, 170)
        plt.ylim(-60, 60)
        plt.axhline(y=0, linewidth=4, linestyle = "--", color='b', label = 'y = 0' )
        plt.plot(self.data['stimulusID'], self.data['responseError'], 'mo', alpha=0.5, markersize=10)
        plt.savefig(filename, dpi=150)

    def Extract_currentCSV(self, nBack, fileName):
        ## FileName: SubjectName_nBack_outlierRemoveornot
        ## Delete rows
        output_data = self.data.copy(deep=True)

        for i in range(nBack):
            output_data = output_data[output_data['trialNumber'] != i + 1]
        # output_data = output_data.reset_index()

        output_data['Stim_diff'] = self.current_stimuliDiff
        del output_data['level_0']
        del output_data['index']
        del output_data['blockType']
        output_data.to_csv(fileName, index=False, header=True)




def vonmise_derivative(xdata, a = 25, kai = 4):
    xdata = xdata / 75 * np.pi
    return - a / (i0(kai) * 2 * np.pi) * exp(kai * cos(xdata)) * kai * sin(xdata) # Derivative of vonmise formula

def polyFunc(x, coeffs):
    y = 0
    order = len(coeffs)
    for i in range(order):
        y += coeffs[i] * (x ** (order - 1 - i))
    return y

def recenter(x, threshold=74):
    for i in range(len(x)):
        if x[i] > threshold - 1:
            x[i] = x[i] - 2 * threshold
        elif x[i] < -threshold:
            x[i] = x[i] + 2 * threshold
    return x

def getRunningMean(stimuxli_diff, filtered_responseError, halfway =74, step = 8):
    RM = [None] * (2 * halfway + 1); # running mean initialization
    xvals = list(range(-halfway, halfway + 1)) # index for running mean -90~90 + -90~90 (avoid error in sep[jj] == 91\92...
    allx_vals = xvals + xvals
    for ii in range(0,len(xvals) - 1): # start running mean calculation 0~180
        if ii - step // 2 >= 0:
            sep = allx_vals[(ii - step // 2) : (ii + step // 2 + 1)] # symmetric to avoid shift
        else:
            sep = allx_vals[(ii - step // 2) : len(allx_vals)] + allx_vals[0 : (ii + step // 2 + 1)]
        sep_sum = []
        for jj in range(0,len(sep)): # match every value in sep to every stimuli_diff point
            for kk in range(0, len(stimuxli_diff)):
                if stimuli_diff[kk] == sep[jj]:
                    sep_sum.insert(0, filtered_responseError[kk])
        RM[ii] = np.mean(sep_sum)
    RM[2 * halfway] = RM[0]
    return RM

def getRegressionLine(stimuxli_diff, filtered_responseError, peak_x):
    stimuxli_diff_filtered = []
    filtered_responseError_new = []
    for i in range(len(stimuxli_diff)):
        if stimuxli_diff[i] < peak_x + 1 or stimuxli_diff[i] > - peak_x + 1:
            stimuxli_diff_filtered.append(stimuxli_diff[i])
            filtered_responseError_new.append(filtered_responseError[i])
    coef = np.polyfit(stimuxli_diff_filtered,filtered_responseError_new,1)
    poly1d_fn = np.poly1d(coef)
    return poly1d_fn, coef


if __name__ == "__main__":
    ### Read data ###
    path = './' ## the folder path containing all experiment csv files
    data = get_multiFrames(path)

    nBack = 1
    outputCSV_name = 'test.csv'

    ### Initialize a subject ###
    subject = Subject(data)

    subject.save_RTfigure('ReactionTime.pdf')
    subject.outlier_removal_RT()
    subject.save_RTfigure('ReactionTime_OutlierRemoved.pdf')
    subject.save_SRfigure('RawData.pdf')

    ### Polynomial Correction ###
    subject.toLinear()
    subject.save_SRfigure('CorrectedData.pdf')
    subject.save_Errorfigure('RawError.pdf')#### CHANGED BY CG - wanted to see error before the removal
    subject.outlier_removal_SD() #### CHANGED BY CG - moved this line here
    subject.save_Errorfigure('ErrorResponse_OutlierRemoved.pdf') #### CHANGED BY CG - wanted to see error after SD removal
    subject.polyCorrection()
    subject.save_Polyfigure('PolyFit.pdf') #### CHANGED BY CG - made a new function to add in the poly line
    subject.fromLinear()
    subject.save_Errorfigure2('BiasRemoved.pdf') #### CHANGED BY CG - wanted to see distribution after polyfit

    ## Compute the stimulus difference ##
    stimuli_diff, loc_diff, filtered_responseError, filtered_RT = subject.getnBack_diff(nBack)
    subject.Extract_currentCSV(nBack, outputCSV_name)


    #### CHANGED BY CG - adapted running mean from old code
    #### RUNNING MEAN ####
    RM = getRunningMean(stimuxli_diff, filtered_responseError)

    ## Von Mise fitting: Shape Similarity##
    init_vals = [25, 4]
    best_vals, covar = curve_fit(vonmise_derivative, stimuli_diff, filtered_responseError, p0=init_vals)
    print('Von Mise Parameters: amplitude {0:.4f}, Kai {1:.4f}.'.format(best_vals[0],best_vals[1]))

    plt.figure()
    plt.title("Derivative Von Mises n Trials Back")
    plt.xlabel('Morph Difference from Previous')
    plt.ylabel('Error on Current Trial')
    plt.plot(stimuli_diff, filtered_responseError, 'co', alpha=0.5, markersize=10)
    x = np.linspace(-75, 75, 300)
    y = [vonmise_derivative(xi,best_vals[0],best_vals[1]) for xi in x]
    plt.plot(x, y, '-', linewidth = 4)
    plt.plot(xvals, RM, label = 'Running Mean', color = 'g', linewidth = 3)
    peak_x = (x[np.argmax(y)]) #### CHANGED BY CG - added to calculate where the peak is of the wave

    ### Regression Line - Needs to be restricted to the width of the peaks of the wave
    poly1d_fn, coef = getRegressionLine(stimuli_diff, filtered_responseError, peak_x)
    xdata = np.linspace(-peak_x, peak_x, 100)
    plt.plot(xdata, poly1d_fn(xdata), '--r', linewidth = 2)
    print(coef[0], coef[1])
    plt.savefig('ShapeDiff_DerivativeVonMises.pdf', dpi=150)

    print('Half Amplitude: {0:.4f}'.format(np.max(y)))
    print('Half Width: {0:.4f}'.format(x[np.argmax(y)]))


#### EVERYTHING AFTER THIS ISN'T SO IMPORTANT RIGHT NOW - CG ####
    ## Trials back and Reaction Time for Shape##
    plt.figure()
    plt.title("Trials Back and Reaction Time")
    plt.xlabel('Morph Difference from Previous')
    plt.ylabel('RT on Current Trial')
    plt.plot(stimuli_diff, filtered_RT, 'co', alpha=0.5, markersize=10)
    x = np.linspace(-75, 75, 300)
    plt.savefig('TrialsBack_RT_Shape.pdf', dpi=150)

     ## Von Mise fitting: Location Similarity##
    init_vals = [25, 4]
    best_vals, covar = curve_fit(vonmise_derivative, loc_diff, filtered_responseError, p0=init_vals)
    print('Von Mise Parameters: amplitude {0:.4f}, Kai {1:.4f}.'.format(best_vals[0],best_vals[1]))

    plt.figure()
    plt.title("Derivative Von Mises n Trials Back")
    plt.xlabel('Angle Location Difference from Previous')
    plt.ylabel('Error on Current Trial')
    plt.plot(loc_diff, filtered_responseError, 'co', alpha=0.5, markersize=10)
    x = np.linspace(-180, 180, 300)
    y = [vonmise_derivative(xi,best_vals[0],best_vals[1]) for xi in x]
    plt.plot(x, y, '-', linewidth = 4)
    plt.savefig('LocationDiff_DerivativeVonMises.pdf', dpi=150)

    print('Half Amplitude: {0:.4f}'.format(np.max(y)))
    print('Half Width: {0:.4f}'.format(x[np.argmax(y)]))

    ## Trials back and Reaction Time for Location##
    plt.figure()
    plt.title("Trials Back and Reaction Time")
    plt.xlabel('Location Difference from Previous')
    plt.ylabel('RT on Current Trial')
    plt.plot(loc_diff, filtered_RT, 'co', alpha=0.5, markersize=10)
    x = np.linspace(-180, 180, 300)
    plt.savefig('TrialsBack_RT_Location.pdf', dpi=150)
