import os
import numpy as np
import pandas as pd

def detect_csvFiles(path):
	"""
	Find csv files under this path. 

    Args:
        path (str): the folder path which contains the data files.

    Returns:
        filenameList (List[str]): list of filenames for experiment data.
    """

    files = os.listdir(path)
	files_csv = list(filter(lambda x: x[-4:] == '.csv' , files))
	filenameList = [path + temp for temp in files_csv]

    return filenameList

def read_Dataframe(path):
	"""
	Read data from a csv file

    Args:
        path (str): the csv file path.

    Returns:
        dataFrame (DataFrame): A DataFrame containing 'stimulusID', 'morphID', 'RT', 'trialNumber', 'blockNumber', 'blockType' of one experiment.
    """

    ### Read useful columns mentioned above ###
    data = pd.read_csv(path)[['stimulusID', 'morphID', 'RT', 'trialNumber', 'blockNumber', 'blockType']]

    ### Remove training blocks ###
    dataFrame = data[data['blockType'] != 'training']

    ### Delete useless columns ###
    dataFrame = dataFrame.reset_index()
	del dataFrame['index']

    return dataFrame

def get_multiFrames(path):
	"""
	Get data from multiple csv files in one folder

    Args:
        path (str): the folder path containing all csv files.

    Returns:
        dataFrames (List[DataFrame]): List of DataFrames from multiple experiments.
    """
    filenameList = detect_csvFiles(path)
    num_subjects = len(filenameList)

    data_list = []
    for file in filenameList:
    	temp = read_Dataframe(file)
    	data_list.append(temp)

    dataFrames = pd.concat(data_list)

    runs = list(dataFrames['blockNumber'])[-1] * subjects
	trials = list(dataFrames['trialNumber'])[-1]
	TotalTrial = runs * trials
	print('Experiment Summary:')
	print(subjects,'subject(s)\t', runs,'runs\t', trials,'trials/run\t', TotalTrial,'total trials')

    return dataFrames